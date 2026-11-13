#pragma once
// OCR engine -- layout/region orchestration (Phase 6).
//
// DEVICE-SAFE header. This module owns:
//   1. PpDocLayoutExecutor: the small anchor-free layout detector graph (folded
//      conv backbone -> class head + box head) recorded from Phase 2 conv/CTC
//      operators, with head-map download for the host CPU decode.
//   2. RegionRecognizer: the per-region recognition interface, with two DISTINCT
//      implementations:
//        * PaddleOcrVlRecognizer: native-resolution NaViT vision + adaptive
//          visual-token reduction + MLP connector + ERNIE dense decoder
//          (the shared DenseVlmExecutor Phase 3 contract). Generative.
//        * OpenDocRecognizer: PP-DocLayoutV2 + UniRec CNN+CTC two-stage region
//          recognizer (the shared CnnExecutor recognizer graph). NOT a dense
//          VLM; no autoregressive KV decoding on this path.
//   3. LayoutVlmExecutor: the full overview -> layout -> crop -> region vision ->
//      decode -> page/document composition orchestration, with token-budgeted
//      region batching (RegionScheduler), detector-output caching, separate
//      overview/crop/output budgets, cancellation/time/memory checks, and a
//      documented CPU layout option.
//
// This header pulls ops_attention.h (via dense_vlm_executor.h) but NEVER
// api.h/model_recipe.h (host AttentionMaskKind clash). Host composition + the
// public RecognizeResult bridge live in the host-only *_result.* modules.
//
// Ownership follows runtime/executor.h: borrows installed immutable WeightStores;
// owns per-request DeviceWorkspaces + TypedShaderCaches. All launches go through
// the safe dispatch adapters; all host transfers use luisa::span. No exceptions /
// RTTI. SYNTHETIC ONLY: no official checkpoint is validated; real artifacts fail
// closed. See MODELS.md.
#include <cstdint>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/models/open_doc.h"
#include "ocr/models/paddle_ocr_vl.h"
#include "ocr/models/pp_doc_layout.h"
#include "ocr/ops/ops_conv.h"
#include "ocr/ops/ops_ctc.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_image.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/ops/ops_vision.h"
#include "ocr/runtime/cnn_executor.h"
#include "ocr/runtime/dense_vlm_executor.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// Layout detector executor
// ---------------------------------------------------------------------------

// One source page input to the layout detector (decoded HWC uint8 RGB).
struct LayoutPageInput {
    luisa::vector<uint8_t> pixels;
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t page_index{0};
};

struct PpDocLayoutExecutorConfig {
    PpDocLayoutConfig model;
    PpDocLayoutTensorNames names;
    luisa::string component{"layout"};
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    uint64_t max_source_pixels{4096u * 4096u};
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

// Head maps from one detector run (host copies): class_probs [num_classes,h,w]
// sigmoid scores; box_deltas [4,h,w].
struct LayoutHeadMaps {
    uint32_t map_width{0};
    uint32_t map_height{0};
    uint32_t num_classes{0};
    luisa::vector<float> class_probs;
    luisa::vector<float> box_deltas;
};

class PpDocLayoutExecutor final {
public:
    PpDocLayoutExecutor() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const PpDocLayoutExecutorConfig &config, const WeightStore &weights);
    [[nodiscard]] Status create_from_packages(compute::Device &device, compute::Stream &stream,
                                              const PpDocLayoutExecutorConfig &config, const WeightStore &weights,
                                              const PackageReader &layout_package);

    // Run the detector graph for one page and download the head maps. The
    // stream is driven internally (host decode needs the maps).
    [[nodiscard]] Result<LayoutHeadMaps> detect(compute::Stream &stream, const LayoutPageInput &page);

    [[nodiscard]] const PpDocLayoutExecutorConfig &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept { return _ws.capacity_bytes(); }

    // Layered-golden download helpers (last detect; caller drains afterwards).
    // These are intentional test APIs: they expose the uploaded preprocessed
    // pixels and the ping-pong backbone feature maps so a test can isolate
    // preprocessing from the conv chain (mirrors CnnExecutor::download_det_*).
    [[nodiscard]] Status download_feat_a(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_feat_b(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_page_pixels(compute::Stream &stream, luisa::vector<float> &out) const;

private:
    struct ConvWeights {
        compute::BufferView<float> weight;
        compute::BufferView<float> bias;
    };
    struct WsRegion {
        uint64_t offset{0};
        uint64_t bytes{0};
    };

    // Run the detector graph on ALREADY-normalized CHW pixels at the detector
    // input size (the shared graph body behind detect(); bypasses host
    // preprocessing). The caller drains.
    [[nodiscard]] Result<LayoutHeadMaps> detect_from_pixels(compute::Stream &stream,
                                                            luisa::span<const float> pixels_chw);

    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();
    [[nodiscard]] Status record_conv(compute::CommandList &list, const PpConvStage &stage,
                                     const ConvWeights &w, compute::BufferView<float> in,
                                     compute::BufferView<float> out, uint32_t n,
                                     uint32_t in_h, uint32_t in_w, uint32_t &out_h, uint32_t &out_w);
    [[nodiscard]] compute::BufferView<float> ws(const WsRegion &a) const noexcept {
        return _ws.buffer().view(static_cast<size_t>(a.offset), static_cast<size_t>(a.bytes)).as<float>();
    }

    PpDocLayoutExecutorConfig _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _ws;
    uint64_t _weight_bytes{0};
    uint32_t _map_h{0}, _map_w{0};
    WsRegion _page_pixels, _feat_a, _feat_b, _class_logits, _class_probs, _box_deltas, _dummy_f;
    luisa::vector<ConvWeights> _backbone;
    ConvWeights _class_head, _box_head;
    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<Conv2dKernel> _conv_cache;
    TypedShaderCache<CnnActKernel> _act_cache;
    TypedShaderCache<SigmoidKernel> _sigmoid_cache;
    const TypedShaderCache<Conv2dKernel>::ShaderT *_sh_conv{nullptr};
    const TypedShaderCache<CnnActKernel>::ShaderT *_sh_act[4]{nullptr, nullptr, nullptr, nullptr};
    const TypedShaderCache<SigmoidKernel>::ShaderT *_sh_sigmoid{nullptr};
};

// ---------------------------------------------------------------------------
// Region recognizer interface
// ---------------------------------------------------------------------------

// One region crop input: decoded HWC uint8 RGB pixels of the crop (source-crop
// size) plus the region task metadata. The recognizer packs/resizes internally.
struct RegionCropInput {
    luisa::vector<uint8_t> pixels;
    uint32_t source_width{0};
    uint32_t source_height{0};
    RegionTask task;
};

// Per-region recognition options (output budget + cancellation/time). The
// scheduler's batch caps are applied by the orchestrator, not here.
struct RegionRecognizeOptions {
 uint32_t max_output_tokens{2048};
 double max_wall_time_ms{0.0};
 // Phase 11 advanced opt-in; unsupported backends report an explicit fallback.
 bool enable_cuda_graph{false};

    luisa::function<bool()> cancel_callback;
};

// Abstract per-region recognizer. Implementations recognize a BATCH of region
// crops (the orchestrator packs them under the scheduler caps) and return one
// RawRegionOutput per input, in input order. Recognition is independent per
// region: no prefix sharing across unrelated crops.
class RegionRecognizer {
public:
    virtual ~RegionRecognizer() noexcept = default;
    [[nodiscard]] virtual Result<luisa::vector<RawRegionOutput>> recognize_batch(
        compute::Stream &stream, luisa::span<const RegionCropInput> crops,
        const RegionRecognizeOptions &options) = 0;
    [[nodiscard]] virtual uint64_t weight_bytes() const noexcept = 0;
    [[nodiscard]] virtual uint64_t workspace_bytes() const noexcept = 0;
    // Peak per-region KV-cache bytes allocated so far (high-water mark across
    // recognize_batch calls). 0 for a CTC-only recognizer (no KV cache). Used by
    // the orchestrator for honest peak/resident device-memory reporting.
    [[nodiscard]] virtual uint64_t kv_cache_peak_bytes() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// PaddleOCR-VL region recognizer (NaViT + reduction + connector + ERNIE)
// ---------------------------------------------------------------------------

struct PaddleOcrVlRecognizerConfig {
    PaddleOcrVlConfig model;
    PaddleOcrVlTensorNames names;
    luisa::string vision_component{"vision"};
    luisa::string decoder_component{"decoder"};
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    // Maximum TOTAL native patch rows the vision workspace is sized for.
    uint32_t max_vision_patches{4096};
    uint32_t prefill_chunk_rows{256};
    // Prompt token ids surrounding the visual tokens (synthetic contract). The
    // visual tokens are inserted between prefix and suffix.
    luisa::vector<uint32_t> prompt_prefix{1u};
    luisa::vector<uint32_t> prompt_suffix{2u};
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

class PaddleOcrVlRecognizer final : public RegionRecognizer {
public:
    PaddleOcrVlRecognizer() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const PaddleOcrVlRecognizerConfig &config, const WeightStore &weights);

    [[nodiscard]] Result<luisa::vector<RawRegionOutput>> recognize_batch(
        compute::Stream &stream, luisa::span<const RegionCropInput> crops,
        const RegionRecognizeOptions &options) override;

    [[nodiscard]] uint64_t weight_bytes() const noexcept override { return _weight_bytes; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept override { return _vision_ws.capacity_bytes() + _text.workspace_bytes(); }
 [[nodiscard]] uint64_t kv_cache_peak_bytes() const noexcept override { return _kv_peak_bytes; }
 [[nodiscard]] uint64_t cuda_graph_steps() const noexcept { return _cuda_graph_steps; }
 [[nodiscard]] uint64_t cuda_graph_updates() const noexcept { return _cuda_graph_updates; }
 [[nodiscard]] uint64_t cuda_graph_fallbacks() const noexcept { return _cuda_graph_fallbacks; }
 [[nodiscard]] const PaddleOcrVlRecognizerConfig &config() const noexcept { return _config; }


    // Download helpers for layered golden tests (last recognize_batch's LAST
    // region; caller drains afterwards). The workspace must not be reset.
    [[nodiscard]] Status download_vision_patches(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_connector_features(compute::Stream &stream, luisa::vector<float> &out) const;

private:
    struct VisionBlockWeights {
        compute::BufferView<float> norm1_w, norm1_b, norm2_w, norm2_b;
        compute::BufferView<float> q_w, k_w, v_w, o_w;
        compute::BufferView<float> q_b, k_b, v_b, o_b;
        compute::BufferView<float> fc1_w, fc1_b, fc2_w, fc2_b;
    };
    struct ConnectorWeights {
        compute::BufferView<float> ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b;
    };

    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();

    // Vision graph for ONE packed crop: patch embed + blocks -> [patches, vh].
    [[nodiscard]] Status record_vision(compute::CommandList &list, compute::BufferView<float> pixels,
                                       compute::BufferView<int32_t> positions, uint32_t rows);
    [[nodiscard]] Status record_vision_block(compute::CommandList &list, uint32_t layer,
                                             compute::BufferView<float> x,
                                             compute::BufferView<int32_t> positions, uint32_t rows);
    // Adaptive reduction + connector for ONE crop: [patches, vh] -> [tokens, th].
    [[nodiscard]] Status record_connector(compute::CommandList &list, compute::BufferView<float> x,
                                          uint32_t grid_h, uint32_t grid_w, uint32_t out_h, uint32_t out_w,
                                          compute::BufferView<float> out);
    [[nodiscard]] Status record_linear(compute::CommandList &list, compute::BufferView<float> A,
                                       compute::BufferView<float> B, compute::BufferView<float> C,
                                       compute::BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
                                       GemmEpilogue epilogue);
    [[nodiscard]] compute::BufferView<float> vws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_vision_ws.buffer());
    }

    PaddleOcrVlRecognizerConfig _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _vision_ws;
    DenseVlmExecutor _text;
    uint64_t _weight_bytes{0};
    uint64_t _kv_peak_bytes{0};
    uint64_t _cuda_graph_steps{0};
    uint64_t _cuda_graph_updates{0};
    uint64_t _cuda_graph_fallbacks{0};

    WorkspaceAllocation _v_pixels, _v_patches, _v_x, _v_norm, _v_q, _v_k, _v_v, _v_qr, _v_kr,
        _v_attn, _v_fc1, _v_fc2, _v_pooled, _v_cnorm, _v_cfc1, _v_features, _v_positions,
        _v_axis, _v_dummy_f, _v_dummy_u, _v_aux;
    uint32_t _last_patches{0};
    uint32_t _last_tokens{0};

    compute::BufferView<float> _patch_embed_w, _patch_embed_b;
    luisa::vector<VisionBlockWeights> _blocks;
    ConnectorWeights _connector;
    luisa::vector<uint32_t> _vision_axis_pending;

    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<ImagePatchifyKernel> _patchify_cache;
    TypedShaderCache<LayerNormKernel> _layer_norm_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<RopeAxialKernel> _rope2d_cache;
    TypedShaderCache<AttentionPrefillKernel> _attn_cache;
    TypedShaderCache<AdaptiveAvgPool2dKernel> _pool_cache;
    TypedShaderCache<ResidualAddKernel> _residual_cache;
    TypedShaderCache<SpatialMergeKernel> _copy_cache;

    const TypedShaderCache<ImagePatchifyKernel>::ShaderT *_sh_patchify{nullptr};
    const TypedShaderCache<LayerNormKernel>::ShaderT *_sh_layer_norm{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_none{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_gelu{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_residual{nullptr};
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope2d{nullptr};
    const TypedShaderCache<AttentionPrefillKernel>::ShaderT *_sh_attn{nullptr};
    const TypedShaderCache<AdaptiveAvgPool2dKernel>::ShaderT *_sh_pool{nullptr};
    const TypedShaderCache<ResidualAddKernel>::ShaderT *_sh_residual{nullptr};
    const TypedShaderCache<SpatialMergeKernel>::ShaderT *_sh_copy{nullptr};
};

// ---------------------------------------------------------------------------
// OpenDoc region recognizer (PP-DocLayoutV2 + UniRec CNN+CTC; NOT a dense VLM)
// ---------------------------------------------------------------------------

struct OpenDocRecognizerConfig {
    OpenDocConfig model;
    luisa::string recognizer_component{"unirec"};
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    output::CtcDecodeOptions ctc;
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

class OpenDocRecognizer final : public RegionRecognizer {
public:
    OpenDocRecognizer() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const OpenDocRecognizerConfig &config, const WeightStore &weights);

    [[nodiscard]] Result<luisa::vector<RawRegionOutput>> recognize_batch(
        compute::Stream &stream, luisa::span<const RegionCropInput> crops,
        const RegionRecognizeOptions &options) override;

    [[nodiscard]] uint64_t weight_bytes() const noexcept override { return _cnn.weight_bytes(); }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept override { return _cnn.workspace_bytes(); }
    // UniRec is CNN+CTC: no autoregressive KV cache on this path.
    [[nodiscard]] uint64_t kv_cache_peak_bytes() const noexcept override { return 0; }
    [[nodiscard]] const OpenDocRecognizerConfig &config() const noexcept { return _config; }

private:
    OpenDocRecognizerConfig _config;
    compute::Device *_device{nullptr};
    // The UniRec recognizer reuses the shared CnnExecutor recognizer graph. The
    // detector part of the CnnExecutor config is a minimal valid placeholder
    // (never run on this path); only recognize_line() is used.
    CnnExecutor _cnn;
};

// ---------------------------------------------------------------------------
// LayoutVlmExecutor orchestration
// ---------------------------------------------------------------------------

// Where the layout detector runs. `device` records the detector graph on the
// GPU. `cpu_reference` runs the layout through the thoroughly-tested CPU
// reference graph: the caller supplies precomputed head maps (from the
// independent host oracle, validated against the device path by the layout
// tests). This is a documented option when GPU residency is the limiting
// factor; the choice is explicit and reported in diagnostics. It is NEVER a
// silent fallback to a different model, and it never fabricates detections.
enum class LayoutDevicePolicy : uint32_t {
    device = 0,
    cpu_reference,
};

struct LayoutVlmExecutorConfig {
    LayoutDevicePolicy layout_policy{LayoutDevicePolicy::device};
    PpDocLayoutDecodeOptions decode;
    // The scheduler limits are the SINGLE source of truth for region batching
    // and the per-region/per-page output-token caps. The orchestrator forwards
    // scheduler.max_output_tokens_per_region to the recognizer and enforces
    // scheduler.max_output_tokens_per_page during accumulation.
    SchedulerLimits scheduler;
    // Overview pixel cap (the layout detector input is fixed; this caps the
    // source page size accepted for the overview pass).
    uint64_t max_overview_pixels{4096u * 4096u};
    uint64_t max_crop_pixels{1024u * 1024u};
    // Detector-output cache capacity (entries). 0 disables caching.
    uint32_t layout_cache_entries{64};

    [[nodiscard]] Status validate() const;
};

// Optional head-map provider for the cpu_reference layout policy. The caller
// supplies the detector head maps for a page (computed by the independent host
// oracle, which the layout tests validate against the device path). Returning
// a failing Status aborts the request; the orchestrator never fabricates maps.
using LayoutHeadMapProvider = luisa::function<Result<LayoutHeadMaps>(const LayoutPageInput &)>;

// The full layout/region orchestrator. Borrows a created layout detector (for
// the device policy) and a created region recognizer; owns the scheduler and
// detector cache. NOT thread-safe.
class LayoutVlmExecutor final {
public:
    LayoutVlmExecutor() noexcept = default;

    // `layout` may be null ONLY when layout_policy == cpu_reference AND a
    // head-map provider is set. `recognizer` must be created. `layout_config`
    // describes the layout geometry for the source transform and CPU decode.
    [[nodiscard]] Status create(const LayoutVlmExecutorConfig &config,
                                PpDocLayoutExecutor *layout,
                                const PpDocLayoutConfig &layout_config,
                                RegionRecognizer *recognizer,
                                LayoutHeadMapProvider cpu_head_maps = {});

    // Run the full pipeline over `pages`. Each page: overview/layout pass ->
    // regions (source coords) -> crops -> token-budgeted region batches ->
    // independent region recognition -> restore page/reading order. Detector
    // outputs are cached for recognition retry under a content/geometry key.
    [[nodiscard]] Result<LayoutRunResult> recognize(compute::Stream &stream,
                                                    luisa::span<const LayoutPageInput> pages);

    [[nodiscard]] const LayoutVlmExecutorConfig &config() const noexcept { return _config; }
    [[nodiscard]] LayoutCache &cache() noexcept { return _cache; }

private:
    // `region_id_base` is the document-global id offset for this page (the count
    // of regions detected on all previously processed pages, in input order).
    [[nodiscard]] Result<luisa::vector<RegionTask>> detect_regions(compute::Stream &stream,
                                                                   const LayoutPageInput &page,
                                                                   uint32_t region_id_base,
                                                                   uint64_t &cache_hits);
    [[nodiscard]] Result<RegionCropInput> make_crop(const LayoutPageInput &page, const RegionTask &task) const;

    LayoutVlmExecutorConfig _config;
    PpDocLayoutExecutor *_layout{nullptr};
    PpDocLayoutConfig _layout_config;
    RegionRecognizer *_recognizer{nullptr};
    LayoutHeadMapProvider _cpu_head_maps;
    RegionScheduler _scheduler{SchedulerLimits{}};
    LayoutCache _cache;
};

}// namespace luisa::ocr
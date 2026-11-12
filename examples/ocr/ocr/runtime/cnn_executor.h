#pragma once
// OCR engine -- conventional CNN OCR executor (Phase 5).
//
// A REAL executing detector -> line-crop -> recognizer -> CTC pipeline with
// SEPARATE stages:
//   1. preprocessing : host STRETCH-resize of the source page to the fixed
//                      detector input size (independent x/y scales, declared
//                      bilinear half-pixel convention, NO aspect preservation
//                      and NO padding), normalized CHW upload. This is a
//                      SYNTHETIC convention, not official PP-OCRv6
//                      preprocessing parity (see MODELS.md).
//   2. detector      : LCNetV4 backbone + RepLKFPN neck + DB head, producing a
//                      sigmoid probability map (device).
//   3. DB postprocess: CPU connected components / convex boxes / unclip /
//                      scoring -> source-coordinate polygons (output/db).
//   4. line crop     : rotated-rectangle polygon crop of each detected box
//                      from the preprocessed page, resized to the recognizer
//                      input (device).
//   5. recognizer    : LCNetV4 backbone + LightSVTR height-collapse + per-frame
//                      SVTR MLP + CTC linear head -> frame logits (device).
//   6. CTC decode    : log-softmax + argmax (device) then host collapse /
//                      character lookup / confidence (output/ctc_decode).
//   7. reading order : deterministic multi-column ordering + canonical join
//                      (output/reading_order).
//
// Ownership follows runtime/executor.h: borrows an installed immutable
// WeightStore; owns a per-request DeviceWorkspace + TypedShaderCaches. All
// launches go through the safe dispatch adapters; all host transfers use
// luisa::span. No exceptions / RTTI. Blank/no-text pages produce a valid EMPTY
// result (no lines, no hallucinated text, no fabricated confidence).
//
// DEVICE-SAFE HEADER: this header pulls only the model contract + ops +
// runtime (no api.h / output/types.h, whose host AttentionMaskKind clashes
// with the ops enum). The public RecognizeResult is assembled by a host-only
// bridge (runtime/cnn_executor_result.h/.cpp) from the plain CnnRunResult.
//
// SYNTHETIC ONLY: geometry/tensor names are the declared synthetic contract
// (models/pp_ocrv6.h). No official checkpoint is validated; real artifacts
// fail closed. See MODELS.md.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/models/pp_ocrv6.h"
#include "ocr/ops/ops_conv.h"
#include "ocr/ops/ops_ctc.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_image.h"
#include "ocr/output/ctc_decode.h"
#include "ocr/output/db_postprocess.h"
#include "ocr/output/reading_order.h"
#include "ocr/runtime/dispatch.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"

namespace luisa::ocr {

// One source page input: decoded HWC uint8 RGB pixels + source size + page
// index. The executor preprocesses on the host (declared convention).
struct CnnPageInput {
    luisa::vector<uint8_t> pixels;// HWC uint8, source size
    uint32_t source_width{0};
    uint32_t source_height{0};
    uint32_t page_index{0};
};

// Configuration for one CNN executor instance.
struct CnnExecutorConfig {
    PpOcrv6Config model;
    PpOcrv6TensorNames names;
    luisa::string detector_component{"detector"};
    luisa::string recognizer_component{"recognizer"};
    // Normalization (RGB, (pixel/255 - mean)/std). Synthetic contract values.
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    // DB postprocess options (source transform is computed per request).
    output::DbPostprocessOptions db;
    // CTC decode options (temperature 1.0 == declared reference).
    output::CtcDecodeOptions ctc;
    // Reading-order options.
    output::ReadingOrderOptions reading_order;
    // Character dictionary (one entry per non-blank class).
    output::CtcDictionary dictionary;
    // Resource limits.
    uint32_t max_lines_per_page{64};   // region cap (0 invalid)
    uint64_t max_source_pixels{4096u * 4096u};// per-page pixel cap (0 invalid)
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

// One recognized line in plain form (mapped to the public OcrLine by the
// host bridge). Polygon/box are in ORIGINAL source coordinates.
struct CnnLine {
    luisa::string text;
    uint32_t page_index{0};
    uint32_t reading_order{0};
    output::DbRect bounds;
    luisa::vector<output::DbPoint> polygon;
    bool has_confidence{false};
    float confidence{0.0f};
    // Number of CTC class tokens kept after collapse for this line (one per
    // dictionary entry emitted). This is a TOKEN count, not a byte count: a
    // CJK ideograph or emoji is ONE token but several UTF-8 bytes.
    uint32_t ctc_token_count{0};
};

// Per-page intermediate + final results (plain; device-free).
struct CnnPageResult {
    uint32_t page_index{0};
    uint32_t source_width{0};
    uint32_t source_height{0};
    // Detector probability map (host copy) + its size.
    uint32_t det_map_width{0};
    uint32_t det_map_height{0};
    luisa::vector<float> det_probabilities;
    // Detected boxes (source coordinates) after DB postprocess.
    luisa::vector<output::DbBox> boxes;
    // Recognized lines in reading order.
    luisa::vector<CnnLine> lines;
    // Canonical joined text.
    luisa::string text;
};

struct CnnRunResult {
    luisa::vector<CnnPageResult> pages;
    uint64_t weight_bytes{0};
    uint64_t workspace_bytes{0};
};

// The conventional CNN OCR executor. NOT thread-safe: one worker drives one
// executor at a time. Reusable across requests (workspace reset per request).
class CnnExecutor final {
public:
    CnnExecutor() noexcept = default;

    // Validate config, resolve every weight view, preflight + reserve the
    // workspace, compile shader variants. `device`/`weights` are borrowed for
    // the executor's lifetime. `stream` is used for the one-time constant
    // uploads and drained before returning.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const CnnExecutorConfig &config, const WeightStore &weights);

    // Package-gated create: additionally validates the ON-DISK tensor tables
    // of both component packages against the declared inventory BEFORE any
    // upload/bind is trusted (exact names/shapes, the on-disk `conv_oihw`
    // layout tag for conv weights and `kn` for linears, fp32-only, and no
    // unexpected tensors). The device-descriptor gate in resolve_weights
    // cannot see the on-disk tag (the runtime maps conv_oihw -> row_major),
    // so this is the active on-disk layout gate for the real load path.
    [[nodiscard]] Status create_from_packages(compute::Device &device, compute::Stream &stream,
                                              const CnnExecutorConfig &config, const WeightStore &weights,
                                              const PackageReader &detector_package,
                                              const PackageReader &recognizer_package);

    // Recognizer-only create: resolves/allocates ONLY the recognizer graph (no
    // detector weights or detector workspace). Used by the OpenDoc two-stage
    // recognizer, whose UniRec stage is a conventional CNN+CTC sequence model
    // applied per region via recognize_line() -- the detector stage is owned by
    // the layout pass, not here. Only recognize_line() is legal after this;
    // recognize() (the full detector->recognizer pipeline) fails closed.
    [[nodiscard]] Status create_recognizer_only(compute::Device &device, compute::Stream &stream,
                                                const CnnExecutorConfig &config, const WeightStore &weights);

    // Run the full pipeline over `pages`. Each page is processed
    // independently; results are returned in input order. The caller's stream
    // is driven internally (commit + synchronize at stage boundaries) because
    // DB postprocess and CTC decode need host data between device stages.
    [[nodiscard]] Result<CnnRunResult> recognize(compute::Stream &stream,
                                                 luisa::span<const CnnPageInput> pages);

    // Recognize ONE already-normalized line crop [3, input_height, input_width]
    // (CHW float) through the recognizer + CTC graph and return the greedy
    // decode. This is the per-region entry point reused by the OpenDoc two-stage
    // recognizer (UniRec is a conventional CNN+CTC sequence model, NOT a dense
    // VLM). The crop must already be normalized to the recognizer input size;
    // the caller owns cropping/resizing. The stream is driven internally. An
    // all-blank decode returns a valid EMPTY result (no hallucinated text).
    [[nodiscard]] Result<output::CtcDecodeResult> recognize_line(compute::Stream &stream,
                                                                 luisa::span<const float> crop_norm);

    [[nodiscard]] const CnnExecutorConfig &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept { return _ws.capacity_bytes(); }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }

    // Download helpers for layered golden tests. Each copies the LAST run's
    // stage output through `stream` (the caller drains afterwards). The
    // workspace must not have been reset since the run. `out` is resized.
    [[nodiscard]] Status download_detector_prob(compute::Stream &stream, luisa::vector<float> &out) const;
    // Debug stage downloads (layered isolation).
    [[nodiscard]] Status download_det_a(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_det_b(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_det_fused(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_det_replk(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_line_crop(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_recognizer_logits(compute::Stream &stream, luisa::vector<float> &out) const;
    // Geometry of the last run's detector map / recognizer frames.
    [[nodiscard]] uint32_t detector_map_height() const noexcept { return _det_map_h; }
    [[nodiscard]] uint32_t detector_map_width() const noexcept { return _det_map_w; }
    [[nodiscard]] uint32_t recognizer_frames() const noexcept { return _rec_frames; }

private:
    struct ConvWeights {
        compute::BufferView<float> weight;
        compute::BufferView<float> bias;
    };

    // Workspace regions (fixed byte offsets into the arena, captured ONCE at
    // create; the per-request reset rewinds the cursor but the regions keep
    // their offsets, so no re-allocation happens inside recognize()).
    struct WsRegion {
        uint64_t offset{0};
        uint64_t bytes{0};
    };

    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();
    [[nodiscard]] Status create_impl(compute::Device &device, compute::Stream &stream,
                                     const CnnExecutorConfig &config, const WeightStore &weights);

    // Run one conv stage (folded Conv+BN) + activation. in/out are workspace
    // views; geometry from the stage. Returns the output spatial size.
    [[nodiscard]] Status record_conv(compute::CommandList &list, const PpConvStage &stage,
                                     const ConvWeights &w, compute::BufferView<float> in,
                                     compute::BufferView<float> out, uint32_t n,
                                     uint32_t in_h, uint32_t in_w,
                                     uint32_t &out_h, uint32_t &out_w);

    // Detector graph for one preprocessed page. Writes the probability map
    // into _det_prob view; returns the map size.
    [[nodiscard]] Status record_detector(compute::CommandList &list, compute::BufferView<float> pixels,
                                         uint32_t &map_h, uint32_t &map_w);

    // Recognizer graph for one line crop. Writes frame logits [frames,classes]
    // into _rec_logits view; returns the frame count.
    [[nodiscard]] Status record_recognizer(compute::CommandList &list, compute::BufferView<float> crop,
                                           uint32_t &frames);

    [[nodiscard]] compute::BufferView<float> ws(const WsRegion &a) const noexcept {
        return _ws.buffer().view(static_cast<size_t>(a.offset), static_cast<size_t>(a.bytes)).as<float>();
    }
    [[nodiscard]] compute::BufferView<int32_t> ws_i32(const WsRegion &a) const noexcept {
        return _ws.buffer().view(static_cast<size_t>(a.offset), static_cast<size_t>(a.bytes)).as<int32_t>();
    }

    CnnExecutorConfig _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _ws;
    uint64_t _weight_bytes{0};
    // Recognizer-only mode (OpenDoc UniRec): no detector weights/workspace; only
    // recognize_line() is legal. Set by create_recognizer_only().
    bool _recognizer_only{false};

    WsRegion _page_pixels, _det_a, _det_b, _det_lat_ws, _det_up, _det_fused,
        _det_replk_ws, _det_prob, _crop_norm, _crop, _rec_a, _rec_b, _rec_seq, _rec_svtr1,
        _rec_svtr2, _rec_logits, _rec_logprob, _rec_ids, _rec_probs, _svtr_aux, _dummy_f;

    // One-time normalization constants (immutable after create).
    compute::Buffer<float> _mean_buf, _std_buf;

    // Weight views.
    luisa::vector<ConvWeights> _det_backbone;
    ConvWeights _det_lateral, _det_replk, _det_db;
    luisa::vector<ConvWeights> _rec_backbone;
    ConvWeights _rec_svtr_fc1, _rec_svtr_fc2, _rec_ctc;

    // Geometry caches (from the validated config).
    uint32_t _det_map_h{0}, _det_map_w{0};
    uint32_t _rec_feat_h{0}, _rec_feat_w{0}, _rec_collapsed{0}, _rec_frames{0};

    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<Conv2dKernel> _conv_cache;
    TypedShaderCache<CnnActKernel> _act_cache;
    TypedShaderCache<FpnResizeKernel> _fpn_cache;
    TypedShaderCache<ConcatChannelsKernel> _concat_cache;
    TypedShaderCache<SigmoidKernel> _sigmoid_cache;
    TypedShaderCache<PolygonCropKernel> _crop_cache;
    TypedShaderCache<ImageResizeNormalizeKernel> _resize_cache;
    TypedShaderCache<HeightCollapseKernel> _collapse_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<CtcLogSoftmaxKernel> _logsoftmax_cache;
    TypedShaderCache<CtcArgmaxProbsKernel> _argmax_cache;

    const TypedShaderCache<Conv2dKernel>::ShaderT *_sh_conv{nullptr};
    const TypedShaderCache<CnnActKernel>::ShaderT *_sh_act[4]{nullptr, nullptr, nullptr, nullptr};
    const TypedShaderCache<FpnResizeKernel>::ShaderT *_sh_fpn{nullptr};
    const TypedShaderCache<ConcatChannelsKernel>::ShaderT *_sh_concat{nullptr};
    const TypedShaderCache<SigmoidKernel>::ShaderT *_sh_sigmoid{nullptr};
    const TypedShaderCache<PolygonCropKernel>::ShaderT *_sh_crop{nullptr};
    const TypedShaderCache<ImageResizeNormalizeKernel>::ShaderT *_sh_resize{nullptr};
    const TypedShaderCache<HeightCollapseKernel>::ShaderT *_sh_collapse{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_gelu{nullptr};
    const TypedShaderCache<CtcLogSoftmaxKernel>::ShaderT *_sh_logsoftmax{nullptr};
    const TypedShaderCache<CtcArgmaxProbsKernel>::ShaderT *_sh_argmax{nullptr};
};

}// namespace luisa::ocr
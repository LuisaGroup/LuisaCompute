#pragma once
// OCR engine -- shared Phase 7 vertical-slice executor.
//
// ONE configurable graph shared by the Phase 7 custom dense/layout variants
// (TeleOCR, GLM-OCR, MinerU, HunyuanOCR). The architecture distinctions are
// EXPLICIT configuration (p7_slice_config.h), never inferred from a model_type
// string:
//   * vision tower: patch embed (+optional learned position), pre-LN blocks,
//     optional QK-Norm, optional 2D RoPE, bidirectional or window/global
//     custom-block-mask attention, GELU-erf or SiLU MLP;
//   * connector: plain MLP / spatial-merge MLP / 2x2 pixel-unshuffle MLP /
//     adaptive-pool MLP / strided-downsample MLP (each a DISTINCT path);
//   * text decoder: the shared DenseVlmExecutor with the model's explicit
//     contract (QKV bias / QK-Norm / 1D / adjacent-MRoPE / interleaved-MRoPE /
//     XD-RoPE / partial rotary).
//
// Each model module (models/tele_ocr.*, glm_ocr.*, mineru.*, hunyuan_ocr.*)
// owns its identity/tensor-name/output contract and instantiates this graph.
// This executor is NOT a model and makes no support claim.
//
// Ownership follows runtime/executor.h: borrows an installed WeightStore, owns
// a per-request DeviceWorkspace + TypedShaderCaches + the embedded text
// executor. All launches go through the safe dispatch adapters; all host
// transfers use luisa::span. No exceptions / RTTI. SYNTHETIC ONLY.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/ops/ops_attention.h"
#include "ocr/ops/ops_conv.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_image.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/ops/ops_vision.h"
#include "ocr/runtime/dense_vlm_executor.h"
#include "ocr/runtime/layout_vlm_executor.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/scheduler.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"

namespace luisa::ocr {

// One image input (P7ImageInput) is the device-safe plain data type in
// p7_slice_config.h, shared with the host preprocessing bridge.

// Vision graph output for one request (device views into the executor-owned
// vision workspace; valid until the next vision run/reset).
struct P7VisionOutput {
    compute::BufferView<float> features;// [total_tokens, text_hidden]
    uint32_t total_tokens{0};
};

// The shared Phase 7 slice executor. NOT thread-safe.
class P7VlmExecutor final {
public:
    P7VlmExecutor() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const P7SliceConfig &config, const P7TensorNames &names,
                                const WeightStore &weights);

    // Additive Phase 9: vision-only create. Runs the SAME vision tower +
    // connector setup as create() but does NOT instantiate the embedded dense
    // text executor. Used by the hybrid GDN model slices (OvisOCR2 / Surya),
    // which pair this vision graph with a separate HybridGdnExecutor for the
    // text decode. Only record_vision()/build_decode_request() are legal after
    // this; text_executor() is not created. The vision path is unchanged.
    [[nodiscard]] Status create_vision_only(compute::Device &device, compute::Stream &stream,
                                            const P7SliceConfig &config, const P7TensorNames &names,
                                            const WeightStore &weights);

    // Record the vision graph for `images` (ragged: each image is processed
    // independently; attention never crosses image boundaries). Host
    // preprocessing through the Phase 1 reference; device patchify + blocks +
    // connector. `list` is caller-owned; the caller commits/drains.
    [[nodiscard]] Result<P7VisionOutput> record_vision(compute::CommandList &list,
                                                       luisa::span<const P7ImageInput> images);

    // Build the multimodal DecodeRequest + position ids for a prompt whose
    // image rows are filled by `vision` (SAME images, in order). At most ONE
    // image per prompt in this slice (the shared DecodeRequest carries a
    // single multimodal range); multi-image prompts fail closed.
    [[nodiscard]] Result<DecodeRequest> build_decode_request(
        luisa::span<const uint32_t> token_ids, uint32_t mm_start, uint32_t mm_count,
        const P7VisionOutput &vision, luisa::span<const P7ImageInput> images,
        luisa::vector<int32_t> &position_ids_out, luisa::vector<int32_t> &next_position_out) const;

    [[nodiscard]] DenseVlmExecutor &text_executor() noexcept { return _text; }
    [[nodiscard]] const DenseVlmExecutor &text_executor() const noexcept { return _text; }
    [[nodiscard]] const P7SliceConfig &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t vision_workspace_bytes() const noexcept { return _vision_ws.capacity_bytes(); }
    [[nodiscard]] uint64_t decoder_workspace_bytes() const noexcept { return _text.workspace_bytes(); }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }

    void reset_vision_workspace() noexcept { _vision_ws.reset(); }

    // Layered-golden download helpers (last record_vision; caller drains).
    [[nodiscard]] Status download_vision_patches(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_vision_final(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_vision_features(compute::Stream &stream, luisa::vector<float> &out) const;
    // Connector reduction output [total_tokens, cin] (the LayerNorm/MLP input).
    // cin depends on the connector kind (vh, vh*m*m, or vh*4).
    [[nodiscard]] Status download_vision_reduced(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] uint32_t connector_input_dim() const noexcept;

private:
    struct VisionBlockWeights {
        compute::BufferView<float> norm1_w, norm1_b, norm2_w, norm2_b;
        compute::BufferView<float> q_w, k_w, v_w, o_w;
        compute::BufferView<float> q_b, k_b, v_b, o_b;
        compute::BufferView<float> q_norm, k_norm;
        compute::BufferView<float> fc1_w, fc1_b, fc2_w, fc2_b;
        // Additive Phase 8: decomposed relative-position bias table (invalid
        // view when rel_pos_window == 0).
        compute::BufferView<float> rel_pos_bias;
    };
    struct ConnectorWeights {
        compute::BufferView<float> ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b;
        compute::BufferView<float> down_w, down_b;// downsample_mlp only
    };

    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();
    // Additive Phase 9: shared create body; with_text=false skips the embedded
    // dense text executor (vision-only mode for the hybrid GDN model slices).
    [[nodiscard]] Status create_impl(compute::Device &device, compute::Stream &stream,
                                     const P7SliceConfig &config, const P7TensorNames &names,
                                     const WeightStore &weights, bool with_text);

    [[nodiscard]] Status record_vision_block(compute::CommandList &list, uint32_t layer,
                                             compute::BufferView<float> x,
                                             compute::BufferView<int32_t> positions, uint32_t rows,
                                             const P7PackedImage &packed);
    // Connector for ONE image: [patch_rows, vh] -> [out_tokens, text_hidden].
    [[nodiscard]] Status record_connector(compute::CommandList &list, compute::BufferView<float> x,
                                          const P7PackedImage &packed, compute::BufferView<float> out);
    [[nodiscard]] Status record_linear(compute::CommandList &list, compute::BufferView<float> A,
                                       compute::BufferView<float> B, compute::BufferView<float> C,
                                       compute::BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
                                       GemmEpilogue epilogue);
    [[nodiscard]] Status copy_rows(compute::CommandList &list, compute::BufferView<float> src,
                                   compute::BufferView<float> dst, uint32_t rows, uint32_t hidden);
    [[nodiscard]] compute::BufferView<float> vws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_vision_ws.buffer());
    }

    P7SliceConfig _config;
    P7TensorNames _names;
    compute::Device *_device{nullptr};
    DeviceWorkspace _vision_ws;
    DenseVlmExecutor _text;
    uint64_t _weight_bytes{0};

    WorkspaceAllocation _v_pixels, _v_patches, _v_x, _v_norm, _v_q, _v_k, _v_v, _v_qn, _v_kn,
        _v_qr, _v_kr, _v_attn, _v_fc1, _v_reduced, _v_cnorm, _v_cfc1, _v_features,
        _v_positions, _v_axis, _v_block_mask, _v_dummy_f, _v_dummy_u, _v_aux, _v_unshuffle,
        _v_merged, _v_rel_bias;
    uint32_t _last_total_patches{0};
    uint32_t _last_total_tokens{0};

    compute::BufferView<float> _patch_embed_w, _patch_embed_b, _position_embed;
    luisa::vector<VisionBlockWeights> _blocks;
    // Additive Phase 8: host copies of the immutable per-layer relative-position
    // bias tables, downloaded ONCE at create (the create-time stream is drained
    // there). record_vision expands them per image grid without device access.
    luisa::vector<luisa::vector<float>> _rel_pos_tables;
    ConnectorWeights _connector;
    luisa::vector<uint32_t> _vision_axis_pending;

    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<ImagePatchifyKernel> _patchify_cache;
    TypedShaderCache<LayerNormKernel> _layer_norm_cache;
    TypedShaderCache<RmsNormKernel> _rms_norm_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<RopeAxialKernel> _rope2d_cache;
    TypedShaderCache<AttentionPrefillKernel> _attn_cache;
    TypedShaderCache<SpatialMergeKernel> _merge_cache;
    TypedShaderCache<PixelUnshuffleKernel> _unshuffle_cache;
    TypedShaderCache<Transpose2dKernel> _transpose_cache;
    TypedShaderCache<AdaptiveAvgPool2dKernel> _pool_cache;
    TypedShaderCache<AddPositionKernel> _add_pos_cache;
    TypedShaderCache<ResidualAddKernel> _residual_cache;

    const TypedShaderCache<ImagePatchifyKernel>::ShaderT *_sh_patchify{nullptr};
    const TypedShaderCache<LayerNormKernel>::ShaderT *_sh_layer_norm{nullptr};
    const TypedShaderCache<RmsNormKernel>::ShaderT *_sh_rms_norm{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_none{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_residual{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_gelu{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_silu{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_residual{nullptr};
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope2d{nullptr};
    const TypedShaderCache<AttentionPrefillKernel>::ShaderT *_sh_attn{nullptr};
    const TypedShaderCache<SpatialMergeKernel>::ShaderT *_sh_merge{nullptr};
    const TypedShaderCache<PixelUnshuffleKernel>::ShaderT *_sh_unshuffle{nullptr};
    const TypedShaderCache<Transpose2dKernel>::ShaderT *_sh_transpose{nullptr};
    const TypedShaderCache<AdaptiveAvgPool2dKernel>::ShaderT *_sh_pool{nullptr};
    const TypedShaderCache<AddPositionKernel>::ShaderT *_sh_add_pos{nullptr};
    const TypedShaderCache<ResidualAddKernel>::ShaderT *_sh_residual{nullptr};
};

// ---------------------------------------------------------------------------
// RegionRecognizer adapter: runs the shared slice over layout region crops.
// The orchestrator (LayoutVlmExecutor) drives overview -> crops -> this
// recognizer -> composition. Generative content carries NO confidence.
// ---------------------------------------------------------------------------

struct P7RegionRecognizerConfig {
    P7SliceConfig model;
    P7TensorNames names;
    // Prompt token ids surrounding the visual tokens (synthetic contract).
    luisa::vector<uint32_t> prompt_prefix{1u};
    luisa::vector<uint32_t> prompt_suffix{2u};
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

class P7RegionRecognizer final : public RegionRecognizer {
public:
    P7RegionRecognizer() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const P7RegionRecognizerConfig &config, const WeightStore &weights);

    [[nodiscard]] Result<luisa::vector<RawRegionOutput>> recognize_batch(
        compute::Stream &stream, luisa::span<const RegionCropInput> crops,
        const RegionRecognizeOptions &options) override;

    [[nodiscard]] uint64_t weight_bytes() const noexcept override { return _slice.weight_bytes(); }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept override {
        return _slice.vision_workspace_bytes() + _slice.decoder_workspace_bytes();
    }
    [[nodiscard]] uint64_t kv_cache_peak_bytes() const noexcept override { return _kv_peak_bytes; }
 // Phase 11 observability: exact-capacity chunks become ragged vision graphs;
 // text decode remains independent batch-1 with optional CUDA graph replay.
 [[nodiscard]] uint64_t vision_graph_count() const noexcept { return _vision_graph_count; }
 [[nodiscard]] uint64_t vision_crop_count() const noexcept { return _vision_crop_count; }
 [[nodiscard]] uint64_t cuda_graph_steps() const noexcept { return _cuda_graph_steps; }
 [[nodiscard]] uint64_t cuda_graph_updates() const noexcept { return _cuda_graph_updates; }
 [[nodiscard]] uint64_t cuda_graph_fallbacks() const noexcept { return _cuda_graph_fallbacks; }

    [[nodiscard]] const P7RegionRecognizerConfig &config() const noexcept { return _config; }
    [[nodiscard]] P7VlmExecutor &slice() noexcept { return _slice; }

private:
    P7RegionRecognizerConfig _config;
    compute::Device *_device{nullptr};
    P7VlmExecutor _slice;
    uint64_t _kv_peak_bytes{0};
    uint64_t _vision_graph_count{0};
    uint64_t _vision_crop_count{0};
 uint64_t _cuda_graph_steps{0};
 uint64_t _cuda_graph_updates{0};
 uint64_t _cuda_graph_fallbacks{0};
};

}// namespace luisa::ocr

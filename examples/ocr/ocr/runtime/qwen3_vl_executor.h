#pragma once
// OCR engine -- Qwen3-VL vertical-slice executor (Phase 4).
//
// The reusable Qwen3-VL graph: image preprocessing -> patch embed -> vision
// blocks (bidirectional per-image attention, QK-Norm, 2D vision RoPE, GELU
// MLP) -> spatial merger -> multimodal embeddings -> Qwen3 dense decoder
// (QK-Norm, GQA, interleaved M-RoPE, DeepStack injection at the exact
// configured early layers) -> deterministic greedy tokens. The text decoder
// is the SHARED DenseVlmExecutor configured with the additive Qwen3 contract
// switches (qk_norm, no qkv_bias, rope_axes=3, deepstack_layers); the vision
// tower + mergers are recorded here from the Phase 2 operators.
//
// FireRed-OCR reuses THIS graph through its own recipe configuration
// (models/fire_red_ocr.*): same executor, different tensor map / prompt /
// output-parser contract. No GRPO/training operators exist anywhere.
//
// SYNTHETIC ONLY: the geometry/tensor names are the declared synthetic
// contract (models/qwen3_vl.h). No official checkpoint is validated; real
// artifacts fail closed through the registry/Session. See MODELS.md.
//
// Ownership follows executor.h: borrows an installed WeightStore, owns a
// per-request DeviceWorkspace + TypedShaderCaches + the embedded text
// executor. All launches go through the safe dispatch adapters; all host
// transfers use luisa::span. No exceptions / RTTI.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>

#include "ocr/models/qwen3_vl.h"
#include "ocr/ops/ops_attention.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_image.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/ops/ops_vision.h"
#include "ocr/runtime/dense_vlm_executor.h"
#include "ocr/runtime/executor.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/qwen3_vl_image.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/workspace.h"

namespace luisa::ocr {

// One image input to the vision graph and its host preprocessing result are
// the shared plain data types in runtime/qwen3_vl_image.h (device-safe).

// Configuration for one Qwen3-VL executor instance.
struct Qwen3VlExecutorConfig {
    Qwen3VlConfig model;
    Qwen3VlTensorNames names;
    // Weight component tags in the WeightStore.
    luisa::string vision_component{"vision"};
    luisa::string decoder_component{"decoder"};
    // Normalization (RGB, (pixel/255 - mean)/std). Synthetic contract values;
    // the official processor statistics are an unmet gate.
    luisa::vector<float> mean{0.5f, 0.5f, 0.5f};
    luisa::vector<float> std_dev{0.5f, 0.5f, 0.5f};
    // Maximum TOTAL patch rows the vision workspace is sized for (declared
    // at create; a larger request fails closed before any allocation). This is
    // the real vision budget. (S10: there is NO per-chunk vision TDR guard in
    // this phase — the vision graph is recorded as one bounded command list;
    // the earlier vision_chunk_patches field was unused and has been removed
    // rather than left as a fake guard.)
    uint32_t max_vision_patches{4096};
    // Maximum prompt rows per decoder prefill chunk (forwarded).
    uint32_t prefill_chunk_rows{256};
    // Opt-in dense residual+RMSNorm fusion A/B. Default preserves the
    // reference command sequence; forwarded to the embedded text executor.
    bool enable_fused_residual_rms_norm{false};
    // Hidden-sized cooperative RMSNorm; false restores the reference kernel.
    bool enable_cooperative_norm{true};
    // Optional byte budget (0 = unset), preflighted before allocation against
    // the executor-owned workspace arenas (vision + decoder).
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

// One multimodal prompt: text token ids with image placeholders already
// expanded to `merged_count` placeholder ids per image (the caller builds the
// sequence; the executor replaces the placeholder rows with vision features).
struct Qwen3VlPrompt {
    luisa::vector<uint32_t> token_ids;// full prompt incl. image placeholder rows
    // Per image: the placeholder row range in token_ids. Must be contiguous
    // and in order; the i-th range corresponds to images[i].
    struct ImageRange {
        uint32_t start{0};
        uint32_t count{0};
    };
    luisa::vector<ImageRange> image_ranges;

    [[nodiscard]] Status validate(uint32_t vocab_size) const;
};

// Result of the vision graph for one request (device views into the
// executor-owned vision workspace; valid until the next vision run/reset).
struct Qwen3VlVisionOutput {
    // Main merger output [total_merged, text_hidden]: the multimodal features.
    compute::BufferView<float> features;
    // DeepStack merger outputs, one [total_merged, text_hidden] view per
    // configured injection (in deepstack order).
    luisa::vector<compute::BufferView<float>> deepstack_features;
    uint32_t total_merged{0};
};

// The Qwen3-VL vertical-slice executor. NOT thread-safe: one worker drives
// one executor at a time. Reusable across requests (scratch overwritten).
class Qwen3VlExecutor final {
public:
    Qwen3VlExecutor() noexcept = default;

    // Validate config, resolve every weight view (vision + decoder), preflight
    // + reserve workspaces, compile shader variants, create the embedded text
    // executor. `device`/`weights` are borrowed for the executor's lifetime.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const Qwen3VlExecutorConfig &config,
                                const WeightStore &weights);

    // Record the vision graph for `images` (ragged: each image is processed
    // independently; attention never crosses image boundaries). Preprocesses
    // on the HOST through the Phase 1 reference (preprocess_image, the exact
    // declared bilinear convention), uploads the normalized CHW pixels,
    // patchifies on device, runs patch embed + vision blocks + mergers. The
    // returned views live in the vision workspace until the next
    // run_vision/reset. `list` is caller-owned; the caller commits/drains.
    // Per-image staging (pixels/positions) is heap-owned and retained through
    // command completion.
    [[nodiscard]] Result<Qwen3VlVisionOutput> record_vision(compute::CommandList &list,
                                                            luisa::span<const Qwen3VlImageInput> images);

    // Build the multimodal DecodeRequest + M-RoPE position ids for a prompt
    // whose image ranges are filled by `vision` (the output of record_vision
    // for the SAME images, in order). The prompt's placeholder ids are
    // replaced by the vision features at the declared ranges.
    // Phase 4 slice supports AT MOST ONE image per prompt (the shared
    // DecodeRequest carries a single multimodal range); multi-image prompts
    // fail closed here rather than silently dropping images.
    [[nodiscard]] Result<DecodeRequest> build_decode_request(const Qwen3VlPrompt &prompt,
                                                             const Qwen3VlVisionOutput &vision,
                                                             luisa::span<const Qwen3VlImageInput> images,
                                                             luisa::vector<int32_t> &position_ids_out) const;

    // Prefill + decode entry points forward to the embedded text executor
    // with the Qwen3 contract (interleaved MRoPE + DeepStack).
    [[nodiscard]] DenseVlmExecutor &text_executor() noexcept { return _text; }
    [[nodiscard]] const DenseVlmExecutor &text_executor() const noexcept { return _text; }

    [[nodiscard]] const Qwen3VlExecutorConfig &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t vision_workspace_bytes() const noexcept { return _vision_ws.capacity_bytes(); }
    [[nodiscard]] uint64_t decoder_workspace_bytes() const noexcept { return _text.workspace_bytes(); }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }

    // Rewind the vision arena between requests (caller drained the stream).
    void reset_vision_workspace() noexcept { _vision_ws.reset(); }

    // Download helpers for layered golden tests (caller drains afterwards).
    [[nodiscard]] Status download_vision_patches(compute::Stream &stream, luisa::vector<float> &out) const;
    // S11: the FINAL vision residual stream (after ALL vision blocks). For an
    // isolated first-block check, run a 1-block executor so this IS block 0.
    [[nodiscard]] Status download_vision_final(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_vision_features(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_deepstack_features(compute::Stream &stream, uint32_t index,
                                                     luisa::vector<float> &out) const;

private:
    struct VisionBlockWeights {
        compute::BufferView<float> norm1_w, norm1_b, norm2_w, norm2_b;
        compute::BufferView<float> q_w, k_w, v_w, o_w;
        compute::BufferView<float> q_b, k_b, v_b, o_b;
        compute::BufferView<float> q_norm, k_norm;
        compute::BufferView<float> fc1_w, fc1_b, fc2_w, fc2_b;
    };
    struct MergerWeights {
        compute::BufferView<float> ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b;
    };

    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();

    // One vision block over `rows` patch tokens of ONE image. Bidirectional
    // attention within the image; positions is the [rows, 2] int32 vision
    // (height, width) position buffer.
    [[nodiscard]] Status record_vision_block(compute::CommandList &list, uint32_t layer,
                                             compute::BufferView<float> x,
                                             compute::BufferView<int32_t> positions,
                                             uint32_t rows);

    // Spatial merge + LayerNorm + 2-layer GELU MLP for one merger, over
    // `patch_rows` patch tokens of one image (grid_w patches per row).
    // Writes [merged_rows, out_dim] into `out` (a subview of a features arena).
    [[nodiscard]] Status record_merger(compute::CommandList &list, const MergerWeights &m,
                                       compute::BufferView<float> x, uint32_t patch_rows,
                                       uint32_t grid_w, uint32_t out_dim,
                                       compute::BufferView<float> out);

    // GEMM helper (vision path; always the prefill GEMM, M can be large).
    [[nodiscard]] Status record_linear(compute::CommandList &list,
                                       compute::BufferView<float> A,
                                       compute::BufferView<float> B,
                                       compute::BufferView<float> C,
                                       compute::BufferView<float> aux,
                                       uint32_t m, uint32_t n, uint32_t k,
                                       GemmEpilogue epilogue);
    [[nodiscard]] compute::BufferView<float> vws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_vision_ws.buffer());
    }

    Qwen3VlExecutorConfig _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _vision_ws;
    DenseVlmExecutor _text;
    uint64_t _weight_bytes{0};

    // Vision workspace regions (fixed offsets; sized for max_vision_patches).
    WorkspaceAllocation _v_pixels, _v_patches, _v_x, _v_norm, _v_q, _v_k, _v_v,
        _v_qn, _v_kn, _v_qr, _v_kr, _v_attn, _v_fc1, _v_fc2, _v_merged,
        _v_mnorm, _v_mfc1, _v_features, _v_positions, _v_axis, _v_dummy_f, _v_dummy_u;
    // Scratch for the bias_residual epilogue aux ([bias N | residual M*N]).
    WorkspaceAllocation _v_aux;
    // One DeepStack feature arena per injection.
    luisa::vector<WorkspaceAllocation> _v_ds_features;
    // Host mirrors of the last vision run (for the download helpers).
    uint32_t _last_total_patches{0};
    uint32_t _last_total_merged{0};

    // Weight views.
    compute::BufferView<float> _patch_embed_w, _patch_embed_b;
    luisa::vector<VisionBlockWeights> _blocks;
    MergerWeights _merger;
    luisa::vector<MergerWeights> _ds_mergers;

    // Constant vision RoPE axis map (uploaded per record from heap staging).
    luisa::vector<uint32_t> _vision_axis_pending;
    // Normalization constants uploaded once at create (device buffers).
    compute::Buffer<float> _mean_buf, _std_buf;

    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<ImageResizeNormalizeKernel> _resize_cache;
    TypedShaderCache<ImagePatchifyKernel> _patchify_cache;
    TypedShaderCache<LayerNormKernel> _layer_norm_cache;
    TypedShaderCache<RmsNormKernel> _rms_norm_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<RopeAxialKernel> _rope2d_cache;
    TypedShaderCache<AttentionPrefillKernel> _attn_cache;
    TypedShaderCache<SpatialMergeKernel> _merge_cache;
    TypedShaderCache<ResidualAddKernel> _residual_cache;

    const TypedShaderCache<ImageResizeNormalizeKernel>::ShaderT *_sh_resize{nullptr};
    const TypedShaderCache<ImagePatchifyKernel>::ShaderT *_sh_patchify{nullptr};
    const TypedShaderCache<LayerNormKernel>::ShaderT *_sh_layer_norm{nullptr};
    const TypedShaderCache<RmsNormKernel>::ShaderT *_sh_rms_norm{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_none{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_gelu{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_residual{nullptr};
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope2d{nullptr};
    const TypedShaderCache<AttentionPrefillKernel>::ShaderT *_sh_attn{nullptr};
    const TypedShaderCache<SpatialMergeKernel>::ShaderT *_sh_merge{nullptr};
    const TypedShaderCache<ResidualAddKernel>::ShaderT *_sh_residual{nullptr};
};

}// namespace luisa::ocr

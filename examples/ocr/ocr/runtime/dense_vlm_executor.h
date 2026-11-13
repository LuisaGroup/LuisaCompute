#pragma once
// OCR engine -- dense autoregressive decoder executor (Phase 3).
//
// A correctness-first dense VLM *language* executor: embedding lookup (+
// multimodal replacement), dense transformer blocks (QKV with bias, 1D RoPE,
// GQA causal attention, output projection + residual, RMSNorm, SwiGLU MLP),
// contiguous-KV append/read, final RMSNorm + LM head. Prefill and single-step
// decode are SEPARATE command-recording loops; host decode scheduling drives
// them (see DenseDecodeController). No vision tower, no MTP/DFlash, no paged
// KV, and no decoder batching beyond batch 1. Phase 11 adds opt-in CUDA graph
// capture/update for the single-sequence decode loop; prefill stays ungraphed.
//
// It is model-agnostic: geometry + tensor names come from the shared
// DenseDecoderConfig / DenseDecoderTensorNames contract (models/qwen2_5_vl.h).
// The synthetic test decoder and (later) real Qwen2.5-VL-derived decoders use
// the SAME executor; nothing here claims a real model is validated.
//
// Ownership (executor.h contract): borrows an installed WeightStore
// (immutable/shared), owns a per-request DeviceWorkspace + TypedShaderCaches.
// All launches go through the Phase 2 safe dispatch adapters; all host
// transfers use luisa::span. No exceptions / RTTI.
#include <cstdint>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/device.h>

#include "ocr/detokenizer.h"
#include "ocr/models/qwen2_5_vl.h"
#include "ocr/ops/ops_attention.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/ops/ops_vision.h"
#include "ocr/runtime/decode_cache.h"
#include "ocr/runtime/executor.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"
#include "ocr/tokenizer.h"

namespace luisa::ocr {

// DenseLinearStorage + dense_linear_storage_name are defined in the device-safe
// models/qwen2_5_vl.h (included above) so host model modules can reference the
// storage policy without pulling the ops AttentionMaskKind.

// Configuration for one dense executor instance.
struct DenseExecutorConfig {
    DenseDecoderConfig decoder;
    DenseDecoderTensorNames names;
    bool tie_word_embeddings{false};
    // Additive Phase 4 contract switches. The DEFAULTS reproduce the exact
    // Phase 3 behavior (Qwen2.5-style: QKV bias, no QK-Norm, 1-axis RoPE, no
    // DeepStack); a Qwen3-configured executor flips them explicitly. They are
    // never inferred from tensor names or model_type strings.
    // Qwen3 text contract: per-head RMSNorm on Q/K (q_norm/k_norm weights
    // required) and NO QKV projection biases (bias tensors must be absent).
    bool qk_norm{false};
    bool qkv_bias{true};
    // RoPE axis count: 1 = plain 1D text positions (Phase 3); 3 = M-RoPE with
    // per-token (temporal, height, width) positions supplied via the explicit
    // prefill/decode entry points; 4 = XD-RoPE (Hunyuan style) with per-token
    // (text, height, width, time) positions. The pair family is selected by
    // decoder.rope_pair_layout: rotate_half + axes 3 = interleaved M-RoPE
    // (Qwen3), adjacent + axes 3 = adjacent-pair M-RoPE (Qwen2-VL/MinerU),
    // adjacent + axes 4 = XD-RoPE. Every multimodal axis count uses the
    // explicit mrope_axis_of_pair map, never a guess.
    uint32_t rope_axes{1};
    // Multimodal per-pair axis map (length head_dim/2, values < rope_axes).
    // Required (non-empty) when rope_axes >= 3; ignored otherwise.
    luisa::vector<uint32_t> mrope_axis_of_pair;
    // Additive Phase 7 partial-rotary factor (GLM-style): only the first
    // head_dim * partial_rotary_num / partial_rotary_den elements of each head
    // are rotated; the remaining tail passes through unchanged. Default 1/1
    // reproduces the exact Phase 3/4 full-rotary behavior. The numerator must
    // be <= denominator, both nonzero, and head_dim*num must be divisible by
    // den*2 so the rotated prefix consists of whole RoPE pairs in the
    // configured pair family (adjacent or rotate-half).
    uint32_t partial_rotary_num{1};
    uint32_t partial_rotary_den{1};
    // Rotated elements per head (derived; validate() checks divisibility).
    [[nodiscard]] uint32_t rotary_dim() const noexcept {
        return partial_rotary_den == 0u
                   ? 0u
                   : static_cast<uint32_t>((static_cast<uint64_t>(decoder.head_dim) * partial_rotary_num) / partial_rotary_den);
    }
    // Additive Phase 8: per-linear weight storage. Default fp32 = the exact
    // Phase 3/4/7 path. int4_* selects the packed W4A32 GEMM/GEMV kernels for
    // q/k/v/o/gate/up/down projections and the standalone LM head. The
    // embedding table, norms, QKV biases, and a TIED LM head stay FP32.
    // `linear_group_size` is the quantization group size (64 or 128) and is
    // required when storage != fp32.
    DenseLinearStorage linear_storage{DenseLinearStorage::fp32};
    uint32_t linear_group_size{0};
    // DeepStack injection: `deepstack_layers[i]` is the EXACT decoder layer
    // AFTER whose residual stream the i-th projected feature is added (image
    // token rows only). Strictly increasing, < decoder.num_layers. The
    // projections themselves are produced by the vision graph and passed per
    // request (Qwen3 prefill entry point); this list only selects the layers.
    luisa::vector<uint32_t> deepstack_layers;
    // Weight component tag in the WeightStore (e.g. "decoder").
    luisa::string component{"decoder"};
    // Maximum prompt rows processed per prefill command chunk (TDR guard).
    // The whole prompt is still prefilled; long prompts are split into
    // causal chunks that read the already-written KV prefix.
    uint32_t prefill_chunk_rows{256};
    // Phase 11 opt-in fusion of output projection residual and post-attention
    // RMSNorm. Default false preserves the reference command sequence.
    bool enable_fused_residual_rms_norm{false};
    // Cooperative hidden RMSNorm for hidden_size >= kCooperativeNormMinHidden.
    // Default on for parity-proven shapes; false is the reference kill switch.
    bool enable_cooperative_norm{true};
    // Optional byte budget (0 = unset). Preflighted BEFORE allocation against
    // the executor-owned device bytes (workspace arena + tied-LM-head transpose
    // buffer when present) and host staging (the tied-head transpose staging).
    // Borrowed shared weights and the caller-owned KV cache are accounted by
    // their own owners, not here.
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

// The dense language executor. One instance serves one model geometry; it is
// reusable across requests (workspace scratch is overwritten per request).
// NOT thread-safe: one worker drives one executor at a time.
class DenseVlmExecutor final : public Executor {
public:
    DenseVlmExecutor() noexcept = default;

    // Validate config, resolve every weight view from `weights` (already
    // installed), preflight + reserve the workspace, compile shader variants.
    // `stream` is used for one-time tied-LM-head transpose (when needed) and
    // constant RoPE-axis-map uploads, drained before create returns.
    // `device`/`weights` are borrowed for the executor's lifetime.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const DenseExecutorConfig &config,
                                const WeightStore &weights);

    // Prefill the prompt; `logits_out` (size vocab_size, host) receives the
    // LAST prompt row's logits after the caller drains the stream.
    [[nodiscard]] Status record_prefill(compute::CommandList &list,
                                        const DecodeRequest &request,
                                        ContiguousKvCache &cache,
                                        luisa::span<float> logits_out) override;

    // Prefill the prompt and write EVERY row's logits into `logits_device`
    // ([prompt_len, vocab_size] device view). Used by layered golden tests.
    [[nodiscard]] Status record_prefill_all_logits(compute::CommandList &list,
                                                   const DecodeRequest &request,
                                                   ContiguousKvCache &cache,
                                                   compute::BufferView<float> logits_device);

    // ------------------------------------------------------------------
    // Additive Phase 4 (Qwen3) entry points. These are ONLY legal when the
    // executor was configured with rope_axes == 3 (interleaved M-RoPE); the
    // Phase 3 entry points above remain the 1-axis path and are unchanged.
    // ------------------------------------------------------------------

    // Qwen3 prefill: `position_ids` is [prompt_len * 3] int32 (temporal,
    // height, width per token, produced by qwen3_vl_build_position_ids).
    // `deepstack_features[i]` is a [mm_count, hidden] device view projected by
    // the i-th DeepStack merger; it is injected into the image rows
    // [mm_start, mm_start+mm_count) AFTER decoder layer deepstack_layers[i].
    // The list length must equal config.deepstack_layers.size(). When
    // all_logits, every row's logits go to logits_device; otherwise only the
    // last row's logits go to last_logits_out.
    [[nodiscard]] Status record_prefill_qwen3(compute::CommandList &list,
                                              const DecodeRequest &request,
                                              luisa::span<const int32_t> position_ids,
                                              luisa::span<const compute::BufferView<float>> deepstack_features,
                                              ContiguousKvCache &cache,
                                              bool all_logits,
                                              luisa::span<float> last_logits_out,
                                              compute::BufferView<float> logits_device);

    // Qwen3 single decode step: `position` is the absolute TEXT position of
    // the new token (all three M-RoPE axes advance together for generated
    // text; the caller tracks it from qwen3_vl_build_position_ids'
    // next_text_position). DeepStack never applies to generated text rows.
    [[nodiscard]] Status record_decode_step_qwen3(compute::CommandList &list,
                                                  uint32_t input_token,
                                                  int32_t position,
                                                  ContiguousKvCache &cache,
                                                  luisa::span<float> logits_out);

    // ------------------------------------------------------------------
    // Additive Phase 7 generic position entry points. Legal ONLY when
    // rope_axes >= 3 (interleaved/adjacent M-RoPE or XD-RoPE); they carry the
    // FULL per-token axis vector so adjacent-pair M-RoPE (Qwen2/MinerU) and
    // 4-axis XD-RoPE (Hunyuan) never borrow the Qwen3 interleaved contract.
    // ------------------------------------------------------------------

    // Prefill with explicit [prompt_len * rope_axes] int32 positions. When
    // all_logits, every row's logits go to logits_device; otherwise only the
    // last row's logits go to last_logits_out. DeepStack (when configured)
    // follows the same contract as record_prefill_qwen3.
    [[nodiscard]] Status record_prefill_positions(compute::CommandList &list,
                                                  const DecodeRequest &request,
                                                  luisa::span<const int32_t> position_ids,
                                                  luisa::span<const compute::BufferView<float>> deepstack_features,
                                                  ContiguousKvCache &cache,
                                                  bool all_logits,
                                                  luisa::span<float> last_logits_out,
                                                  compute::BufferView<float> logits_device);

    // Single decode step with the FULL per-token axis vector (size rope_axes).
    [[nodiscard]] Status record_decode_step_positions(compute::CommandList &list,
                                                      uint32_t input_token,
                                                      luisa::span<const int32_t> position,
                                                      ContiguousKvCache &cache,
                                                      luisa::span<float> logits_out);

    [[nodiscard]] Status record_decode_step(compute::CommandList &list,
                                            uint32_t input_token,
                                            ContiguousKvCache &cache,
                                            luisa::span<float> logits_out) override;

    // Shared decode-step body: `position` is the FULL per-token RoPE axis
    // vector (size rope_axes): 1-axis = the absolute token index; 3-axis =
    // the M-RoPE (temporal, height, width) vector; 4-axis = the XD-RoPE
    // (text, height, width, time) vector. For generated text the caller
    // advances every axis together.
    [[nodiscard]] Status record_decode_step_impl(compute::CommandList &list,
                                                 uint32_t input_token,
                                                 luisa::span<const int32_t> position,
                                                 ContiguousKvCache &cache,
                                                 luisa::span<float> logits_out);

    [[nodiscard]] uint64_t weight_bytes() const noexcept override { return _weight_bytes; }
    // Additive (S2): executor-OWNED device weight bytes (the tied-LM-head
    // transpose buffer when present). weight_bytes() = borrowed component
    // bytes + owned_device_bytes(); measured reporting must not double-count
    // or omit the owned part.
    [[nodiscard]] compute::Device &device() const noexcept { return *_device; }
 [[nodiscard]] uint64_t owned_device_bytes() const noexcept {
        return _tied_lm_head ? _tied_lm_head.size_bytes() : 0ull;
    }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept override { return _workspace.capacity_bytes(); }
    [[nodiscard]] const DenseExecutorConfig &config() const noexcept { return _config; }
    [[nodiscard]] DeviceWorkspace &workspace() noexcept { return _workspace; }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }

    // Rewind the per-request arena between requests (caller drained the
    // stream). Capacity is grow-only; regions keep their offsets.
    void reset_workspace() noexcept { _workspace.reset(); }

private:
    // One linear B operand. FP32 binds `dense`; INT4 binds packed codes +
    // per-group scales (+ zero points for asymmetric). group_size is 0 for the
    // dense path. The storage format is explicit per weight, never inferred.
    struct LinearWeight {
        compute::BufferView<float> dense;
        compute::BufferView<uint32_t> packed;
        compute::BufferView<float> scales;
        compute::BufferView<uint32_t> zero_points;
        DenseLinearStorage storage{DenseLinearStorage::fp32};
        uint32_t group_size{0};
        [[nodiscard]] bool is_packed() const noexcept { return storage != DenseLinearStorage::fp32; }
    };

    struct LayerWeights {
        compute::BufferView<float> input_norm, post_norm;
        LinearWeight q_w, k_w, v_w, o_w;
        compute::BufferView<float> q_b, k_b, v_b;
        LinearWeight gate_w, up_w, down_w;
        // Qwen3 QK-Norm weights (per-head RMSNorm over head_dim); invalid
        // views when qk_norm is disabled.
        compute::BufferView<float> q_norm, k_norm;
    };

    [[nodiscard]] Status resolve_weights(compute::Stream &stream, const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();

    // Shared prefill body. all_logits=false writes only the last row's logits
    // to last_logits_out (host span); all_logits=true writes every row into
    // logits_device. Exactly one of the two outputs is used.
    // `position_ids` (when non-empty) is [prompt_len * rope_axes] int32;
    // empty selects the Phase 3 sequential 1-axis positions. `deepstack`
    // (when non-empty) carries the per-injection projected image features.
    [[nodiscard]] Status record_prefill_impl(compute::CommandList &list,
                                             const DecodeRequest &request,
                                             luisa::span<const int32_t> position_ids,
                                             luisa::span<const compute::BufferView<float>> deepstack,
                                             ContiguousKvCache &cache,
                                             bool all_logits,
                                             luisa::span<float> last_logits_out,
                                             compute::BufferView<float> logits_device);

    // One transformer block over `rows` query tokens at absolute position
    // pos0. x is the [rows, hidden] residual stream (workspace). K/V for the
    // rows are appended to the cache at token offset pos0. decode selects the
    // single-query contiguous-KV attention path; otherwise causal prefill
    // attention reads the cache prefix [0, pos0+rows).
    // `deepstack` (when non-empty) injects feature i into the image rows
    // AFTER this block when layer == deepstack_layers[i]; the injection range
    // is [ds_start, ds_start+ds_count) in absolute token coordinates.
    [[nodiscard]] Status record_block(compute::CommandList &list, uint32_t layer,
                                      compute::BufferView<float> x,
                                      uint32_t rows, uint32_t pos0,
                                      ContiguousKvCache &cache, bool decode,
                                      luisa::span<const compute::BufferView<float>> deepstack = {},
                                      uint32_t ds_start = 0u, uint32_t ds_count = 0u);

    // Final RMSNorm + LM head over `rows` rows of x; writes [rows, vocab] to C.
    [[nodiscard]] Status record_lm_head(compute::CommandList &list,
                                        compute::BufferView<float> x,
                                        uint32_t rows,
                                        compute::BufferView<float> C);

    // GEMM/GEMV helper: C[M,N] = epilogue(A[M,K] @ B[K,N], aux). Picks the
    // decode GEMV path when M <= kGemvSmallM, else the prefill GEMM. `B` carries
    // its explicit storage format (FP32 dense or packed INT4 W4A32).
    [[nodiscard]] Status record_linear(compute::CommandList &list,
                                       compute::BufferView<float> A,
                                       const LinearWeight &B,
                                       compute::BufferView<float> C,
                                       compute::BufferView<float> aux,
                                       uint32_t m, uint32_t n, uint32_t k,
                                       GemmEpilogue epilogue);

    [[nodiscard]] compute::BufferView<float> ws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_workspace.buffer());
    }

    DenseExecutorConfig _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _workspace;
    uint64_t _weight_bytes{0};
    uint32_t _decode_step{0};// host mirror of decode steps since last prefill

    // Workspace regions (fixed offsets; arena reserved to full capacity).
    WorkspaceAllocation _emb, _x, _norm, _norm2, _q, _k, _v, _q_rope, _k_rope,
        _attn, _gate, _up, _mlp, _logits, _ids, _positions, _axis_of_pair,
        _dummy_f, _dummy_u;
    // Qwen3-only regions (zero-sized for the Phase 3 path): QK-Norm outputs
    // (in-place norm is unsafe across threads) and the DeepStack scratch row.
    WorkspaceAllocation _q_normed, _k_normed, _ds_scratch;

    // Weight views. The embedding table + final norm are always FP32. The LM
    // head is FP32 when tied (transposed from the embedding) or standalone
    // (which may be packed INT4 under the W4A32 path).
    compute::BufferView<float> _embed, _final_norm;
    LinearWeight _lm_head;
    luisa::vector<LayerWeights> _layers;
    // Executor-owned transposed LM head for tied embeddings (no standalone
    // lm_head tensor). Immutable after create.
    compute::Buffer<float> _tied_lm_head;

    // Constant RoPE axis map: 1D maps every pair to axis 0; multimodal uses
    // the explicit configured section map TRUNCATED to the rotated prefix
    // (partial rotary). Uploaded once during create; the stable host vector is
    // retained only as the source of that one-time upload.
    luisa::vector<uint32_t> _axis_pending;
    // Host mirror of the tied-LM-head transpose staging size (for accounting).
    uint64_t _tied_head_staging_bytes{0};

    // Shader caches (compiled once per variant; retained for lifetime).
    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<EmbeddingLookupKernel> _embedding_cache;
    TypedShaderCache<MmScatterKernel> _mm_scatter_cache;
    TypedShaderCache<RmsNormKernel> _rms_norm_cache;
 TypedShaderCache<ResidualRmsNormKernel> _residual_rms_norm_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<GemvKernel> _gemv_cache;
    // Additive Phase 8: packed INT4 W4A32 GEMM/GEMV variants (compiled only
    // when linear_storage != fp32). FP32 activations, FP32 accumulation.
    TypedShaderCache<GemmPackedKernel> _gemm_packed_cache;
    TypedShaderCache<GemvPackedKernel> _gemv_packed_cache;
    TypedShaderCache<GatedActKernel> _gated_act_cache;
    TypedShaderCache<RopeAxialKernel> _rope_cache;
    TypedShaderCache<RopeAxialKernel> _rope_rotate_half_cache;
    TypedShaderCache<RopeAxialKernel> _rope_mrope_cache;
    // Additive Phase 7: adjacent-pair M-RoPE / XD-RoPE shader (distinct named
    // factories/keys; the interleaved Qwen3 shader is never reused for them).
    TypedShaderCache<RopeAxialKernel> _rope_axial_multi_cache;
    // Additive Phase 7: partial rotary (GLM) shader; null on the full path.
    TypedShaderCache<RopePartialKernel> _rope_partial_cache;
    TypedShaderCache<AttentionPrefillKernel> _attn_prefill_cache;
    TypedShaderCache<AttentionDecodeKernel> _attn_decode_cache;
    TypedShaderCache<KvAppendKernel> _kv_append_cache;
    TypedShaderCache<DeepStackInjectKernel> _deepstack_cache;

    // Retained shader pointers (stable across cache rehashes).
    const TypedShaderCache<EmbeddingLookupKernel>::ShaderT *_sh_embedding{nullptr};
    const TypedShaderCache<MmScatterKernel>::ShaderT *_sh_mm_scatter{nullptr};
    const TypedShaderCache<RmsNormKernel>::ShaderT *_sh_rms_norm{nullptr};
    // Hidden-sized norm variant; QK-Norm continues to use _sh_rms_norm.
    const TypedShaderCache<RmsNormKernel>::ShaderT *_sh_rms_norm_cooperative{nullptr};
    const TypedShaderCache<ResidualRmsNormKernel>::ShaderT *_sh_residual_rms_norm{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_residual{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_none{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_bias{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_residual{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_none{nullptr};
    // Packed INT4 variants (null on the FP32 path).
    const TypedShaderCache<GemmPackedKernel>::ShaderT *_sh_gemm_packed_bias{nullptr};
    const TypedShaderCache<GemmPackedKernel>::ShaderT *_sh_gemm_packed_residual{nullptr};
    const TypedShaderCache<GemmPackedKernel>::ShaderT *_sh_gemm_packed_none{nullptr};
    const TypedShaderCache<GemvPackedKernel>::ShaderT *_sh_gemv_packed_bias{nullptr};
    const TypedShaderCache<GemvPackedKernel>::ShaderT *_sh_gemv_packed_residual{nullptr};
    const TypedShaderCache<GemvPackedKernel>::ShaderT *_sh_gemv_packed_none{nullptr};
    const TypedShaderCache<GatedActKernel>::ShaderT *_sh_gated_silu{nullptr};
    // Selected RoPE shader (adjacent or rotate-half per config.rope_pair_layout).
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope{nullptr};
    // Partial-rotary shader (Phase 7; mutually exclusive with _sh_rope).
    const TypedShaderCache<RopePartialKernel>::ShaderT *_sh_rope_partial{nullptr};
    const TypedShaderCache<AttentionPrefillKernel>::ShaderT *_sh_attn_prefill{nullptr};
    const TypedShaderCache<AttentionDecodeKernel>::ShaderT *_sh_attn_decode{nullptr};
    const TypedShaderCache<KvAppendKernel>::ShaderT *_sh_kv_append{nullptr};
    const TypedShaderCache<DeepStackInjectKernel>::ShaderT *_sh_deepstack{nullptr};
};

// ---------------------------------------------------------------------------
// Host decode controller: drives prefill + single-step decode with greedy or
// validated deterministic sampling, EOS/stop-string handling via the
// incremental detokenizer, and bounded output/cache. One controller per
// request; it borrows the executor + cache and owns the sampler/detokenizer.
// ---------------------------------------------------------------------------

struct DenseDecodeOptions {
    DecodeSamplingConfig sampling;
    uint32_t max_new_tokens{0};// 0 invalid; hard output cap
    // Additive Phase 7 multimodal position contract. Empty = the Phase 3/4
    // behavior (rope_axes must be 1). Non-empty requires a rope_axes >= 3
    // executor and carries:
    //   prompt_position_ids: [prompt_len * rope_axes] int32 (the caller's
    //     model-specific position builder; validated against the prompt).
    //   next_position: the FULL axis vector of the FIRST generated token.
    //     Each generated step advances EVERY axis by 1 (text continuation).
    //   deepstack_features: per-injection [mm_count, hidden] device views
    //     (must match the executor's configured injection layers).
    luisa::vector<int32_t> prompt_position_ids;
    luisa::vector<int32_t> next_position;
    luisa::vector<compute::BufferView<float>> deepstack_features;
    // Additional EOS ids. The controller ALSO honors the decoder config's
    // eos_token_ids and the tokenizer's own eos (all deduplicated).
    luisa::vector<uint32_t> eos_ids;
    luisa::vector<luisa::string> stop_strings;
    bool skip_special_tokens{true};
 // Phase 11: opt-in CUDA graph capture/replay for single-sequence decode.
 // Unsupported backends use the ordinary stream path and report a fallback.
 // Capture/update failures fail closed; no partial graph output is accepted.
 bool enable_cuda_graph{false};
    // Wall-time budget in milliseconds; 0 = no explicit limit. Must be finite
    // and >= 0. Polled at each per-step synchronization boundary; exceeding it
    // stops the request with StatusCode::timeout (cache released).
    double max_wall_time_ms{0.0};
    // Cancellation: polled at each per-step synchronization boundary (and
    // before prefill); returning true stops the request with
    // StatusCode::cancelled (cache released). Empty = never cancelled.
    luisa::function<bool()> cancel_callback;

    [[nodiscard]] Status validate(uint32_t vocab_size) const;
};

struct DenseDecodeResult {
    luisa::vector<uint32_t> tokens;// generated tokens (excludes prompt)
    luisa::string text;            // detokenized text (stop text never included)
    bool hit_eos{false};
    bool hit_stop_string{false};
    bool truncated{false};// max_new_tokens reached before EOS/stop
 uint32_t steps{0};
 uint32_t cuda_graph_steps{0};
 uint32_t cuda_graph_updates{0};
 uint32_t cuda_graph_fallbacks{0};
 // Diagnostics for later Session integration (e.g. truncation warning).

    luisa::vector<Diagnostic> diagnostics;
};

class DenseDecodeController {
public:
    DenseDecodeController() noexcept = default;

    // `tokenizer` may be null for token-only tests (no detokenization; EOS ids
    // come solely from options). When present it must outlive the controller.
    [[nodiscard]] Status create(DenseVlmExecutor &executor, const Tokenizer *tokenizer,
                                DenseDecodeOptions options);

    // Run one full request: prefill the prompt, then decode until EOS / stop
    // string / max_new_tokens / cache capacity. Synchronizes the stream at
    // each step boundary (host scheduling). Releases the cache sequence on
    // completion. `cache` is borrowed and must be created for the geometry.
    [[nodiscard]] Result<DenseDecodeResult> generate(compute::Stream &stream,
                                                     const DecodeRequest &request,
                                                     ContiguousKvCache &cache);

    // Additive Phase 7: the KV length committed by the last generate() call
    // (prompt + decoded rows). Valid after generate returns; used by layered
    // tests to size KV downloads.
    [[nodiscard]] uint32_t last_committed_length() const noexcept { return _last_committed_length; }

private:
    DenseVlmExecutor *_executor{nullptr};
    const Tokenizer *_tokenizer{nullptr};
    DenseDecodeOptions _options;
    uint32_t _last_committed_length{0};
};

}// namespace luisa::ocr
#pragma once
// OCR engine -- DeepSeek-OCR-2 MoE VLM executor (Phase 10). DEVICE-SAFE header.
//
// Complete synthetic DeepSeek visual/query -> MoE decoder prefill/decode path:
//   VISION: SAM-style local perception (windowed ViT blocks) -> BOTH
//           convolutional downsamplers (two strided Conv2D) -> Qwen2-style
//           visual LM encoder carrying causal-flow QUERIES and the EXACT custom
//           visual/query block mask (visual tokens bidirectional; each query
//           sees all visual tokens + previous queries incl. itself) -> connector
//           (LayerNorm + GELU MLP) over the query rows -> text-hidden features.
//   DECODER: DeepSeek MoE -- embedding + multimodal replacement, per-layer
//           [RMSNorm -> QKV(bias) -> 1D rotate-half RoPE -> causal contiguous-KV
//           attention -> o-proj+residual -> RMSNorm -> MoE(router FP32 +
//           top-6 + 64 routed experts + two shared) + residual], final norm +
//           LM head. Prefill and single-step decode are SEPARATE; the MoE
//           reference loop runs over ACTIVE experts (host reads the routing
//           counts, then records per-expert GEMMs). NO grouped persistent GEMM.
//
// The custom visual/query mask is NEVER a generic causal mask. The text decoder
// attention IS causal (autoregressive); the custom mask lives in the visual LM
// encoder only.
//
// Ownership (runtime/executor.h): borrows an installed immutable WeightStore,
// owns a per-request DeviceWorkspace + TypedShaderCaches. The stream is driven
// INTERNALLY (the MoE active-expert loop needs the routing counts on the host).
// All host transfers use luisa::span. No exceptions / RTTI. SYNTHETIC ONLY.
#include <cstdint>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/detokenizer.h"
#include "ocr/models/deepseek_ocr2.h"
#include "ocr/ops/ops_attention.h"
#include "ocr/ops/ops_conv.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_image.h"
#include "ocr/ops/ops_moe.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/ops/ops_vision.h"
#include "ocr/runtime/decode_cache.h"
#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/executor.h"// DecodeRequest
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"
#include "ocr/tokenizer.h"

namespace luisa::ocr {

// Vision graph output for one request (device views into the executor-owned
// vision workspace; valid until the next vision run/reset).
struct MoeVisionOutput {
    compute::BufferView<float> features;// [num_queries, text_hidden]
    uint32_t visual_tokens{0};          // == config.vision.num_queries
    uint32_t encoder_rows{0};           // visual_after_down + num_queries
};

// Decode options (mirrors the dense/hybrid controllers; reuses the Phase 3
// sampler + Phase 1 detokenizer). Ordinary target decoding only.
struct MoeDecodeOptions {
    DecodeSamplingConfig sampling;
    uint32_t max_new_tokens{0};
    luisa::vector<uint32_t> eos_ids;
    luisa::vector<luisa::string> stop_strings;
    bool skip_special_tokens{true};
    [[nodiscard]] Status validate(uint32_t vocab_size) const;
};

struct MoeDecodeResult {
    luisa::vector<uint32_t> tokens;
    luisa::string text;
    bool hit_eos{false};
    bool hit_stop_string{false};
    bool truncated{false};
    uint32_t steps{0};
    luisa::vector<Diagnostic> diagnostics;
};

struct MoeRunResult {
    MoeDecodeResult decode;
    uint32_t visual_tokens{0};
    uint64_t weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t cache_bytes{0};
};

class MoeVlmExecutor final {
public:
    MoeVlmExecutor() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const DeepSeekOcr2Config &config, const WeightStore &weights);

    // Record the vision graph for ONE image into `list` (caller commits/drains).
    // Produces num_queries text-hidden features via the causal-flow queries.
    [[nodiscard]] Result<MoeVisionOutput> record_vision(compute::CommandList &list, const P7ImageInput &image);

    // Prefill the prompt (drives `stream` internally; syncs at each MoE layer to
    // read routing counts). `request.mm_features` carries the vision features.
    // On return the cache holds prompt_len committed tokens and `logits_out`
    // (size vocab) holds the last-row logits. Single-chunk prefill: prompt_len
    // must fit prefill_chunk_rows (the synthetic slice is small).
    [[nodiscard]] Status prefill(compute::Stream &stream, const DecodeRequest &request,
                                 ContiguousKvCache &cache, luisa::span<float> logits_out);

    // One decode step at absolute position == cache.length(0). Drives `stream`.
    [[nodiscard]] Status decode_step(compute::Stream &stream, uint32_t input_token,
                                     ContiguousKvCache &cache, luisa::span<float> logits_out);

    // Full image -> vision -> MoE decode request. `prompt_tokens` surrounds the
    // visual rows [mm_start, mm_start+num_queries). Drives the stream internally.
    [[nodiscard]] Result<MoeRunResult> run(compute::Stream &stream, const P7ImageInput &image,
                                           luisa::span<const uint32_t> prompt_tokens, uint32_t mm_start,
                                           const MoeDecodeOptions &options, const Tokenizer *tokenizer = nullptr);

    [[nodiscard]] const DeepSeekOcr2Config &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept { return _workspace.capacity_bytes(); }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }
    void reset_workspace() noexcept { _workspace.reset(); }

    // Create a contiguous KV cache sized for this decoder (all layers full-attn).
    [[nodiscard]] Status create_cache(compute::Device &device, compute::Stream &stream,
                                      uint32_t max_seq_len, ContiguousKvCache &cache,
                                      MemoryBudget *budget = nullptr) const;

    // The EXACT custom visual/query block mask (block_size 1) for a given split,
    // exposed so tests can compare the device mask bytes against ONE canonical
    // builder instead of inferring the mask from attention parity only.
    // Visual rows are bidirectional among visual rows and never see queries;
    // query row q sees all visual rows plus query rows <= q (including itself).
    [[nodiscard]] static luisa::vector<uint8_t> build_visual_query_mask(uint32_t visual_rows, uint32_t num_queries);

    // Record ONLY the pixel/position/axis uploads for `image` and then fail.
    // Test-only entry point for the staging-lifetime contract: after this
    // returns an error the caller may still commit `list`, and the uploaded
    // host staging must remain alive until command completion.
    [[nodiscard]] Status record_vision_uploads_then_fail_for_test(compute::CommandList &list,
                                                                 const P7ImageInput &image);

    // ------------------------------------------------------------------
    // Layered-golden download helpers (last prefill/decode; caller drains).
    // ------------------------------------------------------------------
    [[nodiscard]] Status download_vision_features(compute::Stream &stream, luisa::vector<float> &out) const;
    // S4: download the preprocessed CHW pixels the last record_vision uploaded
    // (used to prove an error-path upload read live staging, not freed memory).
    [[nodiscard]] Status download_vision_pixels(compute::Stream &stream, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_router_logits(compute::Stream &stream, uint32_t layer, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_router_probs(compute::Stream &stream, uint32_t layer, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_selected(compute::Stream &stream, uint32_t layer,
                                           luisa::vector<uint32_t> &ids, luisa::vector<float> &weights) const;
    [[nodiscard]] Status download_permutation(compute::Stream &stream, uint32_t layer,
                                              luisa::vector<uint32_t> &counts, luisa::vector<uint32_t> &perm_token,
                                              luisa::vector<uint32_t> &slot_pos) const;
    [[nodiscard]] Status download_expert_out(compute::Stream &stream, uint32_t layer, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_shared(compute::Stream &stream, uint32_t layer,
                                         luisa::vector<float> &shared0, luisa::vector<float> &shared1) const;
    [[nodiscard]] Status download_moe_out(compute::Stream &stream, uint32_t layer, luisa::vector<float> &out) const;
    [[nodiscard]] Status download_encoder_attn(compute::Stream &stream, uint32_t layer, luisa::vector<float> &out) const;
    // S9: download the EXACT custom visual/query mask bytes the device kernels
    // consumed on the last record_vision (block_size 1, enc_rows*enc_rows bytes),
    // so tests compare mask DATA directly, not only attention parity.
    [[nodiscard]] Status download_visual_query_mask(compute::Stream &stream, luisa::vector<uint8_t> &out) const;
    [[nodiscard]] uint32_t last_encoder_rows() const noexcept { return _last_encoder_rows; }
    // The last prefill/decode row count (for sizing per-layer downloads).
    [[nodiscard]] uint32_t last_rows() const noexcept { return _last_rows; }
    [[nodiscard]] uint32_t last_slots() const noexcept { return _last_slots; }

private:
    struct ConvWeights { compute::BufferView<float> weight, bias; };
    struct PerceptionBlock {
        compute::BufferView<float> norm1_w, norm1_b, norm2_w, norm2_b;
        compute::BufferView<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b, o_b;
        compute::BufferView<float> fc1_w, fc1_b, fc2_w, fc2_b;
    };
    struct EncoderBlock {
        compute::BufferView<float> norm1_w, norm1_b, norm2_w, norm2_b;
        compute::BufferView<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b, o_b;
        compute::BufferView<float> fc1_w, fc1_b, fc2_w, fc2_b;
    };
    struct MoeLayer {
        compute::BufferView<float> input_norm, post_norm;
        compute::BufferView<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b;
        compute::BufferView<float> router_w;
        luisa::vector<compute::BufferView<float>> expert_gate, expert_up, expert_down;
        compute::BufferView<float> shared0_gate, shared0_up, shared0_down;
        compute::BufferView<float> shared1_gate, shared1_up, shared1_down;
    };

    [[nodiscard]] Status resolve_weights(const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();

    // Vision sub-stages.
    [[nodiscard]] Status record_perception_block(compute::CommandList &list, uint32_t layer,
                                                 compute::BufferView<float> x, uint32_t rows,
                                                 const P7PackedImage &packed,
                                                 luisa::vector<luisa::vector<uint32_t>> &mask_staging);
    [[nodiscard]] Status record_conv_down(compute::CommandList &list, const ConvWeights &w,
                                          const DeepSeekConvDownsampler &spec,
                                          compute::BufferView<float> tokens_in, uint32_t in_c,
                                          uint32_t gh, uint32_t gw, compute::BufferView<float> tokens_out,
                                          uint32_t &out_gh, uint32_t &out_gw);
    [[nodiscard]] Status record_encoder_block(compute::CommandList &list, uint32_t layer,
                                              compute::BufferView<float> x, uint32_t rows);
    // One MoE decoder layer over `rows` query tokens at absolute pos0. Drives
    // the stream (records attention+router, flushes, reads counts, records
    // experts+scatter, flushes). `decode` selects single-query attention.
    [[nodiscard]] Status record_moe_block(compute::Stream &stream, uint32_t layer,
                                          compute::BufferView<float> x, uint32_t rows, uint32_t pos0,
                                          ContiguousKvCache &cache, bool decode);
    [[nodiscard]] Status record_lm_head(compute::CommandList &list, compute::BufferView<float> x,
                                        uint32_t rows, compute::BufferView<float> C);
    [[nodiscard]] Status record_linear(compute::CommandList &list, compute::BufferView<float> A,
                                       compute::BufferView<float> B, compute::BufferView<float> C,
                                       compute::BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
                                       GemmEpilogue epilogue);
    [[nodiscard]] Status copy_rows(compute::CommandList &list, compute::BufferView<float> src,
                                   compute::BufferView<float> dst, uint32_t rows, uint32_t hidden);
    [[nodiscard]] Status flush(compute::CommandList &list, compute::Stream &stream);

    [[nodiscard]] compute::BufferView<float> ws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_workspace.buffer());
    }

    DeepSeekOcr2Config _config;
    compute::Device *_device{nullptr};
    DeviceWorkspace _workspace;
    uint64_t _weight_bytes{0};
    uint32_t _decode_step{0};
    uint32_t _last_rows{0};
    uint32_t _last_slots{0};
    uint32_t _last_encoder_rows{0};
    uint64_t _last_pixel_elems{0};
    // Per-layer routing counts/offsets read back from the device (host mirror).
    luisa::vector<luisa::vector<uint32_t>> _layer_counts, _layer_offsets;

    // Vision weights.
    compute::BufferView<float> _patch_embed_w, _patch_embed_b;
    luisa::vector<PerceptionBlock> _perception;
    ConvWeights _down1, _down2;
    luisa::vector<EncoderBlock> _encoder;
    compute::BufferView<float> _query_embed;
    compute::BufferView<float> _conn_ln_w, _conn_ln_b, _conn_fc1_w, _conn_fc1_b, _conn_fc2_w, _conn_fc2_b;
    // Decoder weights.
    compute::BufferView<float> _embed, _final_norm, _lm_head;
    luisa::vector<MoeLayer> _layers;
    luisa::vector<uint32_t> _axis_pending;
    // Byte count of the last custom visual/query block mask (block_size 1).
    uint32_t _enc_mask_bytes{0};

    // Workspace regions.
    WorkspaceAllocation _pixels, _patches, _px, _pnorm, _pq, _pk, _pv, _pqr, _pkr, _pattn, _pfc1, _paux,
        _conv_a, _conv_b, _conv_tokens, _ex, _enorm, _eq, _ek, _ev, _eattn, _efc1, _eaux, _enc_mask, _enc_pos,
        _enc_axis, _p_pos, _conn_norm, _conn_fc1, _features,
        _emb, _x, _norm, _norm2, _q, _k, _v, _q_rope, _k_rope, _attn, _logits, _ids_tok, _positions, _axis,
        _dummy_f, _dummy_u;
    // Per-layer MoE routing/expert scratch (one set per layer for golden access).
    struct MoeScratch {
        WorkspaceAllocation router_logits, probs, ids, weights, counts, offsets,
            perm_slot, slot_pos, perm_token, gathered, gate, up, act, expert_out,
            shared0, shared1, moe_out;
    };
    luisa::vector<MoeScratch> _moe;
    // Encoder attention output per layer (golden access).
    luisa::vector<WorkspaceAllocation> _enc_attn_out;

    // Shader caches.
    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<ImagePatchifyKernel> _patchify_cache;
    TypedShaderCache<LayerNormKernel> _layer_norm_cache;
    TypedShaderCache<RmsNormKernel> _rms_norm_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<GemvKernel> _gemv_cache;
    TypedShaderCache<GatedActKernel> _gated_act_cache;
    TypedShaderCache<RopeAxialKernel> _rope2d_cache;
    TypedShaderCache<RopeAxialKernel> _rope_rh_cache;
    TypedShaderCache<AttentionPrefillKernel> _attn_prefill_cache;
    TypedShaderCache<AttentionDecodeKernel> _attn_decode_cache;
    TypedShaderCache<KvAppendKernel> _kv_append_cache;
    TypedShaderCache<EmbeddingLookupKernel> _embedding_cache;
    TypedShaderCache<MmScatterKernel> _mm_scatter_cache;
    TypedShaderCache<Conv2dKernel> _conv_cache;
    TypedShaderCache<Transpose2dKernel> _transpose_cache;
    TypedShaderCache<SpatialMergeKernel> _merge_cache;
    TypedShaderCache<MoeRouterSoftmaxKernel> _moe_softmax_cache;
    TypedShaderCache<MoeTopKKernel> _moe_topk_cache;
    TypedShaderCache<MoeHistogramKernel> _moe_hist_cache;
    TypedShaderCache<MoeOffsetsKernel> _moe_offsets_cache;
    TypedShaderCache<MoePermutationKernel> _moe_perm_cache;
    TypedShaderCache<MoeGatherKernel> _moe_gather_cache;
    TypedShaderCache<MoeScatterKernel> _moe_scatter_cache;
    TypedShaderCache<ResidualAddKernel> _residual_cache;

    const TypedShaderCache<ImagePatchifyKernel>::ShaderT *_sh_patchify{nullptr};
    const TypedShaderCache<LayerNormKernel>::ShaderT *_sh_layer_norm{nullptr};
    const TypedShaderCache<RmsNormKernel>::ShaderT *_sh_rms_norm{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_none{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_residual{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_residual{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias_gelu{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_none{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_bias{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_residual{nullptr};
    const TypedShaderCache<GatedActKernel>::ShaderT *_sh_gated_silu{nullptr};
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope2d{nullptr};
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope_rh{nullptr};
    const TypedShaderCache<AttentionPrefillKernel>::ShaderT *_sh_attn_prefill{nullptr};
    const TypedShaderCache<AttentionDecodeKernel>::ShaderT *_sh_attn_decode{nullptr};
    const TypedShaderCache<KvAppendKernel>::ShaderT *_sh_kv_append{nullptr};
    const TypedShaderCache<EmbeddingLookupKernel>::ShaderT *_sh_embedding{nullptr};
    const TypedShaderCache<MmScatterKernel>::ShaderT *_sh_mm_scatter{nullptr};
    const TypedShaderCache<Conv2dKernel>::ShaderT *_sh_conv{nullptr};
    const TypedShaderCache<Transpose2dKernel>::ShaderT *_sh_transpose{nullptr};
    const TypedShaderCache<SpatialMergeKernel>::ShaderT *_sh_merge{nullptr};
    const TypedShaderCache<MoeRouterSoftmaxKernel>::ShaderT *_sh_moe_softmax{nullptr};
    const TypedShaderCache<MoeTopKKernel>::ShaderT *_sh_moe_topk{nullptr};
    const TypedShaderCache<MoeHistogramKernel>::ShaderT *_sh_moe_hist{nullptr};
    const TypedShaderCache<MoeOffsetsKernel>::ShaderT *_sh_moe_offsets{nullptr};
    const TypedShaderCache<MoePermutationKernel>::ShaderT *_sh_moe_perm{nullptr};
    const TypedShaderCache<MoeGatherKernel>::ShaderT *_sh_moe_gather{nullptr};
    const TypedShaderCache<MoeScatterKernel>::ShaderT *_sh_moe_scatter{nullptr};
    const TypedShaderCache<ResidualAddKernel>::ShaderT *_sh_residual{nullptr};
};

}// namespace luisa::ocr

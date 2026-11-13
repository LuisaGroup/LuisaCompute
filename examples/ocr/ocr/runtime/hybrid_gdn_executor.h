#pragma once
// OCR engine -- hybrid Gated-Delta-Net (GDN) + periodic full-attention decoder
// executor and its hybrid cache (Phase 9).
//
// A correctness-first HYBRID language executor: the layer schedule EXPLICITLY
// mixes GDN layers (causal Conv1D + delta-rule recurrent state) and full GQA
// attention layers (contiguous KV). It is NOT a KV-only relabeling: GDN layers
// carry a per-head [d,d] recurrent matrix + a causal Conv1D ring, full layers
// carry K/V. Dense embedding / RMSNorm / SwiGLU MLP / final norm / LM head are
// shared with the dense decoder conventions. Prefill and single-step decode
// are SEPARATE command-recording loops; a host controller drives greedy /
// validated sampling / EOS / stop handling (reusing the Phase 3 sampler and
// Phase 1 detokenizer).
//
// GDN math is the DECLARED SYNTHETIC contract in ops/ops_gdn.h (no official
// checkpoint is pinned; real artifacts fail closed in the model modules).
//
// Ownership (runtime/executor.h): borrows an installed immutable WeightStore,
// owns a per-request DeviceWorkspace + TypedShaderCaches. The hybrid cache is
// caller-owned (created for the geometry) and borrowed per request. All
// launches go through the safe dispatch adapters; all host transfers use
// luisa::span. No exceptions / RTTI. SYNTHETIC ONLY.
#include <cstdint>

#include <luisa/core/stl/functional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ocr/detokenizer.h"
#include "ocr/models/qwen2_5_vl.h"
#include "ocr/ops/ops_attention.h"
#include "ocr/ops/ops_decode.h"
#include "ocr/ops/ops_gdn.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/ops/ops_rope.h"
#include "ocr/runtime/decode_cache.h"
#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/runtime/executor.h"
#include "ocr/runtime/hybrid_gdn_config.h"
#include "ocr/runtime/memory_budget.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/p7_vlm_executor.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"
#include "ocr/tokenizer.h"

namespace luisa::ocr {

// HybridLayerKind / GdnLayerConfig / HybridCacheConfig / HybridGdnExecutorConfig
// / HybridGdnTensorNames / hybrid_gdn_expected_tensors are declared in the
// host-safe runtime/hybrid_gdn_config.h (included above) so host model modules
// can use the contract without the ops AttentionMaskKind clash.

// ---------------------------------------------------------------------------
// Hybrid cache (plan §7.2 items 1/3/4/5). Distinct stores that never collapse:
//   * contiguous K/V for the FULL-ATTENTION layers only;
//   * per-layer/head/request GDN recurrent state [gdn_heads, d, d];
//   * per-layer/request causal Conv1D ring [3*qk_dim, K-1] (q/k/v groups);
//   * device-visible position/step/state metadata (DecodeState).
// Capacities are checked from the ACTUAL geometry. Paged KV stays unsupported.
// ---------------------------------------------------------------------------
class HybridGdnCache final {
public:
    HybridGdnCache() noexcept = default;

    // Preflight + allocate + synchronously zero all state/metadata (drains
    // `stream`). `budget` (optional) is checked BEFORE allocation and updated
    // with the actual resident bytes on success.
    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const HybridCacheConfig &config, MemoryBudget *budget = nullptr);

    [[nodiscard]] const HybridCacheConfig &config() const noexcept { return _config; }
    [[nodiscard]] bool valid() const noexcept { return _created; }
    [[nodiscard]] uint64_t device_bytes() const noexcept;

    // Host committed-length bookkeeping (mirrors device metadata at sync).
    [[nodiscard]] uint32_t length(uint32_t index) const noexcept;
    [[nodiscard]] Status bind_sequence(uint32_t index, uint32_t prompt_len);
    [[nodiscard]] Status append_tokens(uint32_t index, uint32_t count);
    void release(uint32_t index) noexcept;

    [[nodiscard]] DecodeState &state() noexcept { return _state; }
    [[nodiscard]] const DecodeState &state() const noexcept { return _state; }

    // Full-attention per-layer views (layer is the FULL-ATTENTION-layer index,
    // 0..num_full_layers-1). Shape [batch, max_seq_len, kv_heads, head_dim].
    [[nodiscard]] compute::BufferView<float> k_layer(uint32_t full_layer) const noexcept;
    [[nodiscard]] compute::BufferView<float> v_layer(uint32_t full_layer) const noexcept;

    // GDN per-layer/request views (gdn_layer is 0..num_gdn_layers-1).
    // recurrent: [gdn_heads, d, d]; conv ring: [3*qk_dim, K-1] (q,k,v thirds).
    [[nodiscard]] compute::BufferView<float> recurrent(uint32_t gdn_layer, uint32_t index) const noexcept;
    [[nodiscard]] compute::BufferView<float> conv_ring(uint32_t gdn_layer, uint32_t index) const noexcept;

    // Download helpers (layered goldens; caller drains). Full-layer committed
    // KV prefix; GDN recurrent + conv ring for one request.
    [[nodiscard]] Status download_kv(compute::Stream &stream, uint32_t full_layer, uint32_t index,
                                     luisa::vector<float> &k_out, luisa::vector<float> &v_out) const;
    [[nodiscard]] Status download_recurrent(compute::Stream &stream, uint32_t gdn_layer, uint32_t index,
                                            luisa::vector<float> &out) const;
    [[nodiscard]] Status download_conv_ring(compute::Stream &stream, uint32_t gdn_layer, uint32_t index,
                                            luisa::vector<float> &out) const;

    // Reset ALL sequences: zero recurrent + conv ring + metadata (heap-owned
    // staging retained through completion). Legal only when no in-flight
    // command references the cache (caller drained).
    [[nodiscard]] Status reset(compute::CommandList &list);
    // Record a sequence's completion into device metadata (state=finished,
    // kv_len=0, position=0) so a released slot can never be attended on reuse.
    //
    // METADATA-ONLY BY CONTRACT: this does NOT clear the GDN recurrent matrix or
    // the Conv ring, and release() only frees the host slot. Stale GDN bytes may
    // therefore remain in a released slot. That is safe ONLY because EVERY
    // from-scratch prefill zeroes both stores for its request before any layer
    // work (HybridGdnExecutor::record_state_reset, recorded right after the
    // successful bind_sequence and ahead of every chunk dispatch in the same
    // command list). Full-attention KV needs no zeroing: attention is bounded by
    // kv_len / the causal prefix, so rows beyond the new length are never read.
    // A caller that bypasses prefill (records decode steps directly on a reused
    // slot) MUST call reset() first; reset() clears recurrent + conv + metadata
    // for ALL sequences.
    [[nodiscard]] Status record_finished(compute::CommandList &list, uint32_t index);

private:
    HybridCacheConfig _config;
    DecodeState _state;
    compute::Buffer<float> _k, _v;         // full-attention (empty when none)
    compute::Buffer<float> _recurrent;     // [gdn_layers*batch*gdn_heads*d*d]
    compute::Buffer<float> _conv_ring;     // [gdn_layers*batch*3*qk_dim*(K-1)]
    luisa::vector<uint32_t> _lengths;
    uint64_t _kv_layer_stride{0};          // batch*tokens*kv_heads*head_dim
    uint64_t _recurrent_layer_stride{0};   // batch*gdn_heads*d*d
    uint64_t _conv_layer_stride{0};        // batch*3*qk_dim*(K-1)
    bool _created{false};
    // Retained zero-fill shader (create-time zero init + reset()).
    luisa::unique_ptr<TypedShaderCache<ZeroFillKernel>::ShaderT> _zero_shader;
};

// ---------------------------------------------------------------------------
// Hybrid executor configuration + tensor-name contract + expected-tensor
// inventory are declared in runtime/hybrid_gdn_config.h (included above).
// ---------------------------------------------------------------------------

// The hybrid language executor. One instance serves one geometry; reusable
// across requests (workspace scratch overwritten per request). NOT thread-safe.
class HybridGdnExecutor final {
public:
    HybridGdnExecutor() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const HybridGdnExecutorConfig &config,
                                const HybridGdnTensorNames &names, const WeightStore &weights);

    // FROM-SCRATCH PREFILL CONTRACT (all three entry points below): after the
    // successful bind_sequence and BEFORE any chunk work, the executor zeroes
    // THIS request's GDN recurrent state AND Conv ring (record_state_reset,
    // same command list, ordered ahead of every layer dispatch). A reused cache
    // slot therefore always starts from zero GDN state even though
    // record_finished() is metadata-only -- two generate() calls on ONE cache
    // produce exactly the same logits/state as a fresh cache (device-tested with
    // NONZERO GDN weights). Full-attention KV is not zeroed (bounded by kv_len /
    // the causal prefix). Decode steps CONTINUE the prefilled state; they never
    // reset it.
    //
    // Straight-recurrent full-sequence prefill. logits_out (size vocab) gets
    // the LAST prompt row's logits after the caller drains.
    [[nodiscard]] Status record_prefill(compute::CommandList &list, const DecodeRequest &request,
                                        HybridGdnCache &cache, luisa::span<float> logits_out);
    // Prefill writing EVERY row's logits to logits_device ([prompt_len, vocab]).
    [[nodiscard]] Status record_prefill_all_logits(compute::CommandList &list, const DecodeRequest &request,
                                                   HybridGdnCache &cache, compute::BufferView<float> logits_device);
    // Chunk-aware prefill: GDN layers use the WY/UT chunk kernel over chunks of
    // `chunk_size` tokens (carried recurrent + conv state); full layers use
    // causal prefill over the growing KV. Numerically equivalent to the
    // recurrent path within the declared tolerance. chunk_size==0 => recurrent.
    [[nodiscard]] Status record_prefill_chunked(compute::CommandList &list, const DecodeRequest &request,
                                                HybridGdnCache &cache, uint32_t chunk_size,
                                                bool all_logits, luisa::span<float> last_logits_out,
                                                compute::BufferView<float> logits_device);

    // Single decode step at absolute position == cache.length(0). GDN layers
    // run the recurrent kernel with T=1 (NO sequence x sequence attention).
    [[nodiscard]] Status record_decode_step(compute::CommandList &list, uint32_t input_token,
                                            HybridGdnCache &cache, luisa::span<float> logits_out);

    [[nodiscard]] const HybridGdnExecutorConfig &config() const noexcept { return _config; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }
    [[nodiscard]] uint64_t workspace_bytes() const noexcept { return _workspace.capacity_bytes(); }
    [[nodiscard]] const ShaderKeyRegistry &shader_registry() const noexcept { return _shader_registry; }
    void reset_workspace() noexcept { _workspace.reset(); }

    // Map a global layer index to its full-attention / GDN cache index.
    [[nodiscard]] uint32_t full_index(uint32_t layer) const noexcept { return _full_index[layer]; }
    [[nodiscard]] uint32_t gdn_index(uint32_t layer) const noexcept { return _gdn_index[layer]; }

private:
    struct FullLayerWeights {
        compute::BufferView<float> input_norm, post_norm;
        compute::BufferView<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b, q_norm, k_norm;
        compute::BufferView<float> gate_w, up_w, down_w;
    };
    struct GdnLayerWeights {
        compute::BufferView<float> input_norm, post_norm;
        compute::BufferView<float> q_w, k_w, v_w, z_w, a_w, b_w, o_w;
        compute::BufferView<float> conv_q_w, conv_q_b, conv_k_w, conv_k_b, conv_v_w, conv_v_b;
        compute::BufferView<float> a_log, dt_bias, norm_w;
        compute::BufferView<float> gate_w, up_w, down_w;
    };

    [[nodiscard]] Status resolve_weights(compute::Stream &stream, const WeightStore &weights);
    [[nodiscard]] Status reserve_workspace();
    [[nodiscard]] Status compile_shaders();

    [[nodiscard]] Status record_prefill_impl(compute::CommandList &list, const DecodeRequest &request,
                                             HybridGdnCache &cache, uint32_t chunk_size, bool all_logits,
                                             luisa::span<float> last_logits_out, compute::BufferView<float> logits_device);
    // Zero THIS request's GDN recurrent state + Conv ring through the compiled
    // zero-fill shader. Recorded at the start of every from-scratch prefill
    // (after bind, before any chunk work) so a reused cache slot can never leak
    // the previous request's recurrent/conv state. Full-attention KV is NOT
    // zeroed: attention is bounded by kv_len / the causal prefix.
    [[nodiscard]] Status record_state_reset(compute::CommandList &list, HybridGdnCache &cache, uint32_t index);
    // One GDN layer over `rows` tokens at absolute pos0. chunked selects the
    // WY/UT chunk kernel; otherwise the straight recurrent kernel.
    [[nodiscard]] Status record_gdn_layer(compute::CommandList &list, uint32_t layer,
                                          compute::BufferView<float> x, uint32_t rows, uint32_t pos0,
                                          HybridGdnCache &cache, bool chunked);
    // One full-attention layer over `rows` tokens at absolute pos0.
    [[nodiscard]] Status record_full_layer(compute::CommandList &list, uint32_t layer,
                                           compute::BufferView<float> x, uint32_t rows, uint32_t pos0,
                                           HybridGdnCache &cache, bool decode);
    [[nodiscard]] Status record_mlp(compute::CommandList &list, compute::BufferView<float> x,
                                    compute::BufferView<float> post_norm, compute::BufferView<float> gate_w,
                                    compute::BufferView<float> up_w, compute::BufferView<float> down_w,
                                    uint32_t rows);
    [[nodiscard]] Status record_lm_head(compute::CommandList &list, compute::BufferView<float> x,
                                        uint32_t rows, compute::BufferView<float> C);
    [[nodiscard]] Status record_linear(compute::CommandList &list, compute::BufferView<float> A,
                                       compute::BufferView<float> B, compute::BufferView<float> C,
                                       compute::BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
                                       GemmEpilogue epilogue);
    [[nodiscard]] compute::BufferView<float> ws(const WorkspaceAllocation &a) const noexcept {
        return a.view<float>(_workspace.buffer());
    }

    HybridGdnExecutorConfig _config;
    HybridGdnTensorNames _names;
    compute::Device *_device{nullptr};
    DeviceWorkspace _workspace;
    uint64_t _weight_bytes{0};
    uint32_t _decode_step{0};
    luisa::vector<uint32_t> _full_index, _gdn_index;// global -> cache index

    compute::BufferView<float> _embed, _final_norm, _lm_head;
    compute::Buffer<float> _tied_lm_head;
    luisa::vector<FullLayerWeights> _full_layers;
    luisa::vector<GdnLayerWeights> _gdn_layers;

    // Workspace regions.
    WorkspaceAllocation _emb, _x, _norm, _norm2, _q, _k, _v, _q_rope, _k_rope, _attn,
        _gate, _up, _mlp, _logits, _ids, _positions, _axis_of_pair, _dummy_f, _dummy_u,
        _q_normed, _k_normed,
        _gq, _gk, _gv, _gz, _ga, _gb, _gq_conv, _gk_conv, _gv_conv, _gq_norm, _gk_norm,
        _gbeta, _gg, _go, _gynorm, _gw_scratch, _gu_scratch, _gc_scratch;
    luisa::vector<uint32_t> _axis_pending;

    ShaderKeyRegistry _shader_registry;
    TypedShaderCache<EmbeddingLookupKernel> _embedding_cache;
    TypedShaderCache<MmScatterKernel> _mm_scatter_cache;
    TypedShaderCache<RmsNormKernel> _rms_norm_cache;
    TypedShaderCache<L2NormKernel> _l2_norm_cache;
    TypedShaderCache<GemmKernel> _gemm_cache;
    TypedShaderCache<GemvKernel> _gemv_cache;
    TypedShaderCache<GatedActKernel> _gated_act_cache;
    TypedShaderCache<RopeAxialKernel> _rope_cache;
    TypedShaderCache<AttentionPrefillKernel> _attn_prefill_cache;
    TypedShaderCache<AttentionDecodeKernel> _attn_decode_cache;
    TypedShaderCache<KvAppendKernel> _kv_append_cache;
    TypedShaderCache<GdnConvSiluKernel> _gdn_conv_cache;
    TypedShaderCache<GdnGatesKernel> _gdn_gates_cache;
    TypedShaderCache<GdnRecurrentKernel> _gdn_recurrent_cache;
    TypedShaderCache<GdnChunkKernel> _gdn_chunk_cache;
    TypedShaderCache<GdnGatedNormKernel> _gdn_gated_norm_cache;
    TypedShaderCache<ZeroFillKernel> _zero_cache;

    const TypedShaderCache<EmbeddingLookupKernel>::ShaderT *_sh_embedding{nullptr};
    const TypedShaderCache<MmScatterKernel>::ShaderT *_sh_mm_scatter{nullptr};
    const TypedShaderCache<RmsNormKernel>::ShaderT *_sh_rms_norm{nullptr};
    const TypedShaderCache<L2NormKernel>::ShaderT *_sh_l2_norm{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_bias{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_residual{nullptr};
    const TypedShaderCache<GemmKernel>::ShaderT *_sh_gemm_none{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_bias{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_residual{nullptr};
    const TypedShaderCache<GemvKernel>::ShaderT *_sh_gemv_none{nullptr};
    const TypedShaderCache<GatedActKernel>::ShaderT *_sh_gated_silu{nullptr};
    const TypedShaderCache<RopeAxialKernel>::ShaderT *_sh_rope{nullptr};
    const TypedShaderCache<AttentionPrefillKernel>::ShaderT *_sh_attn_prefill{nullptr};
    const TypedShaderCache<AttentionDecodeKernel>::ShaderT *_sh_attn_decode{nullptr};
    const TypedShaderCache<KvAppendKernel>::ShaderT *_sh_kv_append{nullptr};
    const TypedShaderCache<GdnConvSiluKernel>::ShaderT *_sh_gdn_conv{nullptr};
    const TypedShaderCache<GdnGatesKernel>::ShaderT *_sh_gdn_gates{nullptr};
    const TypedShaderCache<GdnRecurrentKernel>::ShaderT *_sh_gdn_recurrent{nullptr};
    const TypedShaderCache<GdnChunkKernel>::ShaderT *_sh_gdn_chunk{nullptr};
    const TypedShaderCache<GdnGatedNormKernel>::ShaderT *_sh_gdn_gated_norm{nullptr};
    const TypedShaderCache<ZeroFillKernel>::ShaderT *_sh_zero{nullptr};
};

// ---------------------------------------------------------------------------
// Host decode controller (mirrors DenseDecodeController; reuses the Phase 3
// sampler + Phase 1 detokenizer). Drives prefill + single-step decode with
// greedy / validated sampling, EOS / stop-string handling, bounded output, and
// immediate cache release on completion.
// ---------------------------------------------------------------------------
struct HybridDecodeOptions {
    DecodeSamplingConfig sampling;
    uint32_t max_new_tokens{0};
    // 0 = straight-recurrent prefill; >0 = chunk-aware prefill with this chunk.
    uint32_t prefill_chunk_size{0};
    luisa::vector<uint32_t> eos_ids;
    luisa::vector<luisa::string> stop_strings;
    bool skip_special_tokens{true};
    double max_wall_time_ms{0.0};
    luisa::function<bool()> cancel_callback;

    [[nodiscard]] Status validate(uint32_t vocab_size) const;
};

struct HybridDecodeResult {
    luisa::vector<uint32_t> tokens;
    luisa::string text;
    bool hit_eos{false};
    bool hit_stop_string{false};
    bool truncated{false};
    uint32_t steps{0};
    luisa::vector<Diagnostic> diagnostics;
};

class HybridDecodeController final {
public:
    HybridDecodeController() noexcept = default;
    [[nodiscard]] Status create(HybridGdnExecutor &executor, const Tokenizer *tokenizer,
                                HybridDecodeOptions options);
    [[nodiscard]] Result<HybridDecodeResult> generate(compute::Stream &stream,
                                                      const DecodeRequest &request, HybridGdnCache &cache);
    [[nodiscard]] uint32_t last_committed_length() const noexcept { return _last_committed_length; }

private:
    HybridGdnExecutor *_executor{nullptr};
    const Tokenizer *_tokenizer{nullptr};
    HybridDecodeOptions _options;
    uint32_t _last_committed_length{0};
};

// ---------------------------------------------------------------------------
// Additive Phase 9: shared image -> vision -> hybrid-decode graph. Pairs the
// (vision-only) P7 vision tower + connector with the HybridGdnExecutor text
// decoder. Used by the OvisOCR2 / Surya synthetic vertical slices. The vision
// path is the tested P7 graph; the text path is the tested hybrid GDN graph.
// This is NOT a model and makes no support claim. SYNTHETIC ONLY.
// ---------------------------------------------------------------------------
struct HybridVlmGraphConfig {
    // Vision tower + connector + packing. vision.text.hidden_size MUST equal
    // decoder.decoder.hidden_size (the connector emits text-hidden features).
    // vision.text_pos must be rope1d (the hybrid text decoder is 1-axis).
    P7SliceConfig vision;
    P7TensorNames vision_names;
    HybridGdnExecutorConfig decoder;
    HybridGdnTensorNames decoder_names;
    uint32_t max_seq_len{256};
    MemoryLimits memory;

    [[nodiscard]] Status validate() const;
};

struct HybridVlmRunResult {
    HybridDecodeResult decode;
    uint32_t visual_tokens{0};
    uint64_t weight_bytes{0};
    uint64_t workspace_bytes{0};
    uint64_t cache_bytes{0};
};

class HybridVlmGraph final {
public:
    HybridVlmGraph() noexcept = default;

    [[nodiscard]] Status create(compute::Device &device, compute::Stream &stream,
                                const HybridVlmGraphConfig &config, const WeightStore &weights);

    // Run one image -> vision -> hybrid decode request. `prompt_tokens` surrounds
    // the visual rows [mm_start, mm_start+visual_tokens). A full cache reset is
    // recorded before each run as belt-and-braces; cross-request GDN state safety
    // itself is guaranteed by the executor's from-scratch prefill zeroing (see
    // the record_prefill contract above), so a bare executor + reused cache is
    // equally safe. The stream is driven internally.
    // `tokenizer` (BORROWED, optional): must outlive this call; it is used only
    // during generate() and is never retained by the graph. Null = token-only run
    // (decode.text stays empty by design). A text-producing caller MUST pass one.
    [[nodiscard]] Result<HybridVlmRunResult> run(compute::Stream &stream,
                                                 luisa::span<const P7ImageInput> images,
                                                 luisa::span<const uint32_t> prompt_tokens,
                                                 uint32_t mm_start, const HybridDecodeOptions &options,
                                                 const Tokenizer *tokenizer = nullptr);

    [[nodiscard]] const HybridVlmGraphConfig &config() const noexcept { return _config; }
    [[nodiscard]] HybridGdnExecutor &text_executor() noexcept { return _text; }
    [[nodiscard]] P7VlmExecutor &vision_executor() noexcept { return _vision; }
    [[nodiscard]] uint64_t weight_bytes() const noexcept { return _weight_bytes; }

    // Layered-golden downloads (last run; caller drains).
    [[nodiscard]] Status download_vision_features(compute::Stream &stream, luisa::vector<float> &out) const;

private:
    HybridVlmGraphConfig _config;
    compute::Device *_device{nullptr};
    P7VlmExecutor _vision;
    HybridGdnExecutor _text;
    HybridGdnCache _cache;
    uint64_t _weight_bytes{0};
};

}// namespace luisa::ocr

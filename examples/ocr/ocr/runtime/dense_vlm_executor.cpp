// OCR engine -- dense autoregressive decoder executor (Phase 3).
// Correctness-first FP32 dense decoder: separate prefill / decode loops,
// contiguous KV, host decode scheduling. See dense_vlm_executor.h.
#include "ocr/runtime/dense_vlm_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>

#include "ocr/runtime/dispatch_adapter.h"

namespace luisa::ocr {

using namespace luisa::compute;

namespace {

// Bytes of `count` fp32 elements (checked). int32/uint32 staging regions use
// the same 4-byte element size.
[[nodiscard]] Result<uint64_t> f32_bytes(uint64_t count) {
    return tensor_byte_size(TensorDType::f32, count);
}

[[nodiscard]] Status add_region(DeviceWorkspace &ws, WorkspaceAllocation &out, uint64_t elements) {
    auto bytes = f32_bytes(elements);
    if (!bytes.ok()) { return bytes.status; }
    auto alloc = ws.allocate(*bytes.value);
    if (!alloc.ok()) { return alloc.status; }
    out = *alloc.value;
    return Status::success();
}

// Per-record heap-owned upload staging. BufferUploadCommand borrows the host
// pointer, so uploaded ids/positions/axis data must outlive command EXECUTION,
// not just encoding. Each record_* call owns a distinct Staging block released
// by a completion callback, so multiple records into ONE CommandList never
// overwrite each other's borrowed memory (S3).
struct RecordUploadStaging {
    luisa::vector<int32_t> ids;      // token ids uploaded as int32
    luisa::vector<int32_t> positions;// absolute positions
    luisa::vector<uint32_t> axis;    // RoPE axis_of_pair map
};

}// namespace

// ---------------------------------------------------------------------------
// DenseExecutorConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseExecutorConfig::validate() const {
    auto s = decoder.validate();
    if (!s.ok()) { return s; }
    s = names.validate();
    if (!s.ok()) { return s; }
    if (component.empty()) {
        return Status::error(StatusCode::invalid_argument, "weight component tag must not be empty", "component");
    }
    if (prefill_chunk_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "prefill_chunk_rows must be nonzero", "prefill_chunk_rows");
    }
    // Additive Phase 4 switches: the defaults reproduce Phase 3 exactly.
    if (rope_axes != 1u && rope_axes != 3u) {
        return Status::error(StatusCode::invalid_argument, "rope_axes must be 1 (1D) or 3 (interleaved MRoPE)", "rope_axes");
    }
    if (rope_axes == 3u) {
        if (mrope_axis_of_pair.size() != decoder.head_dim / 2u) {
            return Status::error(StatusCode::invalid_argument,
                                 "mrope_axis_of_pair must have head_dim/2 entries", "mrope_axis_of_pair");
        }
        for (auto axis : mrope_axis_of_pair) {
            if (axis >= 3u) {
                return Status::error(StatusCode::invalid_argument,
                                     "mrope axis ids must be 0 (temporal), 1 (height), or 2 (width)", "mrope_axis_of_pair");
            }
        }
        if (decoder.rope_pair_layout != RopePairLayout::rotate_half) {
            return Status::error(StatusCode::invalid_argument,
                                 "interleaved MRoPE requires the rotate-half pair family", "rope_pair_layout");
        }
    }
    if (qk_norm && qkv_bias) {
        // The Qwen3 text contract drops QKV bias; keeping both would silently
        // mix two decoder contracts. Explicit refusal, never a guess.
        return Status::error(StatusCode::invalid_argument,
                             "qk_norm (Qwen3) and qkv_bias (Qwen2.5) are mutually exclusive contracts", "qk_norm/qkv_bias");
    }
    for (uint32_t i = 0; i < deepstack_layers.size(); i++) {
        if (deepstack_layers[i] >= decoder.num_layers) {
            return Status::error(StatusCode::invalid_argument,
                                 "deepstack injection layer is outside the decoder", "deepstack_layers");
        }
        if (i > 0u && deepstack_layers[i] <= deepstack_layers[i - 1u]) {
            return Status::error(StatusCode::invalid_argument,
                                 "deepstack injection layers must be strictly increasing", "deepstack_layers");
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseVlmExecutor::create(Device &device, Stream &stream,
                                              const DenseExecutorConfig &config,
                                              const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    _workspace = DeviceWorkspace{device};
    _embedding_cache = TypedShaderCache<EmbeddingLookupKernel>{device, _shader_registry};
    _mm_scatter_cache = TypedShaderCache<MmScatterKernel>{device, _shader_registry};
    _rms_norm_cache = TypedShaderCache<RmsNormKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _gemv_cache = TypedShaderCache<GemvKernel>{device, _shader_registry};
    _gated_act_cache = TypedShaderCache<GatedActKernel>{device, _shader_registry};
    _rope_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _rope_rotate_half_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _rope_mrope_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _attn_prefill_cache = TypedShaderCache<AttentionPrefillKernel>{device, _shader_registry};
    _attn_decode_cache = TypedShaderCache<AttentionDecodeKernel>{device, _shader_registry};
    _kv_append_cache = TypedShaderCache<KvAppendKernel>{device, _shader_registry};
    _deepstack_cache = TypedShaderCache<DeepStackInjectKernel>{device, _shader_registry};

    auto s = resolve_weights(stream, weights);
    if (!s.ok()) { return s; }
    s = reserve_workspace();
    if (!s.ok()) { return s; }
    return compile_shaders();
}

[[nodiscard]] Status DenseVlmExecutor::resolve_weights(Stream &stream, const WeightStore &weights) {
    const auto &names = _config.names;
    const auto &comp = _config.component;
    const auto &dec = _config.decoder;

    // Build the exact expected tensor inventory and index it by name. Every
    // weight is validated against its expected shape + layout + element count
    // BEFORE a view is bound (M1): an undersized or wrong-layout tensor is
    // refused here, never dispatched (no OOB, no silently-wrong logits).
    // The inventory follows the ACTIVE contract: Phase 3 (QKV bias, no
    // QK-Norm) by default; Qwen3 (QK-Norm, no QKV bias) when configured.
    auto expected = dense_decoder_expected_tensors(dec, names, _config.tie_word_embeddings);
    if (_config.qk_norm) {
        // Drop the Qwen2.5 QKV-bias entries and add the Qwen3 QK-Norm weights.
        luisa::vector<ExpectedTensor> qwen3_expected;
        for (auto &e : expected) {
            bool is_qkv_bias = false;
            for (uint32_t l = 0; l < dec.num_layers && !is_qkv_bias; l++) {
                is_qkv_bias = e.name == names.q_bias(l) || e.name == names.k_bias(l) || e.name == names.v_bias(l);
            }
            if (!is_qkv_bias) { qwen3_expected.push_back(std::move(e)); }
        }
        for (uint32_t l = 0; l < dec.num_layers; l++) {
            qwen3_expected.emplace_back(ExpectedTensor{names.q_norm(l), {dec.head_dim}, TensorLayout::row_major, true});
            qwen3_expected.emplace_back(ExpectedTensor{names.k_norm(l), {dec.head_dim}, TensorLayout::row_major, true});
        }
        expected = std::move(qwen3_expected);
    }
    luisa::unordered_map<luisa::string, const ExpectedTensor *> expected_by_name;
    for (const auto &e : expected) { expected_by_name.emplace(e.name, &e); }

    // Reject unexpected tensors in the decoder component: every installed
    // tensor must be part of the declared inventory (vision/other components
    // live under their own component tag, not here).
    for (const auto &name : weights.tensor_names(comp)) {
        if (expected_by_name.find(name) == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected decoder tensor '{}' in component '{}'", name, comp), name);
        }
    }

    auto view_of = [&](const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("decoder weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("decoder weight '{}' must be dense fp32 in Phase 3", name), name);
        }
        const auto it = expected_by_name.find(name);
        if (it == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("decoder weight '{}' has no expected-tensor contract", name), name);
        }
        // Exact shape + layout + element-count gate before binding.
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        // The bound device buffer must actually hold the declared elements.
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("decoder weight '{}' device buffer is smaller than its element count", name),
                                 name);
        }
        out = w->dense.view();
        return Status::success();
    };
    if (auto s = view_of(names.embed_tokens(), _embed); !s.ok()) { return s; }
    if (auto s = view_of(names.final_norm(), _final_norm); !s.ok()) { return s; }
    // LM head: standalone tensor, else tied to the embedding table. The GEMM
    // B operand needs [hidden, vocab]; the embedding table is [vocab, hidden],
    // so a tied head is transposed ONCE here into an executor-owned buffer.
    if (const auto *head = weights.find(comp, names.lm_head()); head != nullptr) {
        if (auto s = view_of(names.lm_head(), _lm_head); !s.ok()) { return s; }
    } else if (_config.tie_word_embeddings) {
        const uint64_t vocab = dec.vocab_size, hidden = dec.hidden_size;
        uint64_t head_elems = 0;
        if (!checked_mul(vocab, hidden, head_elems)) {
            return Status::error(StatusCode::out_of_memory, "tied LM head element count overflows", "lm_head");
        }
        auto head_bytes = f32_bytes(head_elems);
        if (!head_bytes.ok()) { return head_bytes.status; }
        // S5: preflight the executor-owned transpose buffer (device) and its
        // host staging BEFORE allocation. The staging holds two copies (the
        // downloaded embedding + the transposed upload) at peak.
        const uint64_t staging_bytes = *head_bytes.value * 2ull;
        MemoryBudget head_budget{_config.memory};
        if (auto s = head_budget.require("dense_tied_head", MemoryCategory::weights, *head_bytes.value); !s.ok()) { return s; }
        if (auto s = head_budget.require("dense_tied_head", MemoryCategory::host_staging, staging_bytes); !s.ok()) { return s; }
        if (auto s = head_budget.preflight(); !s.ok()) { return s; }
        luisa::vector<float> emb(static_cast<size_t>(head_elems)), transposed(static_cast<size_t>(head_elems));
        stream << _embed.copy_to(luisa::span<float>{emb}) << synchronize();
        for (uint64_t v = 0; v < vocab; v++) {
            for (uint64_t h = 0; h < hidden; h++) {
                transposed[static_cast<size_t>(h * vocab + v)] = emb[static_cast<size_t>(v * hidden + h)];
            }
        }
        _tied_lm_head = _device->create_buffer<float>(static_cast<size_t>(head_elems));
        if (!_tied_lm_head) {
            return Status::error(StatusCode::out_of_memory, "failed to allocate tied LM head", "lm_head");
        }
        stream << _tied_lm_head.copy_from(luisa::span<const float>{transposed}) << synchronize();
        _lm_head = _tied_lm_head.view();
        _tied_head_staging_bytes = staging_bytes;
    } else {
        return Status::error(StatusCode::not_found,
                             luisa::format("lm_head '{}' is missing and embeddings are not tied", names.lm_head()),
                             names.lm_head());
    }
    _layers.resize(dec.num_layers);
    for (uint32_t l = 0; l < dec.num_layers; l++) {
        auto &L = _layers[l];
        if (auto s = view_of(names.input_layernorm(l), L.input_norm); !s.ok()) { return s; }
        if (auto s = view_of(names.post_attention_layernorm(l), L.post_norm); !s.ok()) { return s; }
        if (auto s = view_of(names.q_proj(l), L.q_w); !s.ok()) { return s; }
        if (auto s = view_of(names.k_proj(l), L.k_w); !s.ok()) { return s; }
        if (auto s = view_of(names.v_proj(l), L.v_w); !s.ok()) { return s; }
        if (auto s = view_of(names.o_proj(l), L.o_w); !s.ok()) { return s; }
        if (_config.qkv_bias) {
            if (auto s = view_of(names.q_bias(l), L.q_b); !s.ok()) { return s; }
            if (auto s = view_of(names.k_bias(l), L.k_b); !s.ok()) { return s; }
            if (auto s = view_of(names.v_bias(l), L.v_b); !s.ok()) { return s; }
        }
        if (auto s = view_of(names.gate_proj(l), L.gate_w); !s.ok()) { return s; }
        if (auto s = view_of(names.up_proj(l), L.up_w); !s.ok()) { return s; }
        if (auto s = view_of(names.down_proj(l), L.down_w); !s.ok()) { return s; }
        if (_config.qk_norm) {
            if (auto s = view_of(names.q_norm(l), L.q_norm); !s.ok()) { return s; }
            if (auto s = view_of(names.k_norm(l), L.k_norm); !s.ok()) { return s; }
        }
    }
    // Weight bytes include the borrowed component weights PLUS the executor-
    // owned tied-LM-head transpose buffer (S5).
    _weight_bytes = weights.component_bytes(comp) + owned_device_bytes();
    return Status::success();
}

[[nodiscard]] Status DenseVlmExecutor::reserve_workspace() {
    const auto &d = _config.decoder;
    const uint64_t chunk = _config.prefill_chunk_rows;
    const uint64_t hidden = d.hidden_size;
    const uint64_t qd = d.q_dim();
    const uint64_t kvd = d.kv_dim();
    const uint64_t inter = d.intermediate_size;
    const uint64_t vocab = d.vocab_size;

    // Plan the full layout first (checked) so the budget preflight sees the
    // exact requirement before any allocation. Region order here MUST match
    // the materialization order below.
    struct RegionSpec {
        const char *name;
        uint64_t elements;
    };
    // Additive Phase 4 regions. These are ALWAYS at least one element so a
    // typed view is legal even on the Phase 3 path (a zero-byte view cannot
    // be reinterpreted); they are only USED when the contract enables them.
    const uint64_t qk_norm_elems = _config.qk_norm ? chunk * qd : 1ull;
    const uint64_t k_norm_elems = _config.qk_norm ? chunk * kvd : 1ull;
    const uint64_t ds_elems = _config.deepstack_layers.empty() ? 1ull : chunk * hidden;
    const RegionSpec specs[] = {
        {"emb", chunk * hidden},
        {"x", chunk * hidden},
        {"norm", chunk * hidden},
        {"norm2", chunk * hidden},
        {"q", chunk * qd},
        {"k", chunk * kvd},
        {"v", chunk * kvd},
        {"q_rope", chunk * qd},
        {"k_rope", chunk * kvd},
        {"attn", chunk * qd},
        {"gate", chunk * inter},
        {"up", chunk * inter},
        {"mlp", chunk * inter},
        {"logits", vocab},
        {"ids", chunk},
        {"positions", chunk * _config.rope_axes},
        {"axis_of_pair", d.head_dim / 2u},
        {"dummy_f", 1},
        {"dummy_u", 1},
        // Additive Phase 4 regions (zero-sized on the Phase 3 path).
        {"q_normed", qk_norm_elems},
        {"k_normed", k_norm_elems},
        {"ds_scratch", ds_elems},
    };
    WorkspacePlan plan;
    for (const auto &spec : specs) {
        auto bytes = f32_bytes(spec.elements);
        if (!bytes.ok()) { return bytes.status; }
        if (auto s = plan.add_bytes(spec.name, *bytes.value); !s.ok()) { return s; }
    }
    // Preflight the workspace arena PLUS any executor-owned device buffer
    // (the tied-LM-head transpose) and its host staging against the limits,
    // so the combined executor footprint is bounded before allocation (S5).
    MemoryBudget budget{_config.memory};
    if (auto s = budget.require("dense_workspace", MemoryCategory::workspace, plan.total_bytes()); !s.ok()) { return s; }
    if (owned_device_bytes() != 0u) {
        if (auto s = budget.require("dense_tied_head", MemoryCategory::weights, owned_device_bytes()); !s.ok()) { return s; }
    }
    if (_tied_head_staging_bytes != 0u) {
        if (auto s = budget.require("dense_tied_head", MemoryCategory::host_staging, _tied_head_staging_bytes); !s.ok()) { return s; }
    }
    if (auto s = budget.preflight(); !s.ok()) { return s; }
    if (auto s = _workspace.reserve(plan.total_bytes()); !s.ok()) { return s; }

    WorkspaceAllocation *outs[] = {
        &_emb, &_x, &_norm, &_norm2, &_q, &_k, &_v, &_q_rope, &_k_rope,
        &_attn, &_gate, &_up, &_mlp, &_logits, &_ids, &_positions,
        &_axis_of_pair, &_dummy_f, &_dummy_u,
        &_q_normed, &_k_normed, &_ds_scratch,
    };
    for (size_t i = 0; i < std::size(specs); i++) {
        if (auto s = add_region(_workspace, *outs[i], specs[i].elements); !s.ok()) { return s; }
    }

    // axis_of_pair is constant per contract: 1D RoPE maps every pair to axis
    // 0; interleaved MRoPE uses the explicit configured section map. Stage it
    // once; every record call uploads it before first use.
    if (_config.rope_axes == 3u) {
        _axis_pending = _config.mrope_axis_of_pair;
    } else {
        _axis_pending.assign(d.head_dim / 2u, 0u);
    }
    return Status::success();
}

[[nodiscard]] Status DenseVlmExecutor::compile_shaders() {
    const auto get = [](auto &cache, const luisa::string &key, auto factory, auto *&out) -> Status {
        auto r = cache.get_or_compile(key, factory);
        if (!r.ok()) { return r.status; }
        out = *r.value;
        return Status::success();
    };
    if (auto s = get(_embedding_cache, decode_variant_key("embedding_lookup"),
                     [] { return make_embedding_lookup_kernel(); }, _sh_embedding); !s.ok()) { return s; }
    if (auto s = get(_mm_scatter_cache, decode_variant_key("mm_scatter"),
                     [] { return make_mm_scatter_kernel(); }, _sh_mm_scatter); !s.ok()) { return s; }
    if (auto s = get(_rms_norm_cache, norm_variant_key("rms_norm"),
                     [] { return make_rms_norm_kernel(); }, _sh_rms_norm); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::residual); }, _sh_gemm_residual); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::none); }, _sh_gemm_none); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, true),
                     [] { return make_gemv_kernel(GemmEpilogue::bias); }, _sh_gemv_bias); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, true),
                     [] { return make_gemv_kernel(GemmEpilogue::residual); }, _sh_gemv_residual); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, true),
                     [] { return make_gemv_kernel(GemmEpilogue::none); }, _sh_gemv_none); !s.ok()) { return s; }
    if (auto s = get(_gated_act_cache, make_shader_key("gated", "act0"),
                     [] { return make_gated_act_kernel(0u); }, _sh_gated_silu); !s.ok()) { return s; }
    // Select the RoPE pair layout explicitly from the config (adjacent vs
    // rotate-half). The two are distinct factories/variant keys; a Qwen2.5
    // rotate-half decoder never silently uses the adjacent synthetic kernel.
    if (_config.rope_axes == 3u) {
        // Interleaved MRoPE: distinct named factory/variant key; never an
        // alias of the 1-axis rotate-half text kernel.
        if (auto s = get(_rope_mrope_cache, rope_variant_key("interleaved_mrope"),
                         [] { return make_rope_interleaved_mrope_kernel(); }, _sh_rope); !s.ok()) { return s; }
    } else if (_config.decoder.rope_pair_layout == RopePairLayout::rotate_half) {
        if (auto s = get(_rope_rotate_half_cache, rope_variant_key("rotate_half"),
                         [] { return make_rope_rotate_half_kernel(); }, _sh_rope); !s.ok()) { return s; }
    } else {
        if (auto s = get(_rope_cache, rope_variant_key("1d"),
                         [] { return make_rope_1d_kernel(); }, _sh_rope); !s.ok()) { return s; }
    }
    if (!_config.deepstack_layers.empty()) {
        if (auto s = get(_deepstack_cache, vision_variant_key("deepstack_inject"),
                         [] { return make_deepstack_inject_kernel(); }, _sh_deepstack); !s.ok()) { return s; }
    }
    if (auto s = get(_attn_prefill_cache, attention_variant_key("prefill_reference"),
                     [] { return make_attention_prefill_reference_kernel(); }, _sh_attn_prefill); !s.ok()) { return s; }
    if (auto s = get(_attn_decode_cache, attention_variant_key("decode"),
                     [] { return make_attention_decode_kernel(); }, _sh_attn_decode); !s.ok()) { return s; }
    if (auto s = get(_kv_append_cache, decode_variant_key("kv_append"),
                     [] { return make_kv_append_kernel(); }, _sh_kv_append); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// linear helper
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseVlmExecutor::record_linear(CommandList &list,
                                                     BufferView<float> A, BufferView<float> B,
                                                     BufferView<float> C, BufferView<float> aux,
                                                     uint32_t m, uint32_t n, uint32_t k,
                                                     GemmEpilogue epilogue) {
    if (m == 0u || n == 0u || k == 0u) {
        return Status::error(StatusCode::invalid_argument, "linear dimensions must be nonzero", "m/n/k");
    }
    const auto aux_view = epilogue == GemmEpilogue::none ? ws(_dummy_f) : aux;
    if (m <= kGemvSmallM) {
        auto plan = plan_gemv_dispatch(m, n, k);
        if (!plan.ok()) { return plan.status; }
        const auto *shader = epilogue == GemmEpilogue::bias ? _sh_gemv_bias
                             : epilogue == GemmEpilogue::residual ? _sh_gemv_residual
                                                                  : _sh_gemv_none;
        if (auto cmd = dispatch_1d((*shader)(A, B, C, aux_view, m, n, k), *plan.value)) {
            list << std::move(cmd);
        }
        return Status::success();
    }
    auto plan = plan_gemm_dispatch_epilogue(m, n, k, epilogue);
    if (!plan.ok()) { return plan.status; }
    const auto *shader = epilogue == GemmEpilogue::bias ? _sh_gemm_bias
                         : epilogue == GemmEpilogue::residual ? _sh_gemm_residual
                                                              : _sh_gemm_none;
    if (auto cmd = dispatch_2d((*shader)(A, B, C, aux_view, m, n, k, plan.value->padded_cols), *plan.value)) {
        list << std::move(cmd);
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// transformer block
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseVlmExecutor::record_block(CommandList &list, uint32_t layer,
                                                    BufferView<float> x,
                                                    uint32_t rows, uint32_t pos0,
                                                    ContiguousKvCache &cache, bool decode,
                                                    luisa::span<const BufferView<float>> deepstack,
                                                    uint32_t ds_start, uint32_t ds_count) {
    const auto &d = _config.decoder;
    const auto &L = _layers[layer];
    const uint32_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t qh = d.num_attention_heads, kvh = d.num_kv_heads, hd = d.head_dim;
    const uint32_t inter = d.intermediate_size;
    const float eps = d.rms_norm_eps;
    const float theta = d.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    auto norm_view = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    auto norm2_view = ws(_norm2).subview(0, static_cast<size_t>(rows) * hidden);
    auto q_view = ws(_q).subview(0, static_cast<size_t>(rows) * qd);
    auto k_view = ws(_k).subview(0, static_cast<size_t>(rows) * kvd);
    auto v_view = ws(_v).subview(0, static_cast<size_t>(rows) * kvd);
    auto q_rope_view = ws(_q_rope).subview(0, static_cast<size_t>(rows) * qd);
    auto k_rope_view = ws(_k_rope).subview(0, static_cast<size_t>(rows) * kvd);
    auto attn_view = ws(_attn).subview(0, static_cast<size_t>(rows) * qd);
    auto gate_view = ws(_gate).subview(0, static_cast<size_t>(rows) * inter);
    auto up_view = ws(_up).subview(0, static_cast<size_t>(rows) * inter);
    auto mlp_view = ws(_mlp).subview(0, static_cast<size_t>(rows) * inter);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, static_cast<size_t>(rows) * _config.rope_axes);
    auto axis_view = ws(_axis_of_pair).as<uint32_t>();
    auto dummy_f = ws(_dummy_f);
    auto dummy_u = ws(_dummy_u).as<uint32_t>();

    // 1. input RMSNorm: norm = rmsnorm(x).
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm_view, L.input_norm, rows, hidden, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. QKV projections (bias only under the Qwen2.5 contract; the Qwen3
    //    contract has no QKV bias).
    const auto qkv_epilogue = _config.qkv_bias ? GemmEpilogue::bias : GemmEpilogue::none;
    if (auto s = record_linear(list, norm_view, L.q_w, q_view, L.q_b, rows, qd, hidden, qkv_epilogue); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm_view, L.k_w, k_view, L.k_b, rows, kvd, hidden, qkv_epilogue); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm_view, L.v_w, v_view, L.v_b, rows, kvd, hidden, qkv_epilogue); !s.ok()) { return s; }
    // 2b. Qwen3 QK-Norm: per-head RMSNorm over head_dim (rows = tokens*heads),
    //     applied BEFORE RoPE. Outputs go to dedicated regions (in-place norm
    //     would race across the per-row threads).
    auto q_for_rope = q_view;
    auto k_for_rope = k_view;
    if (_config.qk_norm) {
        auto q_normed = ws(_q_normed).subview(0, static_cast<size_t>(rows) * qd);
        auto k_normed = ws(_k_normed).subview(0, static_cast<size_t>(rows) * kvd);
        auto plan_qn = plan_norm_dispatch(static_cast<uint64_t>(rows) * qh);
        if (!plan_qn.ok()) { return plan_qn.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(q_view, q_normed, L.q_norm,
                                                   rows * qh, hd, eps, 1u), *plan_qn.value)) {
            list << std::move(cmd);
        }
        auto plan_kn = plan_norm_dispatch(static_cast<uint64_t>(rows) * kvh);
        if (!plan_kn.ok()) { return plan_kn.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(k_view, k_normed, L.k_norm,
                                                   rows * kvh, hd, eps, 1u), *plan_kn.value)) {
            list << std::move(cmd);
        }
        q_for_rope = q_normed;
        k_for_rope = k_normed;
    }
    // 3. RoPE on Q and K (pair layout per config: adjacent (2d,2d+1),
    //    rotate-half (d, d+head_dim/2), or 3-axis interleaved MRoPE).
    {
        auto plan_q = plan_rope_dispatch(rows, qh);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rope)(q_for_rope, q_rope_view, pos_view, axis_view, rows, qh, hd, _config.rope_axes, theta), *plan_q.value)) {
            list << std::move(cmd);
        }
        auto plan_k = plan_rope_dispatch(rows, kvh);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope)(k_for_rope, k_rope_view, pos_view, axis_view, rows, kvh, hd, _config.rope_axes, theta), *plan_k.value)) {
            list << std::move(cmd);
        }
    }
    // 4. Append the RoPE'd K and the V to the contiguous cache at pos0.
    {
        auto s = validate_kv_append(pos0, rows, cache.config().max_seq_len);
        if (!s.ok()) { return s; }
        auto plan = plan_kv_append_dispatch(rows, kvd);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_kv_append)(k_rope_view, v_view, cache.k_layer(layer), cache.v_layer(layer),
                                     rows, kvd, pos0), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 5. Attention.
    if (decode) {
        // Single query over the committed KV prefix [0, pos0+1). The device
        // metadata (kv_len/positions) was recorded by record_decode_step.
        auto plan = plan_attention_decode_dispatch(1, qh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_decode)(q_rope_view, cache.k_layer(layer), cache.v_layer(layer),
                                     attn_view, cache.state().kv_len(), cache.state().positions(), dummy_u,
                                     1u, cache.config().max_seq_len, qh, kvh, hd, scale, 0.0f,
                                     static_cast<uint32_t>(AttentionMaskKind::causal), 0u, 0u, 0u, 0u), *plan.value)) {
            list << std::move(cmd);
        }
    } else {
        // Causal prefill: queries [pos0, pos0+rows) attend keys [0, pos0+rows)
        // already written to the cache. The causal diagonal is end-aligned,
        // so q_abs = qi + (k_len - q_len) = pos0 + qi.
        const uint32_t k_len = pos0 + rows;
        auto plan = plan_attention_prefill_dispatch(rows, qh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_prefill)(q_rope_view, cache.k_layer(layer), cache.v_layer(layer),
                                      attn_view, dummy_u, dummy_f, rows, k_len, qh, kvh, hd, scale, 0.0f,
                                      static_cast<uint32_t>(AttentionMaskKind::causal), 0u, 0u, 0u, 0u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 6. Output projection + residual: x = x + attn @ Wo.
    if (auto s = record_linear(list, attn_view, L.o_w, x, x, rows, hidden, qd, GemmEpilogue::residual); !s.ok()) { return s; }
    // 7. Post-attention RMSNorm: norm2 = rmsnorm(x).
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm2_view, L.post_norm, rows, hidden, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 8. SwiGLU MLP: gate/up projections, silu(gate)*up, down + residual.
    if (auto s = record_linear(list, norm2_view, L.gate_w, gate_view, dummy_f, rows, inter, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm2_view, L.up_w, up_view, dummy_f, rows, inter, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    {
        const uint64_t count = static_cast<uint64_t>(rows) * inter;
        auto plan = plan_gated_act_dispatch(count);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_gated_silu)(gate_view, up_view, mlp_view, static_cast<uint32_t>(count)), *plan.value)) {
            list << std::move(cmd);
        }
    }
    if (auto s = record_linear(list, mlp_view, L.down_w, x, x, rows, hidden, inter, GemmEpilogue::residual); !s.ok()) { return s; }
    // 9. DeepStack injection (Qwen3): AFTER this decoder layer's residual
    //    stream, add the i-th projected visual feature to the image rows
    //    [ds_start, ds_start+ds_count) when this layer is selected. The
    //    injection is chunk-local: only the overlap with [pos0, pos0+rows)
    //    is written, and the feature rows are offset accordingly.
    if (!deepstack.empty()) {
        for (uint32_t i = 0; i < deepstack.size(); i++) {
            if (layer != _config.deepstack_layers[i]) { continue; }
            const uint64_t ds_end = static_cast<uint64_t>(ds_start) + ds_count;
            const uint64_t chunk_end = static_cast<uint64_t>(pos0) + rows;
            if (ds_start >= chunk_end || pos0 >= ds_end) { continue; }
            const uint32_t local_start = ds_start > pos0 ? ds_start - pos0 : 0u;
            const uint32_t local_end = ds_end < chunk_end ? static_cast<uint32_t>(ds_end - pos0) : rows;
            const uint32_t count = local_end - local_start;
            const uint32_t feat_offset = (pos0 + local_start) - ds_start;
            auto proj = deepstack[i].subview(static_cast<size_t>(feat_offset) * hidden,
                                             static_cast<size_t>(count) * hidden);
            auto scratch = ws(_ds_scratch).subview(0, static_cast<size_t>(rows) * hidden);
            auto plan = plan_deepstack_inject_dispatch(rows, hidden);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_deepstack)(x, proj, scratch, rows, hidden, local_start, count), *plan.value)) {
                list << std::move(cmd);
            }
            // Copy the injected residual back into x (the inject kernel
            // writes a separate output; x is the live residual stream).
            // mm_scatter with count=0 is a plain row copy.
            auto plan_copy = plan_mm_scatter_dispatch(rows, hidden);
            if (!plan_copy.ok()) { return plan_copy.status; }
            if (auto cmd = dispatch_1d((*_sh_mm_scatter)(scratch, ws(_dummy_f), x, rows, hidden, 0u, 0u), *plan_copy.value)) {
                list << std::move(cmd);
            }
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// final norm + LM head
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseVlmExecutor::record_lm_head(CommandList &list, BufferView<float> x,
                                                      uint32_t rows, BufferView<float> C) {
    const auto &d = _config.decoder;
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    auto norm_view = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    auto plan = plan_norm_dispatch(rows);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm_view, _final_norm, rows, hidden, d.rms_norm_eps, 1u), *plan.value)) {
        list << std::move(cmd);
    }
    return record_linear(list, norm_view, _lm_head, C, ws(_dummy_f), rows, vocab, hidden, GemmEpilogue::none);
}

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseVlmExecutor::record_prefill(CommandList &list, const DecodeRequest &request,
                                                      ContiguousKvCache &cache, luisa::span<float> logits_out) {
    if (_config.rope_axes != 1u) {
        // M1: the 1-axis entry point cannot supply per-token M-RoPE positions;
        // silently prefilling with sequential positions would produce wrong
        // logits. Mirror the decode guard: refuse, never guess.
        return Status::error(StatusCode::invalid_argument,
                             "a rope_axes!=1 executor requires record_prefill_qwen3 with explicit 3-axis positions",
                             "rope_axes");
    }
    return record_prefill_impl(list, request, {}, {}, cache, false, logits_out, {});
}

[[nodiscard]] Status DenseVlmExecutor::record_prefill_all_logits(CommandList &list, const DecodeRequest &request,
                                                                 ContiguousKvCache &cache, BufferView<float> logits_device) {
    if (_config.rope_axes != 1u) {
        return Status::error(StatusCode::invalid_argument,
                             "a rope_axes!=1 executor requires record_prefill_qwen3 with explicit 3-axis positions",
                             "rope_axes");
    }
    return record_prefill_impl(list, request, {}, {}, cache, true, {}, logits_device);
}

[[nodiscard]] Status DenseVlmExecutor::record_prefill_qwen3(CommandList &list, const DecodeRequest &request,
                                                            luisa::span<const int32_t> position_ids,
                                                            luisa::span<const BufferView<float>> deepstack_features,
                                                            ContiguousKvCache &cache, bool all_logits,
                                                            luisa::span<float> last_logits_out,
                                                            BufferView<float> logits_device) {
    if (_config.rope_axes != 3u) {
        return Status::error(StatusCode::invalid_argument,
                             "record_prefill_qwen3 requires a rope_axes==3 (interleaved MRoPE) executor", "rope_axes");
    }
    if (position_ids.size() != request.prompt_tokens.size() * 3u) {
        return Status::error(StatusCode::invalid_argument,
                             "position_ids must have prompt_len*3 entries (temporal, height, width)", "position_ids");
    }
    if (deepstack_features.size() != _config.deepstack_layers.size()) {
        return Status::error(StatusCode::invalid_argument,
                             "deepstack_features must match the configured injection layers", "deepstack_features");
    }
    if (!deepstack_features.empty() && !request.has_multimodal()) {
        return Status::error(StatusCode::invalid_argument,
                             "deepstack features require an image (multimodal) range in the prompt", "deepstack_features");
    }
    const uint64_t hidden = _config.decoder.hidden_size;
    for (const auto &f : deepstack_features) {
        if (!f || f.size() < static_cast<size_t>(request.mm_count) * hidden) {
            return Status::error(StatusCode::invalid_argument,
                                 "a deepstack feature view is invalid or too small for the image rows", "deepstack_features");
        }
    }
    return record_prefill_impl(list, request, position_ids, deepstack_features, cache, all_logits,
                               last_logits_out, logits_device);
}

[[nodiscard]] Status DenseVlmExecutor::record_prefill_impl(CommandList &list, const DecodeRequest &request,
                                                           luisa::span<const int32_t> position_ids,
                                                           luisa::span<const BufferView<float>> deepstack,
                                                           ContiguousKvCache &cache, bool all_logits,
                                                           luisa::span<float> last_logits_out,
                                                           BufferView<float> logits_device) {
    auto valid = request.validate();
    if (!valid.ok()) { return valid; }
    const auto &d = _config.decoder;
    const uint32_t prompt_len = static_cast<uint32_t>(request.prompt_tokens.size());
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    if (prompt_len > cache.config().max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("prompt length {} exceeds KV capacity {}", prompt_len, cache.config().max_seq_len),
                             "prompt_len");
    }
    if (cache.config().num_layers != d.num_layers || cache.config().num_kv_heads != d.num_kv_heads ||
        cache.config().head_dim != d.head_dim || cache.config().batch != 1u) {
        return Status::error(StatusCode::invalid_argument, "KV cache geometry does not match the decoder", "cache");
    }
    if (!all_logits && last_logits_out.size() != vocab) {
        return Status::error(StatusCode::invalid_argument, "logits_out size must equal vocab_size", "logits_out");
    }
    if (all_logits && logits_device.size() < static_cast<size_t>(prompt_len) * vocab) {
        return Status::error(StatusCode::invalid_argument, "logits_device too small for all prompt rows", "logits_device");
    }
    if (request.has_multimodal() && request.mm_features.size() < static_cast<size_t>(request.mm_count) * hidden) {
        return Status::error(StatusCode::invalid_argument, "mm_features too small for mm_count rows", "mm_features");
    }
    // M2: reject EVERY out-of-vocabulary prompt token before any staging or
    // dispatch (an OOB id would read past the embedding table on the device).
    for (size_t i = 0; i < request.prompt_tokens.size(); i++) {
        if (request.prompt_tokens[i] >= vocab) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("prompt token {} at index {} is outside the vocabulary (vocab_size {})",
                                               request.prompt_tokens[i], i, vocab),
                                 "prompt_tokens");
        }
    }

    // Bind the sequence (host bookkeeping). A completed sequence releases its
    // slot immediately (release()), so a cache can be reused across requests
    // without a device reset; stale KV rows beyond the new length are never
    // read (attention is bounded by kv_len / the causal prefix).
    if (auto s = cache.bind_sequence(0, prompt_len); !s.ok()) { return s; }
    _decode_step = 0;

    // Per-record heap-owned staging (S3): the ids/positions/axis uploads borrow
    // host pointers that must outlive command EXECUTION. One staging block per
    // record_prefill call, retained by a completion callback, so two prefill
    // calls into one CommandList never overwrite each other's borrowed memory.
    auto staging = luisa::make_unique<RecordUploadStaging>();
    staging->ids.assign(request.prompt_tokens.begin(), request.prompt_tokens.end());
    staging->positions.resize(static_cast<size_t>(prompt_len) * _config.rope_axes);
    if (!position_ids.empty()) {
        // Explicit per-token M-RoPE positions (Qwen3 path).
        std::copy(position_ids.begin(), position_ids.end(), staging->positions.begin());
    } else {
        // Phase 3 sequential 1-axis positions.
        for (uint32_t i = 0; i < prompt_len; i++) { staging->positions[i] = static_cast<int32_t>(i); }
    }
    staging->axis = _axis_pending;
    auto *stage = staging.get();

    auto chunks = plan_row_chunks(prompt_len, _config.prefill_chunk_rows);
    if (!chunks.ok()) { return chunks.status; }
    auto x_full = ws(_x);
    auto emb_full = ws(_emb);
    auto ids_full = ws(_ids).as<int32_t>();
    auto pos_full = ws(_positions).as<int32_t>();
    for (uint64_t ci = 0; ci < chunks.value->chunk_count; ci++) {
        auto range = chunks.value->chunk_range(ci);
        if (!range.ok()) { return range.status; }
        const uint32_t pos0 = static_cast<uint32_t>(range.value->first);
        const uint32_t rows = static_cast<uint32_t>(range.value->second - range.value->first);
        auto x = x_full.subview(0, static_cast<size_t>(rows) * hidden);
        auto emb = emb_full.subview(0, static_cast<size_t>(rows) * hidden);

        // Upload this chunk's ids + positions + the constant RoPE axis map
        // from the per-record heap staging (stable through completion).
        list << ids_full.subview(0, rows).copy_from(luisa::span<const int32_t>{stage->ids.data() + pos0, rows})
             << pos_full.subview(0, static_cast<size_t>(rows) * _config.rope_axes)
                    .copy_from(luisa::span<const int32_t>{stage->positions.data() + static_cast<size_t>(pos0) * _config.rope_axes,
                                                          static_cast<size_t>(rows) * _config.rope_axes})
             << ws(_axis_of_pair).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});
        auto ids_view = ids_full.subview(0, rows);
        // Embedding lookup: emb = table[ids].
        {
            auto plan = plan_embedding_dispatch(rows, hidden);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_embedding)(_embed, ids_view, emb, rows, hidden), *plan.value)) {
                list << std::move(cmd);
            }
        }
        // Multimodal replacement (only the chunk overlapping the mm range);
        // otherwise a plain copy of the embeddings into the residual stream.
        uint32_t local_start = 0u, count = 0u, feat_offset = 0u;
        bool overlap = false;
        if (request.has_multimodal()) {
            const uint64_t mm_end = static_cast<uint64_t>(request.mm_start) + request.mm_count;
            const uint64_t chunk_end = static_cast<uint64_t>(pos0) + rows;
            if (request.mm_start < chunk_end && pos0 < mm_end) {
                overlap = true;
                local_start = request.mm_start > pos0 ? request.mm_start - pos0 : 0u;
                const uint32_t local_end = mm_end < chunk_end ? static_cast<uint32_t>(mm_end - pos0) : rows;
                count = local_end - local_start;
                feat_offset = (pos0 + local_start) - request.mm_start;
            }
        }
        {
            auto plan = plan_mm_scatter_dispatch(rows, hidden);
            if (!plan.ok()) { return plan.status; }
            auto feats = overlap ? request.mm_features.subview(static_cast<size_t>(feat_offset) * hidden,
                                                            static_cast<size_t>(count) * hidden)
                                 : ws(_dummy_f);
            if (auto cmd = dispatch_1d((*_sh_mm_scatter)(emb, feats, x, rows, hidden, local_start, count), *plan.value)) {
                list << std::move(cmd);
            }
        }
        // Decoder blocks (with DeepStack injection at the exact configured
        // layers when features are present).
        for (uint32_t l = 0; l < d.num_layers; l++) {
            if (auto s = record_block(list, l, x, rows, pos0, cache, false,
                                      deepstack, request.mm_start, request.mm_count); !s.ok()) { return s; }
        }
        // Logits.
        if (all_logits) {
            auto C = logits_device.subview(static_cast<size_t>(pos0) * vocab, static_cast<size_t>(rows) * vocab);
            if (auto s = record_lm_head(list, x, rows, C); !s.ok()) { return s; }
        } else if (ci + 1u == chunks.value->chunk_count) {
            // Last chunk: only the final row's logits are needed.
            auto last_row = x.subview(static_cast<size_t>(rows - 1u) * hidden, hidden);
            if (auto s = record_lm_head(list, last_row, 1u, ws(_logits)); !s.ok()) { return s; }
        }
    }

    // Record device metadata: prefill state, kv_len = prompt_len, next
    // decode position = prompt_len, step = 0. DecodeState::record copies the
    // snapshot into its own completion-retained staging, so these stack-local
    // vectors are safe.
    const luisa::vector<uint32_t> meta_kv_len = {prompt_len};
    const luisa::vector<int32_t> meta_positions = {static_cast<int32_t>(prompt_len)};
    const luisa::vector<uint32_t> meta_step = {0u};
    const luisa::vector<uint32_t> meta_state = {static_cast<uint32_t>(DecodeSeqState::prefill)};
    if (auto s = cache.state().record(list, meta_kv_len, meta_positions, meta_step, meta_state); !s.ok()) { return s; }

    // Download last-row logits (host span; caller drains before reading).
    if (!all_logits) {
        list << ws(_logits).copy_to(last_logits_out);
    }
    // Retain the upload staging through command completion (S3).
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return Status::success();
}

// ---------------------------------------------------------------------------
// single-step decode
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseVlmExecutor::record_decode_step(CommandList &list, uint32_t input_token,
                                                          ContiguousKvCache &cache, luisa::span<float> logits_out) {
    if (_config.rope_axes == 3u) {
        // The 1-axis entry point cannot know the M-RoPE text position; the
        // Qwen3 entry point must be used (never a silent position guess).
        return Status::error(StatusCode::invalid_argument,
                             "a rope_axes==3 executor requires record_decode_step_qwen3 with an explicit position",
                             "rope_axes");
    }
    return record_decode_step_impl(list, input_token, static_cast<int32_t>(cache.length(0)), cache, logits_out);
}

[[nodiscard]] Status DenseVlmExecutor::record_decode_step_qwen3(CommandList &list, uint32_t input_token,
                                                                int32_t position, ContiguousKvCache &cache,
                                                                luisa::span<float> logits_out) {
    if (_config.rope_axes != 3u) {
        return Status::error(StatusCode::invalid_argument,
                             "record_decode_step_qwen3 requires a rope_axes==3 executor", "rope_axes");
    }
    if (position < 0) {
        return Status::error(StatusCode::invalid_argument, "decode position must be nonnegative", "position");
    }
    return record_decode_step_impl(list, input_token, position, cache, logits_out);
}

[[nodiscard]] Status DenseVlmExecutor::record_decode_step_impl(CommandList &list, uint32_t input_token,
                                                               int32_t position, ContiguousKvCache &cache,
                                                               luisa::span<float> logits_out) {
    const auto &d = _config.decoder;
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    if (logits_out.size() != vocab) {
        return Status::error(StatusCode::invalid_argument, "logits_out size must equal vocab_size", "logits_out");
    }
    if (cache.config().batch != 1u) {
        return Status::error(StatusCode::unsupported, "Phase 3 decode is batch 1 only", "batch");
    }
    const uint32_t pos = cache.length(0);
    if (pos == 0u) {
        return Status::error(StatusCode::invalid_argument, "decode step before prefill (empty KV)", "position");
    }
    if (pos >= cache.config().max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("decode position {} reached KV capacity {}", pos, cache.config().max_seq_len),
                             "position");
    }
    if (input_token >= vocab) {
        return Status::error(StatusCode::invalid_argument, "decode input token outside the vocabulary", "input_token");
    }
    if (_config.rope_axes == 1u && position != static_cast<int32_t>(pos)) {
        // The 1-axis path derives the position from the KV length; a caller
        // passing a different one would silently desynchronize RoPE.
        return Status::error(StatusCode::invalid_argument,
                             "1-axis decode position must equal the committed KV length", "position");
    }
    // Record device metadata BEFORE the blocks: the decode attention reads
    // kv_len = pos+1 and position = pos (the new row is appended below).
    // DecodeState::record copies into completion-retained staging, so these
    // stack-local vectors are safe across multiple records in one list.
    const luisa::vector<uint32_t> meta_kv_len = {pos + 1u};
    const luisa::vector<int32_t> meta_positions = {static_cast<int32_t>(pos)};
    const luisa::vector<uint32_t> meta_step = {_decode_step + 1u};
    const luisa::vector<uint32_t> meta_state = {static_cast<uint32_t>(DecodeSeqState::decoding)};
    if (auto s = cache.state().record(list, meta_kv_len, meta_positions, meta_step, meta_state); !s.ok()) { return s; }

    // Per-record heap-owned staging for the single token id + position + axis
    // map (S3): retained through completion so two decode steps recorded into
    // one CommandList never overwrite each other's borrowed upload memory.
    auto staging = luisa::make_unique<RecordUploadStaging>();
    staging->ids = {static_cast<int32_t>(input_token)};
    staging->positions.assign(_config.rope_axes, position);
    staging->axis = _axis_pending;
    auto *stage = staging.get();
    auto ids_view = ws(_ids).as<int32_t>().subview(0, 1);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, _config.rope_axes);
    list << ids_view.copy_from(luisa::span<const int32_t>{stage->ids})
         << pos_view.copy_from(luisa::span<const int32_t>{stage->positions})
         << ws(_axis_of_pair).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});

    auto x = ws(_x).subview(0, hidden);
    auto emb = ws(_emb).subview(0, hidden);
    {
        auto plan = plan_embedding_dispatch(1, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_embedding)(_embed, ids_view, emb, 1u, hidden), *plan.value)) {
            list << std::move(cmd);
        }
    }
    {
        auto plan = plan_mm_scatter_dispatch(1, hidden);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_mm_scatter)(emb, ws(_dummy_f), x, 1u, hidden, 0u, 0u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (auto s = record_block(list, l, x, 1u, pos, cache, true); !s.ok()) { return s; }
    }
    if (auto s = record_lm_head(list, x, 1u, ws(_logits)); !s.ok()) { return s; }
    list << ws(_logits).copy_to(logits_out);

    // Advance host bookkeeping (mirrors the device metadata recorded above).
    if (auto s = cache.append_tokens(0, 1u); !s.ok()) { return s; }
    _decode_step++;
    // Retain the upload staging through command completion (S3).
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return Status::success();
}

// ---------------------------------------------------------------------------
// DenseDecodeOptions / Controller
// ---------------------------------------------------------------------------

[[nodiscard]] Status DenseDecodeOptions::validate(uint32_t vocab_size) const {
    if (max_new_tokens == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_new_tokens must be nonzero", "max_new_tokens");
    }
    auto s = validate_decode_sampling(sampling, vocab_size);
    if (!s.ok()) { return s; }
    for (auto id : eos_ids) {
        if (id >= vocab_size) {
            return Status::error(StatusCode::invalid_argument, "EOS id outside the vocabulary", "eos_ids");
        }
    }
    for (const auto &stop : stop_strings) {
        if (stop.empty()) {
            return Status::error(StatusCode::invalid_argument, "stop strings must not be empty", "stop_strings");
        }
    }
    if (max_wall_time_ms < 0.0 || !std::isfinite(max_wall_time_ms)) {
        return Status::error(StatusCode::invalid_argument,
                             "max_wall_time_ms must be finite and >= 0", "max_wall_time_ms");
    }
    return Status::success();
}

[[nodiscard]] Status DenseDecodeController::create(DenseVlmExecutor &executor, const Tokenizer *tokenizer,
                                                   DenseDecodeOptions options) {
    auto valid = options.validate(executor.config().decoder.vocab_size);
    if (!valid.ok()) { return valid; }
    _executor = &executor;
    _tokenizer = tokenizer;
    _options = std::move(options);
    return Status::success();
}

[[nodiscard]] Result<DenseDecodeResult> DenseDecodeController::generate(Stream &stream,
                                                                        const DecodeRequest &request,
                                                                        ContiguousKvCache &cache) {
    using R = Result<DenseDecodeResult>;
    if (_executor == nullptr) {
        return R::failure(StatusCode::invalid_argument, "decode controller is not created", "controller");
    }
    auto valid = request.validate();
    if (!valid.ok()) { return R::failure(valid); }
    const auto &d = _executor->config().decoder;
    const uint32_t vocab = d.vocab_size;

    // S7: cancellation is polled before prefill and at every per-step
    // synchronization boundary; the wall-time budget is checked at the same
    // boundaries. The clock starts here (after argument validation).
    Clock clock;
    const auto cancelled = [&]() {
        return _options.cancel_callback && _options.cancel_callback();
    };
    const auto timed_out = [&]() {
        return _options.max_wall_time_ms > 0.0 && clock.toc() > _options.max_wall_time_ms;
    };
    if (cancelled()) {
        return R::failure(StatusCode::cancelled, "request cancelled before prefill", "cancel");
    }
    if (timed_out()) {
        return R::failure(StatusCode::timeout, "wall-time budget exceeded before prefill", "max_wall_time_ms");
    }

    DenseDecodeResult result;
    luisa::vector<float> logits(vocab);

    // S1: EOS ids = decoder config eos_token_ids + explicit options + the
    // tokenizer's own eos, deduplicated (order-preserving).
    luisa::vector<uint32_t> eos_ids;
    const auto add_eos = [&eos_ids](uint32_t id) {
        if (std::find(eos_ids.begin(), eos_ids.end(), id) == eos_ids.end()) {
            eos_ids.push_back(id);
        }
    };
    for (auto id : d.eos_token_ids) { add_eos(id); }
    for (auto id : _options.eos_ids) { add_eos(id); }
    if (_tokenizer != nullptr && _tokenizer->eos_id().has_value()) {
        add_eos(*_tokenizer->eos_id());
    }
    const auto is_eos = [&eos_ids](uint32_t id) {
        return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
    };

    // Incremental detokenizer (only when a tokenizer is present).
    luisa::optional<IncrementalDetokenizer> detok;
    if (_tokenizer != nullptr) {
        DetokenizerOptions dopts;
        dopts.skip_special_tokens = _options.skip_special_tokens;
        dopts.stop_strings = _options.stop_strings;
        dopts.eos_ids = eos_ids;
        detok.emplace(*_tokenizer, std::move(dopts));
    }

    // Prefill. record_prefill binds the cache slot; once bound, EVERY exit
    // path must release it (S2). The RAII guard is ARMED only after a
    // successful prefill (the slot is bound exactly when record_prefill
    // succeeds). It releases the host slot on any return (success or failure)
    // and records the device-side finished state through a real API (S8) so a
    // released slot can never be attended on reuse. The finished record is
    // committed + synchronized in the guard before the caller drains.
    struct ReleaseGuard {
        ContiguousKvCache *cache;
        Stream *stream;
        bool armed{false};
        ~ReleaseGuard() noexcept {
            if (!armed) { return; }
            auto list = CommandList::create();
            if (cache->record_finished(list, 0).ok()) {
                *stream << list.commit() << synchronize();
            }
            cache->release(0);
        }
    } guard{&cache, &stream, false};
    {
        auto list = CommandList::create();
        auto s = _executor->record_prefill(list, request, cache, logits);
        if (!s.ok()) { return R::failure(s); }
        stream << list.commit() << synchronize();
        guard.armed = true;
    }

    // Deterministic sampler over the full prompt history.
    luisa::vector<uint32_t> prompt32(request.prompt_tokens.begin(), request.prompt_tokens.end());
    DeterministicSampler sampler{_options.sampling, vocab, prompt32};

    for (uint32_t step = 0; step < _options.max_new_tokens; step++) {
        auto picked = sampler.sample(logits);
        if (!picked.ok()) { return R::failure(picked.status); }
        const uint32_t token = *picked.value;
        result.tokens.push_back(token);
        result.steps = step + 1u;

        // EOS: stop BEFORE emitting (the EOS token is never detokenized).
        if (is_eos(token)) {
            result.hit_eos = true;
            break;
        }
        // Detokenize incrementally; stop strings never leak.
        if (detok.has_value()) {
            auto text = detok->push(token);
            if (!text.ok()) { return R::failure(text.status); }
            result.text += *text.value;
            if (detok->stopped()) {
                result.hit_stop_string = true;
                break;
            }
        }
        // Cache capacity: the next decode step would write at position
        // cache.length(0); stop if that reaches capacity.
        if (cache.length(0) >= cache.config().max_seq_len) {
            result.truncated = true;
            break;
        }
        if (step + 1u == _options.max_new_tokens) {
            result.truncated = true;
            break;
        }
        // S7: poll cancellation + wall-time at the per-step synchronization
        // boundary BEFORE recording/dispatching the next decode step. The
        // guard releases the cache on these failures.
        if (cancelled()) {
            return R::failure(StatusCode::cancelled, "request cancelled at a decode boundary", "cancel");
        }
        if (timed_out()) {
            return R::failure(StatusCode::timeout, "wall-time budget exceeded at a decode boundary", "max_wall_time_ms");
        }
        // Decode one step to produce the next logits.
        auto list = CommandList::create();
        auto s = _executor->record_decode_step(list, token, cache, logits);
        if (!s.ok()) { return R::failure(s); }
        stream << list.commit() << synchronize();
    }

    // Flush any pending detokenizer text (complete UTF-8 only).
    if (detok.has_value() && !detok->finished()) {
        auto tail = detok->finish();
        if (!tail.ok()) { return R::failure(tail.status); }
        result.text += *tail.value;
    }
    if (result.truncated) {
        result.diagnostics.push_back(Diagnostic{
            DiagnosticSeverity::warning, "output_truncated",
            luisa::format("generation stopped at the {}-token output budget before EOS/stop",
                          _options.max_new_tokens)});
    }
    // The ReleaseGuard releases the cache slot + records finished state on
    // return (success or failure).
    return R::success(std::move(result));
}

}// namespace luisa::ocr
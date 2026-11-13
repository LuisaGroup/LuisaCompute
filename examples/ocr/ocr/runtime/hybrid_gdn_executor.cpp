// OCR engine -- hybrid GDN + periodic full-attention decoder executor (Phase 9).
// The GDN math is the declared synthetic contract in ops/ops_gdn.h. The layer
// schedule EXPLICITLY mixes GDN and full GQA attention layers. SYNTHETIC ONLY.
#include "ocr/runtime/hybrid_gdn_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>

#include "ocr/quantization.h"

namespace luisa::ocr {

using namespace luisa::compute;

// hybrid_layer_kind_name / GdnLayerConfig / HybridCacheConfig capacity math /
// HybridGdnExecutorConfig / HybridGdnTensorNames / hybrid_gdn_expected_tensors
// live in runtime/hybrid_gdn_config.cpp (the shared host-safe contract). This
// TU implements the device cache, executor, and host decode controller.

// ---------------------------------------------------------------------------
// HybridGdnCache
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnCache::create(Device &device, Stream &stream,
                                            const HybridCacheConfig &config, MemoryBudget *budget) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    auto total = config.total_bytes();
    if (!total.ok()) { return total.status; }
    if (budget != nullptr) {
        auto pre = budget->preflight_additional_device(*total.value);
        if (!pre.ok()) { return pre; }
    }
    auto kv = config.kv_bytes();
    auto rec = config.recurrent_bytes();
    auto conv = config.conv_bytes();
    if (!kv.ok() || !rec.ok() || !conv.ok()) {
        return Status::error(StatusCode::invalid_argument, "hybrid cache byte plan failed", "capacity");
    }
    // Element counts (FP32). K and V are equal halves of the KV bytes.
    const uint64_t kv_elems = *kv.value / 4ull / 2ull;// per K (or V) buffer
    const uint64_t rec_elems = *rec.value / 4ull;
    const uint64_t conv_elems = *conv.value / 4ull;
    Buffer<float> k, v, recurrent, conv_ring;
    if (kv_elems != 0u) {
        k = device.create_buffer<float>(static_cast<size_t>(kv_elems));
        v = device.create_buffer<float>(static_cast<size_t>(kv_elems));
        if (!k || !v) {
            return Status::error(StatusCode::out_of_memory, "failed to allocate hybrid KV cache", "kv");
        }
    }
    if (rec_elems != 0u) {
        recurrent = device.create_buffer<float>(static_cast<size_t>(rec_elems));
        if (!recurrent) {
            return Status::error(StatusCode::out_of_memory, "failed to allocate GDN recurrent state", "recurrent");
        }
    }
    if (conv_elems != 0u) {
        conv_ring = device.create_buffer<float>(static_cast<size_t>(conv_elems));
        if (!conv_ring) {
            return Status::error(StatusCode::out_of_memory, "failed to allocate GDN conv ring state", "conv");
        }
    }
    // Synchronous zero-init of the device metadata (drains the stream).
    auto state_status = _state.create(device, stream, config.batch);
    if (!state_status.ok()) { return state_status; }
    // Compile + retain the zero-fill shader for create-time init and reset().
    auto zero_shader = device.compile(make_zero_fill_kernel());
    if (!zero_shader || !zero_shader.compile_ok()) {
        return Status::error(StatusCode::internal_error, "failed to compile hybrid zero-fill shader", "zero");
    }
    _zero_shader = luisa::make_unique<TypedShaderCache<ZeroFillKernel>::ShaderT>(std::move(zero_shader));
    // Zero the recurrent + conv state so an unbound request never reads stale
    // state. The staging-free zero shader writes constants; drain before return.
    if (rec_elems != 0u) {
        auto plan = plan_gdn_elementwise_dispatch(rec_elems);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_zero_shader)(recurrent, static_cast<uint32_t>(rec_elems)), *plan.value)) {
            stream << std::move(cmd);
        }
    }
    if (conv_elems != 0u) {
        auto plan = plan_gdn_elementwise_dispatch(conv_elems);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_zero_shader)(conv_ring, static_cast<uint32_t>(conv_elems)), *plan.value)) {
            stream << std::move(cmd);
        }
    }
    stream << synchronize();

    _config = config;
    _k = std::move(k);
    _v = std::move(v);
    _recurrent = std::move(recurrent);
    _conv_ring = std::move(conv_ring);
    _kv_layer_stride = static_cast<uint64_t>(config.batch) * config.max_seq_len * config.num_kv_heads * config.head_dim;
    _recurrent_layer_stride = static_cast<uint64_t>(config.batch) * config.gdn_heads * config.gdn_head_dim * config.gdn_head_dim;
    _conv_layer_stride = static_cast<uint64_t>(config.batch) * 3ull * config.gdn_qk_dim() * config.conv_ring_width();
    _lengths.assign(config.batch, 0u);
    _created = true;
    if (budget != nullptr) {
        auto s = budget->require("hybrid_cache", MemoryCategory::kv_cache, device_bytes());
        if (!s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] uint64_t HybridGdnCache::device_bytes() const noexcept {
    if (!_created) { return 0u; }
    uint64_t bytes = _state.device_bytes();
    if (_k) { bytes += _k.size_bytes() + _v.size_bytes(); }
    if (_recurrent) { bytes += _recurrent.size_bytes(); }
    if (_conv_ring) { bytes += _conv_ring.size_bytes(); }
    return bytes;
}

[[nodiscard]] uint32_t HybridGdnCache::length(uint32_t index) const noexcept {
    return index < _lengths.size() ? _lengths[index] : 0u;
}
[[nodiscard]] Status HybridGdnCache::bind_sequence(uint32_t index, uint32_t prompt_len) {
    if (!_created || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "hybrid sequence index out of range", "index");
    }
    if (_lengths[index] != 0u) {
        return Status::error(StatusCode::invalid_argument, "hybrid sequence slot is occupied; release() it first", "index");
    }
    if (prompt_len > _config.max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("prompt length {} exceeds hybrid capacity {}", prompt_len, _config.max_seq_len),
                             "prompt_len");
    }
    _lengths[index] = prompt_len;
    return Status::success();
}
[[nodiscard]] Status HybridGdnCache::append_tokens(uint32_t index, uint32_t count) {
    if (index >= _lengths.size()) {
        return Status::error(StatusCode::invalid_argument, "hybrid sequence index out of range", "index");
    }
    const uint64_t next = static_cast<uint64_t>(_lengths[index]) + count;
    if (next > _config.max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("hybrid append of {} tokens overruns capacity {}", count, _config.max_seq_len),
                             "count");
    }
    _lengths[index] = static_cast<uint32_t>(next);
    return Status::success();
}
void HybridGdnCache::release(uint32_t index) noexcept {
    if (index < _lengths.size()) { _lengths[index] = 0u; }
}

[[nodiscard]] BufferView<float> HybridGdnCache::k_layer(uint32_t full_layer) const noexcept {
    return _k.view(static_cast<size_t>(full_layer) * static_cast<size_t>(_kv_layer_stride),
                   static_cast<size_t>(_kv_layer_stride));
}
[[nodiscard]] BufferView<float> HybridGdnCache::v_layer(uint32_t full_layer) const noexcept {
    return _v.view(static_cast<size_t>(full_layer) * static_cast<size_t>(_kv_layer_stride),
                   static_cast<size_t>(_kv_layer_stride));
}
[[nodiscard]] BufferView<float> HybridGdnCache::recurrent(uint32_t gdn_layer, uint32_t index) const noexcept {
    const uint64_t per_req = static_cast<uint64_t>(_config.gdn_heads) * _config.gdn_head_dim * _config.gdn_head_dim;
    const uint64_t base = static_cast<uint64_t>(gdn_layer) * _recurrent_layer_stride + static_cast<uint64_t>(index) * per_req;
    return _recurrent.view(static_cast<size_t>(base), static_cast<size_t>(per_req));
}
[[nodiscard]] BufferView<float> HybridGdnCache::conv_ring(uint32_t gdn_layer, uint32_t index) const noexcept {
    const uint64_t per_req = 3ull * _config.gdn_qk_dim() * _config.conv_ring_width();
    const uint64_t base = static_cast<uint64_t>(gdn_layer) * _conv_layer_stride + static_cast<uint64_t>(index) * per_req;
    return _conv_ring.view(static_cast<size_t>(base), static_cast<size_t>(per_req));
}

[[nodiscard]] Status HybridGdnCache::download_kv(Stream &stream, uint32_t full_layer, uint32_t index,
                                                 luisa::vector<float> &k_out, luisa::vector<float> &v_out) const {
    if (!_created || full_layer >= _config.num_full_layers || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "hybrid KV download index out of range", "layer/index");
    }
    const uint32_t len = _lengths[index];
    const size_t row = static_cast<size_t>(_config.num_kv_heads) * _config.head_dim;
    const size_t count = static_cast<size_t>(len) * row;
    k_out.resize(count);
    v_out.resize(count);
    if (count == 0u) { return Status::success(); }
    const size_t base = static_cast<size_t>(full_layer) * static_cast<size_t>(_kv_layer_stride) +
                        static_cast<size_t>(index) * static_cast<size_t>(_config.max_seq_len) * row;
    stream << _k.view(base, count).copy_to(luisa::span<float>{k_out})
           << _v.view(base, count).copy_to(luisa::span<float>{v_out});
    return Status::success();
}
[[nodiscard]] Status HybridGdnCache::download_recurrent(Stream &stream, uint32_t gdn_layer, uint32_t index,
                                                        luisa::vector<float> &out) const {
    if (!_created || gdn_layer >= _config.num_gdn_layers || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "hybrid recurrent download index out of range", "layer/index");
    }
    auto view = recurrent(gdn_layer, index);
    out.resize(view.size());
    stream << view.copy_to(luisa::span<float>{out});
    return Status::success();
}
[[nodiscard]] Status HybridGdnCache::download_conv_ring(Stream &stream, uint32_t gdn_layer, uint32_t index,
                                                        luisa::vector<float> &out) const {
    if (!_created || gdn_layer >= _config.num_gdn_layers || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "hybrid conv download index out of range", "layer/index");
    }
    auto view = conv_ring(gdn_layer, index);
    out.resize(view.size());
    stream << view.copy_to(luisa::span<float>{out});
    return Status::success();
}

[[nodiscard]] Status HybridGdnCache::reset(CommandList &list) {
    if (!_created) {
        return Status::error(StatusCode::invalid_argument, "hybrid cache is not created", "cache");
    }
    // Zero recurrent + conv state through the retained zero shader.
    if (_recurrent) {
        const uint64_t n = _recurrent.size();
        auto plan = plan_gdn_elementwise_dispatch(n);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_zero_shader)(_recurrent, static_cast<uint32_t>(n)), *plan.value)) {
            list << std::move(cmd);
        }
    }
    if (_conv_ring) {
        const uint64_t n = _conv_ring.size();
        auto plan = plan_gdn_elementwise_dispatch(n);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_zero_shader)(_conv_ring, static_cast<uint32_t>(n)), *plan.value)) {
            list << std::move(cmd);
        }
    }
    const luisa::vector<uint32_t> zeros(_config.batch, 0u);
    const luisa::vector<int32_t> pos_zeros(_config.batch, 0);
    const luisa::vector<uint32_t> states(_config.batch, static_cast<uint32_t>(DecodeSeqState::empty));
    auto s = _state.record(list, zeros, pos_zeros, zeros, states);
    if (!s.ok()) { return s; }
    std::fill(_lengths.begin(), _lengths.end(), 0u);
    return Status::success();
}

[[nodiscard]] Status HybridGdnCache::record_finished(CommandList &list, uint32_t index) {
    if (!_created || index >= _config.batch) {
        return Status::error(StatusCode::invalid_argument, "hybrid sequence index out of range", "index");
    }
    auto s = _state.record_sequence(list, index, 0u, 0, DecodeSeqState::finished);
    if (!s.ok()) { return s; }
    if (index < _lengths.size()) { _lengths[index] = 0u; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// HybridGdnExecutor: create / resolve / reserve / compile
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnExecutor::create(Device &device, Stream &stream,
                                               const HybridGdnExecutorConfig &config,
                                               const HybridGdnTensorNames &names, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    valid = names.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _names = names;
    _device = &device;
    _workspace = DeviceWorkspace{device};
    _embedding_cache = TypedShaderCache<EmbeddingLookupKernel>{device, _shader_registry};
    _mm_scatter_cache = TypedShaderCache<MmScatterKernel>{device, _shader_registry};
    _rms_norm_cache = TypedShaderCache<RmsNormKernel>{device, _shader_registry};
    _l2_norm_cache = TypedShaderCache<L2NormKernel>{device, _shader_registry};
    _gemm_cache = TypedShaderCache<GemmKernel>{device, _shader_registry};
    _gemv_cache = TypedShaderCache<GemvKernel>{device, _shader_registry};
    _gated_act_cache = TypedShaderCache<GatedActKernel>{device, _shader_registry};
    _rope_cache = TypedShaderCache<RopeAxialKernel>{device, _shader_registry};
    _attn_prefill_cache = TypedShaderCache<AttentionPrefillKernel>{device, _shader_registry};
    _attn_decode_cache = TypedShaderCache<AttentionDecodeKernel>{device, _shader_registry};
    _kv_append_cache = TypedShaderCache<KvAppendKernel>{device, _shader_registry};
    _gdn_conv_cache = TypedShaderCache<GdnConvSiluKernel>{device, _shader_registry};
    _gdn_gates_cache = TypedShaderCache<GdnGatesKernel>{device, _shader_registry};
    _gdn_recurrent_cache = TypedShaderCache<GdnRecurrentKernel>{device, _shader_registry};
    _gdn_chunk_cache = TypedShaderCache<GdnChunkKernel>{device, _shader_registry};
    _gdn_gated_norm_cache = TypedShaderCache<GdnGatedNormKernel>{device, _shader_registry};
    _zero_cache = TypedShaderCache<ZeroFillKernel>{device, _shader_registry};

    // Global -> cache index maps.
    _full_index.assign(config.decoder.num_layers, 0u);
    _gdn_index.assign(config.decoder.num_layers, 0u);
    uint32_t fi = 0, gi = 0;
    for (uint32_t l = 0; l < config.decoder.num_layers; l++) {
        if (config.layer_kinds[l] == HybridLayerKind::full_attention) { _full_index[l] = fi++; }
        else { _gdn_index[l] = gi++; }
    }
    // 1-axis RoPE axis map (all pairs -> axis 0).
    _axis_pending.assign(config.decoder.head_dim / 2u, 0u);

    if (auto s = resolve_weights(stream, weights); !s.ok()) { return s; }
    if (auto s = reserve_workspace(); !s.ok()) { return s; }
    return compile_shaders();
}

[[nodiscard]] Status HybridGdnExecutor::resolve_weights(Stream &stream, const WeightStore &weights) {
    const auto &comp = _config.component;
    auto expected = hybrid_gdn_expected_tensors(_config, _names);
    luisa::unordered_map<luisa::string, const ExpectedTensor *> expected_by_name;
    for (const auto &e : expected) { expected_by_name.emplace(e.name, &e); }
    for (const auto &name : weights.tensor_names(comp)) {
        if (expected_by_name.find(name) == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected hybrid tensor '{}' in component '{}'", name, comp), name);
        }
    }
    auto view_of = [&](const luisa::string &name, BufferView<float> &out) -> Status {
        const auto *w = weights.find(comp, name);
        if (w == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("hybrid weight '{}' not found in component '{}'", name, comp), name);
        }
        if (w->encoding != TensorEncoding::fp32 || !w->dense) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("hybrid weight '{}' must be dense fp32 in the Phase 9 slice", name), name);
        }
        const auto it = expected_by_name.find(name);
        if (it == expected_by_name.end()) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("hybrid weight '{}' has no expected-tensor contract", name), name);
        }
        if (auto s = validate_decoder_weight_desc(w->logical, w->element_count, *it->second); !s.ok()) { return s; }
        if (w->dense.size() < w->element_count) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("hybrid weight '{}' device buffer is smaller than its element count", name), name);
        }
        out = w->dense.view();
        return Status::success();
    };
    if (auto s = view_of(_names.dense.embed_tokens(), _embed); !s.ok()) { return s; }
    if (auto s = view_of(_names.dense.final_norm(), _final_norm); !s.ok()) { return s; }
    // LM head: standalone, else tied (transposed embedding, owned buffer).
    if (const auto *w = weights.find(comp, _names.dense.lm_head()); w != nullptr) {
        if (auto s = view_of(_names.dense.lm_head(), _lm_head); !s.ok()) { return s; }
    } else if (_config.tie_word_embeddings) {
        const uint64_t hidden = _config.decoder.hidden_size, vocab = _config.decoder.vocab_size;
        uint64_t head_elems = 0;
        if (!checked_mul(hidden, vocab, head_elems)) {
            return Status::error(StatusCode::out_of_memory, "tied LM head element count overflows", "lm_head");
        }
        luisa::vector<float> emb(static_cast<size_t>(head_elems)), transposed(static_cast<size_t>(head_elems));
        stream << _embed.copy_to(luisa::span<float>{emb}) << synchronize();
        for (uint32_t v = 0; v < vocab; v++) {
            for (uint32_t h = 0; h < hidden; h++) {
                transposed[static_cast<size_t>(h) * vocab + v] = emb[static_cast<size_t>(v) * hidden + h];
            }
        }
        _tied_lm_head = _device->create_buffer<float>(static_cast<size_t>(head_elems));
        if (!_tied_lm_head) {
            return Status::error(StatusCode::out_of_memory, "failed to allocate tied LM head", "lm_head");
        }
        stream << _tied_lm_head.copy_from(luisa::span<const float>{transposed}) << synchronize();
        _lm_head = _tied_lm_head.view();
    } else {
        return Status::error(StatusCode::not_found,
                             luisa::format("lm_head '{}' is missing and embeddings are not tied", _names.dense.lm_head()),
                             "lm_head");
    }

    _full_layers.resize(_config.num_full_layers());
    _gdn_layers.resize(_config.num_gdn_layers());
    for (uint32_t l = 0; l < _config.decoder.num_layers; l++) {
        if (_config.layer_kinds[l] == HybridLayerKind::full_attention) {
            auto &L = _full_layers[_full_index[l]];
            if (auto s = view_of(_names.dense.input_layernorm(l), L.input_norm); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.post_attention_layernorm(l), L.post_norm); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.q_proj(l), L.q_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.k_proj(l), L.k_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.v_proj(l), L.v_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.o_proj(l), L.o_w); !s.ok()) { return s; }
            if (_config.qkv_bias) {
                if (auto s = view_of(_names.dense.q_bias(l), L.q_b); !s.ok()) { return s; }
                if (auto s = view_of(_names.dense.k_bias(l), L.k_b); !s.ok()) { return s; }
                if (auto s = view_of(_names.dense.v_bias(l), L.v_b); !s.ok()) { return s; }
            }
            if (_config.qk_norm) {
                if (auto s = view_of(_names.dense.q_norm(l), L.q_norm); !s.ok()) { return s; }
                if (auto s = view_of(_names.dense.k_norm(l), L.k_norm); !s.ok()) { return s; }
            }
            if (auto s = view_of(_names.dense.gate_proj(l), L.gate_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.up_proj(l), L.up_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.down_proj(l), L.down_w); !s.ok()) { return s; }
        } else {
            auto &L = _gdn_layers[_gdn_index[l]];
            if (auto s = view_of(_names.dense.input_layernorm(l), L.input_norm); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.post_attention_layernorm(l), L.post_norm); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_q_proj(l), L.q_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_k_proj(l), L.k_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_v_proj(l), L.v_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_z_proj(l), L.z_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_a_proj(l), L.a_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_b_proj(l), L.b_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_o_proj(l), L.o_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_conv_q_weight(l), L.conv_q_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_conv_q_bias(l), L.conv_q_b); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_conv_k_weight(l), L.conv_k_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_conv_k_bias(l), L.conv_k_b); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_conv_v_weight(l), L.conv_v_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_conv_v_bias(l), L.conv_v_b); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_a_log(l), L.a_log); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_dt_bias(l), L.dt_bias); !s.ok()) { return s; }
            if (auto s = view_of(_names.gdn_norm_weight(l), L.norm_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.gate_proj(l), L.gate_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.up_proj(l), L.up_w); !s.ok()) { return s; }
            if (auto s = view_of(_names.dense.down_proj(l), L.down_w); !s.ok()) { return s; }
        }
    }
    _weight_bytes = weights.component_bytes(comp) + (_tied_lm_head ? _tied_lm_head.size_bytes() : 0ull);
    return Status::success();
}

[[nodiscard]] Status HybridGdnExecutor::reserve_workspace() {
    const auto &d = _config.decoder;
    const auto &g = _config.gdn;
    const uint64_t mr = _config.prefill_chunk_rows;
    const uint64_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim(), inter = d.intermediate_size;
    const uint64_t qk = g.qk_dim(), gh = g.num_heads;
    const auto align = [](uint64_t bytes) { return (bytes + kWorkspaceAlignment - 1u) & ~(kWorkspaceAlignment - 1u); };
    struct Spec { WorkspaceAllocation *out; uint64_t elems; };
    const Spec specs[] = {
        {&_emb, mr * hidden}, {&_x, mr * hidden}, {&_norm, mr * hidden}, {&_norm2, mr * hidden},
        {&_q, mr * qd}, {&_k, mr * kvd}, {&_v, mr * kvd}, {&_q_rope, mr * qd}, {&_k_rope, mr * kvd},
        {&_attn, mr * qd}, {&_gate, mr * inter}, {&_up, mr * inter}, {&_mlp, mr * inter},
        {&_logits, d.vocab_size}, {&_ids, mr}, {&_positions, mr}, {&_axis_of_pair, d.head_dim / 2u},
        {&_dummy_f, 1}, {&_dummy_u, 1}, {&_q_normed, mr * qd}, {&_k_normed, mr * kvd},
        {&_gq, mr * qk}, {&_gk, mr * qk}, {&_gv, mr * qk}, {&_gz, mr * qk},
        {&_ga, mr * gh}, {&_gb, mr * gh},
        {&_gq_conv, mr * qk}, {&_gk_conv, mr * qk}, {&_gv_conv, mr * qk},
        {&_gq_norm, mr * qk}, {&_gk_norm, mr * qk},
        {&_gbeta, mr * gh}, {&_gg, mr * gh}, {&_go, mr * qk}, {&_gynorm, mr * qk},
        {&_gw_scratch, mr * qk}, {&_gu_scratch, mr * qk}, {&_gc_scratch, mr * gh},
    };
    uint64_t total = 0;
    for (const auto &s : specs) {
        uint64_t bytes = 0;
        if (!checked_mul(s.elems, 4ull, bytes)) {
            return Status::error(StatusCode::out_of_memory, "hybrid workspace region overflows", "workspace");
        }
        if (!checked_add(total, align(bytes), total)) {
            return Status::error(StatusCode::out_of_memory, "hybrid workspace total overflows", "workspace");
        }
    }
    if (auto s = _workspace.reserve(total); !s.ok()) { return s; }
    for (const auto &s : specs) {
        auto r = _workspace.allocate(s.elems * 4ull);
        if (!r.ok()) { return r.status; }
        *s.out = *r.value;
    }
    return Status::success();
}

[[nodiscard]] Status HybridGdnExecutor::compile_shaders() {
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
    if (auto s = get(_l2_norm_cache, norm_variant_key("l2_norm"),
                     [] { return make_l2_norm_kernel(); }, _sh_l2_norm); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::none); }, _sh_gemm_none); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::bias); }, _sh_gemm_bias); !s.ok()) { return s; }
    if (auto s = get(_gemm_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, false),
                     [] { return make_gemm_kernel(GemmEpilogue::residual); }, _sh_gemm_residual); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, true),
                     [] { return make_gemv_kernel(GemmEpilogue::none); }, _sh_gemv_none); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::bias, GemmWeightFormat::fp32, true),
                     [] { return make_gemv_kernel(GemmEpilogue::bias); }, _sh_gemv_bias); !s.ok()) { return s; }
    if (auto s = get(_gemv_cache, gemm_variant_key(GemmEpilogue::residual, GemmWeightFormat::fp32, true),
                     [] { return make_gemv_kernel(GemmEpilogue::residual); }, _sh_gemv_residual); !s.ok()) { return s; }
    if (auto s = get(_gated_act_cache, gemm_variant_key(GemmEpilogue::none, GemmWeightFormat::fp32, false) + "_gated_silu",
                     [] { return make_gated_act_kernel(0u); }, _sh_gated_silu); !s.ok()) { return s; }
    const bool rotate_half = _config.decoder.rope_pair_layout == RopePairLayout::rotate_half;
    if (rotate_half) {
        if (auto s = get(_rope_cache, rope_variant_key("rotate_half"),
                         [] { return make_rope_rotate_half_kernel(); }, _sh_rope); !s.ok()) { return s; }
    } else {
        if (auto s = get(_rope_cache, rope_variant_key("1d"),
                         [] { return make_rope_1d_kernel(); }, _sh_rope); !s.ok()) { return s; }
    }
    if (auto s = get(_attn_prefill_cache, attention_variant_key("prefill_reference"),
                     [] { return make_attention_prefill_reference_kernel(); }, _sh_attn_prefill); !s.ok()) { return s; }
    if (auto s = get(_attn_decode_cache, attention_variant_key("decode"),
                     [] { return make_attention_decode_kernel(); }, _sh_attn_decode); !s.ok()) { return s; }
    if (auto s = get(_kv_append_cache, decode_variant_key("kv_append"),
                     [] { return make_kv_append_kernel(); }, _sh_kv_append); !s.ok()) { return s; }
    if (auto s = get(_gdn_conv_cache, gdn_variant_key("conv_silu"),
                     [] { return make_gdn_conv_silu_kernel(); }, _sh_gdn_conv); !s.ok()) { return s; }
    if (auto s = get(_gdn_gates_cache, gdn_variant_key("gates"),
                     [] { return make_gdn_gates_kernel(); }, _sh_gdn_gates); !s.ok()) { return s; }
    if (auto s = get(_gdn_recurrent_cache, gdn_variant_key("recurrent"),
                     [] { return make_gdn_recurrent_kernel(); }, _sh_gdn_recurrent); !s.ok()) { return s; }
    if (auto s = get(_gdn_chunk_cache, gdn_variant_key("chunk"),
                     [] { return make_gdn_chunk_kernel(); }, _sh_gdn_chunk); !s.ok()) { return s; }
    if (auto s = get(_gdn_gated_norm_cache, gdn_variant_key("gated_norm"),
                     [] { return make_gdn_gated_norm_kernel(); }, _sh_gdn_gated_norm); !s.ok()) { return s; }
    if (auto s = get(_zero_cache, gdn_variant_key("zero_fill"),
                     [] { return make_zero_fill_kernel(); }, _sh_zero); !s.ok()) { return s; }
    return Status::success();
}

// ---------------------------------------------------------------------------
// linear / MLP / LM head helpers
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnExecutor::record_linear(CommandList &list, BufferView<float> A,
                                                      BufferView<float> B, BufferView<float> C,
                                                      BufferView<float> aux, uint32_t m, uint32_t n, uint32_t k,
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

[[nodiscard]] Status HybridGdnExecutor::record_mlp(CommandList &list, BufferView<float> x,
                                                   BufferView<float> post_norm, BufferView<float> gate_w,
                                                   BufferView<float> up_w, BufferView<float> down_w, uint32_t rows) {
    const auto &d = _config.decoder;
    const uint32_t hidden = d.hidden_size, inter = d.intermediate_size;
    auto norm2 = ws(_norm2).subview(0, static_cast<size_t>(rows) * hidden);
    auto gate = ws(_gate).subview(0, static_cast<size_t>(rows) * inter);
    auto up = ws(_up).subview(0, static_cast<size_t>(rows) * inter);
    auto mlp = ws(_mlp).subview(0, static_cast<size_t>(rows) * inter);
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm2, post_norm, rows, hidden, d.rms_norm_eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    if (auto s = record_linear(list, norm2, gate_w, gate, ws(_dummy_f), rows, inter, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm2, up_w, up, ws(_dummy_f), rows, inter, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    {
        const uint64_t count = static_cast<uint64_t>(rows) * inter;
        auto plan = plan_gated_act_dispatch(count);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_gated_silu)(gate, up, mlp, static_cast<uint32_t>(count)), *plan.value)) {
            list << std::move(cmd);
        }
    }
    return record_linear(list, mlp, down_w, x, x, rows, hidden, inter, GemmEpilogue::residual);
}

[[nodiscard]] Status HybridGdnExecutor::record_lm_head(CommandList &list, BufferView<float> x,
                                                       uint32_t rows, BufferView<float> C) {
    const auto &d = _config.decoder;
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    auto norm = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    auto plan = plan_norm_dispatch(rows);
    if (!plan.ok()) { return plan.status; }
    if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm, _final_norm, rows, hidden, d.rms_norm_eps, 1u), *plan.value)) {
        list << std::move(cmd);
    }
    return record_linear(list, norm, _lm_head, C, ws(_dummy_f), rows, vocab, hidden, GemmEpilogue::none);
}

// ---------------------------------------------------------------------------
// full-attention layer
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnExecutor::record_full_layer(CommandList &list, uint32_t layer,
                                                          BufferView<float> x, uint32_t rows, uint32_t pos0,
                                                          HybridGdnCache &cache, bool decode) {
    const auto &d = _config.decoder;
    const auto &L = _full_layers[_full_index[layer]];
    const uint32_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t qh = d.num_attention_heads, kvh = d.num_kv_heads, hd = d.head_dim;
    const float eps = d.rms_norm_eps, theta = d.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const uint32_t full_idx = _full_index[layer];

    auto norm = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    auto q = ws(_q).subview(0, static_cast<size_t>(rows) * qd);
    auto k = ws(_k).subview(0, static_cast<size_t>(rows) * kvd);
    auto v = ws(_v).subview(0, static_cast<size_t>(rows) * kvd);
    auto q_rope = ws(_q_rope).subview(0, static_cast<size_t>(rows) * qd);
    auto k_rope = ws(_k_rope).subview(0, static_cast<size_t>(rows) * kvd);
    auto attn = ws(_attn).subview(0, static_cast<size_t>(rows) * qd);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, rows);
    auto axis_view = ws(_axis_of_pair).as<uint32_t>();
    auto dummy_f = ws(_dummy_f);
    auto dummy_u = ws(_dummy_u).as<uint32_t>();

    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm, L.input_norm, rows, hidden, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    const auto qkv_epi = _config.qkv_bias ? GemmEpilogue::bias : GemmEpilogue::none;
    if (auto s = record_linear(list, norm, L.q_w, q, L.q_b, rows, qd, hidden, qkv_epi); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.k_w, k, L.k_b, rows, kvd, hidden, qkv_epi); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.v_w, v, L.v_b, rows, kvd, hidden, qkv_epi); !s.ok()) { return s; }
    auto q_for_rope = q;
    auto k_for_rope = k;
    if (_config.qk_norm) {
        auto qn = ws(_q_normed).subview(0, static_cast<size_t>(rows) * qd);
        auto kn = ws(_k_normed).subview(0, static_cast<size_t>(rows) * kvd);
        auto plan_q = plan_norm_dispatch(static_cast<uint64_t>(rows) * qh);
        if (!plan_q.ok()) { return plan_q.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(q, qn, L.q_norm, rows * qh, hd, eps, 1u), *plan_q.value)) {
            list << std::move(cmd);
        }
        auto plan_k = plan_norm_dispatch(static_cast<uint64_t>(rows) * kvh);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(k, kn, L.k_norm, rows * kvh, hd, eps, 1u), *plan_k.value)) {
            list << std::move(cmd);
        }
        q_for_rope = qn;
        k_for_rope = kn;
    }
    {
        auto plan_q = plan_rope_dispatch(rows, qh);
        if (!plan_q.ok()) { return plan_q.status; }
        auto plan_k = plan_rope_dispatch(rows, kvh);
        if (!plan_k.ok()) { return plan_k.status; }
        if (auto cmd = dispatch_1d((*_sh_rope)(q_for_rope, q_rope, pos_view, axis_view, rows, qh, hd, 1u, theta), *plan_q.value)) {
            list << std::move(cmd);
        }
        if (auto cmd = dispatch_1d((*_sh_rope)(k_for_rope, k_rope, pos_view, axis_view, rows, kvh, hd, 1u, theta), *plan_k.value)) {
            list << std::move(cmd);
        }
    }
    {
        auto s = validate_kv_append(pos0, rows, cache.config().max_seq_len);
        if (!s.ok()) { return s; }
        auto plan = plan_kv_append_dispatch(rows, kvd);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_kv_append)(k_rope, v, cache.k_layer(full_idx), cache.v_layer(full_idx),
                                     rows, kvd, pos0), *plan.value)) {
            list << std::move(cmd);
        }
    }
    if (decode) {
        auto plan = plan_attention_decode_dispatch(1, qh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_decode)(q_rope, cache.k_layer(full_idx), cache.v_layer(full_idx),
                                     attn, cache.state().kv_len(), cache.state().positions(), dummy_u,
                                     1u, cache.config().max_seq_len, qh, kvh, hd, scale, 0.0f,
                                     static_cast<uint32_t>(AttentionMaskKind::causal), 0u, 0u, 0u, 0u), *plan.value)) {
            list << std::move(cmd);
        }
    } else {
        const uint32_t k_len = pos0 + rows;
        auto plan = plan_attention_prefill_dispatch(rows, qh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_attn_prefill)(q_rope, cache.k_layer(full_idx), cache.v_layer(full_idx),
                                      attn, dummy_u, dummy_f, rows, k_len, qh, kvh, hd, scale, 0.0f,
                                      static_cast<uint32_t>(AttentionMaskKind::causal), 0u, 0u, 0u, 0u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    if (auto s = record_linear(list, attn, L.o_w, x, x, rows, hidden, qd, GemmEpilogue::residual); !s.ok()) { return s; }
    return record_mlp(list, x, L.post_norm, L.gate_w, L.up_w, L.down_w, rows);
}

// ---------------------------------------------------------------------------
// GDN layer
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnExecutor::record_gdn_layer(CommandList &list, uint32_t layer,
                                                         BufferView<float> x, uint32_t rows, uint32_t pos0,
                                                         HybridGdnCache &cache, bool chunked) {
    const auto &d = _config.decoder;
    const auto &g = _config.gdn;
    const auto &L = _gdn_layers[_gdn_index[layer]];
    const uint32_t hidden = d.hidden_size;
    const uint32_t qk = g.qk_dim(), gh = g.num_heads, gd = g.head_dim, K = g.conv_kernel_size;
    const uint32_t gdn_idx = _gdn_index[layer];
    const float eps = d.rms_norm_eps;

    auto norm = ws(_norm).subview(0, static_cast<size_t>(rows) * hidden);
    auto gq = ws(_gq).subview(0, static_cast<size_t>(rows) * qk);
    auto gk = ws(_gk).subview(0, static_cast<size_t>(rows) * qk);
    auto gv = ws(_gv).subview(0, static_cast<size_t>(rows) * qk);
    auto gz = ws(_gz).subview(0, static_cast<size_t>(rows) * qk);
    auto ga = ws(_ga).subview(0, static_cast<size_t>(rows) * gh);
    auto gb = ws(_gb).subview(0, static_cast<size_t>(rows) * gh);
    auto gq_conv = ws(_gq_conv).subview(0, static_cast<size_t>(rows) * qk);
    auto gk_conv = ws(_gk_conv).subview(0, static_cast<size_t>(rows) * qk);
    auto gv_conv = ws(_gv_conv).subview(0, static_cast<size_t>(rows) * qk);
    auto gq_norm = ws(_gq_norm).subview(0, static_cast<size_t>(rows) * qk);
    auto gk_norm = ws(_gk_norm).subview(0, static_cast<size_t>(rows) * qk);
    auto gbeta = ws(_gbeta).subview(0, static_cast<size_t>(rows) * gh);
    auto gg = ws(_gg).subview(0, static_cast<size_t>(rows) * gh);
    auto go = ws(_go).subview(0, static_cast<size_t>(rows) * qk);
    auto gynorm = ws(_gynorm).subview(0, static_cast<size_t>(rows) * qk);

    // 1. input RMSNorm.
    {
        auto plan = plan_norm_dispatch(rows);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_rms_norm)(x, norm, L.input_norm, rows, hidden, eps, 1u), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 2. Q/K/V/gate projections (no bias).
    if (auto s = record_linear(list, norm, L.q_w, gq, ws(_dummy_f), rows, qk, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.k_w, gk, ws(_dummy_f), rows, qk, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.v_w, gv, ws(_dummy_f), rows, qk, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.z_w, gz, ws(_dummy_f), rows, qk, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.a_w, ga, ws(_dummy_f), rows, gh, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    if (auto s = record_linear(list, norm, L.b_w, gb, ws(_dummy_f), rows, gh, hidden, GemmEpilogue::none); !s.ok()) { return s; }
    // 3. Causal depthwise Conv1D + SiLU with ring state (q/k/v thirds).
    {
        auto s = validate_gdn_conv_inputs(rows, qk, K, pos0);
        if (!s.ok()) { return s; }
        auto ring = cache.conv_ring(gdn_idx, 0);
        const uint64_t third = static_cast<uint64_t>(qk) * (K - 1u);
        auto ring_q = ring.subview(0, static_cast<size_t>(third));
        auto ring_k = ring.subview(static_cast<size_t>(third), static_cast<size_t>(third));
        auto ring_v = ring.subview(static_cast<size_t>(2u * third), static_cast<size_t>(third));
        auto plan = plan_gdn_conv_dispatch(qk);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_gdn_conv)(gq, L.conv_q_w, L.conv_q_b, ring_q, gq_conv, rows, qk, K, pos0), *plan.value)) {
            list << std::move(cmd);
        }
        if (auto cmd = dispatch_1d((*_sh_gdn_conv)(gk, L.conv_k_w, L.conv_k_b, ring_k, gk_conv, rows, qk, K, pos0), *plan.value)) {
            list << std::move(cmd);
        }
        if (auto cmd = dispatch_1d((*_sh_gdn_conv)(gv, L.conv_v_w, L.conv_v_b, ring_v, gv_conv, rows, qk, K, pos0), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 4. L2 / QK normalization (per head over d).
    {
        auto plan = plan_norm_dispatch(static_cast<uint64_t>(rows) * gh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_l2_norm)(gq_conv, gq_norm, rows * gh, gd, g.l2_eps), *plan.value)) {
            list << std::move(cmd);
        }
        if (auto cmd = dispatch_1d((*_sh_l2_norm)(gk_conv, gk_norm, rows * gh, gd, g.l2_eps), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 5. beta/gate.
    {
        auto plan = plan_gdn_gates_dispatch(rows, gh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_gdn_gates)(ga, gb, L.a_log, L.dt_bias, gbeta, gg, rows, gh, g.softplus_threshold), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 6. Recurrent (straight) or chunked (WY/UT) state update.
    {
        auto s = validate_gdn_state_inputs(rows, gh, gd);
        if (!s.ok()) { return s; }
        auto state = cache.recurrent(gdn_idx, 0);
        auto plan = plan_gdn_heads_dispatch(gh);
        if (!plan.ok()) { return plan.status; }
        if (chunked) {
            auto w_scratch = ws(_gw_scratch).subview(0, static_cast<size_t>(rows) * qk);
            auto u_scratch = ws(_gu_scratch).subview(0, static_cast<size_t>(rows) * qk);
            auto c_scratch = ws(_gc_scratch).subview(0, static_cast<size_t>(rows) * gh);
            if (auto cmd = dispatch_1d((*_sh_gdn_chunk)(gq_norm, gk_norm, gv_conv, gbeta, gg, state, go,
                                         w_scratch, u_scratch, c_scratch, rows, gh, gd), *plan.value)) {
                list << std::move(cmd);
            }
        } else {
            if (auto cmd = dispatch_1d((*_sh_gdn_recurrent)(gq_norm, gk_norm, gv_conv, gbeta, gg, state, go,
                                             rows, gh, gd), *plan.value)) {
                list << std::move(cmd);
            }
        }
    }
    // 7. Gated output normalization.
    {
        auto plan = plan_gdn_gated_norm_dispatch(rows, gh);
        if (!plan.ok()) { return plan.status; }
        if (auto cmd = dispatch_1d((*_sh_gdn_gated_norm)(go, gz, L.norm_w, gynorm, rows, gh, gd, g.norm_eps), *plan.value)) {
            list << std::move(cmd);
        }
    }
    // 8. Output projection + residual.
    if (auto s = record_linear(list, gynorm, L.o_w, x, x, rows, hidden, qk, GemmEpilogue::residual); !s.ok()) { return s; }
    // 9. SwiGLU MLP.
    return record_mlp(list, x, L.post_norm, L.gate_w, L.up_w, L.down_w, rows);
}

// ---------------------------------------------------------------------------
// prefill / decode
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnExecutor::record_state_reset(CommandList &list, HybridGdnCache &cache, uint32_t index) {
    if (!cache.valid() || index >= cache.config().batch) {
        return Status::error(StatusCode::invalid_argument, "hybrid state reset index out of range", "index");
    }
    if (_sh_zero == nullptr) {
        return Status::error(StatusCode::internal_error, "hybrid zero-fill shader not compiled", "zero");
    }
    // Zero every GDN layer's recurrent matrix and Conv ring for THIS request.
    // Recorded ahead of all layer dispatches in the same list, so the zeroing
    // is ordered before the first delta-rule / conv read without a host sync.
    for (uint32_t gl = 0; gl < cache.config().num_gdn_layers; gl++) {
        const auto state = cache.recurrent(gl, index);
        const auto ring = cache.conv_ring(gl, index);
        if (state) {
            auto count = checked_u32(state.size(), "hybrid recurrent reset count");
            if (!count.ok()) { return count.status; }
            auto plan = plan_gdn_elementwise_dispatch(state.size());
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_zero)(state, *count.value), *plan.value)) {
                list << std::move(cmd);
            }
        }
        if (ring) {
            auto count = checked_u32(ring.size(), "hybrid conv ring reset count");
            if (!count.ok()) { return count.status; }
            auto plan = plan_gdn_elementwise_dispatch(ring.size());
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_zero)(ring, *count.value), *plan.value)) {
                list << std::move(cmd);
            }
        }
    }
    return Status::success();
}

namespace {
struct HybridRecordStaging {
    luisa::vector<int32_t> ids;
    luisa::vector<int32_t> positions;
    luisa::vector<uint32_t> axis;
};
}// namespace

[[nodiscard]] Status HybridGdnExecutor::record_prefill(CommandList &list, const DecodeRequest &request,
                                                       HybridGdnCache &cache, luisa::span<float> logits_out) {
    return record_prefill_impl(list, request, cache, 0u, false, logits_out, {});
}
[[nodiscard]] Status HybridGdnExecutor::record_prefill_all_logits(CommandList &list, const DecodeRequest &request,
                                                                  HybridGdnCache &cache, BufferView<float> logits_device) {
    return record_prefill_impl(list, request, cache, 0u, true, {}, logits_device);
}
[[nodiscard]] Status HybridGdnExecutor::record_prefill_chunked(CommandList &list, const DecodeRequest &request,
                                                               HybridGdnCache &cache, uint32_t chunk_size,
                                                               bool all_logits, luisa::span<float> last_logits_out,
                                                               BufferView<float> logits_device) {
    return record_prefill_impl(list, request, cache, chunk_size, all_logits, last_logits_out, logits_device);
}

[[nodiscard]] Status HybridGdnExecutor::record_prefill_impl(CommandList &list, const DecodeRequest &request,
                                                            HybridGdnCache &cache, uint32_t chunk_size, bool all_logits,
                                                            luisa::span<float> last_logits_out, BufferView<float> logits_device) {
    auto valid = request.validate();
    if (!valid.ok()) { return valid; }
    const auto &d = _config.decoder;
    const uint32_t prompt_len = static_cast<uint32_t>(request.prompt_tokens.size());
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    if (prompt_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "hybrid prefill needs a nonempty prompt", "prompt_len");
    }
    if (prompt_len > cache.config().max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("prompt length {} exceeds hybrid capacity {}", prompt_len, cache.config().max_seq_len),
                             "prompt_len");
    }
    if (cache.config().num_full_layers != _config.num_full_layers() ||
        cache.config().num_gdn_layers != _config.num_gdn_layers() ||
        cache.config().num_kv_heads != d.num_kv_heads || cache.config().head_dim != d.head_dim ||
        cache.config().gdn_heads != _config.gdn.num_heads || cache.config().gdn_head_dim != _config.gdn.head_dim ||
        cache.config().conv_kernel_size != _config.gdn.conv_kernel_size || cache.config().batch != 1u) {
        return Status::error(StatusCode::invalid_argument, "hybrid cache geometry does not match the decoder", "cache");
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
    for (size_t i = 0; i < request.prompt_tokens.size(); i++) {
        if (request.prompt_tokens[i] >= vocab) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("prompt token {} at index {} is outside the vocabulary", request.prompt_tokens[i], i),
                                 "prompt_tokens");
        }
    }
    // Effective chunk: recurrent path uses prefill_chunk_rows; chunked path uses
    // chunk_size (bounded by the workspace row capacity prefill_chunk_rows).
    uint32_t effective_chunk = _config.prefill_chunk_rows;
    if (chunk_size != 0u) {
        if (chunk_size > _config.prefill_chunk_rows) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("chunk_size {} exceeds the workspace row capacity {}; raise prefill_chunk_rows",
                                               chunk_size, _config.prefill_chunk_rows),
                                 "chunk_size");
        }
        effective_chunk = chunk_size;
    }
    if (auto s = cache.bind_sequence(0, prompt_len); !s.ok()) { return s; }
    _decode_step = 0;

    // S1 (Phase 9 review): a from-scratch prefill MUST start from ZERO GDN state.
    // The executor is reusable and the cache slot may still hold the previous
    // request's recurrent matrix / Conv ring: record_finished() is metadata-only
    // (kv_len/position/state) and release() only frees the host slot, so WITHOUT
    // this reset the delta rule would accumulate onto stale state and the Conv
    // would read the previous request's taps -- a cross-request state leak.
    // Zero BOTH stores for THIS request here, after the successful bind and
    // BEFORE any chunk work, through the already-compiled zero-fill shader. The
    // commands are recorded into the SAME list ahead of every layer dispatch, so
    // ordering is guaranteed by the command list (no host sync needed).
    // Full-attention KV needs no zeroing: attention is bounded by kv_len / the
    // causal prefix, so stale rows beyond the new length are never read.
    if (auto s = record_state_reset(list, cache, 0); !s.ok()) { return s; }

    auto staging = luisa::make_unique<HybridRecordStaging>();
    staging->ids.assign(request.prompt_tokens.begin(), request.prompt_tokens.end());
    staging->positions.resize(prompt_len);
    for (uint32_t i = 0; i < prompt_len; i++) { staging->positions[i] = static_cast<int32_t>(i); }
    staging->axis = _axis_pending;
    auto *stage = staging.get();

    auto chunks = plan_row_chunks(prompt_len, effective_chunk);
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
        list << ids_full.subview(0, rows).copy_from(luisa::span<const int32_t>{stage->ids.data() + pos0, rows})
             << pos_full.subview(0, rows).copy_from(luisa::span<const int32_t>{stage->positions.data() + pos0, rows})
             << ws(_axis_of_pair).as<uint32_t>().copy_from(luisa::span<const uint32_t>{stage->axis});
        auto ids_view = ids_full.subview(0, rows);
        {
            auto plan = plan_embedding_dispatch(rows, hidden);
            if (!plan.ok()) { return plan.status; }
            if (auto cmd = dispatch_1d((*_sh_embedding)(_embed, ids_view, emb, rows, hidden), *plan.value)) {
                list << std::move(cmd);
            }
        }
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
        for (uint32_t l = 0; l < d.num_layers; l++) {
            if (_config.layer_kinds[l] == HybridLayerKind::full_attention) {
                if (auto s = record_full_layer(list, l, x, rows, pos0, cache, false); !s.ok()) { return s; }
            } else {
                if (auto s = record_gdn_layer(list, l, x, rows, pos0, cache, chunk_size != 0u); !s.ok()) { return s; }
            }
        }
        if (all_logits) {
            auto C = logits_device.subview(static_cast<size_t>(pos0) * vocab, static_cast<size_t>(rows) * vocab);
            if (auto s = record_lm_head(list, x, rows, C); !s.ok()) { return s; }
        } else if (ci + 1u == chunks.value->chunk_count) {
            auto last_row = x.subview(static_cast<size_t>(rows - 1u) * hidden, hidden);
            if (auto s = record_lm_head(list, last_row, 1u, ws(_logits)); !s.ok()) { return s; }
        }
    }
    const luisa::vector<uint32_t> meta_kv_len = {prompt_len};
    const luisa::vector<int32_t> meta_positions = {static_cast<int32_t>(prompt_len)};
    const luisa::vector<uint32_t> meta_step = {0u};
    const luisa::vector<uint32_t> meta_state = {static_cast<uint32_t>(DecodeSeqState::prefill)};
    if (auto s = cache.state().record(list, meta_kv_len, meta_positions, meta_step, meta_state); !s.ok()) { return s; }
    if (!all_logits) {
        list << ws(_logits).copy_to(last_logits_out);
    }
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return Status::success();
}

[[nodiscard]] Status HybridGdnExecutor::record_decode_step(CommandList &list, uint32_t input_token,
                                                           HybridGdnCache &cache, luisa::span<float> logits_out) {
    const auto &d = _config.decoder;
    const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
    if (logits_out.size() != vocab) {
        return Status::error(StatusCode::invalid_argument, "logits_out size must equal vocab_size", "logits_out");
    }
    if (cache.config().batch != 1u) {
        return Status::error(StatusCode::unsupported, "Phase 9 hybrid decode is batch 1 only", "batch");
    }
    const uint32_t pos = cache.length(0);
    if (pos == 0u) {
        return Status::error(StatusCode::invalid_argument, "decode step before prefill (empty state)", "position");
    }
    if (pos >= cache.config().max_seq_len) {
        return Status::error(StatusCode::out_of_memory,
                             luisa::format("decode position {} reached hybrid capacity {}", pos, cache.config().max_seq_len),
                             "position");
    }
    if (input_token >= vocab) {
        return Status::error(StatusCode::invalid_argument, "decode input token outside the vocabulary", "input_token");
    }
    // Device metadata BEFORE the layers: decode attention reads kv_len=pos+1.
    const luisa::vector<uint32_t> meta_kv_len = {pos + 1u};
    const luisa::vector<int32_t> meta_positions = {static_cast<int32_t>(pos)};
    const luisa::vector<uint32_t> meta_step = {_decode_step + 1u};
    const luisa::vector<uint32_t> meta_state = {static_cast<uint32_t>(DecodeSeqState::decoding)};
    if (auto s = cache.state().record(list, meta_kv_len, meta_positions, meta_step, meta_state); !s.ok()) { return s; }

    auto staging = luisa::make_unique<HybridRecordStaging>();
    staging->ids = {static_cast<int32_t>(input_token)};
    staging->positions = {static_cast<int32_t>(pos)};
    staging->axis = _axis_pending;
    auto *stage = staging.get();
    auto ids_view = ws(_ids).as<int32_t>().subview(0, 1);
    auto pos_view = ws(_positions).as<int32_t>().subview(0, 1);
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
        if (_config.layer_kinds[l] == HybridLayerKind::full_attention) {
            if (auto s = record_full_layer(list, l, x, 1u, pos, cache, true); !s.ok()) { return s; }
        } else {
            // GDN decode: straight recurrent T=1 (NO sequence x sequence attention).
            if (auto s = record_gdn_layer(list, l, x, 1u, pos, cache, false); !s.ok()) { return s; }
        }
    }
    if (auto s = record_lm_head(list, x, 1u, ws(_logits)); !s.ok()) { return s; }
    list << ws(_logits).copy_to(logits_out);
    if (auto s = cache.append_tokens(0, 1u); !s.ok()) { return s; }
    _decode_step++;
    list.add_callback([staging = std::move(staging)]() mutable { staging.reset(); });
    return Status::success();
}

// ---------------------------------------------------------------------------
// Host decode controller
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridDecodeOptions::validate(uint32_t vocab_size) const {
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
        return Status::error(StatusCode::invalid_argument, "max_wall_time_ms must be finite and >= 0", "max_wall_time_ms");
    }
    return Status::success();
}

[[nodiscard]] Status HybridDecodeController::create(HybridGdnExecutor &executor, const Tokenizer *tokenizer,
                                                    HybridDecodeOptions options) {
    auto valid = options.validate(executor.config().decoder.vocab_size);
    if (!valid.ok()) { return valid; }
    _executor = &executor;
    _tokenizer = tokenizer;
    _options = std::move(options);
    return Status::success();
}

[[nodiscard]] Result<HybridDecodeResult> HybridDecodeController::generate(Stream &stream,
                                                                          const DecodeRequest &request, HybridGdnCache &cache) {
    using R = Result<HybridDecodeResult>;
    if (_executor == nullptr) {
        return R::failure(StatusCode::invalid_argument, "hybrid decode controller is not created", "controller");
    }
    auto valid = request.validate();
    if (!valid.ok()) { return R::failure(valid); }
    const auto &d = _executor->config().decoder;
    const uint32_t vocab = d.vocab_size;
    _last_committed_length = 0;

    Clock clock;
    const auto cancelled = [&]() { return _options.cancel_callback && _options.cancel_callback(); };
    const auto timed_out = [&]() { return _options.max_wall_time_ms > 0.0 && clock.toc() > _options.max_wall_time_ms; };
    if (cancelled()) { return R::failure(StatusCode::cancelled, "request cancelled before prefill", "cancel"); }
    if (timed_out()) { return R::failure(StatusCode::timeout, "wall-time budget exceeded before prefill", "max_wall_time_ms"); }

    HybridDecodeResult result;
    luisa::vector<float> logits(vocab);

    luisa::vector<uint32_t> eos_ids;
    const auto add_eos = [&eos_ids](uint32_t id) {
        if (std::find(eos_ids.begin(), eos_ids.end(), id) == eos_ids.end()) { eos_ids.push_back(id); }
    };
    for (auto id : d.eos_token_ids) { add_eos(id); }
    for (auto id : _options.eos_ids) { add_eos(id); }
    if (_tokenizer != nullptr && _tokenizer->eos_id().has_value()) { add_eos(*_tokenizer->eos_id()); }
    const auto is_eos = [&eos_ids](uint32_t id) { return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end(); };

    luisa::optional<IncrementalDetokenizer> detok;
    if (_tokenizer != nullptr) {
        DetokenizerOptions dopts;
        dopts.skip_special_tokens = _options.skip_special_tokens;
        dopts.stop_strings = _options.stop_strings;
        dopts.eos_ids = eos_ids;
        detok.emplace(*_tokenizer, std::move(dopts));
    }

    // Release guard: armed after a successful prefill; releases the slot and
    // records the device finished state on EVERY exit path.
    struct ReleaseGuard {
        HybridGdnCache *cache;
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
        Status s = _executor->record_prefill_chunked(list, request, cache, _options.prefill_chunk_size,
                                                     false, logits, {});
        if (!s.ok()) { return R::failure(s); }
        stream << list.commit() << synchronize();
        guard.armed = true;
    }

    luisa::vector<uint32_t> prompt32(request.prompt_tokens.begin(), request.prompt_tokens.end());
    DeterministicSampler sampler{_options.sampling, vocab, prompt32};

    for (uint32_t step = 0; step < _options.max_new_tokens; step++) {
        auto picked = sampler.sample(logits);
        if (!picked.ok()) { return R::failure(picked.status); }
        const uint32_t token = *picked.value;
        result.tokens.push_back(token);
        result.steps = step + 1u;
        if (is_eos(token)) { result.hit_eos = true; break; }
        if (detok.has_value()) {
            auto text = detok->push(token);
            if (!text.ok()) { return R::failure(text.status); }
            result.text += *text.value;
            if (detok->stopped()) { result.hit_stop_string = true; break; }
        }
        if (cache.length(0) >= cache.config().max_seq_len) { result.truncated = true; break; }
        if (step + 1u == _options.max_new_tokens) { result.truncated = true; break; }
        if (cancelled()) { return R::failure(StatusCode::cancelled, "request cancelled at a decode boundary", "cancel"); }
        if (timed_out()) { return R::failure(StatusCode::timeout, "wall-time budget exceeded at a decode boundary", "max_wall_time_ms"); }
        auto list = CommandList::create();
        if (auto s = _executor->record_decode_step(list, token, cache, logits); !s.ok()) { return R::failure(s); }
        stream << list.commit() << synchronize();
    }
    _last_committed_length = cache.length(0);
    if (detok.has_value() && !detok->finished()) {
        auto tail = detok->finish();
        if (!tail.ok()) { return R::failure(tail.status); }
        result.text += *tail.value;
    }
    if (result.truncated) {
        result.diagnostics.push_back(Diagnostic{
            DiagnosticSeverity::warning, "output_truncated",
            luisa::format("generation stopped at the {}-token output budget before EOS/stop", _options.max_new_tokens)});
    }
    return R::success(std::move(result));
}

// ---------------------------------------------------------------------------
// Additive Phase 9: HybridVlmGraph (image -> vision -> hybrid decode).
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridVlmGraphConfig::validate() const {
    auto s = vision.validate();
    if (!s.ok()) { return s; }
    s = decoder.validate();
    if (!s.ok()) { return s; }
    if (vision.text_pos != P7TextPosKind::rope1d) {
        return Status::error(StatusCode::invalid_argument,
                             "the hybrid text decoder is 1-axis; vision.text_pos must be rope1d", "text_pos");
    }
    if (vision.text.hidden_size != decoder.decoder.hidden_size) {
        return Status::error(StatusCode::invalid_argument,
                             "vision.text.hidden_size must equal the hybrid decoder hidden_size (connector output)", "hidden_size");
    }
    if (vision.text.vocab_size != decoder.decoder.vocab_size) {
        return Status::error(StatusCode::invalid_argument,
                             "vision.text.vocab_size must equal the hybrid decoder vocab_size", "vocab_size");
    }
    if (max_seq_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "max_seq_len must be nonzero", "max_seq_len");
    }
    return Status::success();
}

[[nodiscard]] Status HybridVlmGraph::create(Device &device, Stream &stream,
                                            const HybridVlmGraphConfig &config, const WeightStore &weights) {
    auto valid = config.validate();
    if (!valid.ok()) { return valid; }
    _config = config;
    _device = &device;
    // Vision tower + connector ONLY (no embedded dense text executor).
    if (auto s = _vision.create_vision_only(device, stream, config.vision, config.vision_names, weights); !s.ok()) { return s; }
    // Hybrid GDN text decoder.
    if (auto s = _text.create(device, stream, config.decoder, config.decoder_names, weights); !s.ok()) { return s; }
    // Hybrid cache for the decoder geometry.
    HybridCacheConfig cc;
    cc.num_full_layers = config.decoder.num_full_layers();
    cc.num_gdn_layers = config.decoder.num_gdn_layers();
    cc.batch = 1;
    cc.num_kv_heads = config.decoder.decoder.num_kv_heads;
    cc.head_dim = config.decoder.decoder.head_dim;
    cc.max_seq_len = config.max_seq_len;
    cc.gdn_heads = config.decoder.gdn.num_heads;
    cc.gdn_head_dim = config.decoder.gdn.head_dim;
    cc.conv_kernel_size = config.decoder.gdn.conv_kernel_size;
    cc.dtype_bytes = 4;
    if (auto s = _cache.create(device, stream, cc); !s.ok()) { return s; }
    _weight_bytes = _vision.weight_bytes() + _text.weight_bytes();
    return Status::success();
}

[[nodiscard]] Result<HybridVlmRunResult> HybridVlmGraph::run(Stream &stream,
                                                             luisa::span<const P7ImageInput> images,
                                                             luisa::span<const uint32_t> prompt_tokens,
                                                             uint32_t mm_start, const HybridDecodeOptions &options,
                                                             const Tokenizer *tokenizer) {
    using R = Result<HybridVlmRunResult>;
    if (_device == nullptr) { return R::failure(StatusCode::invalid_argument, "hybrid vlm graph not created", "graph"); }
    if (prompt_tokens.empty()) { return R::failure(StatusCode::invalid_argument, "empty prompt", "prompt_tokens"); }
    const uint32_t vocab = _config.decoder.decoder.vocab_size;
    for (auto t : prompt_tokens) {
        if (t >= vocab) { return R::failure(StatusCode::invalid_argument, "prompt token outside the vocabulary", "prompt_tokens"); }
    }
    // Belt-and-braces full reset (ALL sequences: recurrent + conv + metadata).
    // The previous run drained the stream (generate synchronizes), so reset is
    // legal. NOTE: cross-request GDN state safety does NOT depend on this call:
    // every from-scratch prefill already zeroes THIS request's recurrent + conv
    // state after bind (record_state_reset), which is what makes a bare
    // HybridGdnExecutor + reused HybridGdnCache safe too.
    {
        auto list = CommandList::create();
        if (auto s = _cache.reset(list); !s.ok()) { return R::failure(s); }
        stream << list.commit() << synchronize();
    }
    // Vision (optional: text-only when no images).
    uint32_t visual_tokens = 0;
    BufferView<float> features;
    if (!images.empty()) {
        auto list = CommandList::create();
        auto v = _vision.record_vision(list, images);
        if (!v.ok()) { return R::failure(v.status); }
        stream << list.commit() << synchronize();
        visual_tokens = v.value->total_tokens;
        features = v.value->features;
    }
    if (static_cast<uint64_t>(mm_start) + visual_tokens > prompt_tokens.size()) {
        return R::failure(StatusCode::invalid_argument, "image range exceeds the prompt", "mm_start");
    }
    DecodeRequest request;
    request.prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());
    if (visual_tokens != 0u) {
        request.mm_features = features;
        request.mm_start = mm_start;
        request.mm_count = visual_tokens;
    }
    HybridDecodeController controller;
    if (auto s = controller.create(_text, tokenizer, options); !s.ok()) { return R::failure(s); }
    auto gen = controller.generate(stream, request, _cache);
    if (!gen.ok()) { return R::failure(gen.status); }
    HybridVlmRunResult result;
    result.decode = std::move(*gen.value);
    result.visual_tokens = visual_tokens;
    result.weight_bytes = _weight_bytes;
    result.workspace_bytes = _vision.vision_workspace_bytes() + _text.workspace_bytes();
    result.cache_bytes = _cache.device_bytes();
    return R::success(std::move(result));
}

[[nodiscard]] Status HybridVlmGraph::download_vision_features(Stream &stream, luisa::vector<float> &out) const {
    return _vision.download_vision_features(stream, out);
}

}// namespace luisa::ocr

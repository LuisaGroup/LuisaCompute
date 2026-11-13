// OCR engine -- hybrid GDN decoder CONFIG + tensor-name contract (Phase 9).
// Host-side validation / capacity math / tensor inventory. The device executor
// and GDN kernels consume this data; no DSL/device code lives here.
#include "ocr/runtime/hybrid_gdn_config.h"

#include <cmath>

#include <luisa/core/stl/format.h>

#include "ocr/quantization.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string_view hybrid_layer_kind_name(HybridLayerKind kind) noexcept {
    switch (kind) {
        case HybridLayerKind::gdn: return "gdn";
        case HybridLayerKind::full_attention: return "full_attention";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// GdnLayerConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status GdnLayerConfig::validate() const {
    if (num_heads == 0u || head_dim == 0u) {
        return Status::error(StatusCode::invalid_argument, "GDN heads/head_dim must be nonzero", "geometry");
    }
    if (conv_kernel_size < 2u || conv_kernel_size > 16u) {
        return Status::error(StatusCode::invalid_argument,
                             "GDN conv kernel size must be in [2, 16] (ring holds K-1 taps)", "conv_kernel_size");
    }
    if (!(l2_eps > 0.0f) || !std::isfinite(l2_eps)) {
        return Status::error(StatusCode::invalid_argument, "GDN l2_eps must be positive and finite", "l2_eps");
    }
    if (!(norm_eps > 0.0f) || !std::isfinite(norm_eps)) {
        return Status::error(StatusCode::invalid_argument, "GDN norm_eps must be positive and finite", "norm_eps");
    }
    if (!(softplus_threshold > 0.0f) || !std::isfinite(softplus_threshold)) {
        return Status::error(StatusCode::invalid_argument,
                             "GDN softplus_threshold must be positive and finite", "softplus_threshold");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// HybridCacheConfig
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridCacheConfig::validate() const {
    if (batch == 0u || max_seq_len == 0u) {
        return Status::error(StatusCode::invalid_argument, "hybrid cache batch/max_seq_len must be nonzero", "config");
    }
    if (num_full_layers == 0u && num_gdn_layers == 0u) {
        return Status::error(StatusCode::invalid_argument, "hybrid cache needs at least one layer", "config");
    }
    if (dtype_bytes != 4u) {
        return Status::error(StatusCode::unsupported, "Phase 9 hybrid cache storage is FP32 (dtype_bytes == 4)", "dtype_bytes");
    }
    if (num_full_layers != 0u && (num_kv_heads == 0u || head_dim == 0u)) {
        return Status::error(StatusCode::invalid_argument, "full-attention KV geometry must be nonzero", "kv");
    }
    if (num_gdn_layers != 0u) {
        if (gdn_heads == 0u || gdn_head_dim == 0u) {
            return Status::error(StatusCode::invalid_argument, "GDN heads/head_dim must be nonzero", "gdn");
        }
        if (conv_kernel_size < 2u || conv_kernel_size > 16u) {
            return Status::error(StatusCode::invalid_argument, "GDN conv kernel size must be in [2, 16]", "conv_kernel_size");
        }
    }
    return Status::success();
}

[[nodiscard]] Result<uint64_t> HybridCacheConfig::kv_bytes() const {
    if (num_full_layers == 0u) { return Result<uint64_t>::success(0u); }
    return plan_kv_cache_bytes(num_full_layers, batch, max_seq_len, num_kv_heads, head_dim, dtype_bytes);
}
[[nodiscard]] Result<uint64_t> HybridCacheConfig::recurrent_bytes() const {
    uint64_t per = 0, total = 0;
    if (!checked_mul(gdn_heads, gdn_head_dim, per) || !checked_mul(per, gdn_head_dim, per) ||
        !checked_mul(per, batch, per) || !checked_mul(per, num_gdn_layers, total) ||
        !checked_mul(total, dtype_bytes, total)) {
        return Result<uint64_t>::failure(StatusCode::invalid_argument, "GDN recurrent byte count overflows", "recurrent");
    }
    return Result<uint64_t>::success(total);
}
[[nodiscard]] Result<uint64_t> HybridCacheConfig::conv_bytes() const {
    const uint64_t channels = 3ull * gdn_qk_dim();
    uint64_t per = 0, total = 0;
    if (!checked_mul(channels, conv_ring_width(), per) || !checked_mul(per, batch, per) ||
        !checked_mul(per, num_gdn_layers, total) || !checked_mul(total, dtype_bytes, total)) {
        return Result<uint64_t>::failure(StatusCode::invalid_argument, "GDN conv byte count overflows", "conv");
    }
    return Result<uint64_t>::success(total);
}
[[nodiscard]] Result<uint64_t> HybridCacheConfig::metadata_bytes() const {
    return Result<uint64_t>::success(4ull * batch * 4ull);
}
[[nodiscard]] Result<uint64_t> HybridCacheConfig::total_bytes() const {
    auto s = validate();
    if (!s.ok()) { return Result<uint64_t>::failure(s); }
    auto kv = kv_bytes();
    if (!kv.ok()) { return kv; }
    auto rec = recurrent_bytes();
    if (!rec.ok()) { return rec; }
    auto conv = conv_bytes();
    if (!conv.ok()) { return conv; }
    auto meta = metadata_bytes();
    uint64_t total = 0;
    if (!checked_add(*kv.value, *rec.value, total) || !checked_add(total, *conv.value, total) ||
        !checked_add(total, *meta.value, total)) {
        return Result<uint64_t>::failure(StatusCode::invalid_argument, "hybrid cache total byte count overflows", "total");
    }
    return Result<uint64_t>::success(total);
}

// ---------------------------------------------------------------------------
// HybridGdnExecutorConfig
// ---------------------------------------------------------------------------

[[nodiscard]] uint32_t HybridGdnExecutorConfig::num_full_layers() const noexcept {
    uint32_t n = 0;
    for (auto k : layer_kinds) { n += (k == HybridLayerKind::full_attention) ? 1u : 0u; }
    return n;
}
[[nodiscard]] uint32_t HybridGdnExecutorConfig::num_gdn_layers() const noexcept {
    uint32_t n = 0;
    for (auto k : layer_kinds) { n += (k == HybridLayerKind::gdn) ? 1u : 0u; }
    return n;
}
[[nodiscard]] Status HybridGdnExecutorConfig::validate() const {
    auto s = decoder.validate();
    if (!s.ok()) { return s; }
    s = gdn.validate();
    if (!s.ok()) { return s; }
    if (layer_kinds.size() != decoder.num_layers) {
        return Status::error(StatusCode::invalid_argument, "layer_kinds size must equal decoder.num_layers", "layer_kinds");
    }
    if (num_gdn_layers() == 0u) {
        return Status::error(StatusCode::invalid_argument,
                             "a hybrid executor needs at least one GDN layer (pure dense uses DenseVlmExecutor)", "layer_kinds");
    }
    if (qk_norm && qkv_bias) {
        return Status::error(StatusCode::invalid_argument,
                             "qk_norm (Qwen3) and qkv_bias (Qwen2.5) are mutually exclusive contracts", "qk_norm/qkv_bias");
    }
    if (prefill_chunk_rows == 0u) {
        return Status::error(StatusCode::invalid_argument, "prefill_chunk_rows must be nonzero", "prefill_chunk_rows");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// HybridGdnTensorNames
// ---------------------------------------------------------------------------

[[nodiscard]] Status HybridGdnTensorNames::validate() const { return dense.validate(); }
[[nodiscard]] luisa::string HybridGdnTensorNames::layer_prefix(uint32_t layer) const { return dense.layer_prefix(layer); }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_q_proj(uint32_t l) const { return layer_prefix(l) + "gdn.q_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_k_proj(uint32_t l) const { return layer_prefix(l) + "gdn.k_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_v_proj(uint32_t l) const { return layer_prefix(l) + "gdn.v_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_z_proj(uint32_t l) const { return layer_prefix(l) + "gdn.z_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_a_proj(uint32_t l) const { return layer_prefix(l) + "gdn.a_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_b_proj(uint32_t l) const { return layer_prefix(l) + "gdn.b_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_o_proj(uint32_t l) const { return layer_prefix(l) + "gdn.o_proj.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_conv_q_weight(uint32_t l) const { return layer_prefix(l) + "gdn.conv_q.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_conv_q_bias(uint32_t l) const { return layer_prefix(l) + "gdn.conv_q.bias"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_conv_k_weight(uint32_t l) const { return layer_prefix(l) + "gdn.conv_k.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_conv_k_bias(uint32_t l) const { return layer_prefix(l) + "gdn.conv_k.bias"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_conv_v_weight(uint32_t l) const { return layer_prefix(l) + "gdn.conv_v.weight"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_conv_v_bias(uint32_t l) const { return layer_prefix(l) + "gdn.conv_v.bias"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_a_log(uint32_t l) const { return layer_prefix(l) + "gdn.A_log"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_dt_bias(uint32_t l) const { return layer_prefix(l) + "gdn.dt_bias"; }
[[nodiscard]] luisa::string HybridGdnTensorNames::gdn_norm_weight(uint32_t l) const { return layer_prefix(l) + "gdn.norm.weight"; }

// ---------------------------------------------------------------------------
// Expected tensor inventory
// ---------------------------------------------------------------------------

[[nodiscard]] luisa::vector<ExpectedTensor> hybrid_gdn_expected_tensors(
    const HybridGdnExecutorConfig &config, const HybridGdnTensorNames &names) {
    const auto &d = config.decoder;
    const auto &g = config.gdn;
    const uint32_t hidden = d.hidden_size, inter = d.intermediate_size;
    const uint32_t qd = d.q_dim(), kvd = d.kv_dim(), hd = d.head_dim;
    const uint32_t qk = g.qk_dim(), gh = g.num_heads, gd = g.head_dim, K = g.conv_kernel_size;
    luisa::vector<ExpectedTensor> out;
    const auto kn = TensorLayout::kn;
    const auto rm = TensorLayout::row_major;
    out.emplace_back(ExpectedTensor{names.dense.embed_tokens(), {d.vocab_size, hidden}, rm, true});
    out.emplace_back(ExpectedTensor{names.dense.final_norm(), {hidden}, rm, true});
    if (!config.tie_word_embeddings) {
        out.emplace_back(ExpectedTensor{names.dense.lm_head(), {hidden, d.vocab_size}, kn, true});
    }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        out.emplace_back(ExpectedTensor{names.dense.input_layernorm(l), {hidden}, rm, true});
        out.emplace_back(ExpectedTensor{names.dense.post_attention_layernorm(l), {hidden}, rm, true});
        out.emplace_back(ExpectedTensor{names.dense.gate_proj(l), {hidden, inter}, kn, true});
        out.emplace_back(ExpectedTensor{names.dense.up_proj(l), {hidden, inter}, kn, true});
        out.emplace_back(ExpectedTensor{names.dense.down_proj(l), {inter, hidden}, kn, true});
        if (config.layer_kinds[l] == HybridLayerKind::full_attention) {
            out.emplace_back(ExpectedTensor{names.dense.q_proj(l), {hidden, qd}, kn, true});
            out.emplace_back(ExpectedTensor{names.dense.k_proj(l), {hidden, kvd}, kn, true});
            out.emplace_back(ExpectedTensor{names.dense.v_proj(l), {hidden, kvd}, kn, true});
            out.emplace_back(ExpectedTensor{names.dense.o_proj(l), {qd, hidden}, kn, true});
            if (config.qkv_bias) {
                out.emplace_back(ExpectedTensor{names.dense.q_bias(l), {qd}, rm, true});
                out.emplace_back(ExpectedTensor{names.dense.k_bias(l), {kvd}, rm, true});
                out.emplace_back(ExpectedTensor{names.dense.v_bias(l), {kvd}, rm, true});
            }
            if (config.qk_norm) {
                out.emplace_back(ExpectedTensor{names.dense.q_norm(l), {hd}, rm, true});
                out.emplace_back(ExpectedTensor{names.dense.k_norm(l), {hd}, rm, true});
            }
        } else {
            out.emplace_back(ExpectedTensor{names.gdn_q_proj(l), {hidden, qk}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_k_proj(l), {hidden, qk}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_v_proj(l), {hidden, qk}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_z_proj(l), {hidden, qk}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_a_proj(l), {hidden, gh}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_b_proj(l), {hidden, gh}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_o_proj(l), {qk, hidden}, kn, true});
            out.emplace_back(ExpectedTensor{names.gdn_conv_q_weight(l), {qk, K}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_conv_q_bias(l), {qk}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_conv_k_weight(l), {qk, K}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_conv_k_bias(l), {qk}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_conv_v_weight(l), {qk, K}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_conv_v_bias(l), {qk}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_a_log(l), {gh}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_dt_bias(l), {gh}, rm, true});
            out.emplace_back(ExpectedTensor{names.gdn_norm_weight(l), {gd}, rm, true});
        }
    }
    return out;
}

[[nodiscard]] Status hybrid_gdn_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                 luisa::span<const ExpectedTensor> expected) {
    return validate_dense_decoder_tensors(tensors, expected);
}

}// namespace luisa::ocr

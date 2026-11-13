#pragma once
// Shared synthetic Phase 9 fixtures + INDEPENDENT CPU oracle for the hybrid GDN
// executor device tests. Synthetic only: deterministic weights, declared small
// geometries, no model download, no real-model parity claim.
//
// The oracle is composed ONLY from the Phase 2 host references (cpu_reference.h)
// and the Phase 9 GDN host oracles (ops_gdn.h) -- it shares NO code with the
// device executor. It processes the decoder TOKEN BY TOKEN (the straight
// recurrent path), maintaining host KV / recurrent / conv-ring state, so it is a
// valid independent reference for the device's recurrent prefill, chunked
// prefill, AND single-step decode.
//
// DEVICE-SAFE: includes the device executor (ops_attention.h) but NEVER
// api.h/model_recipe.h (host AttentionMaskKind clash).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "ut/ut.hpp"
#include "ocr_test_utils.h"
#include "ocr_p7_test_utils.h"

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_gdn.h"
#include "ocr/package.h"
#include "ocr/runtime/hybrid_gdn_executor.h"
#include "ocr/runtime/weights.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Synthetic hybrid decoder weights (config-driven, FP32).
// ---------------------------------------------------------------------------
struct GdnFullLayer {
    luisa::vector<float> input_norm, post_norm;
    luisa::vector<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b, q_norm, k_norm;
    luisa::vector<float> gate_w, up_w, down_w;
};
struct GdnGdnLayer {
    luisa::vector<float> input_norm, post_norm;
    luisa::vector<float> q_w, k_w, v_w, z_w, a_w, b_w, o_w;
    luisa::vector<float> conv_q_w, conv_q_b, conv_k_w, conv_k_b, conv_v_w, conv_v_b;
    luisa::vector<float> a_log, dt_bias, norm_w;
    luisa::vector<float> gate_w, up_w, down_w;
};
struct HybridSyntheticWeights {
    luisa::vector<float> embed, final_norm, lm_head;// lm_head [hidden, vocab] kn
    bool tied{true};
    luisa::vector<GdnFullLayer> full;
    luisa::vector<GdnGdnLayer> gdn;
};

[[nodiscard]] inline HybridSyntheticWeights generate_hybrid_weights(const HybridGdnExecutorConfig &cfg,
                                                                    uint32_t seed = 0x9E37u) {
    HybridSyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    const auto &d = cfg.decoder;
    const auto &g = cfg.gdn;
    const uint32_t hidden = d.hidden_size, inter = d.intermediate_size;
    const uint32_t qd = d.q_dim(), kvd = d.kv_dim(), hd = d.head_dim;
    const uint32_t qk = g.qk_dim(), gh = g.num_heads, gd = g.head_dim, K = g.conv_kernel_size;
    fill(w.embed, static_cast<size_t>(d.vocab_size) * hidden, -0.6f, 0.6f);
    fill(w.final_norm, hidden, 0.7f, 1.3f);
    w.tied = cfg.tie_word_embeddings;
    // Standalone lm_head = transposed embedding [hidden, vocab] (same effective
    // weights as the tied arm).
    w.lm_head.resize(static_cast<size_t>(hidden) * d.vocab_size);
    for (uint32_t v = 0; v < d.vocab_size; v++) {
        for (uint32_t h = 0; h < hidden; h++) {
            w.lm_head[static_cast<size_t>(h) * d.vocab_size + v] = w.embed[static_cast<size_t>(v) * hidden + h];
        }
    }
    uint32_t fi = 0, gi = 0;
    w.full.resize(cfg.num_full_layers());
    w.gdn.resize(cfg.num_gdn_layers());
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (cfg.layer_kinds[l] == HybridLayerKind::full_attention) {
            auto &L = w.full[fi++];
            fill(L.input_norm, hidden, 0.7f, 1.3f);
            fill(L.post_norm, hidden, 0.7f, 1.3f);
            fill(L.q_w, static_cast<size_t>(hidden) * qd, -0.3f, 0.3f);
            fill(L.k_w, static_cast<size_t>(hidden) * kvd, -0.3f, 0.3f);
            fill(L.v_w, static_cast<size_t>(hidden) * kvd, -0.3f, 0.3f);
            fill(L.o_w, static_cast<size_t>(qd) * hidden, -0.3f, 0.3f);
            if (cfg.qkv_bias) {
                fill(L.q_b, qd, -0.1f, 0.1f);
                fill(L.k_b, kvd, -0.1f, 0.1f);
                fill(L.v_b, kvd, -0.1f, 0.1f);
            }
            if (cfg.qk_norm) {
                fill(L.q_norm, hd, 0.7f, 1.3f);
                fill(L.k_norm, hd, 0.7f, 1.3f);
            }
            fill(L.gate_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
            fill(L.up_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
            fill(L.down_w, static_cast<size_t>(inter) * hidden, -0.3f, 0.3f);
        } else {
            auto &L = w.gdn[gi++];
            fill(L.input_norm, hidden, 0.7f, 1.3f);
            fill(L.post_norm, hidden, 0.7f, 1.3f);
            fill(L.q_w, static_cast<size_t>(hidden) * qk, -0.3f, 0.3f);
            fill(L.k_w, static_cast<size_t>(hidden) * qk, -0.3f, 0.3f);
            fill(L.v_w, static_cast<size_t>(hidden) * qk, -0.3f, 0.3f);
            fill(L.z_w, static_cast<size_t>(hidden) * qk, -0.3f, 0.3f);
            fill(L.a_w, static_cast<size_t>(hidden) * gh, -0.3f, 0.3f);
            fill(L.b_w, static_cast<size_t>(hidden) * gh, -0.3f, 0.3f);
            fill(L.o_w, static_cast<size_t>(qk) * hidden, -0.3f, 0.3f);
            fill(L.conv_q_w, static_cast<size_t>(qk) * K, -0.4f, 0.4f);
            fill(L.conv_q_b, qk, -0.1f, 0.1f);
            fill(L.conv_k_w, static_cast<size_t>(qk) * K, -0.4f, 0.4f);
            fill(L.conv_k_b, qk, -0.1f, 0.1f);
            fill(L.conv_v_w, static_cast<size_t>(qk) * K, -0.4f, 0.4f);
            fill(L.conv_v_b, qk, -0.1f, 0.1f);
            fill(L.a_log, gh, -1.0f, 0.0f);
            fill(L.dt_bias, gh, -0.3f, 0.3f);
            fill(L.norm_w, gd, 0.7f, 1.3f);
            fill(L.gate_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
            fill(L.up_w, static_cast<size_t>(hidden) * inter, -0.3f, 0.3f);
            fill(L.down_w, static_cast<size_t>(inter) * hidden, -0.3f, 0.3f);
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// Package writing (decoder component, FP32).
// ---------------------------------------------------------------------------
[[nodiscard]] inline Status write_hybrid_decoder_package(const luisa::filesystem::path &path,
                                                         const HybridGdnExecutorConfig &cfg,
                                                         const HybridGdnTensorNames &names,
                                                         const HybridSyntheticWeights &w) {
    auto writer = PackageWriter::create(path);
    if (!writer.ok()) { return writer.status; }
    auto &wr = *writer.value;
    const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                           TensorLayout layout, luisa::span<const float> data) -> Status {
        PackageWriter::TensorSpec spec;
        spec.name = name;
        spec.encoding = TensorEncoding::fp32;
        spec.layout = layout;
        spec.shape = std::move(shape);
        spec.payload = p7_f32_bytes(data);
        return wr->add_tensor(spec);
    };
    const auto &d = cfg.decoder;
    const auto &g = cfg.gdn;
    const uint32_t hidden = d.hidden_size, inter = d.intermediate_size;
    const uint32_t qd = d.q_dim(), kvd = d.kv_dim(), hd = d.head_dim;
    const uint32_t qk = g.qk_dim(), gh = g.num_heads, gd = g.head_dim, K = g.conv_kernel_size;
    const auto kn = TensorLayout::kn;
    const auto rm = TensorLayout::row_major;
    if (auto s = add(names.dense.embed_tokens(), {d.vocab_size, hidden}, rm, w.embed); !s.ok()) { return s; }
    if (auto s = add(names.dense.final_norm(), {hidden}, rm, w.final_norm); !s.ok()) { return s; }
    if (!cfg.tie_word_embeddings) {
        if (auto s = add(names.dense.lm_head(), {hidden, d.vocab_size}, kn, w.lm_head); !s.ok()) { return s; }
    }
    uint32_t fi = 0, gi = 0;
    for (uint32_t l = 0; l < d.num_layers; l++) {
        if (cfg.layer_kinds[l] == HybridLayerKind::full_attention) {
            const auto &L = w.full[fi++];
            if (auto s = add(names.dense.input_layernorm(l), {hidden}, rm, L.input_norm); !s.ok()) { return s; }
            if (auto s = add(names.dense.post_attention_layernorm(l), {hidden}, rm, L.post_norm); !s.ok()) { return s; }
            if (auto s = add(names.dense.q_proj(l), {hidden, qd}, kn, L.q_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.k_proj(l), {hidden, kvd}, kn, L.k_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.v_proj(l), {hidden, kvd}, kn, L.v_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.o_proj(l), {qd, hidden}, kn, L.o_w); !s.ok()) { return s; }
            if (cfg.qkv_bias) {
                if (auto s = add(names.dense.q_bias(l), {qd}, rm, L.q_b); !s.ok()) { return s; }
                if (auto s = add(names.dense.k_bias(l), {kvd}, rm, L.k_b); !s.ok()) { return s; }
                if (auto s = add(names.dense.v_bias(l), {kvd}, rm, L.v_b); !s.ok()) { return s; }
            }
            if (cfg.qk_norm) {
                if (auto s = add(names.dense.q_norm(l), {hd}, rm, L.q_norm); !s.ok()) { return s; }
                if (auto s = add(names.dense.k_norm(l), {hd}, rm, L.k_norm); !s.ok()) { return s; }
            }
            if (auto s = add(names.dense.gate_proj(l), {hidden, inter}, kn, L.gate_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.up_proj(l), {hidden, inter}, kn, L.up_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.down_proj(l), {inter, hidden}, kn, L.down_w); !s.ok()) { return s; }
        } else {
            const auto &L = w.gdn[gi++];
            if (auto s = add(names.dense.input_layernorm(l), {hidden}, rm, L.input_norm); !s.ok()) { return s; }
            if (auto s = add(names.dense.post_attention_layernorm(l), {hidden}, rm, L.post_norm); !s.ok()) { return s; }
            if (auto s = add(names.gdn_q_proj(l), {hidden, qk}, kn, L.q_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_k_proj(l), {hidden, qk}, kn, L.k_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_v_proj(l), {hidden, qk}, kn, L.v_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_z_proj(l), {hidden, qk}, kn, L.z_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_a_proj(l), {hidden, gh}, kn, L.a_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_b_proj(l), {hidden, gh}, kn, L.b_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_o_proj(l), {qk, hidden}, kn, L.o_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_conv_q_weight(l), {qk, K}, rm, L.conv_q_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_conv_q_bias(l), {qk}, rm, L.conv_q_b); !s.ok()) { return s; }
            if (auto s = add(names.gdn_conv_k_weight(l), {qk, K}, rm, L.conv_k_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_conv_k_bias(l), {qk}, rm, L.conv_k_b); !s.ok()) { return s; }
            if (auto s = add(names.gdn_conv_v_weight(l), {qk, K}, rm, L.conv_v_w); !s.ok()) { return s; }
            if (auto s = add(names.gdn_conv_v_bias(l), {qk}, rm, L.conv_v_b); !s.ok()) { return s; }
            if (auto s = add(names.gdn_a_log(l), {gh}, rm, L.a_log); !s.ok()) { return s; }
            if (auto s = add(names.gdn_dt_bias(l), {gh}, rm, L.dt_bias); !s.ok()) { return s; }
            if (auto s = add(names.gdn_norm_weight(l), {gd}, rm, L.norm_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.gate_proj(l), {hidden, inter}, kn, L.gate_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.up_proj(l), {hidden, inter}, kn, L.up_w); !s.ok()) { return s; }
            if (auto s = add(names.dense.down_proj(l), {inter, hidden}, kn, L.down_w); !s.ok()) { return s; }
        }
    }
    return wr->finish(p7_manifest("decoder", "ovis_ocr2"));
}

[[nodiscard]] inline Status upload_hybrid_weights(compute::Device &device, compute::Stream &stream,
                                                  const luisa::filesystem::path &decoder_path, WeightStore &store) {
    auto reader = PackageReader::open(decoder_path);
    if (!reader.ok()) { return reader.status; }
    return store.upload_component_sync(device, stream, **reader.value, "decoder");
}

// ---------------------------------------------------------------------------
// Independent token-by-token CPU oracle for the full hybrid decoder.
// ---------------------------------------------------------------------------
class HybridCpuOracle {
public:
    HybridCpuOracle(const HybridGdnExecutorConfig &cfg, const HybridSyntheticWeights &w)
        : _cfg(cfg), _w(w) {
        const auto &d = cfg.decoder;
        const auto &g = cfg.gdn;
        _capacity = d.max_position_embeddings;
        _k.assign(cfg.num_full_layers(), luisa::vector<float>(static_cast<size_t>(_capacity) * d.kv_dim(), 0.0f));
        _v.assign(cfg.num_full_layers(), luisa::vector<float>(static_cast<size_t>(_capacity) * d.kv_dim(), 0.0f));
        _rec.assign(cfg.num_gdn_layers(), luisa::vector<float>(static_cast<size_t>(g.num_heads) * g.head_dim * g.head_dim, 0.0f));
        _ring.assign(cfg.num_gdn_layers(), luisa::vector<float>(static_cast<size_t>(3) * g.qk_dim() * (g.conv_kernel_size - 1u), 0.0f));
        _kv_len.assign(cfg.num_full_layers(), 0u);
    }

    void reset() {
        for (auto &b : _k) { std::fill(b.begin(), b.end(), 0.0f); }
        for (auto &b : _v) { std::fill(b.begin(), b.end(), 0.0f); }
        for (auto &b : _rec) { std::fill(b.begin(), b.end(), 0.0f); }
        for (auto &b : _ring) { std::fill(b.begin(), b.end(), 0.0f); }
        std::fill(_kv_len.begin(), _kv_len.end(), 0u);
        _pos = 0;
    }

    // Process one token at the current position; returns its logits row.
    [[nodiscard]] luisa::vector<float> step(uint32_t token, luisa::span<const float> mm_features = {},
                                          uint32_t mm_start = 0, uint32_t mm_count = 0) {
        const auto &d = _cfg.decoder;
        const auto &g = _cfg.gdn;
        const uint32_t hidden = d.hidden_size, vocab = d.vocab_size;
        const uint32_t p = _pos;
        // Embedding (+ optional multimodal replacement for this row).
        luisa::vector<float> x(hidden);
        for (uint32_t h = 0; h < hidden; h++) { x[h] = _w.embed[static_cast<size_t>(token) * hidden + h]; }
        if (mm_count != 0 && p >= mm_start && p < mm_start + mm_count) {
            const uint32_t fr = p - mm_start;
            for (uint32_t h = 0; h < hidden; h++) { x[h] = mm_features[static_cast<size_t>(fr) * hidden + h]; }
        }
        uint32_t fi = 0, gi = 0;
        for (uint32_t l = 0; l < d.num_layers; l++) {
            if (_cfg.layer_kinds[l] == HybridLayerKind::full_attention) {
                x = full_layer(fi++, l, x, p);
            } else {
                x = gdn_layer(gi++, l, x, p);
            }
        }
        // Final norm + LM head.
        luisa::vector<float> norm, logits;
        LUISA_ASSERT(ref_rms_norm(x, _w.final_norm, 1, hidden, d.rms_norm_eps, norm).ok(), "oracle final norm");
        LUISA_ASSERT(ref_gemm(norm, _w.lm_head, {}, 1, vocab, hidden, RefGemmEpilogue::none, logits).ok(), "oracle lm head");
        _pos++;
        return logits;
    }

    [[nodiscard]] uint32_t position() const noexcept { return _pos; }
    [[nodiscard]] const luisa::vector<float> &recurrent(uint32_t gdn_layer) const { return _rec[gdn_layer]; }
    [[nodiscard]] const luisa::vector<float> &ring(uint32_t gdn_layer) const { return _ring[gdn_layer]; }
    [[nodiscard]] const luisa::vector<float> &kv_k(uint32_t full_layer) const { return _k[full_layer]; }
    [[nodiscard]] const luisa::vector<float> &kv_v(uint32_t full_layer) const { return _v[full_layer]; }
    [[nodiscard]] uint32_t kv_len(uint32_t full_layer) const { return _kv_len[full_layer]; }

private:
    [[nodiscard]] luisa::vector<float> mlp(luisa::span<const float> x, luisa::span<const float> post_norm,
                                         luisa::span<const float> gate_w, luisa::span<const float> up_w,
                                         luisa::span<const float> down_w) {
        const auto &d = _cfg.decoder;
        const uint32_t hidden = d.hidden_size, inter = d.intermediate_size;
        luisa::vector<float> norm, gate, up, act, down, out;
        LUISA_ASSERT(ref_rms_norm(x, post_norm, 1, hidden, d.rms_norm_eps, norm).ok(), "oracle mlp norm");
        LUISA_ASSERT(ref_gemm(norm, gate_w, {}, 1, inter, hidden, RefGemmEpilogue::none, gate).ok(), "oracle gate");
        LUISA_ASSERT(ref_gemm(norm, up_w, {}, 1, inter, hidden, RefGemmEpilogue::none, up).ok(), "oracle up");
        LUISA_ASSERT(ref_gated_act(gate, up, 0u, act).ok(), "oracle silu");
        LUISA_ASSERT(ref_gemm(act, down_w, {}, 1, hidden, inter, RefGemmEpilogue::none, down).ok(), "oracle down");
        LUISA_ASSERT(ref_residual_add(x, down, out).ok(), "oracle mlp residual");
        return out;
    }

    [[nodiscard]] luisa::vector<float> full_layer(uint32_t fi, uint32_t /*l*/, luisa::span<const float> x, uint32_t p) {
        const auto &d = _cfg.decoder;
        const auto &L = _w.full[fi];
        const uint32_t hidden = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
        const uint32_t qh = d.num_attention_heads, kvh = d.num_kv_heads, hd = d.head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        luisa::vector<float> norm, q, k, v;
        LUISA_ASSERT(ref_rms_norm(x, L.input_norm, 1, hidden, d.rms_norm_eps, norm).ok(), "oracle full norm");
        const auto epi = _cfg.qkv_bias ? RefGemmEpilogue::bias : RefGemmEpilogue::none;
        LUISA_ASSERT(ref_gemm(norm, L.q_w, _cfg.qkv_bias ? luisa::span<const float>{L.q_b} : luisa::span<const float>{}, 1, qd, hidden, epi, q).ok(), "oracle q");
        LUISA_ASSERT(ref_gemm(norm, L.k_w, _cfg.qkv_bias ? luisa::span<const float>{L.k_b} : luisa::span<const float>{}, 1, kvd, hidden, epi, k).ok(), "oracle k");
        LUISA_ASSERT(ref_gemm(norm, L.v_w, _cfg.qkv_bias ? luisa::span<const float>{L.v_b} : luisa::span<const float>{}, 1, kvd, hidden, epi, v).ok(), "oracle v");
        if (_cfg.qk_norm) {
            luisa::vector<float> qn, kn;
            LUISA_ASSERT(ref_rms_norm(q, L.q_norm, qh, hd, d.rms_norm_eps, qn).ok(), "oracle qnorm");
            LUISA_ASSERT(ref_rms_norm(k, L.k_norm, kvh, hd, d.rms_norm_eps, kn).ok(), "oracle knorm");
            q = qn;
            k = kn;
        }
        // RoPE (1-axis).
        const luisa::vector<int32_t> pos = {static_cast<int32_t>(p)};
        luisa::vector<float> qr, kr;
        if (d.rope_pair_layout == RopePairLayout::rotate_half) {
            LUISA_ASSERT(ref_rope_rotate_half(q, pos, 1, qh, hd, d.rope_theta, qr).ok(), "oracle q rope");
            LUISA_ASSERT(ref_rope_rotate_half(k, pos, 1, kvh, hd, d.rope_theta, kr).ok(), "oracle k rope");
        } else {
            LUISA_ASSERT(ref_rope_1d(q, pos, 1, qh, hd, d.rope_theta, qr).ok(), "oracle q rope");
            LUISA_ASSERT(ref_rope_1d(k, pos, 1, kvh, hd, d.rope_theta, kr).ok(), "oracle k rope");
        }
        // Append to host KV at position p.
        for (uint32_t i = 0; i < kvd; i++) {
            _k[fi][static_cast<size_t>(p) * kvd + i] = kr[i];
            _v[fi][static_cast<size_t>(p) * kvd + i] = v[i];
        }
        _kv_len[fi] = p + 1u;
        // Decode-style attention over the committed prefix [0, p].
        const luisa::vector<uint32_t> kv_len = {p + 1u};
        luisa::vector<float> attn;
        RefAttentionMask mask;
        mask.kind = RefMaskKind::causal;
        LUISA_ASSERT(ref_attention_decode(qr, _k[fi], _v[fi], kv_len, pos, 1, _capacity, qh, kvh, hd,
                                          scale, 0.0f, mask, attn).ok(), "oracle full attn");
        luisa::vector<float> o, res;
        LUISA_ASSERT(ref_gemm(attn, L.o_w, {}, 1, hidden, qd, RefGemmEpilogue::none, o).ok(), "oracle o proj");
        LUISA_ASSERT(ref_residual_add(x, o, res).ok(), "oracle full residual");
        return mlp(res, L.post_norm, L.gate_w, L.up_w, L.down_w);
    }

    [[nodiscard]] luisa::vector<float> gdn_layer(uint32_t gi, uint32_t /*l*/, luisa::span<const float> x, uint32_t p) {
        const auto &d = _cfg.decoder;
        const auto &g = _cfg.gdn;
        const auto &L = _w.gdn[gi];
        const uint32_t hidden = d.hidden_size;
        const uint32_t qk = g.qk_dim(), gh = g.num_heads, gd = g.head_dim, K = g.conv_kernel_size;
        luisa::vector<float> norm, gq, gk, gv, gz, ga, gb;
        LUISA_ASSERT(ref_rms_norm(x, L.input_norm, 1, hidden, d.rms_norm_eps, norm).ok(), "oracle gdn norm");
        LUISA_ASSERT(ref_gemm(norm, L.q_w, {}, 1, qk, hidden, RefGemmEpilogue::none, gq).ok(), "oracle gdn q");
        LUISA_ASSERT(ref_gemm(norm, L.k_w, {}, 1, qk, hidden, RefGemmEpilogue::none, gk).ok(), "oracle gdn k");
        LUISA_ASSERT(ref_gemm(norm, L.v_w, {}, 1, qk, hidden, RefGemmEpilogue::none, gv).ok(), "oracle gdn v");
        LUISA_ASSERT(ref_gemm(norm, L.z_w, {}, 1, qk, hidden, RefGemmEpilogue::none, gz).ok(), "oracle gdn z");
        LUISA_ASSERT(ref_gemm(norm, L.a_w, {}, 1, gh, hidden, RefGemmEpilogue::none, ga).ok(), "oracle gdn a");
        LUISA_ASSERT(ref_gemm(norm, L.b_w, {}, 1, gh, hidden, RefGemmEpilogue::none, gb).ok(), "oracle gdn b");
        // Conv+SiLU+ring (q/k/v thirds of the ring).
        const size_t third = static_cast<size_t>(qk) * (K - 1u);
        luisa::vector<float> ring_q(_ring[gi].begin(), _ring[gi].begin() + third);
        luisa::vector<float> ring_k(_ring[gi].begin() + third, _ring[gi].begin() + 2 * third);
        luisa::vector<float> ring_v(_ring[gi].begin() + 2 * third, _ring[gi].begin() + 3 * third);
        luisa::vector<float> gq_conv, gk_conv, gv_conv;
        LUISA_ASSERT(ref_gdn_conv_silu(gq, L.conv_q_w, L.conv_q_b, ring_q, 1, qk, K, p, gq_conv).ok(), "oracle conv q");
        LUISA_ASSERT(ref_gdn_conv_silu(gk, L.conv_k_w, L.conv_k_b, ring_k, 1, qk, K, p, gk_conv).ok(), "oracle conv k");
        LUISA_ASSERT(ref_gdn_conv_silu(gv, L.conv_v_w, L.conv_v_b, ring_v, 1, qk, K, p, gv_conv).ok(), "oracle conv v");
        std::copy(ring_q.begin(), ring_q.end(), _ring[gi].begin());
        std::copy(ring_k.begin(), ring_k.end(), _ring[gi].begin() + third);
        std::copy(ring_v.begin(), ring_v.end(), _ring[gi].begin() + 2 * third);
        // L2 normalize q/k per head.
        luisa::vector<float> gq_norm, gk_norm;
        LUISA_ASSERT(ref_l2_norm(gq_conv, gh, gd, g.l2_eps, gq_norm).ok(), "oracle l2 q");
        LUISA_ASSERT(ref_l2_norm(gk_conv, gh, gd, g.l2_eps, gk_norm).ok(), "oracle l2 k");
        // Gates.
        luisa::vector<float> beta, gg;
        LUISA_ASSERT(ref_gdn_gates(ga, gb, L.a_log, L.dt_bias, 1, gh, g.softplus_threshold, beta, gg).ok(), "oracle gates");
        // Recurrent delta rule (T=1), carrying the state.
        luisa::vector<float> go;
        LUISA_ASSERT(ref_gdn_recurrent(gq_norm, gk_norm, gv_conv, beta, gg, _rec[gi], 1, gh, gd, go).ok(), "oracle recurrent");
        // Gated output norm.
        luisa::vector<float> gynorm;
        LUISA_ASSERT(ref_gdn_gated_norm(go, gz, L.norm_w, 1, gh, gd, g.norm_eps, gynorm).ok(), "oracle gated norm");
        luisa::vector<float> o, res;
        LUISA_ASSERT(ref_gemm(gynorm, L.o_w, {}, 1, hidden, qk, RefGemmEpilogue::none, o).ok(), "oracle gdn o");
        LUISA_ASSERT(ref_residual_add(x, o, res).ok(), "oracle gdn residual");
        return mlp(res, L.post_norm, L.gate_w, L.up_w, L.down_w);
    }

    HybridGdnExecutorConfig _cfg;
    HybridSyntheticWeights _w;
    uint32_t _capacity{0};
    uint32_t _pos{0};
    luisa::vector<luisa::vector<float>> _k, _v, _rec, _ring;
    luisa::vector<uint32_t> _kv_len;
};

// A small synthetic hybrid geometry: 4 layers = [gdn, full, gdn, full] (2 GDN +
// 2 full), hidden 16, GQA 4Q/2KV head_dim 4, GDN 2 heads head_dim 4 conv K=4.
[[nodiscard]] inline HybridGdnExecutorConfig hybrid_synthetic_config() {
    HybridGdnExecutorConfig cfg;
    auto &d = cfg.decoder;
    d.vocab_size = 24;
    d.hidden_size = 16;
    d.num_layers = 4;
    d.num_attention_heads = 4;
    d.num_kv_heads = 2;
    d.head_dim = 4;
    d.intermediate_size = 32;
    d.rms_norm_eps = 1e-5f;
    d.rope_theta = 10000.0f;
    d.rope_pair_layout = RopePairLayout::adjacent;
    d.max_position_embeddings = 64;
    d.eos_token_ids = {23u};
    cfg.gdn.num_heads = 2;
    cfg.gdn.head_dim = 4;
    cfg.gdn.conv_kernel_size = 4;
    cfg.gdn.l2_eps = 1e-6f;
    cfg.gdn.norm_eps = 1e-5f;
    cfg.gdn.softplus_threshold = 20.0f;
    cfg.layer_kinds = {HybridLayerKind::gdn, HybridLayerKind::full_attention,
                       HybridLayerKind::gdn, HybridLayerKind::full_attention};
    cfg.tie_word_embeddings = true;
    cfg.qkv_bias = true;
    cfg.qk_norm = false;
    cfg.component = "decoder";
    cfg.prefill_chunk_rows = 64;
    return cfg;
}

[[nodiscard]] inline HybridCacheConfig hybrid_cache_config(const HybridGdnExecutorConfig &cfg, uint32_t max_seq_len) {
    HybridCacheConfig c;
    c.num_full_layers = cfg.num_full_layers();
    c.num_gdn_layers = cfg.num_gdn_layers();
    c.batch = 1;
    c.num_kv_heads = cfg.decoder.num_kv_heads;
    c.head_dim = cfg.decoder.head_dim;
    c.max_seq_len = max_seq_len;
    c.gdn_heads = cfg.gdn.num_heads;
    c.gdn_head_dim = cfg.gdn.head_dim;
    c.conv_kernel_size = cfg.gdn.conv_kernel_size;
    c.dtype_bytes = 4;
    return c;
}

// Controlled weights for controller-wiring tests: ZERO every per-layer tensor
// (so each layer's residual contribution is exactly 0 and the residual stream
// stays equal to the token embedding), set final_norm = 1, and use ONE-HOT
// embeddings (embed[v] = e_{v % hidden}). Then logits[v] = <rmsnorm(embed[last]),
// embed[v]> is nonzero ONLY for v % hidden == last % hidden, and greedy argmax
// (lowest-index tie-break) returns `last` exactly for last < hidden: a
// deterministic SELF-LOOP on the last prompt token. This lets a test force early
// EOS (eos == last prompt token) or a repeated-text stop string without needing
// to script a full transformer. The conv/gate/recurrent math still runs (on
// zeros) so the device path is exercised end to end.
[[nodiscard]] inline HybridSyntheticWeights make_self_loop_weights(const HybridGdnExecutorConfig &cfg,
                                                                   uint32_t seed = 0x51EEu) {
    auto w = generate_hybrid_weights(cfg, seed);
    const auto zero = [](luisa::vector<float> &v) { std::fill(v.begin(), v.end(), 0.0f); };
    for (auto &L : w.full) {
        zero(L.input_norm); zero(L.post_norm);
        zero(L.q_w); zero(L.k_w); zero(L.v_w); zero(L.o_w);
        zero(L.q_b); zero(L.k_b); zero(L.v_b); zero(L.q_norm); zero(L.k_norm);
        zero(L.gate_w); zero(L.up_w); zero(L.down_w);
    }
    for (auto &L : w.gdn) {
        zero(L.input_norm); zero(L.post_norm);
        zero(L.q_w); zero(L.k_w); zero(L.v_w); zero(L.z_w); zero(L.a_w); zero(L.b_w); zero(L.o_w);
        zero(L.conv_q_w); zero(L.conv_q_b); zero(L.conv_k_w); zero(L.conv_k_b); zero(L.conv_v_w); zero(L.conv_v_b);
        zero(L.a_log); zero(L.dt_bias); zero(L.norm_w);
        zero(L.gate_w); zero(L.up_w); zero(L.down_w);
    }
    std::fill(w.final_norm.begin(), w.final_norm.end(), 1.0f);
    // One-hot embeddings (exact self-loop for tokens < hidden).
    const uint32_t hidden = cfg.decoder.hidden_size, vocab = cfg.decoder.vocab_size;
    std::fill(w.embed.begin(), w.embed.end(), 0.0f);
    for (uint32_t v = 0; v < vocab; v++) {
        w.embed[static_cast<size_t>(v) * hidden + (v % hidden)] = 1.0f;
    }
    // Recompute the tied LM head from the one-hot embedding.
    w.lm_head.resize(static_cast<size_t>(hidden) * vocab);
    for (uint32_t v = 0; v < vocab; v++) {
        for (uint32_t h = 0; h < hidden; h++) {
            w.lm_head[static_cast<size_t>(h) * vocab + v] = w.embed[static_cast<size_t>(v) * hidden + h];
        }
    }
    return w;
}

// A 24-piece Unigram tokenizer matching the synthetic hybrid vocab (24): ids
// 0..22 -> lowercase letters 'a'..'w'; id 23 -> "</s>" (special EOS). Lets the
// model tests detokenize generated ids into deterministic visible text for the
// end-to-end Markdown path. Generated in-memory; no fixture file / download.
[[nodiscard]] inline luisa::unique_ptr<Tokenizer, TokenizerDeleter> load_hybrid_tokenizer() {
    luisa::string vocab = "[";
    for (uint32_t i = 0; i < 23u; i++) {
        const char c = static_cast<char>('a' + i);
        vocab += luisa::string{"[\""} + c + "\",-1.0]";
        vocab += ",";
    }
    vocab += "[\"</s>\",-1.0]]";
    luisa::string json = "{";
    json += "\"version\":\"1.0\",";
    json += "\"added_tokens\":[{\"id\":23,\"content\":\"</s>\",\"special\":true,"
            "\"lstrip\":false,\"rstrip\":false,\"single_word\":false}],";
    json += "\"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,";
    json += "\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"byte_fallback\":false,"
            "\"add_bos_token\":false,\"add_eos_token\":false,";
    json += "\"vocab\":" + vocab + "}}";
    auto r = Tokenizer::parse(json);
    if (!r.ok()) {
        LUISA_WARNING("hybrid tokenizer parse failed: {}", r.status.message);
        return nullptr;
    }
    return std::move(*r.value);
}

}// namespace luisa::ocr::test

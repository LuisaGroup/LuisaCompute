#pragma once
// Shared synthetic Phase 10 fixtures + INDEPENDENT CPU oracle for the DeepSeek
// MoE VLM device tests. Synthetic only: deterministic weights, declared small
// geometries (EXACT 64 routed / top-6 / two-shared), no download, no real-model
// parity claim.
//
// The oracle is composed ONLY from the Phase 2 host references (cpu_reference.h)
// and the Phase 10 MoE host oracles (ops_moe.h) -- it shares NO code with the
// device executor. The decoder oracle processes tokens ONE AT A TIME (maintaining
// host KV), so it is a valid independent reference for the device's causal
// prefill AND single-step decode. The vision oracle mirrors the SAM perception +
// BOTH conv downsamplers + custom-mask visual LM encoder + connector exactly.
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

#include "ocr/models/deepseek_ocr2.h"
#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_moe.h"
#include "ocr/package.h"
#include "ocr/runtime/moe_vlm_executor.h"
#include "ocr/runtime/weights.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Synthetic weights (config-driven, FP32).
// ---------------------------------------------------------------------------
struct DeepSeekVisionBlock {
    luisa::vector<float> norm1_w, norm1_b, norm2_w, norm2_b;
    luisa::vector<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b, o_b;
    luisa::vector<float> fc1_w, fc1_b, fc2_w, fc2_b;
};
struct DeepSeekConvW { luisa::vector<float> weight, bias; };
struct DeepSeekMoeLayerW {
    luisa::vector<float> input_norm, post_norm;
    luisa::vector<float> q_w, k_w, v_w, o_w, q_b, k_b, v_b;
    luisa::vector<float> router_w;
    luisa::vector<MoeExpertWeights> experts;
    MoeExpertWeights shared0, shared1;
};
struct DeepSeekSyntheticWeights {
    // Vision.
    luisa::vector<float> patch_embed_w, patch_embed_b;
    luisa::vector<DeepSeekVisionBlock> perception;
    DeepSeekConvW down1, down2;
    luisa::vector<DeepSeekVisionBlock> encoder;
    luisa::vector<float> query_embed;
    luisa::vector<float> conn_ln_w, conn_ln_b, conn_fc1_w, conn_fc1_b, conn_fc2_w, conn_fc2_b;
    // Decoder.
    luisa::vector<float> embed, final_norm, lm_head;// lm_head [hidden, vocab] kn
    luisa::vector<DeepSeekMoeLayerW> layers;
};

[[nodiscard]] inline DeepSeekSyntheticWeights generate_deepseek_weights(const DeepSeekOcr2Config &cfg, uint32_t seed = 0xD5EEu) {
    DeepSeekSyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    const auto &v = cfg.vision;
    const auto &d = cfg.decoder.decoder;
    const uint32_t vh = v.perception.hidden_size, vi = v.perception.intermediate_size;
    const uint32_t patch_dim = v.perception.patch_dim();
    const uint32_t eh = v.encoder_hidden(), ei = v.encoder_intermediate_size;
    const uint32_t th = d.hidden_size;
    fill(w.patch_embed_w, static_cast<size_t>(patch_dim) * vh, -0.3f, 0.3f);
    fill(w.patch_embed_b, vh, -0.1f, 0.1f);
    const auto fill_block = [&](DeepSeekVisionBlock &B, uint32_t h, uint32_t inter) {
        fill(B.norm1_w, h, 0.7f, 1.3f); fill(B.norm1_b, h, -0.1f, 0.1f);
        fill(B.norm2_w, h, 0.7f, 1.3f); fill(B.norm2_b, h, -0.1f, 0.1f);
        fill(B.q_w, static_cast<size_t>(h) * h, -0.3f, 0.3f);
        fill(B.k_w, static_cast<size_t>(h) * h, -0.3f, 0.3f);
        fill(B.v_w, static_cast<size_t>(h) * h, -0.3f, 0.3f);
        fill(B.o_w, static_cast<size_t>(h) * h, -0.3f, 0.3f);
        fill(B.q_b, h, -0.1f, 0.1f); fill(B.k_b, h, -0.1f, 0.1f);
        fill(B.v_b, h, -0.1f, 0.1f); fill(B.o_b, h, -0.1f, 0.1f);
        fill(B.fc1_w, static_cast<size_t>(h) * inter, -0.3f, 0.3f);
        fill(B.fc1_b, inter, -0.1f, 0.1f);
        fill(B.fc2_w, static_cast<size_t>(inter) * h, -0.3f, 0.3f);
        fill(B.fc2_b, h, -0.1f, 0.1f);
    };
    w.perception.resize(v.perception.num_layers);
    for (auto &B : w.perception) { fill_block(B, vh, vi); }
    fill(w.down1.weight, static_cast<size_t>(v.down1.out_channels) * vh * v.down1.kernel * v.down1.kernel, -0.2f, 0.2f);
    fill(w.down1.bias, v.down1.out_channels, -0.1f, 0.1f);
    fill(w.down2.weight, static_cast<size_t>(v.down2.out_channels) * v.down1.out_channels * v.down2.kernel * v.down2.kernel, -0.2f, 0.2f);
    fill(w.down2.bias, v.down2.out_channels, -0.1f, 0.1f);
    w.encoder.resize(v.encoder_layers);
    for (auto &B : w.encoder) { fill_block(B, eh, ei); }
    fill(w.query_embed, static_cast<size_t>(v.num_queries) * eh, -0.3f, 0.3f);
    fill(w.conn_ln_w, eh, 0.7f, 1.3f); fill(w.conn_ln_b, eh, -0.1f, 0.1f);
    fill(w.conn_fc1_w, static_cast<size_t>(eh) * v.connector_intermediate_size, -0.3f, 0.3f);
    fill(w.conn_fc1_b, v.connector_intermediate_size, -0.1f, 0.1f);
    fill(w.conn_fc2_w, static_cast<size_t>(v.connector_intermediate_size) * th, -0.3f, 0.3f);
    fill(w.conn_fc2_b, th, -0.1f, 0.1f);
    // Decoder.
    fill(w.embed, static_cast<size_t>(d.vocab_size) * th, -0.6f, 0.6f);
    fill(w.final_norm, th, 0.7f, 1.3f);
    w.lm_head.resize(static_cast<size_t>(th) * d.vocab_size);
    for (uint32_t vv = 0; vv < d.vocab_size; vv++) {
        for (uint32_t h = 0; h < th; h++) {
            w.lm_head[static_cast<size_t>(h) * d.vocab_size + vv] = w.embed[static_cast<size_t>(vv) * th + h];
        }
    }
    const uint32_t qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t mi = cfg.decoder.moe_intermediate_size, si = cfg.decoder.shared_intermediate_size;
    const uint32_t ne = cfg.decoder.num_experts;
    w.layers.resize(d.num_layers);
    for (auto &L : w.layers) {
        fill(L.input_norm, th, 0.7f, 1.3f); fill(L.post_norm, th, 0.7f, 1.3f);
        fill(L.q_w, static_cast<size_t>(th) * qd, -0.3f, 0.3f);
        fill(L.k_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill(L.v_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill(L.o_w, static_cast<size_t>(qd) * th, -0.3f, 0.3f);
        fill(L.q_b, qd, -0.1f, 0.1f); fill(L.k_b, kvd, -0.1f, 0.1f); fill(L.v_b, kvd, -0.1f, 0.1f);
        fill(L.router_w, static_cast<size_t>(th) * ne, -0.4f, 0.4f);
        L.experts.resize(ne);
        for (auto &e : L.experts) {
            fill(e.gate_w, static_cast<size_t>(th) * mi, -0.3f, 0.3f);
            fill(e.up_w, static_cast<size_t>(th) * mi, -0.3f, 0.3f);
            fill(e.down_w, static_cast<size_t>(mi) * th, -0.3f, 0.3f);
        }
        fill(L.shared0.gate_w, static_cast<size_t>(th) * si, -0.3f, 0.3f);
        fill(L.shared0.up_w, static_cast<size_t>(th) * si, -0.3f, 0.3f);
        fill(L.shared0.down_w, static_cast<size_t>(si) * th, -0.3f, 0.3f);
        fill(L.shared1.gate_w, static_cast<size_t>(th) * si, -0.3f, 0.3f);
        fill(L.shared1.up_w, static_cast<size_t>(th) * si, -0.3f, 0.3f);
        fill(L.shared1.down_w, static_cast<size_t>(si) * th, -0.3f, 0.3f);
    }
    return w;
}

// Self-loop weights: zero attention + MoE, one-hot embeddings, final_norm=1,
// lm_head=transpose(embed). Greedy decode then repeats the last prompt token,
// giving deterministic NON-EMPTY text for the end-to-end grounded Markdown path.
[[nodiscard]] inline DeepSeekSyntheticWeights make_deepseek_self_loop_weights(const DeepSeekOcr2Config &cfg, uint32_t seed = 0x51EEu) {
    auto w = generate_deepseek_weights(cfg, seed);
    const auto zero = [](luisa::vector<float> &v) { std::fill(v.begin(), v.end(), 0.0f); };
    const auto zero_block = [&](DeepSeekVisionBlock &B) {
        zero(B.norm1_w); zero(B.norm1_b); zero(B.norm2_w); zero(B.norm2_b);
        zero(B.q_w); zero(B.k_w); zero(B.v_w); zero(B.o_w);
        zero(B.q_b); zero(B.k_b); zero(B.v_b); zero(B.o_b);
        zero(B.fc1_w); zero(B.fc1_b); zero(B.fc2_w); zero(B.fc2_b);
    };
    // Keep vision functional (it only feeds features); zero the DECODER so the
    // text self-loops. Vision stays as generated.
    for (auto &L : w.layers) {
        zero(L.input_norm); zero(L.post_norm);
        zero(L.q_w); zero(L.k_w); zero(L.v_w); zero(L.o_w);
        zero(L.q_b); zero(L.k_b); zero(L.v_b); zero(L.router_w);
        for (auto &e : L.experts) { zero(e.gate_w); zero(e.up_w); zero(e.down_w); }
        zero(L.shared0.gate_w); zero(L.shared0.up_w); zero(L.shared0.down_w);
        zero(L.shared1.gate_w); zero(L.shared1.up_w); zero(L.shared1.down_w);
    }
    std::fill(w.final_norm.begin(), w.final_norm.end(), 1.0f);
    const uint32_t th = cfg.decoder.decoder.hidden_size, vocab = cfg.decoder.decoder.vocab_size;
    std::fill(w.embed.begin(), w.embed.end(), 0.0f);
    for (uint32_t vv = 0; vv < vocab; vv++) { w.embed[static_cast<size_t>(vv) * th + (vv % th)] = 1.0f; }
    w.lm_head.assign(static_cast<size_t>(th) * vocab, 0.0f);
    for (uint32_t vv = 0; vv < vocab; vv++) {
        for (uint32_t h = 0; h < th; h++) {
            w.lm_head[static_cast<size_t>(h) * vocab + vv] = w.embed[static_cast<size_t>(vv) * th + h];
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// Package writers (vision + decoder components).
// ---------------------------------------------------------------------------
[[nodiscard]] inline Status write_deepseek_vision_package(const luisa::filesystem::path &path,
                                                          const DeepSeekOcr2Config &cfg,
                                                          const DeepSeekSyntheticWeights &w) {
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
    const auto &v = cfg.vision;
    const auto &names = cfg.vision_names;
    const auto kn = TensorLayout::kn;
    const auto rm = TensorLayout::row_major;
    // Conv weights: row-major tag with OIHW shape (matches the shared WeightStore
    // mapping and the Phase 5/9 CNN conv convention).
    const auto oi = TensorLayout::row_major;
    const uint32_t vh = v.perception.hidden_size, vi = v.perception.intermediate_size;
    const uint32_t eh = v.encoder_hidden(), ei = v.encoder_intermediate_size;
    const uint32_t th = cfg.decoder.decoder.hidden_size;
    const auto &pn = names.perception;
    if (auto s = add(pn.patch_embed_weight(), {v.perception.patch_dim(), vh}, kn, w.patch_embed_w); !s.ok()) { return s; }
    if (auto s = add(pn.patch_embed_bias(), {vh}, rm, w.patch_embed_b); !s.ok()) { return s; }
    const auto write_block = [&](const DeepSeekVisionBlock &B, uint32_t l, bool perception) -> Status {
        const auto n1w = perception ? pn.block_norm1_weight(l) : names.encoder_norm1_weight(l);
        const auto n1b = perception ? pn.block_norm1_bias(l) : names.encoder_norm1_bias(l);
        const auto n2w = perception ? pn.block_norm2_weight(l) : names.encoder_norm2_weight(l);
        const auto n2b = perception ? pn.block_norm2_bias(l) : names.encoder_norm2_bias(l);
        const auto qw = perception ? pn.block_q_proj(l) : names.encoder_q_proj(l);
        const auto kw = perception ? pn.block_k_proj(l) : names.encoder_k_proj(l);
        const auto vw = perception ? pn.block_v_proj(l) : names.encoder_v_proj(l);
        const auto ow = perception ? pn.block_o_proj(l) : names.encoder_o_proj(l);
        const auto qb = perception ? pn.block_q_bias(l) : names.encoder_q_bias(l);
        const auto kb = perception ? pn.block_k_bias(l) : names.encoder_k_bias(l);
        const auto vb = perception ? pn.block_v_bias(l) : names.encoder_v_bias(l);
        const auto ob = perception ? pn.block_o_bias(l) : names.encoder_o_bias(l);
        const auto f1 = perception ? pn.block_fc1(l) : names.encoder_fc1(l);
        const auto f1b = perception ? pn.block_fc1_bias(l) : names.encoder_fc1_bias(l);
        const auto f2 = perception ? pn.block_fc2(l) : names.encoder_fc2(l);
        const auto f2b = perception ? pn.block_fc2_bias(l) : names.encoder_fc2_bias(l);
        const uint32_t h = perception ? vh : eh, inter = perception ? vi : ei;
        if (auto s = add(n1w, {h}, rm, B.norm1_w); !s.ok()) { return s; }
        if (auto s = add(n1b, {h}, rm, B.norm1_b); !s.ok()) { return s; }
        if (auto s = add(n2w, {h}, rm, B.norm2_w); !s.ok()) { return s; }
        if (auto s = add(n2b, {h}, rm, B.norm2_b); !s.ok()) { return s; }
        if (auto s = add(qw, {h, h}, kn, B.q_w); !s.ok()) { return s; }
        if (auto s = add(kw, {h, h}, kn, B.k_w); !s.ok()) { return s; }
        if (auto s = add(vw, {h, h}, kn, B.v_w); !s.ok()) { return s; }
        if (auto s = add(ow, {h, h}, kn, B.o_w); !s.ok()) { return s; }
        if (auto s = add(qb, {h}, rm, B.q_b); !s.ok()) { return s; }
        if (auto s = add(kb, {h}, rm, B.k_b); !s.ok()) { return s; }
        if (auto s = add(vb, {h}, rm, B.v_b); !s.ok()) { return s; }
        if (auto s = add(ob, {h}, rm, B.o_b); !s.ok()) { return s; }
        if (auto s = add(f1, {h, inter}, kn, B.fc1_w); !s.ok()) { return s; }
        if (auto s = add(f1b, {inter}, rm, B.fc1_b); !s.ok()) { return s; }
        if (auto s = add(f2, {inter, h}, kn, B.fc2_w); !s.ok()) { return s; }
        if (auto s = add(f2b, {h}, rm, B.fc2_b); !s.ok()) { return s; }
        return Status::success();
    };
    for (uint32_t l = 0; l < v.perception.num_layers; l++) {
        if (auto s = write_block(w.perception[l], l, true); !s.ok()) { return s; }
    }
    if (auto s = add(names.down_weight(1), {v.down1.out_channels, vh, v.down1.kernel, v.down1.kernel}, oi, w.down1.weight); !s.ok()) { return s; }
    if (auto s = add(names.down_bias(1), {v.down1.out_channels}, rm, w.down1.bias); !s.ok()) { return s; }
    if (auto s = add(names.down_weight(2), {v.down2.out_channels, v.down1.out_channels, v.down2.kernel, v.down2.kernel}, oi, w.down2.weight); !s.ok()) { return s; }
    if (auto s = add(names.down_bias(2), {v.down2.out_channels}, rm, w.down2.bias); !s.ok()) { return s; }
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        if (auto s = write_block(w.encoder[l], l, false); !s.ok()) { return s; }
    }
    if (auto s = add(names.query_embed(), {v.num_queries, eh}, rm, w.query_embed); !s.ok()) { return s; }
    if (auto s = add(names.connector_ln_weight(), {eh}, rm, w.conn_ln_w); !s.ok()) { return s; }
    if (auto s = add(names.connector_ln_bias(), {eh}, rm, w.conn_ln_b); !s.ok()) { return s; }
    if (auto s = add(names.connector_fc1(), {eh, v.connector_intermediate_size}, kn, w.conn_fc1_w); !s.ok()) { return s; }
    if (auto s = add(names.connector_fc1_bias(), {v.connector_intermediate_size}, rm, w.conn_fc1_b); !s.ok()) { return s; }
    if (auto s = add(names.connector_fc2(), {v.connector_intermediate_size, th}, kn, w.conn_fc2_w); !s.ok()) { return s; }
    if (auto s = add(names.connector_fc2_bias(), {th}, rm, w.conn_fc2_b); !s.ok()) { return s; }
    return wr->finish(p7_manifest("vision", "deepseek_ocr_2"));
}

[[nodiscard]] inline Status write_deepseek_decoder_package(const luisa::filesystem::path &path,
                                                           const DeepSeekOcr2Config &cfg,
                                                           const DeepSeekSyntheticWeights &w) {
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
    const auto &d = cfg.decoder.decoder;
    const auto &names = cfg.decoder_names;
    const auto kn = TensorLayout::kn;
    const auto rm = TensorLayout::row_major;
    const uint32_t th = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t mi = cfg.decoder.moe_intermediate_size, si = cfg.decoder.shared_intermediate_size;
    if (auto s = add(names.dense.embed_tokens(), {d.vocab_size, th}, rm, w.embed); !s.ok()) { return s; }
    if (auto s = add(names.dense.final_norm(), {th}, rm, w.final_norm); !s.ok()) { return s; }
    if (auto s = add(names.dense.lm_head(), {th, d.vocab_size}, kn, w.lm_head); !s.ok()) { return s; }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        const auto &L = w.layers[l];
        if (auto s = add(names.dense.input_layernorm(l), {th}, rm, L.input_norm); !s.ok()) { return s; }
        if (auto s = add(names.dense.post_attention_layernorm(l), {th}, rm, L.post_norm); !s.ok()) { return s; }
        if (auto s = add(names.dense.q_proj(l), {th, qd}, kn, L.q_w); !s.ok()) { return s; }
        if (auto s = add(names.dense.k_proj(l), {th, kvd}, kn, L.k_w); !s.ok()) { return s; }
        if (auto s = add(names.dense.v_proj(l), {th, kvd}, kn, L.v_w); !s.ok()) { return s; }
        if (auto s = add(names.dense.o_proj(l), {qd, th}, kn, L.o_w); !s.ok()) { return s; }
        if (auto s = add(names.dense.q_bias(l), {qd}, rm, L.q_b); !s.ok()) { return s; }
        if (auto s = add(names.dense.k_bias(l), {kvd}, rm, L.k_b); !s.ok()) { return s; }
        if (auto s = add(names.dense.v_bias(l), {kvd}, rm, L.v_b); !s.ok()) { return s; }
        if (auto s = add(names.router_weight(l), {th, cfg.decoder.num_experts}, kn, L.router_w); !s.ok()) { return s; }
        for (uint32_t e = 0; e < cfg.decoder.num_experts; e++) {
            if (auto s = add(names.expert_gate(l, e), {th, mi}, kn, L.experts[e].gate_w); !s.ok()) { return s; }
            if (auto s = add(names.expert_up(l, e), {th, mi}, kn, L.experts[e].up_w); !s.ok()) { return s; }
            if (auto s = add(names.expert_down(l, e), {mi, th}, kn, L.experts[e].down_w); !s.ok()) { return s; }
        }
        if (auto s = add(names.shared_gate(l, 0), {th, si}, kn, L.shared0.gate_w); !s.ok()) { return s; }
        if (auto s = add(names.shared_up(l, 0), {th, si}, kn, L.shared0.up_w); !s.ok()) { return s; }
        if (auto s = add(names.shared_down(l, 0), {si, th}, kn, L.shared0.down_w); !s.ok()) { return s; }
        if (auto s = add(names.shared_gate(l, 1), {th, si}, kn, L.shared1.gate_w); !s.ok()) { return s; }
        if (auto s = add(names.shared_up(l, 1), {th, si}, kn, L.shared1.up_w); !s.ok()) { return s; }
        if (auto s = add(names.shared_down(l, 1), {si, th}, kn, L.shared1.down_w); !s.ok()) { return s; }
    }
    return wr->finish(p7_manifest("decoder", "deepseek_ocr_2"));
}

[[nodiscard]] inline Status upload_deepseek_weights(compute::Device &device, compute::Stream &stream,
                                                    const luisa::filesystem::path &vision_path,
                                                    const luisa::filesystem::path &decoder_path, WeightStore &store) {
    {
        auto reader = PackageReader::open(vision_path);
        if (!reader.ok()) { return reader.status; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "vision"); !s.ok()) { return s; }
    }
    {
        auto reader = PackageReader::open(decoder_path);
        if (!reader.ok()) { return reader.status; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "decoder"); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Independent vision oracle (SAM perception + 2 conv downsamplers + custom-mask
// visual LM encoder + connector). Mirrors the device path exactly.
// ---------------------------------------------------------------------------
struct DeepSeekVisionOracle {
    luisa::vector<float> patches;
    luisa::vector<float> patch_embed;
    luisa::vector<float> perception_out;
    luisa::vector<float> down1_out, down2_out;// token-major
    uint32_t visual_rows{0};
    luisa::vector<float> encoder_input;// visual + queries
    luisa::vector<luisa::vector<float>> encoder_attn;// per-layer attention output
    luisa::vector<float> encoder_out;
    luisa::vector<float> features;// [num_queries, text_hidden]
};

// token-major [rows, C] -> NCHW [C, rows]
[[nodiscard]] inline luisa::vector<float> ds_tokens_to_nchw(luisa::span<const float> tokens, uint32_t rows, uint32_t c) {
    luisa::vector<float> out(static_cast<size_t>(c) * rows);
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            out[static_cast<size_t>(ci) * rows + t] = tokens[static_cast<size_t>(t) * c + ci];
        }
    }
    return out;
}
[[nodiscard]] inline luisa::vector<float> ds_nchw_to_tokens(luisa::span<const float> nchw, uint32_t c, uint32_t rows) {
    luisa::vector<float> out(static_cast<size_t>(rows) * c);
    for (uint32_t t = 0; t < rows; t++) {
        for (uint32_t ci = 0; ci < c; ci++) {
            out[static_cast<size_t>(t) * c + ci] = nchw[static_cast<size_t>(ci) * rows + t];
        }
    }
    return out;
}

// The EXACT custom visual/query block mask (block_size 1).
// S9: the mask BYTES come from ONE canonical builder shared with the executor
// (MoeVlmExecutor::build_visual_query_mask), so the oracle and the device consume
// identical data and a test can compare the downloaded device mask byte-for-byte.
[[nodiscard]] inline RefAttentionMask ds_visual_query_mask(uint32_t visual_rows, uint32_t num_queries) {
    RefAttentionMask mask;
    mask.kind = RefMaskKind::custom_block;
    mask.block_size = 1u;
    mask.block_mask = MoeVlmExecutor::build_visual_query_mask(visual_rows, num_queries);
    return mask;
}

// Independent STRUCTURAL expectation of the mask (no shared code with the
// canonical builder): used to prove the builder itself encodes the declared
// reference semantics, not just that both sides agree.
[[nodiscard]] inline luisa::vector<uint8_t> ds_expected_visual_query_mask(uint32_t visual_rows, uint32_t num_queries) {
    const uint32_t rows = visual_rows + num_queries;
    luisa::vector<uint8_t> want(static_cast<size_t>(rows) * rows, 0u);
    for (uint32_t q = 0; q < rows; q++) {
        for (uint32_t k = 0; k < rows; k++) {
            const bool q_is_visual = q < visual_rows;
            const bool k_is_visual = k < visual_rows;
            bool allow = false;
            if (q_is_visual) {
                allow = k_is_visual;
            } else if (k_is_visual) {
                allow = true;
            } else {
                allow = (k - visual_rows) <= (q - visual_rows);
            }
            want[static_cast<size_t>(q) * rows + k] = allow ? 1u : 0u;
        }
    }
    return want;
}

[[nodiscard]] inline Status ds_oracle_vision_block(const DeepSeekVisionBlock &B, luisa::span<const float> x_in,
                                                   uint32_t rows, uint32_t h, uint32_t inter, uint32_t heads,
                                                   float eps, const RefAttentionMask &mask,
                                                   luisa::span<const int32_t> positions, luisa::span<const uint32_t> axis,
                                                   bool rope2d, float theta, luisa::vector<float> &x_out,
                                                   luisa::vector<float> &attn_out) {
    const uint32_t hd = h / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    luisa::vector<float> norm, q, k, vv, qr, kr, attn, after_o, norm2, fc1, o_proj, fc2_proj;
    auto s = ref_layer_norm(x_in, B.norm1_w, B.norm1_b, rows, h, eps, norm);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, B.q_w, B.q_b, rows, h, h, RefGemmEpilogue::bias, q); if (!s.ok()) { return s; }
    s = ref_gemm(norm, B.k_w, B.k_b, rows, h, h, RefGemmEpilogue::bias, k); if (!s.ok()) { return s; }
    s = ref_gemm(norm, B.v_w, B.v_b, rows, h, h, RefGemmEpilogue::bias, vv); if (!s.ok()) { return s; }
    auto q_attn = q, k_attn = k;
    if (rope2d) {
        s = ref_rope_axial(q, positions, axis, rows, heads, hd, 2u, theta, qr); if (!s.ok()) { return s; }
        s = ref_rope_axial(k, positions, axis, rows, heads, hd, 2u, theta, kr); if (!s.ok()) { return s; }
        q_attn = qr; k_attn = kr;
    }
    s = ref_attention(q_attn, k_attn, vv, rows, rows, heads, heads, hd, scale, 0.0f, mask, attn); if (!s.ok()) { return s; }
    attn_out = attn;
    s = ref_gemm(attn, B.o_w, B.o_b, rows, h, h, RefGemmEpilogue::bias, o_proj); if (!s.ok()) { return s; }
    s = ref_residual_add(x_in, o_proj, after_o); if (!s.ok()) { return s; }
    s = ref_layer_norm(after_o, B.norm2_w, B.norm2_b, rows, h, eps, norm2); if (!s.ok()) { return s; }
    s = ref_gemm(norm2, B.fc1_w, B.fc1_b, rows, inter, h, RefGemmEpilogue::bias_gelu_erf, fc1); if (!s.ok()) { return s; }
    s = ref_gemm(fc1, B.fc2_w, B.fc2_b, rows, h, inter, RefGemmEpilogue::bias, fc2_proj); if (!s.ok()) { return s; }
    return ref_residual_add(after_o, fc2_proj, x_out);
}

[[nodiscard]] inline Status ds_oracle_vision(const DeepSeekOcr2Config &cfg, const DeepSeekSyntheticWeights &w,
                                             luisa::span<const float> pixels_chw, const P7PackedImage &packed,
                                             DeepSeekVisionOracle &out) {
    const auto &v = cfg.vision;
    const uint32_t vh = v.perception.hidden_size, vi = v.perception.intermediate_size;
    const uint32_t heads = v.perception.num_heads, hd = v.perception.head_dim;
    const uint32_t rows = static_cast<uint32_t>(packed.patch_count);
    out.patches = p7_oracle_patchify(pixels_chw, packed.resized_height, packed.resized_width, v.perception.patch_size);
    auto s = ref_gemm(out.patches, w.patch_embed_w, w.patch_embed_b, rows, vh, v.perception.patch_dim(), RefGemmEpilogue::bias, out.patch_embed);
    if (!s.ok()) { return s; }
    luisa::vector<float> x = out.patch_embed;
    auto positions = p7_vision_positions(packed);
    auto axis = p7_vision_axis_of_pair(v.perception);
    // SAM perception blocks (window/global mask, 2D RoPE).
    for (uint32_t l = 0; l < v.perception.num_layers; l++) {
        const bool global = std::find(v.perception.global_layers.begin(), v.perception.global_layers.end(), l) != v.perception.global_layers.end();
        auto mask = p7_oracle_window_mask(rows, packed.grid_w, v.perception.window_patch_size, global);
        luisa::vector<float> xo, attn;
        s = ds_oracle_vision_block(w.perception[l], x, rows, vh, vi, heads, v.perception.layer_norm_eps, mask,
                                   positions, axis, true, v.perception.rope_theta, xo, attn);
        if (!s.ok()) { return s; }
        x = std::move(xo);
    }
    out.perception_out = x;
    // BOTH conv downsamplers.
    uint32_t gh = packed.grid_h, gw = packed.grid_w, cur_c = vh;
    {
        auto nchw = ds_tokens_to_nchw(x, rows, cur_c);
        luisa::vector<float> conv_out;
        uint32_t oh = 0, ow = 0;
        s = ref_conv2d(nchw, w.down1.weight, w.down1.bias, 1u, cur_c, gh, gw, v.down1.out_channels,
                       v.down1.kernel, v.down1.kernel, v.down1.stride, v.down1.stride, v.down1.pad, v.down1.pad,
                       1u, 1u, 1u, conv_out, oh, ow);
        if (!s.ok()) { return s; }
        gh = oh; gw = ow; cur_c = v.down1.out_channels;
        out.down1_out = ds_nchw_to_tokens(conv_out, cur_c, gh * gw);
        x = out.down1_out;
    }
    {
        auto nchw = ds_tokens_to_nchw(x, gh * gw, cur_c);
        luisa::vector<float> conv_out;
        uint32_t oh = 0, ow = 0;
        s = ref_conv2d(nchw, w.down2.weight, w.down2.bias, 1u, cur_c, gh, gw, v.down2.out_channels,
                       v.down2.kernel, v.down2.kernel, v.down2.stride, v.down2.stride, v.down2.pad, v.down2.pad,
                       1u, 1u, 1u, conv_out, oh, ow);
        if (!s.ok()) { return s; }
        gh = oh; gw = ow; cur_c = v.down2.out_channels;
        out.down2_out = ds_nchw_to_tokens(conv_out, cur_c, gh * gw);
        x = out.down2_out;
    }
    const uint32_t visual_rows = gh * gw;
    out.visual_rows = visual_rows;
    const uint32_t eh = v.encoder_hidden(), ei = v.encoder_intermediate_size;
    const uint32_t nq = v.num_queries;
    const uint32_t enc_rows = visual_rows + nq;
    // Append query embeddings.
    out.encoder_input.assign(static_cast<size_t>(enc_rows) * eh, 0.0f);
    for (size_t i = 0; i < x.size(); i++) { out.encoder_input[i] = x[i]; }
    for (uint32_t q = 0; q < nq; q++) {
        for (uint32_t c = 0; c < eh; c++) {
            out.encoder_input[static_cast<size_t>(visual_rows + q) * eh + c] = w.query_embed[static_cast<size_t>(q) * eh + c];
        }
    }
    // Visual LM encoder (custom mask, NO rope).
    auto enc_mask = ds_visual_query_mask(visual_rows, nq);
    luisa::vector<int32_t> no_pos;
    luisa::vector<uint32_t> no_axis;
    x = out.encoder_input;
    out.encoder_attn.resize(v.encoder_layers);
    for (uint32_t l = 0; l < v.encoder_layers; l++) {
        luisa::vector<float> xo, attn;
        s = ds_oracle_vision_block(w.encoder[l], x, enc_rows, eh, ei, heads, v.encoder_norm_eps, enc_mask,
                                   no_pos, no_axis, false, 0.0f, xo, attn);
        if (!s.ok()) { return s; }
        out.encoder_attn[l] = attn;
        x = std::move(xo);
    }
    out.encoder_out = x;
    // Connector over the query rows.
    const uint32_t th = cfg.decoder.decoder.hidden_size;
    luisa::vector<float> query_rows(static_cast<size_t>(nq) * eh);
    for (uint32_t q = 0; q < nq; q++) {
        for (uint32_t c = 0; c < eh; c++) {
            query_rows[static_cast<size_t>(q) * eh + c] = x[static_cast<size_t>(visual_rows + q) * eh + c];
        }
    }
    luisa::vector<float> cnorm, cfc1;
    s = ref_layer_norm(query_rows, w.conn_ln_w, w.conn_ln_b, nq, eh, v.connector_norm_eps, cnorm); if (!s.ok()) { return s; }
    s = ref_gemm(cnorm, w.conn_fc1_w, w.conn_fc1_b, nq, v.connector_intermediate_size, eh, RefGemmEpilogue::bias_gelu_erf, cfc1); if (!s.ok()) { return s; }
    return ref_gemm(cfc1, w.conn_fc2_w, w.conn_fc2_b, nq, th, v.connector_intermediate_size, RefGemmEpilogue::bias, out.features);
}

// ---------------------------------------------------------------------------
// Independent token-by-token MoE decoder oracle (host KV; causal). Validates
// prefill AND decode. Reuses the independent ref_moe_route / ref_moe_layer.
// ---------------------------------------------------------------------------
struct DeepSeekDecoderOracle {
    luisa::vector<luisa::vector<float>> k_cache, v_cache;// [layer][capacity*kv_dim]
    uint32_t capacity{0};
    uint32_t committed{0};
    luisa::vector<float> last_logits;
    // Per-layer routing/MoE intermediates for the LAST processed token batch.
    luisa::vector<MoeRouting> routing;
    luisa::vector<luisa::vector<float>> expert_out, shared0, shared1, moe_out;
};

[[nodiscard]] inline Status ds_oracle_decoder_step(const DeepSeekOcr2Config &cfg, const DeepSeekSyntheticWeights &w,
                                                   DeepSeekDecoderOracle &st, luisa::span<const float> emb_rows,
                                                   uint32_t rows, uint32_t pos0, luisa::vector<float> &logits_out) {
    const auto &d = cfg.decoder.decoder;
    const uint32_t th = d.hidden_size, qd = d.q_dim(), kvd = d.kv_dim();
    const uint32_t qh = d.num_attention_heads, kvh = d.num_kv_heads, hd = d.head_dim;
    const float eps = d.rms_norm_eps, theta = d.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const uint32_t top_k = cfg.decoder.top_k, ne = cfg.decoder.num_experts;
    const uint32_t mi = cfg.decoder.moe_intermediate_size, si = cfg.decoder.shared_intermediate_size;
    if (st.capacity == 0u) {
        st.capacity = d.max_position_embeddings;
        st.k_cache.assign(d.num_layers, luisa::vector<float>(static_cast<size_t>(st.capacity) * kvd, 0.0f));
        st.v_cache.assign(d.num_layers, luisa::vector<float>(static_cast<size_t>(st.capacity) * kvd, 0.0f));
        st.routing.resize(d.num_layers);
        st.expert_out.resize(d.num_layers);
        st.shared0.resize(d.num_layers);
        st.shared1.resize(d.num_layers);
        st.moe_out.resize(d.num_layers);
    }
    luisa::vector<float> x(emb_rows.begin(), emb_rows.end());
    luisa::vector<int32_t> positions(rows);
    for (uint32_t i = 0; i < rows; i++) { positions[i] = static_cast<int32_t>(pos0 + i); }
    for (uint32_t l = 0; l < d.num_layers; l++) {
        const auto &L = w.layers[l];
        luisa::vector<float> norm, q, k, v, qr, kr, attn, o_proj, after_o, norm2;
        auto s = ref_rms_norm(x, L.input_norm, rows, th, eps, norm); if (!s.ok()) { return s; }
        s = ref_gemm(norm, L.q_w, L.q_b, rows, qd, th, RefGemmEpilogue::bias, q); if (!s.ok()) { return s; }
        s = ref_gemm(norm, L.k_w, L.k_b, rows, kvd, th, RefGemmEpilogue::bias, k); if (!s.ok()) { return s; }
        s = ref_gemm(norm, L.v_w, L.v_b, rows, kvd, th, RefGemmEpilogue::bias, v); if (!s.ok()) { return s; }
        s = ref_rope_rotate_half(q, positions, rows, qh, hd, theta, qr); if (!s.ok()) { return s; }
        s = ref_rope_rotate_half(k, positions, rows, kvh, hd, theta, kr); if (!s.ok()) { return s; }
        // Append K/V to host cache at pos0.
        for (uint32_t i = 0; i < rows; i++) {
            for (uint32_t c = 0; c < kvd; c++) {
                st.k_cache[l][static_cast<size_t>(pos0 + i) * kvd + c] = kr[static_cast<size_t>(i) * kvd + c];
                st.v_cache[l][static_cast<size_t>(pos0 + i) * kvd + c] = v[static_cast<size_t>(i) * kvd + c];
            }
        }
        const uint32_t k_len = pos0 + rows;
        // Causal attention over the committed prefix [0, k_len). ref_attention
        // validates the exact k_len*kv_dim size, so pass a prefix subview of the
        // host cache (not the whole capacity-sized buffer).
        RefAttentionMask causal;
        causal.kind = RefMaskKind::causal;
        luisa::span<const float> k_prefix{st.k_cache[l].data(), static_cast<size_t>(k_len) * kvd};
        luisa::span<const float> v_prefix{st.v_cache[l].data(), static_cast<size_t>(k_len) * kvd};
        s = ref_attention(qr, k_prefix, v_prefix, rows, k_len, qh, kvh, hd, scale, 0.0f, causal, attn);
        if (!s.ok()) { return s; }
        s = ref_gemm(attn, L.o_w, {}, rows, th, qd, RefGemmEpilogue::none, o_proj); if (!s.ok()) { return s; }
        s = ref_residual_add(x, o_proj, after_o); if (!s.ok()) { return s; }
        s = ref_rms_norm(after_o, L.post_norm, rows, th, eps, norm2); if (!s.ok()) { return s; }
        // MoE.
        MoeRouterConfig rc{ne, top_k};
        s = ref_moe_route(norm2, L.router_w, rows, th, rc, st.routing[l]); if (!s.ok()) { return s; }
        luisa::vector<float> weighted;
        s = ref_moe_layer(norm2, st.routing[l], L.experts, L.shared0, L.shared1, rows, th, mi, si,
                          st.expert_out[l], weighted, st.shared0[l], st.shared1[l], st.moe_out[l]);
        if (!s.ok()) { return s; }
        s = ref_residual_add(after_o, st.moe_out[l], x); if (!s.ok()) { return s; }
    }
    st.committed = pos0 + rows;
    // Final norm + LM head over the LAST row.
    luisa::vector<float> last_row(th), fnorm, logits;
    for (uint32_t c = 0; c < th; c++) { last_row[c] = x[static_cast<size_t>(rows - 1u) * th + c]; }
    auto s = ref_rms_norm(last_row, w.final_norm, 1u, th, eps, fnorm); if (!s.ok()) { return s; }
    s = ref_gemm(fnorm, w.lm_head, {}, 1u, d.vocab_size, th, RefGemmEpilogue::none, logits); if (!s.ok()) { return s; }
    st.last_logits = logits;
    logits_out = std::move(logits);
    return Status::success();
}

// A 24-piece Unigram tokenizer matching the synthetic vocab (24): ids 0..22 ->
// 'a'..'w'; id 23 -> "</s>" (special EOS). Generated in-memory; no fixture file.
[[nodiscard]] inline luisa::unique_ptr<Tokenizer, TokenizerDeleter> load_deepseek_tokenizer() {
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
        LUISA_WARNING("deepseek tokenizer parse failed: {}", r.status.message);
        return nullptr;
    }
    return std::move(*r.value);
}

}// namespace luisa::ocr::test

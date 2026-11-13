#pragma once
// Shared synthetic Phase 8 fixtures + INDEPENDENT CPU oracles for the
// dots.mocr / GOT-OCR2.0 / olmOCR-2 device tests. Synthetic only: deterministic
// weights, declared small geometries, no model download, no real-model parity
// claim.
//
// The oracles are composed ONLY from the Phase 2 host references
// (cpu_reference.h) plus independent host math, so they share NO code with the
// device executors. A test compares two independent implementations.
//
// DEVICE-SAFE: this header includes the device executors (ops_attention.h) but
// NEVER api.h/model_recipe.h (host AttentionMaskKind clash).
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
#include "ocr/package.h"
#include "ocr/quantization.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/p7_vlm_executor.h"
#include "ocr/runtime/weights.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Quantized weight storage (per-linear INT4) for the W4A32 path.
// ---------------------------------------------------------------------------

// One linear weight's storage: either FP32 dense or packed grouped INT4. The
// dequantized FP32 values are retained for the independent oracle arm.
struct P8LinearStorage {
    DenseLinearStorage storage{DenseLinearStorage::fp32};
    uint32_t group_size{0};
    luisa::vector<float> dense;       // original FP32 values (oracle B operand)
    luisa::vector<float> dequantized; // dequantized values (quantized oracle arm)
    luisa::vector<uint8_t> packed;    // packed INT4 bytes (low nibble first)
    luisa::vector<float> scales;
    luisa::vector<uint8_t> zero_points;
    [[nodiscard]] bool is_quantized() const noexcept { return storage != DenseLinearStorage::fp32; }
};

// Quantize a dense linear weight to grouped INT4 (Phase 1 semantics) and keep
// the dequantized values for the oracle. Deterministic. quantize_int4 returns
// PACKED bytes directly (low nibble first); dequantize_int4 consumes them.
[[nodiscard]] inline Status p8_quantize_linear(luisa::span<const float> values, DenseLinearStorage storage,
                                               uint32_t group_size, P8LinearStorage &out) {
    out.storage = storage;
    out.group_size = group_size;
    out.dense.assign(values.begin(), values.end());
    if (storage == DenseLinearStorage::fp32) {
        out.dequantized = out.dense;
        return Status::success();
    }
    const bool symmetric = storage == DenseLinearStorage::int4_sym;
    auto s = quantize_int4(values, group_size, symmetric, out.packed, out.scales, out.zero_points);
    if (!s.ok()) { return s; }
    out.dequantized.resize(values.size());
    return dequantize_int4(out.packed, values.size(), out.scales, out.zero_points, group_size, symmetric, out.dequantized);
}

// The B operand the oracle should use for a linear weight: the dequantized
// values for a quantized weight, the original dense values otherwise.
[[nodiscard]] inline luisa::span<const float> p8_oracle_b(const P8LinearStorage &w) {
    return w.is_quantized() ? luisa::span<const float>{w.dequantized} : luisa::span<const float>{w.dense};
}

// ---------------------------------------------------------------------------
// Synthetic weights for the Phase 8 graphs (config-driven, per-linear storage).
// ---------------------------------------------------------------------------

struct P8TextLayer {
    luisa::vector<float> input_norm, post_norm;
    P8LinearStorage q_w, k_w, v_w, o_w;
    luisa::vector<float> q_b, k_b, v_b;
    luisa::vector<float> q_norm, k_norm;
    P8LinearStorage gate_w, up_w, down_w;
};

struct P8SyntheticWeights {
    // Vision (always FP32).
    luisa::vector<float> patch_embed_w, patch_embed_b;
    luisa::vector<float> position_embed;
    luisa::vector<P7VisionBlock> blocks;
    // Per-layer decomposed relative-position bias tables (GOT). Empty when the
    // vision tower has no relative bias.
    luisa::vector<luisa::vector<float>> rel_pos_bias;
    // Connector (always FP32).
    luisa::vector<float> conn_ln_w, conn_ln_b, conn_fc1_w, conn_fc1_b, conn_fc2_w, conn_fc2_b;
    luisa::vector<float> conn_down_w, conn_down_b;
    // Text decoder.
    luisa::vector<float> embed, final_norm;
    P8LinearStorage lm_head;
    bool tied{true};
    luisa::vector<P8TextLayer> layers;
};

// Generate deterministic weights. `linear_storage`/`group_size` select the
// per-linear decoder storage (FP32 baseline or packed INT4). The vision tower,
// connector, embeddings, norms, and biases stay FP32.
[[nodiscard]] inline P8SyntheticWeights generate_p8_weights(const P7SliceConfig &cfg,
                                                           DenseLinearStorage linear_storage = DenseLinearStorage::fp32,
                                                           uint32_t group_size = 0, uint32_t seed = 0x5EEDu) {
    P8SyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    // Quantize a freshly filled linear weight.
    const auto fill_linear = [&](P8LinearStorage &out, size_t n, float lo, float hi) {
        luisa::vector<float> tmp;
        fill(tmp, n, lo, hi);
        auto s = p8_quantize_linear(tmp, linear_storage, group_size, out);
        LUISA_ASSERT(s.ok(), "p8 quantize failed");
    };
    const uint32_t vh = cfg.vision.hidden_size, vi = cfg.vision.intermediate_size;
    const uint32_t th = cfg.text.hidden_size, ti = cfg.text.intermediate_size;
    const uint32_t ci = cfg.connector.intermediate_size;
    const uint32_t patch_dim = cfg.vision.patch_dim();
    fill(w.patch_embed_w, static_cast<size_t>(patch_dim) * vh, -0.3f, 0.3f);
    fill(w.patch_embed_b, vh, -0.1f, 0.1f);
    if (cfg.vision.position_rows != 0u) {
        fill(w.position_embed, static_cast<size_t>(cfg.vision.position_rows) * vh, -0.2f, 0.2f);
    }
    w.blocks.resize(cfg.vision.num_layers);
    for (auto &B : w.blocks) {
        fill(B.norm1_w, vh, 0.7f, 1.3f);
        fill(B.norm1_b, vh, -0.1f, 0.1f);
        fill(B.norm2_w, vh, 0.7f, 1.3f);
        fill(B.norm2_b, vh, -0.1f, 0.1f);
        fill(B.q_w, static_cast<size_t>(vh) * vh, -0.3f, 0.3f);
        fill(B.k_w, static_cast<size_t>(vh) * vh, -0.3f, 0.3f);
        fill(B.v_w, static_cast<size_t>(vh) * vh, -0.3f, 0.3f);
        fill(B.o_w, static_cast<size_t>(vh) * vh, -0.3f, 0.3f);
        if (cfg.vision.qkv_bias) {
            fill(B.q_b, vh, -0.1f, 0.1f);
            fill(B.k_b, vh, -0.1f, 0.1f);
            fill(B.v_b, vh, -0.1f, 0.1f);
            fill(B.o_b, vh, -0.1f, 0.1f);
        }
        if (cfg.vision.qk_norm) {
            fill(B.q_norm, cfg.vision.head_dim, 0.7f, 1.3f);
            fill(B.k_norm, cfg.vision.head_dim, 0.7f, 1.3f);
        }
        fill(B.fc1_w, static_cast<size_t>(vh) * vi, -0.3f, 0.3f);
        fill(B.fc1_b, vi, -0.1f, 0.1f);
        fill(B.fc2_w, static_cast<size_t>(vi) * vh, -0.3f, 0.3f);
        fill(B.fc2_b, vh, -0.1f, 0.1f);
    }
    // Relative-position bias tables (GOT).
    if (cfg.vision.rel_pos_window != 0u) {
        const uint32_t side = 2u * cfg.vision.rel_pos_window - 1u;
        w.rel_pos_bias.resize(cfg.vision.num_layers);
        for (auto &t : w.rel_pos_bias) {
            fill(t, static_cast<size_t>(cfg.vision.num_heads) * side * side, -0.2f, 0.2f);
        }
    }
    // Connector input dim.
    uint32_t cin = vh;
    if (cfg.connector.kind == P7ConnectorKind::spatial_mlp) {
        cin = vh * cfg.connector.spatial_factor * cfg.connector.spatial_factor;
    } else if (cfg.connector.kind == P7ConnectorKind::unshuffle_mlp) {
        cin = vh * 4u;
    } else if (cfg.connector.kind == P7ConnectorKind::downsample_mlp) {
        const uint32_t md = vh * cfg.connector.spatial_factor * cfg.connector.spatial_factor;
        fill(w.conn_down_w, static_cast<size_t>(md) * vh, -0.3f, 0.3f);
        fill(w.conn_down_b, vh, -0.1f, 0.1f);
        cin = vh;
    }
    fill(w.conn_ln_w, cin, 0.7f, 1.3f);
    if (!cfg.connector.rms_norm) { fill(w.conn_ln_b, cin, -0.1f, 0.1f); }
    fill(w.conn_fc1_w, static_cast<size_t>(cin) * ci, -0.3f, 0.3f);
    fill(w.conn_fc1_b, ci, -0.1f, 0.1f);
    fill(w.conn_fc2_w, static_cast<size_t>(ci) * th, -0.3f, 0.3f);
    fill(w.conn_fc2_b, th, -0.1f, 0.1f);
    // Text.
    fill(w.embed, static_cast<size_t>(cfg.text.vocab_size) * th, -0.6f, 0.6f);
    fill(w.final_norm, th, 0.7f, 1.3f);
    w.tied = cfg.tie_word_embeddings;
    // The standalone lm_head is the TRANSPOSED EMBEDDING TABLE ([K=hidden,
    // N=vocab]) -- the SAME effective weights the tied FP32 arm uses. The
    // quantized arm then quantizes exactly those weights, so the A/B delta
    // isolates quantization error only (no different random head, no LCG
    // stream divergence between tied/untied arms).
    {
        luisa::vector<float> head_kn(static_cast<size_t>(th) * cfg.text.vocab_size);
        for (uint32_t v = 0; v < cfg.text.vocab_size; v++) {
            for (uint32_t h = 0; h < th; h++) {
                head_kn[static_cast<size_t>(h) * cfg.text.vocab_size + v] = w.embed[static_cast<size_t>(v) * th + h];
            }
        }
        auto qs = p8_quantize_linear(head_kn, linear_storage, group_size, w.lm_head);
        LUISA_ASSERT(qs.ok(), "p8 lm_head quantize failed");
    }
    w.layers.resize(cfg.text.num_layers);
    const uint32_t qd = cfg.text.q_dim(), kvd = cfg.text.kv_dim();
    const bool qwen3_text = cfg.text_pos == P7TextPosKind::mrope_interleaved && !cfg.text_qwen2_5_contract;
    for (auto &L : w.layers) {
        fill(L.input_norm, th, 0.7f, 1.3f);
        fill(L.post_norm, th, 0.7f, 1.3f);
        fill_linear(L.q_w, static_cast<size_t>(th) * qd, -0.3f, 0.3f);
        fill_linear(L.k_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill_linear(L.v_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill_linear(L.o_w, static_cast<size_t>(qd) * th, -0.3f, 0.3f);
        if (!qwen3_text) {
            fill(L.q_b, qd, -0.1f, 0.1f);
            fill(L.k_b, kvd, -0.1f, 0.1f);
            fill(L.v_b, kvd, -0.1f, 0.1f);
        } else {
            fill(L.q_norm, cfg.text.head_dim, 0.7f, 1.3f);
            fill(L.k_norm, cfg.text.head_dim, 0.7f, 1.3f);
        }
        fill_linear(L.gate_w, static_cast<size_t>(th) * ti, -0.3f, 0.3f);
        fill_linear(L.up_w, static_cast<size_t>(th) * ti, -0.3f, 0.3f);
        fill_linear(L.down_w, static_cast<size_t>(ti) * th, -0.3f, 0.3f);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Package writing (vision FP32 + decoder with per-linear storage).
// ---------------------------------------------------------------------------

// Write one linear weight tensor: FP32 dense or packed INT4 with metadata.
[[nodiscard]] inline Status p8_add_linear(PackageWriter &wr, const luisa::string &name,
                                          luisa::vector<uint64_t> shape, const P8LinearStorage &w) {
    PackageWriter::TensorSpec spec;
    spec.name = name;
    spec.shape = std::move(shape);
    spec.layout = TensorLayout::kn;
    if (!w.is_quantized()) {
        spec.encoding = TensorEncoding::fp32;
        spec.payload = p7_f32_bytes(w.dense);
    } else {
        spec.encoding = TensorEncoding::int4_packed;
        spec.quant_kind = w.storage == DenseLinearStorage::int4_sym ? QuantKind::int4_symmetric : QuantKind::int4_asymmetric;
        spec.group_size = w.group_size;
        spec.scales = w.scales;
        spec.zero_points = w.zero_points;
        // quantize_int4 already produced packed bytes (low nibble first).
        spec.payload.resize(w.packed.size());
        if (!w.packed.empty()) { std::memcpy(spec.payload.data(), w.packed.data(), w.packed.size()); }
    }
    return wr.add_tensor(spec);
}

[[nodiscard]] inline Status write_p8_packages(const luisa::filesystem::path &vision_path,
                                              const luisa::filesystem::path &decoder_path,
                                              const P7SliceConfig &cfg, const P7TensorNames &names,
                                              const P8SyntheticWeights &w, luisa::string_view model) {
    const uint32_t vh = cfg.vision.hidden_size, vi = cfg.vision.intermediate_size;
    const uint32_t th = cfg.text.hidden_size, ti = cfg.text.intermediate_size;
    const uint32_t ci = cfg.connector.intermediate_size;
    const uint32_t patch_dim = cfg.vision.patch_dim();
    const uint32_t qd = cfg.text.q_dim(), kvd = cfg.text.kv_dim();
    const bool qwen3_text = cfg.text_pos == P7TextPosKind::mrope_interleaved && !cfg.text_qwen2_5_contract;
    {
        auto writer = PackageWriter::create(vision_path);
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
        if (auto s = add(names.patch_embed_weight(), {patch_dim, vh}, TensorLayout::kn, w.patch_embed_w); !s.ok()) { return s; }
        if (auto s = add(names.patch_embed_bias(), {vh}, TensorLayout::row_major, w.patch_embed_b); !s.ok()) { return s; }
        if (cfg.vision.position_rows != 0u) {
            if (auto s = add(names.position_embed(), {cfg.vision.position_rows, vh}, TensorLayout::row_major, w.position_embed); !s.ok()) { return s; }
        }
        if (cfg.vision.rel_pos_window != 0u) {
            const uint32_t side = 2u * cfg.vision.rel_pos_window - 1u;
            for (uint32_t l = 0; l < cfg.vision.num_layers; l++) {
                if (auto s = add(names.block_rel_pos_bias(l), {cfg.vision.num_heads, side, side}, TensorLayout::row_major, w.rel_pos_bias[l]); !s.ok()) { return s; }
            }
        }
        for (uint32_t l = 0; l < w.blocks.size(); l++) {
            const auto &B = w.blocks[l];
            if (auto s = add(names.block_norm1_weight(l), {vh}, TensorLayout::row_major, B.norm1_w); !s.ok()) { return s; }
            if (auto s = add(names.block_norm1_bias(l), {vh}, TensorLayout::row_major, B.norm1_b); !s.ok()) { return s; }
            if (auto s = add(names.block_norm2_weight(l), {vh}, TensorLayout::row_major, B.norm2_w); !s.ok()) { return s; }
            if (auto s = add(names.block_norm2_bias(l), {vh}, TensorLayout::row_major, B.norm2_b); !s.ok()) { return s; }
            if (auto s = add(names.block_q_proj(l), {vh, vh}, TensorLayout::kn, B.q_w); !s.ok()) { return s; }
            if (auto s = add(names.block_k_proj(l), {vh, vh}, TensorLayout::kn, B.k_w); !s.ok()) { return s; }
            if (auto s = add(names.block_v_proj(l), {vh, vh}, TensorLayout::kn, B.v_w); !s.ok()) { return s; }
            if (auto s = add(names.block_o_proj(l), {vh, vh}, TensorLayout::kn, B.o_w); !s.ok()) { return s; }
            if (cfg.vision.qkv_bias) {
                if (auto s = add(names.block_q_bias(l), {vh}, TensorLayout::row_major, B.q_b); !s.ok()) { return s; }
                if (auto s = add(names.block_k_bias(l), {vh}, TensorLayout::row_major, B.k_b); !s.ok()) { return s; }
                if (auto s = add(names.block_v_bias(l), {vh}, TensorLayout::row_major, B.v_b); !s.ok()) { return s; }
                if (auto s = add(names.block_o_bias(l), {vh}, TensorLayout::row_major, B.o_b); !s.ok()) { return s; }
            }
            if (cfg.vision.qk_norm) {
                if (auto s = add(names.block_q_norm(l), {cfg.vision.head_dim}, TensorLayout::row_major, B.q_norm); !s.ok()) { return s; }
                if (auto s = add(names.block_k_norm(l), {cfg.vision.head_dim}, TensorLayout::row_major, B.k_norm); !s.ok()) { return s; }
            }
            if (auto s = add(names.block_fc1(l), {vh, vi}, TensorLayout::kn, B.fc1_w); !s.ok()) { return s; }
            if (auto s = add(names.block_fc1_bias(l), {vi}, TensorLayout::row_major, B.fc1_b); !s.ok()) { return s; }
            if (auto s = add(names.block_fc2(l), {vi, vh}, TensorLayout::kn, B.fc2_w); !s.ok()) { return s; }
            if (auto s = add(names.block_fc2_bias(l), {vh}, TensorLayout::row_major, B.fc2_b); !s.ok()) { return s; }
        }
        uint32_t cin = vh;
        if (cfg.connector.kind == P7ConnectorKind::spatial_mlp) { cin = vh * cfg.connector.spatial_factor * cfg.connector.spatial_factor; }
        else if (cfg.connector.kind == P7ConnectorKind::unshuffle_mlp) { cin = vh * 4u; }
        else if (cfg.connector.kind == P7ConnectorKind::downsample_mlp) {
            const uint32_t md = vh * cfg.connector.spatial_factor * cfg.connector.spatial_factor;
            if (auto s = add(names.connector_down_weight(), {md, vh}, TensorLayout::kn, w.conn_down_w); !s.ok()) { return s; }
            if (auto s = add(names.connector_down_bias(), {vh}, TensorLayout::row_major, w.conn_down_b); !s.ok()) { return s; }
            cin = vh;
        }
        if (auto s = add(names.connector_ln_weight(), {cin}, TensorLayout::row_major, w.conn_ln_w); !s.ok()) { return s; }
        if (!cfg.connector.rms_norm) {
            if (auto s = add(names.connector_ln_bias(), {cin}, TensorLayout::row_major, w.conn_ln_b); !s.ok()) { return s; }
        }
        if (auto s = add(names.connector_fc1(), {cin, ci}, TensorLayout::kn, w.conn_fc1_w); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc1_bias(), {ci}, TensorLayout::row_major, w.conn_fc1_b); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc2(), {ci, th}, TensorLayout::kn, w.conn_fc2_w); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc2_bias(), {th}, TensorLayout::row_major, w.conn_fc2_b); !s.ok()) { return s; }
        if (auto s = wr->finish(p7_manifest("vision", model)); !s.ok()) { return s; }
    }
    {
        auto writer = PackageWriter::create(decoder_path);
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
        const auto &tn = names.text;
        if (auto s = add(tn.embed_tokens(), {cfg.text.vocab_size, th}, TensorLayout::row_major, w.embed); !s.ok()) { return s; }
        if (auto s = add(tn.final_norm(), {th}, TensorLayout::row_major, w.final_norm); !s.ok()) { return s; }
        if (!w.tied) {
            if (auto s = p8_add_linear(*wr, tn.lm_head(), {th, cfg.text.vocab_size}, w.lm_head); !s.ok()) { return s; }
        }
        for (uint32_t l = 0; l < w.layers.size(); l++) {
            const auto &L = w.layers[l];
            if (auto s = add(tn.input_layernorm(l), {th}, TensorLayout::row_major, L.input_norm); !s.ok()) { return s; }
            if (auto s = add(tn.post_attention_layernorm(l), {th}, TensorLayout::row_major, L.post_norm); !s.ok()) { return s; }
            if (auto s = p8_add_linear(*wr, tn.q_proj(l), {th, qd}, L.q_w); !s.ok()) { return s; }
            if (auto s = p8_add_linear(*wr, tn.k_proj(l), {th, kvd}, L.k_w); !s.ok()) { return s; }
            if (auto s = p8_add_linear(*wr, tn.v_proj(l), {th, kvd}, L.v_w); !s.ok()) { return s; }
            if (auto s = p8_add_linear(*wr, tn.o_proj(l), {qd, th}, L.o_w); !s.ok()) { return s; }
            if (!qwen3_text) {
                if (auto s = add(tn.q_bias(l), {qd}, TensorLayout::row_major, L.q_b); !s.ok()) { return s; }
                if (auto s = add(tn.k_bias(l), {kvd}, TensorLayout::row_major, L.k_b); !s.ok()) { return s; }
                if (auto s = add(tn.v_bias(l), {kvd}, TensorLayout::row_major, L.v_b); !s.ok()) { return s; }
            } else {
                if (auto s = add(tn.q_norm(l), {cfg.text.head_dim}, TensorLayout::row_major, L.q_norm); !s.ok()) { return s; }
                if (auto s = add(tn.k_norm(l), {cfg.text.head_dim}, TensorLayout::row_major, L.k_norm); !s.ok()) { return s; }
            }
            if (auto s = p8_add_linear(*wr, tn.gate_proj(l), {th, ti}, L.gate_w); !s.ok()) { return s; }
            if (auto s = p8_add_linear(*wr, tn.up_proj(l), {th, ti}, L.up_w); !s.ok()) { return s; }
            if (auto s = p8_add_linear(*wr, tn.down_proj(l), {ti, th}, L.down_w); !s.ok()) { return s; }
        }
        if (auto s = wr->finish(p7_manifest("decoder", model)); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Independent relative-position bias expansion oracle (GOT).
// ---------------------------------------------------------------------------
[[nodiscard]] inline luisa::vector<float> p8_oracle_expand_rel_bias(luisa::span<const float> table,
                                                                    uint32_t heads, uint32_t window,
                                                                    uint32_t grid_h, uint32_t grid_w) {
    const uint32_t rows = grid_h * grid_w;
    const uint32_t side = 2u * window - 1u;
    luisa::vector<float> out(static_cast<size_t>(heads) * rows * rows, 0.0f);
    for (uint32_t h = 0; h < heads; h++) {
        for (uint32_t q = 0; q < rows; q++) {
            const int32_t qh = static_cast<int32_t>(q / grid_w);
            const int32_t qw = static_cast<int32_t>(q % grid_w);
            for (uint32_t k = 0; k < rows; k++) {
                const int32_t kh = static_cast<int32_t>(k / grid_w);
                const int32_t kw = static_cast<int32_t>(k % grid_w);
                int32_t dy = qh - kh + static_cast<int32_t>(window) - 1;
                int32_t dx = qw - kw + static_cast<int32_t>(window) - 1;
                dy = std::clamp(dy, 0, static_cast<int32_t>(side) - 1);
                dx = std::clamp(dx, 0, static_cast<int32_t>(side) - 1);
                out[(static_cast<size_t>(h) * rows + q) * rows + k] =
                    table[(static_cast<size_t>(h) * side + static_cast<uint32_t>(dy)) * side + static_cast<uint32_t>(dx)];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Vision oracle with relative-position bias + RMSNorm connector (Phase 8).
// Independent of the device executor; uses only Phase 2 host references.
// ---------------------------------------------------------------------------
struct P8VisionOracle {
    luisa::vector<float> patches;
    luisa::vector<float> patch_embed;
    luisa::vector<luisa::vector<float>> block_out;
    luisa::vector<float> reduced;
    luisa::vector<float> features;
};

[[nodiscard]] inline Status p8_oracle_vision(const P7SliceConfig &cfg, const P8SyntheticWeights &w,
                                             luisa::span<const float> pixels_chw, const P7PackedImage &packed,
                                             P8VisionOracle &out) {
    const auto &v = cfg.vision;
    const uint32_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint32_t heads = v.num_heads, hd = v.head_dim;
    const uint32_t patch_dim = v.patch_dim();
    const uint32_t rows = static_cast<uint32_t>(packed.patch_count);
    out.patches = p7_oracle_patchify(pixels_chw, packed.resized_height, packed.resized_width, v.patch_size);
    auto s = ref_gemm(out.patches, w.patch_embed_w, w.patch_embed_b, rows, vh, patch_dim, RefGemmEpilogue::bias, out.patch_embed);
    if (!s.ok()) { return s; }
    luisa::vector<float> x = out.patch_embed;
    if (v.position_rows != 0u) {
        luisa::vector<int32_t> ids(rows);
        for (uint32_t t = 0; t < rows; t++) { ids[t] = static_cast<int32_t>(t); }
        luisa::vector<float> added;
        s = ref_add_position(x, w.position_embed, ids, rows, vh, added);
        if (!s.ok()) { return s; }
        x = std::move(added);
    }
    auto positions = p7_vision_positions(packed);
    auto axis = p7_vision_axis_of_pair(v);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const uint32_t num_blocks = static_cast<uint32_t>(w.blocks.size());
    out.block_out.resize(num_blocks);
    for (uint32_t l = 0; l < num_blocks; l++) {
        const auto &B = w.blocks[l];
        luisa::vector<float> norm, q, k, vv, qr, kr, attn, after_o, norm2, fc1;
        s = ref_layer_norm(x, B.norm1_w, B.norm1_b, rows, vh, v.layer_norm_eps, norm);
        if (!s.ok()) { return s; }
        const auto qkv_epi = v.qkv_bias ? RefGemmEpilogue::bias : RefGemmEpilogue::none;
        s = ref_gemm(norm, B.q_w, B.q_b, rows, vh, vh, qkv_epi, q);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.k_w, B.k_b, rows, vh, vh, qkv_epi, k);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.v_w, B.v_b, rows, vh, vh, qkv_epi, vv);
        if (!s.ok()) { return s; }
        auto q_for_rope = q;
        auto k_for_rope = k;
        if (v.qk_norm) {
            luisa::vector<float> qn, kn;
            s = ref_rms_norm(q, B.q_norm, rows * heads, hd, v.layer_norm_eps, qn);
            if (!s.ok()) { return s; }
            s = ref_rms_norm(k, B.k_norm, rows * heads, hd, v.layer_norm_eps, kn);
            if (!s.ok()) { return s; }
            q_for_rope = std::move(qn);
            k_for_rope = std::move(kn);
        }
        auto q_for_attn = q_for_rope;
        auto k_for_attn = k_for_rope;
        if (v.rope_kind == P7VisionRopeKind::axial2d) {
            s = ref_rope_axial(q_for_rope, positions, axis, rows, heads, hd, 2u, v.rope_theta, qr);
            if (!s.ok()) { return s; }
            s = ref_rope_axial(k_for_rope, positions, axis, rows, heads, hd, 2u, v.rope_theta, kr);
            if (!s.ok()) { return s; }
            q_for_attn = qr;
            k_for_attn = kr;
        }
        RefAttentionMask mask;
        mask.kind = RefMaskKind::bidirectional;
        if (v.mask_kind == P7VisionMaskKind::window_global) {
            const bool global = std::find(v.global_layers.begin(), v.global_layers.end(), l) != v.global_layers.end();
            mask = p7_oracle_window_mask(rows, packed.grid_w, v.window_patch_size, global);
        }
        // Relative-position bias (GOT): expand the per-layer table and add it.
        if (v.rel_pos_window != 0u) {
            mask.rel_bias = p8_oracle_expand_rel_bias(w.rel_pos_bias[l], heads, v.rel_pos_window, packed.grid_h, packed.grid_w);
        }
        s = ref_attention(q_for_attn, k_for_attn, vv, rows, rows, heads, heads, hd, scale, 0.0f, mask, attn);
        if (!s.ok()) { return s; }
        luisa::vector<float> o_proj;
        if (v.qkv_bias) {
            s = ref_gemm(attn, B.o_w, B.o_b, rows, vh, vh, RefGemmEpilogue::bias, o_proj);
        } else {
            s = ref_gemm(attn, B.o_w, {}, rows, vh, vh, RefGemmEpilogue::none, o_proj);
        }
        if (!s.ok()) { return s; }
        s = ref_residual_add(x, o_proj, after_o);
        if (!s.ok()) { return s; }
        s = ref_layer_norm(after_o, B.norm2_w, B.norm2_b, rows, vh, v.layer_norm_eps, norm2);
        if (!s.ok()) { return s; }
        const auto act_epi = v.mlp_act == 1u ? RefGemmEpilogue::bias_silu : RefGemmEpilogue::bias_gelu_erf;
        s = ref_gemm(norm2, B.fc1_w, B.fc1_b, rows, vi, vh, act_epi, fc1);
        if (!s.ok()) { return s; }
        luisa::vector<float> fc2_proj;
        s = ref_gemm(fc1, B.fc2_w, B.fc2_b, rows, vh, vi, RefGemmEpilogue::bias, fc2_proj);
        if (!s.ok()) { return s; }
        s = ref_residual_add(after_o, fc2_proj, x);
        if (!s.ok()) { return s; }
        out.block_out[l] = x;
    }
    // Connector reduction.
    const auto &c = cfg.connector;
    const uint32_t tokens = static_cast<uint32_t>(packed.visual_tokens);
    uint32_t cin = vh;
    if (c.kind == P7ConnectorKind::spatial_mlp) {
        cin = vh * c.spatial_factor * c.spatial_factor;
        out.reduced = p7_oracle_spatial_merge(x, rows, vh, packed.grid_w, c.spatial_factor);
    } else if (c.kind == P7ConnectorKind::unshuffle_mlp) {
        cin = vh * 4u;
        out.reduced = p7_oracle_unshuffle_tokens(x, packed.grid_h, packed.grid_w, vh, 2u);
    } else {
        cin = vh;
        out.reduced = x;
    }
    // Normalization: RMSNorm (Qwen2.5-VL merger) or LayerNorm.
    luisa::vector<float> cnorm, cfc1;
    if (c.rms_norm) {
        s = ref_rms_norm(out.reduced, w.conn_ln_w, tokens, cin, c.layer_norm_eps, cnorm);
    } else {
        s = ref_layer_norm(out.reduced, w.conn_ln_w, w.conn_ln_b, tokens, cin, c.layer_norm_eps, cnorm);
    }
    if (!s.ok()) { return s; }
    s = ref_gemm(cnorm, w.conn_fc1_w, w.conn_fc1_b, tokens, c.intermediate_size, cin, RefGemmEpilogue::bias_gelu_erf, cfc1);
    if (!s.ok()) { return s; }
    return ref_gemm(cfc1, w.conn_fc2_w, w.conn_fc2_b, tokens, cfg.text.hidden_size, c.intermediate_size, RefGemmEpilogue::bias, out.features);
}

// ---------------------------------------------------------------------------
// Text decoder oracle with per-linear storage (Phase 8). The B operand for each
// linear is the dequantized weight (quantized arm) or the dense weight (FP32).
// ---------------------------------------------------------------------------
struct P8TextOracle {
    luisa::vector<float> embeddings;
    luisa::vector<luisa::vector<float>> layer_out;
    luisa::vector<float> final_norm;
    luisa::vector<float> logits;
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
};

[[nodiscard]] inline Status p8_oracle_text_block(const P7SliceConfig &cfg, const P8SyntheticWeights &w,
                                                 uint32_t layer, luisa::span<const float> x, uint32_t rows,
                                                 uint32_t pos0, luisa::span<const int32_t> position_ids,
                                                 luisa::vector<float> &k_cache, luisa::vector<float> &v_cache,
                                                 luisa::vector<float> &x_out) {
    const auto &t = cfg.text;
    const auto &L = w.layers[layer];
    const uint32_t th = t.hidden_size, ti = t.intermediate_size;
    const uint32_t qh = t.num_attention_heads, kvh = t.num_kv_heads, hd = t.head_dim;
    const uint32_t qd = t.q_dim(), kvd = t.kv_dim();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const uint32_t axes = cfg.text_rope_axes();
    const bool qwen3 = cfg.text_pos == P7TextPosKind::mrope_interleaved && !cfg.text_qwen2_5_contract;
    auto axis = cfg.text_axis_of_pair();
    luisa::vector<float> norm, q, k, v, qr, kr, attn, after_o, norm2, gate, up, mlp;
    auto s = ref_rms_norm(x, L.input_norm, rows, th, t.rms_norm_eps, norm);
    if (!s.ok()) { return s; }
    const auto qkv_epi = qwen3 ? RefGemmEpilogue::none : RefGemmEpilogue::bias;
    s = ref_gemm(norm, p8_oracle_b(L.q_w), L.q_b, rows, qd, th, qkv_epi, q);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, p8_oracle_b(L.k_w), L.k_b, rows, kvd, th, qkv_epi, k);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, p8_oracle_b(L.v_w), L.v_b, rows, kvd, th, qkv_epi, v);
    if (!s.ok()) { return s; }
    auto q_for_rope = q;
    auto k_for_rope = k;
    if (qwen3) {
        luisa::vector<float> qn, kn;
        s = ref_rms_norm(q, L.q_norm, rows * qh, hd, t.rms_norm_eps, qn);
        if (!s.ok()) { return s; }
        s = ref_rms_norm(k, L.k_norm, rows * kvh, hd, t.rms_norm_eps, kn);
        if (!s.ok()) { return s; }
        q_for_rope = std::move(qn);
        k_for_rope = std::move(kn);
    }
    luisa::vector<int32_t> pos(position_ids.begin(), position_ids.begin() + static_cast<size_t>(rows) * axes);
    if (t.rope_pair_layout == RopePairLayout::rotate_half) {
        s = ref_rope_interleaved(q_for_rope, pos, axis, rows, qh, hd, axes, t.rope_theta, qr);
        if (!s.ok()) { return s; }
        s = ref_rope_interleaved(k_for_rope, pos, axis, rows, kvh, hd, axes, t.rope_theta, kr);
        if (!s.ok()) { return s; }
    } else {
        s = ref_rope_axial(q_for_rope, pos, axis, rows, qh, hd, axes, t.rope_theta, qr);
        if (!s.ok()) { return s; }
        s = ref_rope_axial(k_for_rope, pos, axis, rows, kvh, hd, axes, t.rope_theta, kr);
        if (!s.ok()) { return s; }
    }
    s = ref_kv_append(kr, v, k_cache, v_cache, rows, kvd, pos0, pos0 + rows);
    if (!s.ok()) { return s; }
    RefAttentionMask mask;
    mask.kind = RefMaskKind::causal;
    const uint32_t k_len = pos0 + rows;
    s = ref_attention(qr, k_cache, v_cache, rows, k_len, qh, kvh, hd, scale, 0.0f, mask, attn);
    if (!s.ok()) { return s; }
    s = ref_gemm(attn, p8_oracle_b(L.o_w), x, rows, th, qd, RefGemmEpilogue::residual, after_o);
    if (!s.ok()) { return s; }
    s = ref_rms_norm(after_o, L.post_norm, rows, th, t.rms_norm_eps, norm2);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm2, p8_oracle_b(L.gate_w), {}, rows, ti, th, RefGemmEpilogue::none, gate);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm2, p8_oracle_b(L.up_w), {}, rows, ti, th, RefGemmEpilogue::none, up);
    if (!s.ok()) { return s; }
    s = ref_gated_act(gate, up, 0u, mlp);
    if (!s.ok()) { return s; }
    return ref_gemm(mlp, p8_oracle_b(L.down_w), after_o, rows, th, ti, RefGemmEpilogue::residual, x_out);
}

[[nodiscard]] inline Status p8_oracle_text_forward(const P7SliceConfig &cfg, const P8SyntheticWeights &w,
                                                   luisa::span<const uint32_t> tokens,
                                                   luisa::span<const int32_t> position_ids,
                                                   luisa::span<const float> mm_features,
                                                   uint32_t mm_start, uint32_t mm_count, uint32_t kv_prefix,
                                                   luisa::vector<luisa::vector<float>> &k_caches,
                                                   luisa::vector<luisa::vector<float>> &v_caches,
                                                   P8TextOracle &result) {
    const auto &t = cfg.text;
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    const uint32_t th = t.hidden_size, kvd = t.kv_dim();
    luisa::vector<int32_t> ids(tokens.begin(), tokens.end());
    auto s = ref_embedding_lookup(w.embed, ids, t.vocab_size, th, result.embeddings);
    if (!s.ok()) { return s; }
    if (!mm_features.empty()) {
        luisa::vector<float> replaced;
        s = ref_mm_scatter(result.embeddings, mm_features, rows, th, mm_start, replaced);
        if (!s.ok()) { return s; }
        result.embeddings = std::move(replaced);
    }
    const uint32_t num_layers = static_cast<uint32_t>(w.layers.size());
    if (k_caches.empty()) {
        k_caches.resize(num_layers);
        v_caches.resize(num_layers);
    }
    const uint32_t total = kv_prefix + rows;
    for (uint32_t l = 0; l < num_layers; l++) {
        k_caches[l].resize(static_cast<size_t>(total) * kvd);
        v_caches[l].resize(static_cast<size_t>(total) * kvd);
    }
    result.layer_out.resize(num_layers);
    luisa::vector<float> x = result.embeddings;
    for (uint32_t l = 0; l < num_layers; l++) {
        luisa::vector<float> x_out;
        s = p8_oracle_text_block(cfg, w, l, x, rows, kv_prefix, position_ids, k_caches[l], v_caches[l], x_out);
        if (!s.ok()) { return s; }
        x = std::move(x_out);
        result.layer_out[l] = x;
    }
    s = ref_rms_norm(x, w.final_norm, rows, th, t.rms_norm_eps, result.final_norm);
    if (!s.ok()) { return s; }
    // LM head: standalone (possibly quantized) or tied (FP32 embedding).
    luisa::vector<float> head_t(static_cast<size_t>(th) * t.vocab_size);
    if (w.tied) {
        for (uint32_t v = 0; v < t.vocab_size; v++) {
            for (uint32_t h = 0; h < th; h++) {
                head_t[static_cast<size_t>(h) * t.vocab_size + v] = w.embed[static_cast<size_t>(v) * th + h];
            }
        }
    } else {
        // The standalone lm_head is stored packed [K=hidden, N=vocab].
        const auto b = p8_oracle_b(w.lm_head);
        std::copy(b.begin(), b.end(), head_t.begin());
    }
    s = ref_gemm(result.final_norm, head_t, {}, rows, t.vocab_size, th, RefGemmEpilogue::none, result.logits);
    if (!s.ok()) { return s; }
    result.k_cache = k_caches;
    result.v_cache = v_caches;
    return Status::success();
}

}// namespace luisa::ocr::test

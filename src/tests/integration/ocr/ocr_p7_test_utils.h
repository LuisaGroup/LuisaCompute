#pragma once
// Shared synthetic Phase 7 fixtures + INDEPENDENT CPU oracles for the TeleOCR /
// GLM-OCR / MinerU / HunyuanOCR device tests. Synthetic only: deterministic
// weights, the declared small geometries, no model download, no real-model
// parity claim.
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

#include "ocr/ops/cpu_reference.h"
#include "ocr/package.h"
#include "ocr/runtime/p7_slice_config.h"
#include "ocr/runtime/p7_vlm_executor.h"
#include "ocr/runtime/weights.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Temp dir + package helpers
// ---------------------------------------------------------------------------
struct P7TempDir {
    luisa::filesystem::path path;
    P7TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_p7_test_" + luisa::format("{}", reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~P7TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] inline luisa::vector<std::byte> p7_f32_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> out(v.size() * 4);
    if (!v.empty()) { std::memcpy(out.data(), v.data(), out.size()); }
    return out;
}

[[nodiscard]] inline PackageManifest p7_manifest(luisa::string_view component, luisa::string_view model) {
    PackageManifest m;
    m.model = luisa::string{model};
    m.recipe_version = "1";
    m.checkpoint_revision = "synthetic-rev";
    m.config_fingerprint = "synthetic-fp";
    m.license = "test";
    m.producer = "ocr_p7_test";
    m.producer_version = "1";
    m.preprocessing_version = "1";
    m.tokenizer_version = "1";
    m.output_parser_version = "1";
    m.component = luisa::string{component};
    return m;
}

// ---------------------------------------------------------------------------
// Synthetic weights for the shared Phase 7 graph (config-driven).
// ---------------------------------------------------------------------------
struct P7VisionBlock {
    luisa::vector<float> norm1_w, norm1_b, norm2_w, norm2_b;
    luisa::vector<float> q_w, k_w, v_w, o_w;
    luisa::vector<float> q_b, k_b, v_b, o_b;
    luisa::vector<float> q_norm, k_norm;
    luisa::vector<float> fc1_w, fc1_b, fc2_w, fc2_b;
};
struct P7SyntheticWeights {
    luisa::vector<float> patch_embed_w, patch_embed_b;
    luisa::vector<float> position_embed;// [position_rows, vh] (empty when none)
    luisa::vector<P7VisionBlock> blocks;
    luisa::vector<float> conn_ln_w, conn_ln_b, conn_fc1_w, conn_fc1_b, conn_fc2_w, conn_fc2_b;
    luisa::vector<float> conn_down_w, conn_down_b;// downsample_mlp only
    // Text decoder.
    luisa::vector<float> embed, final_norm, lm_head;
    bool tied{true};
    struct TextLayer {
        luisa::vector<float> input_norm, post_norm;
        luisa::vector<float> q_w, k_w, v_w, o_w;
        luisa::vector<float> q_b, k_b, v_b;
        luisa::vector<float> q_norm, k_norm;
        luisa::vector<float> gate_w, up_w, down_w;
    };
    luisa::vector<TextLayer> layers;
};

[[nodiscard]] inline P7SyntheticWeights generate_p7_weights(const P7SliceConfig &cfg, uint32_t seed = 0x5EEDu) {
    P7SyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
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
    fill(w.conn_ln_b, cin, -0.1f, 0.1f);
    fill(w.conn_fc1_w, static_cast<size_t>(cin) * ci, -0.3f, 0.3f);
    fill(w.conn_fc1_b, ci, -0.1f, 0.1f);
    fill(w.conn_fc2_w, static_cast<size_t>(ci) * th, -0.3f, 0.3f);
    fill(w.conn_fc2_b, th, -0.1f, 0.1f);
    // Text.
    fill(w.embed, static_cast<size_t>(cfg.text.vocab_size) * th, -0.6f, 0.6f);
    w.lm_head = w.embed;
    fill(w.final_norm, th, 0.7f, 1.3f);
    w.layers.resize(cfg.text.num_layers);
    const uint32_t qd = cfg.text.q_dim(), kvd = cfg.text.kv_dim();
    const bool qwen3_text = cfg.text_pos == P7TextPosKind::mrope_interleaved;
    for (auto &L : w.layers) {
        fill(L.input_norm, th, 0.7f, 1.3f);
        fill(L.post_norm, th, 0.7f, 1.3f);
        fill(L.q_w, static_cast<size_t>(th) * qd, -0.3f, 0.3f);
        fill(L.k_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill(L.v_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill(L.o_w, static_cast<size_t>(qd) * th, -0.3f, 0.3f);
        if (!qwen3_text) {
            fill(L.q_b, qd, -0.1f, 0.1f);
            fill(L.k_b, kvd, -0.1f, 0.1f);
            fill(L.v_b, kvd, -0.1f, 0.1f);
        } else {
            fill(L.q_norm, cfg.text.head_dim, 0.7f, 1.3f);
            fill(L.k_norm, cfg.text.head_dim, 0.7f, 1.3f);
        }
        fill(L.gate_w, static_cast<size_t>(th) * ti, -0.3f, 0.3f);
        fill(L.up_w, static_cast<size_t>(th) * ti, -0.3f, 0.3f);
        fill(L.down_w, static_cast<size_t>(ti) * th, -0.3f, 0.3f);
    }
    return w;
}

// Write the vision + decoder packages under the given tensor-name map.
[[nodiscard]] inline Status write_p7_packages(const luisa::filesystem::path &vision_path,
                                              const luisa::filesystem::path &decoder_path,
                                              const P7SliceConfig &cfg, const P7TensorNames &names,
                                              const P7SyntheticWeights &w, luisa::string_view model) {
    const uint32_t vh = cfg.vision.hidden_size, vi = cfg.vision.intermediate_size;
    const uint32_t th = cfg.text.hidden_size, ti = cfg.text.intermediate_size;
    const uint32_t ci = cfg.connector.intermediate_size;
    const uint32_t patch_dim = cfg.vision.patch_dim();
    const uint32_t qd = cfg.text.q_dim(), kvd = cfg.text.kv_dim();
    const bool qwen3_text = cfg.text_pos == P7TextPosKind::mrope_interleaved;
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
        if (auto s = add(names.connector_ln_bias(), {cin}, TensorLayout::row_major, w.conn_ln_b); !s.ok()) { return s; }
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
            luisa::vector<float> head_t(static_cast<size_t>(th) * cfg.text.vocab_size);
            for (uint32_t v = 0; v < cfg.text.vocab_size; v++) {
                for (uint32_t h = 0; h < th; h++) {
                    head_t[static_cast<size_t>(h) * cfg.text.vocab_size + v] = w.lm_head[static_cast<size_t>(v) * th + h];
                }
            }
            if (auto s = add(tn.lm_head(), {th, cfg.text.vocab_size}, TensorLayout::kn, head_t); !s.ok()) { return s; }
        }
        for (uint32_t l = 0; l < w.layers.size(); l++) {
            const auto &L = w.layers[l];
            if (auto s = add(tn.input_layernorm(l), {th}, TensorLayout::row_major, L.input_norm); !s.ok()) { return s; }
            if (auto s = add(tn.post_attention_layernorm(l), {th}, TensorLayout::row_major, L.post_norm); !s.ok()) { return s; }
            if (auto s = add(tn.q_proj(l), {th, qd}, TensorLayout::kn, L.q_w); !s.ok()) { return s; }
            if (auto s = add(tn.k_proj(l), {th, kvd}, TensorLayout::kn, L.k_w); !s.ok()) { return s; }
            if (auto s = add(tn.v_proj(l), {th, kvd}, TensorLayout::kn, L.v_w); !s.ok()) { return s; }
            if (auto s = add(tn.o_proj(l), {qd, th}, TensorLayout::kn, L.o_w); !s.ok()) { return s; }
            if (!qwen3_text) {
                if (auto s = add(tn.q_bias(l), {qd}, TensorLayout::row_major, L.q_b); !s.ok()) { return s; }
                if (auto s = add(tn.k_bias(l), {kvd}, TensorLayout::row_major, L.k_b); !s.ok()) { return s; }
                if (auto s = add(tn.v_bias(l), {kvd}, TensorLayout::row_major, L.v_b); !s.ok()) { return s; }
            } else {
                if (auto s = add(tn.q_norm(l), {cfg.text.head_dim}, TensorLayout::row_major, L.q_norm); !s.ok()) { return s; }
                if (auto s = add(tn.k_norm(l), {cfg.text.head_dim}, TensorLayout::row_major, L.k_norm); !s.ok()) { return s; }
            }
            if (auto s = add(tn.gate_proj(l), {th, ti}, TensorLayout::kn, L.gate_w); !s.ok()) { return s; }
            if (auto s = add(tn.up_proj(l), {th, ti}, TensorLayout::kn, L.up_w); !s.ok()) { return s; }
            if (auto s = add(tn.down_proj(l), {ti, th}, TensorLayout::kn, L.down_w); !s.ok()) { return s; }
        }
        if (auto s = wr->finish(p7_manifest("decoder", model)); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] inline Status upload_p7_weights(compute::Device &device, compute::Stream &stream,
                                              const luisa::filesystem::path &vision_path,
                                              const luisa::filesystem::path &decoder_path,
                                              WeightStore &store) {
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
// Independent preprocessing mirror (float bilinear, half-pixel centers).
// ---------------------------------------------------------------------------
[[nodiscard]] inline luisa::vector<float> p7_oracle_preprocess(luisa::span<const uint8_t> pixels,
                                                               uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh,
                                                               luisa::span<const float> mean,
                                                               luisa::span<const float> std_dev) {
    luisa::vector<float> out(static_cast<size_t>(3) * dw * dh);
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    for (uint32_t y = 0; y < dh; y++) {
        for (uint32_t x = 0; x < dw; x++) {
            const float sx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const float sy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
            const float x0f = std::floor(sx), y0f = std::floor(sy);
            const int32_t x0 = static_cast<int32_t>(x0f), y0 = static_cast<int32_t>(y0f);
            const float fx = sx - x0f, fy = sy - y0f;
            for (uint32_t c = 0; c < 3u; c++) {
                const auto at = [&](int32_t px, int32_t py) -> float {
                    const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(sw) - 1);
                    const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(sh) - 1);
                    return static_cast<float>(pixels[(static_cast<size_t>(cy) * sw + cx) * 3u + c]);
                };
                const float v00 = at(x0, y0), v10 = at(x0 + 1, y0);
                const float v01 = at(x0, y0 + 1), v11 = at(x0 + 1, y0 + 1);
                const float top = v00 + (v10 - v00) * fx;
                const float bottom = v01 + (v11 - v01) * fy;
                const float v = (top + (bottom - top) * fy) / 255.0f;
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] =
                    (v - mean[c]) / std_dev[c];
            }
        }
    }
    return out;
}

// Patchify CHW -> [rows, patch_dim] in (c, py, px) order.
[[nodiscard]] inline luisa::vector<float> p7_oracle_patchify(luisa::span<const float> pixels_chw,
                                                             uint32_t H, uint32_t W, uint32_t patch) {
    const uint32_t gh = H / patch, gw = W / patch;
    const uint32_t rows = gh * gw;
    const uint32_t pd = 3 * patch * patch;
    luisa::vector<float> out(static_cast<size_t>(rows) * pd);
    for (uint32_t ph = 0; ph < gh; ph++) {
        for (uint32_t pw = 0; pw < gw; pw++) {
            const uint32_t t = ph * gw + pw;
            for (uint32_t c = 0; c < 3u; c++) {
                for (uint32_t py = 0; py < patch; py++) {
                    for (uint32_t px = 0; px < patch; px++) {
                        const uint32_t y = ph * patch + py;
                        const uint32_t x = pw * patch + px;
                        const uint32_t d = (c * patch + py) * patch + px;
                        out[static_cast<size_t>(t) * pd + d] =
                            pixels_chw[(static_cast<size_t>(c) * H + y) * W + x];
                    }
                }
            }
        }
    }
    return out;
}

// Spatial merge oracle: [T, H] -> [T/(m*m), H*m*m], row-major m x m blocks.
[[nodiscard]] inline luisa::vector<float> p7_oracle_spatial_merge(luisa::span<const float> x,
                                                                  uint32_t T, uint32_t H,
                                                                  uint32_t tokens_w, uint32_t m) {
    const uint32_t mm = m * m;
    const uint32_t out_tokens = T / mm;
    const uint32_t out_hidden = H * mm;
    const uint32_t blocks_w = tokens_w / m;
    luisa::vector<float> out(static_cast<size_t>(out_tokens) * out_hidden);
    for (uint32_t ot = 0; ot < out_tokens; ot++) {
        const uint32_t bx = ot % blocks_w;
        const uint32_t by = ot / blocks_w;
        for (uint32_t sub = 0; sub < mm; sub++) {
            const uint32_t sy = sub / m, sx = sub % m;
            const uint32_t src_token = (by * m + sy) * tokens_w + (bx * m + sx);
            for (uint32_t d = 0; d < H; d++) {
                out[static_cast<size_t>(ot) * out_hidden + sub * H + d] =
                    x[static_cast<size_t>(src_token) * H + d];
            }
        }
    }
    return out;
}

// Pixel-unshuffle oracle producing TOKEN rows [tokens, C*r*r] directly. The
// device transposes the token-major vision output to channel-major NCHW,
// runs the Phase 2 pixel_unshuffle (output channel index ci*r*r + ry*r + rx),
// then transposes back to token rows; the linear channel index is preserved,
// so the token channel order is (ci, ry, rx) -- matching ref_pixel_unshuffle.
// This is the declared MinerU synthetic contract (the official channel
// ordering is an unmet gate).
[[nodiscard]] inline luisa::vector<float> p7_oracle_unshuffle_tokens(luisa::span<const float> x_tokens,
                                                                     uint32_t gh, uint32_t gw, uint32_t C, uint32_t r) {
    const uint32_t oh = gh / r, ow = gw / r;
    const uint32_t tokens = oh * ow;
    const uint32_t cin = C * r * r;
    luisa::vector<float> out(static_cast<size_t>(tokens) * cin);
    for (uint32_t oy = 0; oy < oh; oy++) {
        for (uint32_t ox = 0; ox < ow; ox++) {
            const uint32_t ot = oy * ow + ox;
            for (uint32_t ci = 0; ci < C; ci++) {
                for (uint32_t ry = 0; ry < r; ry++) {
                    for (uint32_t rx = 0; rx < r; rx++) {
                        const uint32_t y = oy * r + ry, xx = ox * r + rx;
                        const uint32_t src_token = y * gw + xx;
                        const uint32_t oc = ci * r * r + ry * r + rx;
                        out[static_cast<size_t>(ot) * cin + oc] =
                            x_tokens[static_cast<size_t>(src_token) * C + ci];
                    }
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Vision oracle (config-driven; window/global mask, QK-Norm, 2D RoPE, MLP act).
// ---------------------------------------------------------------------------
struct P7VisionOracle {
    luisa::vector<float> patches;
    luisa::vector<float> patch_embed;
    luisa::vector<luisa::vector<float>> block_out;
    luisa::vector<float> reduced;// connector reduction output [tokens, cin]
    luisa::vector<float> features;// [tokens, th]
};

// Build the window/global mask as a RefAttentionMask (custom_block) matching
// the device convention: global = single all-ones block; window = block_size 1
// per-patch same-tile.
[[nodiscard]] inline RefAttentionMask p7_oracle_window_mask(uint32_t rows, uint32_t grid_w,
                                                            uint32_t window, bool global) {
    RefAttentionMask mask;
    mask.kind = RefMaskKind::custom_block;
    if (global) {
        mask.block_size = rows == 0u ? 1u : rows;
        mask.block_mask.assign(1u, 1u);
        return mask;
    }
    mask.block_size = 1u;
    const uint32_t tiles_w = grid_w == 0u ? 1u : (grid_w + window - 1u) / window;
    luisa::vector<uint32_t> tile_of(rows);
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t ph = grid_w == 0u ? 0u : t / grid_w;
        const uint32_t pw = grid_w == 0u ? 0u : t % grid_w;
        tile_of[t] = (ph / window) * tiles_w + (pw / window);
    }
    mask.block_mask.resize(static_cast<size_t>(rows) * rows);
    for (uint32_t q = 0; q < rows; q++) {
        for (uint32_t k = 0; k < rows; k++) {
            mask.block_mask[static_cast<size_t>(q) * rows + k] = (tile_of[q] == tile_of[k]) ? 1u : 0u;
        }
    }
    return mask;
}

[[nodiscard]] inline Status p7_oracle_vision(const P7SliceConfig &cfg, const P7SyntheticWeights &w,
                                             luisa::span<const float> pixels_chw, const P7PackedImage &packed,
                                             P7VisionOracle &out) {
    const auto &v = cfg.vision;
    const uint32_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint32_t heads = v.num_heads, hd = v.head_dim;
    const uint32_t patch_dim = v.patch_dim();
    const uint32_t rows = static_cast<uint32_t>(packed.patch_count);
    out.patches = p7_oracle_patchify(pixels_chw, packed.resized_height, packed.resized_width, v.patch_size);
    auto s = ref_gemm(out.patches, w.patch_embed_w, w.patch_embed_b, rows, vh, patch_dim, RefGemmEpilogue::bias, out.patch_embed);
    if (!s.ok()) { return s; }
    luisa::vector<float> x = out.patch_embed;
    // Learned absolute position.
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
        s = ref_attention(q_for_attn, k_for_attn, vv, rows, rows, heads, heads, hd, scale, 0.0f, mask, attn);
        if (!s.ok()) { return s; }
        // o-proj (+bias) + residual.
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
    } else if (c.kind == P7ConnectorKind::adaptive_mlp) {
        cin = vh;
        // True per-channel spatial adaptive pool over the TOKEN-major vision
        // output [patch_rows, vh] (patch t = (ph, pw), channel ci). This is an
        // independent direct computation -- NOT ref_adaptive_avg_pool2d on the
        // token-major buffer (which would pool across the wrong axis). Bin
        // edges follow PyTorch: start = floor(i*in/out), end = ceil((i+1)*in/out).
        const uint32_t gh = packed.grid_h, gw = packed.grid_w;
        const uint32_t oh = packed.out_h, ow = packed.out_w;
        out.reduced.assign(static_cast<size_t>(tokens) * vh, 0.0f);
        for (uint32_t oy = 0; oy < oh; oy++) {
            const uint32_t y0 = static_cast<uint32_t>(static_cast<uint64_t>(oy) * gh / oh);
            const uint32_t y1 = static_cast<uint32_t>((static_cast<uint64_t>(oy + 1) * gh + oh - 1u) / oh);
            for (uint32_t ox = 0; ox < ow; ox++) {
                const uint32_t x0 = static_cast<uint32_t>(static_cast<uint64_t>(ox) * gw / ow);
                const uint32_t x1 = static_cast<uint32_t>((static_cast<uint64_t>(ox + 1) * gw + ow - 1u) / ow);
                const uint32_t ot = oy * ow + ox;
                for (uint32_t ci = 0; ci < vh; ci++) {
                    double sum = 0.0;
                    uint32_t cnt = 0;
                    for (uint32_t yy = y0; yy < y1; yy++) {
                        for (uint32_t xx = x0; xx < x1; xx++) {
                            sum += x[static_cast<size_t>(yy * gw + xx) * vh + ci];
                            cnt++;
                        }
                    }
                    out.reduced[static_cast<size_t>(ot) * vh + ci] = cnt > 0u ? static_cast<float>(sum / cnt) : 0.0f;
                }
            }
        }
    } else if (c.kind == P7ConnectorKind::downsample_mlp) {
        const uint32_t md = vh * c.spatial_factor * c.spatial_factor;
        auto merged = p7_oracle_spatial_merge(x, rows, vh, packed.grid_w, c.spatial_factor);
        cin = vh;
        s = ref_gemm(merged, w.conn_down_w, w.conn_down_b, tokens, vh, md, RefGemmEpilogue::bias, out.reduced);
        if (!s.ok()) { return s; }
    } else {
        cin = vh;
        out.reduced = x;
    }
    // LN + 2-layer GELU MLP.
    luisa::vector<float> cnorm, cfc1;
    s = ref_layer_norm(out.reduced, w.conn_ln_w, w.conn_ln_b, tokens, cin, c.layer_norm_eps, cnorm);
    if (!s.ok()) { return s; }
    s = ref_gemm(cnorm, w.conn_fc1_w, w.conn_fc1_b, tokens, c.intermediate_size, cin, RefGemmEpilogue::bias_gelu_erf, cfc1);
    if (!s.ok()) { return s; }
    return ref_gemm(cfc1, w.conn_fc2_w, w.conn_fc2_b, tokens, cfg.text.hidden_size, c.intermediate_size, RefGemmEpilogue::bias, out.features);
}

// ---------------------------------------------------------------------------
// Text decoder oracle (config-driven: QKV bias / QK-Norm / 1D / MRoPE / XD /
// partial rotary). Independent of the device executor.
// ---------------------------------------------------------------------------
struct P7TextOracle {
    luisa::vector<float> embeddings;
    luisa::vector<luisa::vector<float>> layer_out;
    luisa::vector<float> final_norm;
    luisa::vector<float> logits;// [rows, vocab]
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
};

[[nodiscard]] inline Status p7_oracle_text_block(const P7SliceConfig &cfg, const P7SyntheticWeights &w,
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
    const bool qwen3 = cfg.text_pos == P7TextPosKind::mrope_interleaved;
    auto axis = cfg.text_axis_of_pair();
    luisa::vector<float> norm, q, k, v, qr, kr, attn, after_o, norm2, gate, up, mlp;
    auto s = ref_rms_norm(x, L.input_norm, rows, th, t.rms_norm_eps, norm);
    if (!s.ok()) { return s; }
    const auto qkv_epi = qwen3 ? RefGemmEpilogue::none : RefGemmEpilogue::bias;
    s = ref_gemm(norm, L.q_w, L.q_b, rows, qd, th, qkv_epi, q);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.k_w, L.k_b, rows, kvd, th, qkv_epi, k);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.v_w, L.v_b, rows, kvd, th, qkv_epi, v);
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
    // RoPE: positions for the NEW rows only (position_ids covers exactly rows).
    luisa::vector<int32_t> pos(position_ids.begin(), position_ids.begin() + static_cast<size_t>(rows) * axes);
    const bool partial = cfg.partial_rotary_num != cfg.partial_rotary_den;
    const uint32_t rotary_dim = partial
                                    ? static_cast<uint32_t>((static_cast<uint64_t>(hd) * cfg.partial_rotary_num) / cfg.partial_rotary_den)
                                    : hd;
    if (partial) {
        // axis_of_pair truncated to rotary_dim/2.
        luisa::vector<uint32_t> paxis(axis.begin(), axis.begin() + rotary_dim / 2u);
        if (t.rope_pair_layout == RopePairLayout::rotate_half) {
            s = ref_rope_partial_rotate_half(q_for_rope, pos, paxis, rows, qh, hd, rotary_dim, axes, t.rope_theta, qr);
            if (!s.ok()) { return s; }
            s = ref_rope_partial_rotate_half(k_for_rope, pos, paxis, rows, kvh, hd, rotary_dim, axes, t.rope_theta, kr);
            if (!s.ok()) { return s; }
        } else {
            s = ref_rope_partial_adjacent(q_for_rope, pos, paxis, rows, qh, hd, rotary_dim, axes, t.rope_theta, qr);
            if (!s.ok()) { return s; }
            s = ref_rope_partial_adjacent(k_for_rope, pos, paxis, rows, kvh, hd, rotary_dim, axes, t.rope_theta, kr);
            if (!s.ok()) { return s; }
        }
    } else if (t.rope_pair_layout == RopePairLayout::rotate_half) {
        // Interleaved (Qwen3 MRoPE) or 1-axis rotate-half.
        s = ref_rope_interleaved(q_for_rope, pos, axis, rows, qh, hd, axes, t.rope_theta, qr);
        if (!s.ok()) { return s; }
        s = ref_rope_interleaved(k_for_rope, pos, axis, rows, kvh, hd, axes, t.rope_theta, kr);
        if (!s.ok()) { return s;
        }
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
    s = ref_gemm(attn, L.o_w, x, rows, th, qd, RefGemmEpilogue::residual, after_o);
    if (!s.ok()) { return s; }
    s = ref_rms_norm(after_o, L.post_norm, rows, th, t.rms_norm_eps, norm2);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm2, L.gate_w, {}, rows, ti, th, RefGemmEpilogue::none, gate);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm2, L.up_w, {}, rows, ti, th, RefGemmEpilogue::none, up);
    if (!s.ok()) { return s; }
    s = ref_gated_act(gate, up, 0u, mlp);
    if (!s.ok()) { return s; }
    return ref_gemm(mlp, L.down_w, after_o, rows, th, ti, RefGemmEpilogue::residual, x_out);
}

[[nodiscard]] inline Status p7_oracle_text_forward(const P7SliceConfig &cfg, const P7SyntheticWeights &w,
                                                   luisa::span<const uint32_t> tokens,
                                                   luisa::span<const int32_t> position_ids,
                                                   luisa::span<const float> mm_features,
                                                   uint32_t mm_start, uint32_t mm_count, uint32_t kv_prefix,
                                                   luisa::vector<luisa::vector<float>> &k_caches,
                                                   luisa::vector<luisa::vector<float>> &v_caches,
                                                   P7TextOracle &result) {
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
        s = p7_oracle_text_block(cfg, w, l, x, rows, kv_prefix, position_ids, k_caches[l], v_caches[l], x_out);
        if (!s.ok()) { return s; }
        x = std::move(x_out);
        result.layer_out[l] = x;
    }
    s = ref_rms_norm(x, w.final_norm, rows, th, t.rms_norm_eps, result.final_norm);
    if (!s.ok()) { return s; }
    const luisa::vector<float> &head = w.tied ? w.embed : w.lm_head;
    luisa::vector<float> head_t(static_cast<size_t>(th) * t.vocab_size);
    for (uint32_t v = 0; v < t.vocab_size; v++) {
        for (uint32_t h = 0; h < th; h++) {
            head_t[static_cast<size_t>(h) * t.vocab_size + v] = head[static_cast<size_t>(v) * th + h];
        }
    }
    s = ref_gemm(result.final_norm, head_t, {}, rows, t.vocab_size, th, RefGemmEpilogue::none, result.logits);
    if (!s.ok()) { return s; }
    result.k_cache = k_caches;
    result.v_cache = v_caches;
    return Status::success();
}

// Greedy argmax over a logits row (lowest-index tie-break).
[[nodiscard]] inline uint32_t p7_oracle_argmax(luisa::span<const float> logits) {
    uint32_t best = 0u;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<uint32_t>(i); }
    }
    return best;
}

// A deterministic region crop (bright bar on dark noise).
[[nodiscard]] inline RegionCropInput p7_make_crop(uint32_t w, uint32_t h, uint32_t region_id, uint32_t seed = 11u) {
    RegionCropInput crop;
    crop.source_width = w;
    crop.source_height = h;
    crop.task.region_id = region_id;
    crop.task.page_index = 0;
    crop.task.reading_order = region_id;
    crop.task.block_type = LayoutBlockType::text;
    crop.task.source_box = output::DbRect{0, 0, static_cast<float>(w), static_cast<float>(h)};
    crop.pixels.resize(static_cast<size_t>(w) * h * 3u);
    Lcg lcg;
    lcg.state = seed;
    for (auto &b : crop.pixels) { b = static_cast<uint8_t>(lcg.next() % 60u); }
    // A bright horizontal bar.
    for (uint32_t y = h / 3u; y < 2u * h / 3u; y++) {
        for (uint32_t x = w / 6u; x < 5u * w / 6u; x++) {
            for (uint32_t c = 0; c < 3u; c++) {
                crop.pixels[(static_cast<size_t>(y) * w + x) * 3u + c] = 230u;
            }
        }
    }
    return crop;
}

}// namespace luisa::ocr::test

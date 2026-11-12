#pragma once
// Shared synthetic Phase 6 fixtures + INDEPENDENT CPU oracles for the layout /
// region-scheduler / PaddleOCR-VL / OpenDoc device tests. Synthetic only:
// deterministic weights, the declared small geometries, no model download, no
// real-model parity claim.
//
// The oracles are composed ONLY from the Phase 2 host references
// (cpu_reference.h) plus the Phase 5/6 host oracles, so they share NO code with
// the device executors. A test compares two independent implementations.
//
// DEVICE-SAFE: this header includes the device executors (ops_attention.h) but
// NEVER api.h/model_recipe.h (host AttentionMaskKind clash). Host composition
// bridges are exercised by separate host-only tests.
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
#include "ocr_pp_test_utils.h"

#include "ocr/models/open_doc.h"
#include "ocr/models/paddle_ocr_vl.h"
#include "ocr/models/pp_doc_layout.h"
#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_ctc.h"
#include "ocr/package.h"
#include "ocr/runtime/layout_vlm_executor.h"
#include "ocr/runtime/weights.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Temp dir
// ---------------------------------------------------------------------------
struct P6TempDir {
    luisa::filesystem::path path;
    P6TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_p6_test_" + luisa::format("{}", reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~P6TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] inline luisa::vector<std::byte> p6_f32_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> out(v.size() * 4);
    if (!v.empty()) { std::memcpy(out.data(), v.data(), out.size()); }
    return out;
}

[[nodiscard]] inline PackageManifest p6_manifest(luisa::string_view component) {
    PackageManifest m;
    m.model = "paddle_ocr_vl_1_6";// known semantic target; synthetic test artifact
    m.recipe_version = "1";
    m.checkpoint_revision = "synthetic-rev";
    m.config_fingerprint = "synthetic-fp";
    m.license = "test";
    m.producer = "ocr_p6_test";
    m.producer_version = "1";
    m.preprocessing_version = "1";
    m.tokenizer_version = "1";
    m.output_parser_version = "1";
    m.component = luisa::string{component};
    return m;
}

// ---------------------------------------------------------------------------
// PP-DocLayout synthetic weights + package + oracle
// ---------------------------------------------------------------------------
struct LayoutConvWeights {
    luisa::vector<float> weight;// [out_c, in_c/groups, kh, kw]
    luisa::vector<float> bias;  // [out_c]
};

struct LayoutSyntheticWeights {
    luisa::vector<LayoutConvWeights> backbone;
    LayoutConvWeights class_head;
    LayoutConvWeights box_head;
};

// Deterministic LCG weight generation. The class/box heads are calibrated so a
// bright-bar page produces deterministic detections (fixture design, not a
// model claim). The SAME weights feed the device path and the oracle.
[[nodiscard]] inline LayoutSyntheticWeights generate_layout_weights(const PpDocLayoutConfig &cfg,
                                                                    uint32_t seed = 0x5EEDu) {
    LayoutSyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill_conv = [&](LayoutConvWeights &c, const PpConvStage &s) {
        const uint32_t icg = s.in_c / s.groups;
        const float base = 1.0f / static_cast<float>(icg * s.kh * s.kw);
        c.weight.resize(static_cast<size_t>(s.weight_elements()));
        for (auto &x : c.weight) { x = base + lcg.uniform(-0.002f, 0.002f); }
        c.bias.resize(s.out_c);
        for (auto &x : c.bias) { x = lcg.uniform(0.0f, 0.02f); }
    };
    for (const auto &s : cfg.backbone) {
        LayoutConvWeights c;
        fill_conv(c, s);
        w.backbone.push_back(std::move(c));
    }
    fill_conv(w.class_head, cfg.class_head);
    fill_conv(w.box_head, cfg.box_head);
    return w;
}

// Write the layout detector package (conv_oihw weights, row_major biases).
[[nodiscard]] inline Status write_layout_package(const luisa::filesystem::path &path,
                                                 const PpDocLayoutConfig &cfg,
                                                 const PpDocLayoutTensorNames &names,
                                                 const LayoutSyntheticWeights &w) {
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
        spec.payload = p6_f32_bytes(data);
        return wr->add_tensor(spec);
    };
    const auto conv_shape = [](const PpConvStage &s) {
        return luisa::vector<uint64_t>{s.out_c, s.in_c / s.groups, s.kh, s.kw};
    };
    for (uint32_t i = 0; i < cfg.backbone.size(); i++) {
        if (auto s = add(names.backbone_w(i), conv_shape(cfg.backbone[i]), TensorLayout::conv_oihw, w.backbone[i].weight); !s.ok()) { return s; }
        if (auto s = add(names.backbone_b(i), {cfg.backbone[i].out_c}, TensorLayout::row_major, w.backbone[i].bias); !s.ok()) { return s; }
    }
    if (auto s = add(names.class_head_w(), conv_shape(cfg.class_head), TensorLayout::conv_oihw, w.class_head.weight); !s.ok()) { return s; }
    if (auto s = add(names.class_head_b(), {cfg.class_head.out_c}, TensorLayout::row_major, w.class_head.bias); !s.ok()) { return s; }
    if (auto s = add(names.box_head_w(), conv_shape(cfg.box_head), TensorLayout::conv_oihw, w.box_head.weight); !s.ok()) { return s; }
    if (auto s = add(names.box_head_b(), {cfg.box_head.out_c}, TensorLayout::row_major, w.box_head.bias); !s.ok()) { return s; }
    return wr->finish(p6_manifest("layout"));
}

// Independent CPU oracle: conv backbone -> class head (sigmoid) + box head.
// Uses ref_conv2d (double accumulate) + ref_sigmoid, NOT the device kernels.
struct LayoutOracleMaps {
    uint32_t map_h{0}, map_w{0}, num_classes{0};
    luisa::vector<float> class_probs;
    luisa::vector<float> box_deltas;
};

[[nodiscard]] inline Status oracle_layout_detect(const PpDocLayoutConfig &cfg,
                                                 const LayoutSyntheticWeights &w,
                                                 luisa::span<const float> pixels_chw,
                                                 LayoutOracleMaps &out) {
    luisa::vector<float> cur{pixels_chw.begin(), pixels_chw.end()};
    uint32_t h = cfg.input_height, wd = cfg.input_width;
    for (uint32_t i = 0; i < cfg.backbone.size(); i++) {
        const auto &s = cfg.backbone[i];
        luisa::vector<float> next;
        uint32_t oh = 0, ow = 0;
        if (auto st = ref_conv2d(cur, w.backbone[i].weight, w.backbone[i].bias, 1u, s.in_c, h, wd,
                                 s.out_c, s.kh, s.kw, s.stride_h, s.stride_w, s.pad_h, s.pad_w,
                                 s.dil_h, s.dil_w, s.groups, next, oh, ow); !st.ok()) { return st; }
        luisa::vector<float> act;
        if (auto st = ref_cnn_act(next, s.act, act); !st.ok()) { return st; }
        cur = std::move(act);
        h = oh;
        wd = ow;
    }
    out.map_h = h;
    out.map_w = wd;
    out.num_classes = cfg.num_classes();
    // Class head -> sigmoid.
    {
        const auto &s = cfg.class_head;
        luisa::vector<float> logits;
        uint32_t oh = 0, ow = 0;
        if (auto st = ref_conv2d(cur, w.class_head.weight, w.class_head.bias, 1u, s.in_c, h, wd,
                                 s.out_c, s.kh, s.kw, s.stride_h, s.stride_w, s.pad_h, s.pad_w,
                                 s.dil_h, s.dil_w, s.groups, logits, oh, ow); !st.ok()) { return st; }
        if (auto st = ref_sigmoid(logits, out.class_probs); !st.ok()) { return st; }
    }
    // Box head (no activation).
    {
        const auto &s = cfg.box_head;
        uint32_t oh = 0, ow = 0;
        if (auto st = ref_conv2d(cur, w.box_head.weight, w.box_head.bias, 1u, s.in_c, h, wd,
                                 s.out_c, s.kh, s.kw, s.stride_h, s.stride_w, s.pad_h, s.pad_w,
                                 s.dil_h, s.dil_w, s.groups, out.box_deltas, oh, ow); !st.ok()) { return st; }
    }
    return Status::success();
}

// Host stretch-resize + normalize mirror of the executor's declared convention.
// Mirrors the device's FLOAT arithmetic exactly (half-pixel centers, clamp
// edges, float bilinear), so the conv-chain parity test isolates the conv graph
// from preprocessing. This is an independent re-derivation of the same declared
// convention, NOT a call into the device helper.
[[nodiscard]] inline luisa::vector<float> p6_oracle_preprocess(luisa::span<const uint8_t> pixels,
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
                const float bottom = v01 + (v11 - v01) * fx;
                const float v = (top + (bottom - top) * fy) / 255.0f;
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] =
                    (v - mean[c]) / std_dev[c];
            }
        }
    }
    return out;
}

// A deterministic synthetic page: bright bars on a dark background.
[[nodiscard]] inline LayoutPageInput p6_make_page(uint32_t w, uint32_t h, uint32_t seed = 7u) {
    LayoutPageInput page;
    page.source_width = w;
    page.source_height = h;
    page.page_index = 0;
    page.pixels.resize(static_cast<size_t>(w) * h * 3u);
    Lcg lcg;
    lcg.state = seed;
    for (auto &b : page.pixels) { b = static_cast<uint8_t>(lcg.next() % 40u); }
    const auto bar = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
        for (uint32_t y = y0; y < y1 && y < h; y++) {
            for (uint32_t x = x0; x < x1 && x < w; x++) {
                for (uint32_t c = 0; c < 3u; c++) {
                    page.pixels[(static_cast<size_t>(y) * w + x) * 3u + c] = 230u;
                }
            }
        }
    };
    bar(w / 8u, h / 6u, w / 2u, h / 6u + h / 8u);
    bar(w / 8u, h / 2u, 7u * w / 8u, h / 2u + h / 8u);
    return page;
}

// ---------------------------------------------------------------------------
// PaddleOCR-VL synthetic weights (NaViT vision + connector + ERNIE decoder)
// ---------------------------------------------------------------------------
struct PaddleVisionBlock {
    luisa::vector<float> norm1_w, norm1_b, norm2_w, norm2_b;// [vh]
    luisa::vector<float> q_w, k_w, v_w, o_w;// [vh, vh] packed [K,N]
    luisa::vector<float> q_b, k_b, v_b, o_b;// [vh]
    luisa::vector<float> fc1_w;// [vh, vi]
    luisa::vector<float> fc1_b;// [vi]
    luisa::vector<float> fc2_w;// [vi, vh]
    luisa::vector<float> fc2_b;// [vh]
};

struct PaddleSyntheticWeights {
    // Vision.
    luisa::vector<float> patch_embed_w;// [patch_dim, vh]
    luisa::vector<float> patch_embed_b;// [vh]
    luisa::vector<PaddleVisionBlock> blocks;
    // Connector.
    luisa::vector<float> conn_ln_w, conn_ln_b;// [vh]
    luisa::vector<float> conn_fc1_w;// [vh, ci]
    luisa::vector<float> conn_fc1_b;// [ci]
    luisa::vector<float> conn_fc2_w;// [ci, th]
    luisa::vector<float> conn_fc2_b;// [th]
    // ERNIE decoder (Phase 3 contract: QKV bias, no QK-Norm, rotate-half RoPE).
    luisa::vector<float> embed;// [vocab, th]
    luisa::vector<float> final_norm;// [th]
    luisa::vector<float> lm_head;// [vocab, th] (== embed when tied)
    bool tied{true};
    struct TextLayer {
        luisa::vector<float> input_norm, post_norm;// [th]
        luisa::vector<float> q_w, k_w, v_w, o_w;// [K,N]
        luisa::vector<float> q_b, k_b, v_b;// [N]
        luisa::vector<float> gate_w, up_w, down_w;// [K,N]
    };
    luisa::vector<TextLayer> layers;
};

[[nodiscard]] inline PaddleSyntheticWeights generate_paddle_weights(const PaddleOcrVlConfig &cfg,
                                                                    uint32_t seed = 0x5EEDu) {
    PaddleSyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    const uint32_t vh = cfg.vision.hidden_size, vi = cfg.vision.intermediate_size;
    const uint32_t th = cfg.text.hidden_size, ti = cfg.text.intermediate_size;
    const uint32_t ci = cfg.connector.connector_intermediate;
    const uint32_t patch_dim = cfg.vision.patch_dim();
    fill(w.patch_embed_w, static_cast<size_t>(patch_dim) * vh, -0.3f, 0.3f);
    fill(w.patch_embed_b, vh, -0.1f, 0.1f);
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
        fill(B.q_b, vh, -0.1f, 0.1f);
        fill(B.k_b, vh, -0.1f, 0.1f);
        fill(B.v_b, vh, -0.1f, 0.1f);
        fill(B.o_b, vh, -0.1f, 0.1f);
        fill(B.fc1_w, static_cast<size_t>(vh) * vi, -0.3f, 0.3f);
        fill(B.fc1_b, vi, -0.1f, 0.1f);
        fill(B.fc2_w, static_cast<size_t>(vi) * vh, -0.3f, 0.3f);
        fill(B.fc2_b, vh, -0.1f, 0.1f);
    }
    fill(w.conn_ln_w, vh, 0.7f, 1.3f);
    fill(w.conn_ln_b, vh, -0.1f, 0.1f);
    fill(w.conn_fc1_w, static_cast<size_t>(vh) * ci, -0.3f, 0.3f);
    fill(w.conn_fc1_b, ci, -0.1f, 0.1f);
    fill(w.conn_fc2_w, static_cast<size_t>(ci) * th, -0.3f, 0.3f);
    fill(w.conn_fc2_b, th, -0.1f, 0.1f);
    fill(w.embed, static_cast<size_t>(cfg.text.vocab_size) * th, -0.6f, 0.6f);
    w.lm_head = w.embed;
    fill(w.final_norm, th, 0.7f, 1.3f);
    w.layers.resize(cfg.text.num_layers);
    const uint32_t qd = cfg.text.q_dim(), kvd = cfg.text.kv_dim();
    for (auto &L : w.layers) {
        fill(L.input_norm, th, 0.7f, 1.3f);
        fill(L.post_norm, th, 0.7f, 1.3f);
        fill(L.q_w, static_cast<size_t>(th) * qd, -0.3f, 0.3f);
        fill(L.k_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill(L.v_w, static_cast<size_t>(th) * kvd, -0.3f, 0.3f);
        fill(L.o_w, static_cast<size_t>(qd) * th, -0.3f, 0.3f);
        fill(L.q_b, qd, -0.1f, 0.1f);
        fill(L.k_b, kvd, -0.1f, 0.1f);
        fill(L.v_b, kvd, -0.1f, 0.1f);
        fill(L.gate_w, static_cast<size_t>(th) * ti, -0.3f, 0.3f);
        fill(L.up_w, static_cast<size_t>(th) * ti, -0.3f, 0.3f);
        fill(L.down_w, static_cast<size_t>(ti) * th, -0.3f, 0.3f);
    }
    return w;
}

// Write the Paddle vision + decoder packages under the given tensor-name map.
[[nodiscard]] inline Status write_paddle_packages(const luisa::filesystem::path &vision_path,
                                                  const luisa::filesystem::path &decoder_path,
                                                  const PaddleOcrVlConfig &cfg,
                                                  const PaddleOcrVlTensorNames &names,
                                                  const PaddleSyntheticWeights &w) {
    const uint32_t vh = cfg.vision.hidden_size, vi = cfg.vision.intermediate_size;
    const uint32_t th = cfg.text.hidden_size, ti = cfg.text.intermediate_size;
    const uint32_t ci = cfg.connector.connector_intermediate;
    const uint32_t patch_dim = cfg.vision.patch_dim();
    const uint32_t qd = cfg.text.q_dim(), kvd = cfg.text.kv_dim();
    // Vision package.
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
            spec.payload = p6_f32_bytes(data);
            return wr->add_tensor(spec);
        };
        if (auto s = add(names.patch_embed_weight(), {patch_dim, vh}, TensorLayout::kn, w.patch_embed_w); !s.ok()) { return s; }
        if (auto s = add(names.patch_embed_bias(), {vh}, TensorLayout::row_major, w.patch_embed_b); !s.ok()) { return s; }
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
            if (auto s = add(names.block_q_bias(l), {vh}, TensorLayout::row_major, B.q_b); !s.ok()) { return s; }
            if (auto s = add(names.block_k_bias(l), {vh}, TensorLayout::row_major, B.k_b); !s.ok()) { return s; }
            if (auto s = add(names.block_v_bias(l), {vh}, TensorLayout::row_major, B.v_b); !s.ok()) { return s; }
            if (auto s = add(names.block_o_bias(l), {vh}, TensorLayout::row_major, B.o_b); !s.ok()) { return s; }
            if (auto s = add(names.block_fc1(l), {vh, vi}, TensorLayout::kn, B.fc1_w); !s.ok()) { return s; }
            if (auto s = add(names.block_fc1_bias(l), {vi}, TensorLayout::row_major, B.fc1_b); !s.ok()) { return s; }
            if (auto s = add(names.block_fc2(l), {vi, vh}, TensorLayout::kn, B.fc2_w); !s.ok()) { return s; }
            if (auto s = add(names.block_fc2_bias(l), {vh}, TensorLayout::row_major, B.fc2_b); !s.ok()) { return s; }
        }
        if (auto s = add(names.connector_ln_weight(), {vh}, TensorLayout::row_major, w.conn_ln_w); !s.ok()) { return s; }
        if (auto s = add(names.connector_ln_bias(), {vh}, TensorLayout::row_major, w.conn_ln_b); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc1(), {vh, ci}, TensorLayout::kn, w.conn_fc1_w); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc1_bias(), {ci}, TensorLayout::row_major, w.conn_fc1_b); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc2(), {ci, th}, TensorLayout::kn, w.conn_fc2_w); !s.ok()) { return s; }
        if (auto s = add(names.connector_fc2_bias(), {th}, TensorLayout::row_major, w.conn_fc2_b); !s.ok()) { return s; }
        if (auto s = wr->finish(p6_manifest("vision")); !s.ok()) { return s; }
    }
    // Decoder package.
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
            spec.payload = p6_f32_bytes(data);
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
            if (auto s = add(tn.q_bias(l), {qd}, TensorLayout::row_major, L.q_b); !s.ok()) { return s; }
            if (auto s = add(tn.k_bias(l), {kvd}, TensorLayout::row_major, L.k_b); !s.ok()) { return s; }
            if (auto s = add(tn.v_bias(l), {kvd}, TensorLayout::row_major, L.v_b); !s.ok()) { return s; }
            if (auto s = add(tn.gate_proj(l), {th, ti}, TensorLayout::kn, L.gate_w); !s.ok()) { return s; }
            if (auto s = add(tn.up_proj(l), {th, ti}, TensorLayout::kn, L.up_w); !s.ok()) { return s; }
            if (auto s = add(tn.down_proj(l), {ti, th}, TensorLayout::kn, L.down_w); !s.ok()) { return s; }
        }
        if (auto s = wr->finish(p6_manifest("decoder")); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] inline Status upload_paddle_weights(compute::Device &device, compute::Stream &stream,
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
// PaddleOCR-VL independent CPU oracle (NaViT + reduction + connector + ERNIE)
// ---------------------------------------------------------------------------

// Greedy argmax over a logits row (lowest-index tie-break).
[[nodiscard]] inline uint32_t oracle_argmax(luisa::span<const float> logits) {
    uint32_t best = 0u;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

struct PaddleVisionOracle {
    luisa::vector<float> patches;// [patch_count, patch_dim]
    luisa::vector<float> vision_final;// [patch_count, vh]
    luisa::vector<float> features;// [visual_tokens, th]
};

// Patchify CHW -> [rows, patch_dim] in (c, py, px) order (independent oracle).
[[nodiscard]] inline luisa::vector<float> paddle_oracle_patchify(luisa::span<const float> pixels_chw,
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

// NaViT vision oracle for ONE packed crop.
[[nodiscard]] inline Status paddle_oracle_vision(const PaddleOcrVlConfig &cfg,
                                                 const PaddleSyntheticWeights &w,
                                                 luisa::span<const float> pixels_chw,
                                                 const PaddlePackedImage &packed,
                                                 PaddleVisionOracle &out) {
    const auto &v = cfg.vision;
    const uint32_t vh = v.hidden_size, vi = v.intermediate_size;
    const uint32_t heads = v.num_heads, hd = v.head_dim;
    const uint32_t rows = static_cast<uint32_t>(packed.patch_count);
    const float eps = v.layer_norm_eps, theta = v.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    out.patches = paddle_oracle_patchify(pixels_chw, packed.resized_height, packed.resized_width, v.patch_size);
    // Patch embed.
    luisa::vector<float> x;
    auto s = ref_gemm(out.patches, w.patch_embed_w, w.patch_embed_b, rows, vh, v.patch_dim(),
                      RefGemmEpilogue::bias, x);
    if (!s.ok()) { return s; }
    auto positions = paddle_vision_positions(packed);
    auto axis = paddle_vision_axis_of_pair(v);
    RefAttentionMask mask;
    mask.kind = RefMaskKind::bidirectional;
    for (uint32_t l = 0; l < v.num_layers; l++) {
        const auto &B = w.blocks[l];
        luisa::vector<float> norm, q, k, vv, qr, kr, attn, o_proj, after_o, norm2, fc1, fc2;
        s = ref_layer_norm(x, B.norm1_w, B.norm1_b, rows, vh, eps, norm);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.q_w, B.q_b, rows, vh, vh, RefGemmEpilogue::bias, q);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.k_w, B.k_b, rows, vh, vh, RefGemmEpilogue::bias, k);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm, B.v_w, B.v_b, rows, vh, vh, RefGemmEpilogue::bias, vv);
        if (!s.ok()) { return s; }
        s = ref_rope_axial(q, positions, axis, rows, heads, hd, 2u, theta, qr);
        if (!s.ok()) { return s; }
        s = ref_rope_axial(k, positions, axis, rows, heads, hd, 2u, theta, kr);
        if (!s.ok()) { return s; }
        s = ref_attention(qr, kr, vv, rows, rows, heads, heads, hd, scale, 0.0f, mask, attn);
        if (!s.ok()) { return s; }
        s = ref_gemm(attn, B.o_w, B.o_b, rows, vh, vh, RefGemmEpilogue::bias, o_proj);
        if (!s.ok()) { return s; }
        s = ref_residual_add(x, o_proj, after_o);
        if (!s.ok()) { return s; }
        s = ref_layer_norm(after_o, B.norm2_w, B.norm2_b, rows, vh, eps, norm2);
        if (!s.ok()) { return s; }
        s = ref_gemm(norm2, B.fc1_w, B.fc1_b, rows, vi, vh, RefGemmEpilogue::bias_gelu_erf, fc1);
        if (!s.ok()) { return s; }
        s = ref_gemm(fc1, B.fc2_w, B.fc2_b, rows, vh, vi, RefGemmEpilogue::bias, fc2);
        if (!s.ok()) { return s; }
        s = ref_residual_add(after_o, fc2, x);
        if (!s.ok()) { return s; }
    }
    out.vision_final = x;
    // Adaptive reduction: adaptive avg pool over the patch grid [1, vh, gh, gw]
    // -> [1, vh, oh, ow], then flatten to [tokens, vh].
    const uint32_t oh = packed.reduced_h, ow = packed.reduced_w;
    const uint32_t tokens = oh * ow;
    luisa::vector<float> pooled;
    s = ref_adaptive_avg_pool2d(x, 1u, vh, packed.grid_h, packed.grid_w, oh, ow, pooled);
    if (!s.ok()) { return s; }
    // Connector: LayerNorm + fc1(GELU) + fc2.
    luisa::vector<float> cnorm, cfc1;
    s = ref_layer_norm(pooled, w.conn_ln_w, w.conn_ln_b, tokens, vh, cfg.connector.layer_norm_eps, cnorm);
    if (!s.ok()) { return s; }
    s = ref_gemm(cnorm, w.conn_fc1_w, w.conn_fc1_b, tokens, cfg.connector.connector_intermediate, vh,
                 RefGemmEpilogue::bias_gelu_erf, cfc1);
    if (!s.ok()) { return s; }
    return ref_gemm(cfc1, w.conn_fc2_w, w.conn_fc2_b, tokens, cfg.text.hidden_size,
                    cfg.connector.connector_intermediate, RefGemmEpilogue::bias, out.features);
}

// ERNIE decoder oracle (Phase 3 contract: QKV bias, no QK-Norm, rotate-half 1D
// RoPE, RMSNorm, SwiGLU). Prefill + greedy decode. Independent of the device.
struct PaddleTextOracle {
    luisa::vector<float> logits;// [rows, vocab]
    luisa::vector<luisa::vector<float>> k_cache, v_cache;
};

[[nodiscard]] inline Status paddle_oracle_block(const PaddleOcrVlConfig &cfg,
                                                const PaddleSyntheticWeights &w, uint32_t layer,
                                                luisa::span<const float> x, uint32_t rows, uint32_t pos0,
                                                luisa::vector<float> &k_cache, luisa::vector<float> &v_cache,
                                                luisa::vector<float> &x_out) {
    const auto &t = cfg.text;
    const auto &L = w.layers[layer];
    const uint32_t th = t.hidden_size, ti = t.intermediate_size;
    const uint32_t qd = t.q_dim(), kvd = t.kv_dim();
    const uint32_t qheads = t.num_attention_heads, kvheads = t.num_kv_heads, hd = t.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    luisa::vector<float> norm, q, k, v, qr, kr, attn, after_o, norm2, gate, up, mlp;
    auto s = ref_rms_norm(x, L.input_norm, rows, th, t.rms_norm_eps, norm);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.q_w, L.q_b, rows, qd, th, RefGemmEpilogue::bias, q);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.k_w, L.k_b, rows, kvd, th, RefGemmEpilogue::bias, k);
    if (!s.ok()) { return s; }
    s = ref_gemm(norm, L.v_w, L.v_b, rows, kvd, th, RefGemmEpilogue::bias, v);
    if (!s.ok()) { return s; }
    // Rotate-half 1D RoPE: positions pos0..pos0+rows-1.
    luisa::vector<int32_t> pos(rows);
    for (uint32_t i = 0; i < rows; i++) { pos[i] = static_cast<int32_t>(pos0 + i); }
    s = ref_rope_rotate_half(q, pos, rows, qheads, hd, t.rope_theta, qr);
    if (!s.ok()) { return s; }
    s = ref_rope_rotate_half(k, pos, rows, kvheads, hd, t.rope_theta, kr);
    if (!s.ok()) { return s; }
    s = ref_kv_append(kr, v, k_cache, v_cache, rows, kvd, pos0, pos0 + rows);
    if (!s.ok()) { return s; }
    RefAttentionMask mask;
    mask.kind = RefMaskKind::causal;
    const uint32_t k_len = pos0 + rows;
    s = ref_attention(qr, k_cache, v_cache, rows, k_len, qheads, kvheads, hd, scale, 0.0f, mask, attn);
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

[[nodiscard]] inline Status paddle_oracle_forward(const PaddleOcrVlConfig &cfg,
                                                  const PaddleSyntheticWeights &w,
                                                  luisa::span<const uint32_t> tokens,
                                                  luisa::span<const float> mm_features,
                                                  uint32_t mm_start, uint32_t mm_count,
                                                  uint32_t kv_prefix,
                                                  luisa::vector<luisa::vector<float>> &k_caches,
                                                  luisa::vector<luisa::vector<float>> &v_caches,
                                                  PaddleTextOracle &result) {
    const auto &t = cfg.text;
    const uint32_t th = t.hidden_size, vocab = t.vocab_size;
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    const uint32_t kvd = t.kv_dim();
    luisa::vector<int32_t> ids(tokens.begin(), tokens.end());
    luisa::vector<float> emb;
    auto s = ref_embedding_lookup(w.embed, ids, vocab, th, emb);
    if (!s.ok()) { return s; }
    if (!mm_features.empty()) {
        luisa::vector<float> replaced;
        s = ref_mm_scatter(emb, mm_features, rows, th, mm_start, replaced);
        if (!s.ok()) { return s; }
        emb = std::move(replaced);
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
    luisa::vector<float> x = emb;
    for (uint32_t l = 0; l < num_layers; l++) {
        luisa::vector<float> x_out;
        s = paddle_oracle_block(cfg, w, l, x, rows, kv_prefix, k_caches[l], v_caches[l], x_out);
        if (!s.ok()) { return s; }
        x = std::move(x_out);
    }
    luisa::vector<float> final_norm;
    s = ref_rms_norm(x, w.final_norm, rows, th, t.rms_norm_eps, final_norm);
    if (!s.ok()) { return s; }
    const luisa::vector<float> &head = w.tied ? w.embed : w.lm_head;
    luisa::vector<float> head_t(static_cast<size_t>(th) * vocab);
    for (uint32_t v = 0; v < vocab; v++) {
        for (uint32_t h = 0; h < th; h++) {
            head_t[static_cast<size_t>(h) * vocab + v] = head[static_cast<size_t>(v) * th + h];
        }
    }
    s = ref_gemm(final_norm, head_t, {}, rows, vocab, th, RefGemmEpilogue::none, result.logits);
    if (!s.ok()) { return s; }
    result.k_cache = k_caches;
    result.v_cache = v_caches;
    return Status::success();
}

// ---------------------------------------------------------------------------
// OpenDoc synthetic fixture: PP-DocLayoutV2 layout + UniRec CNN+CTC recognizer.
// The UniRec recognizer reuses the shared PP-OCRv6 recognizer graph; weights are
// generated/folded with the PP helpers and written under PP-OCRv6 recognizer
// tensor names (the CnnExecutor resolves them). The layout detector uses the
// Phase 6 layout package. Synthetic only; no real-model claim.
// ---------------------------------------------------------------------------
struct OpenDocFixtureWeights {
    // Layout.
    PpDocLayoutConfig layout_cfg;
    PpDocLayoutTensorNames layout_names;
    LayoutSyntheticWeights layout_weights;
    // UniRec recognizer (PP-OCRv6 tiny geometry + dictionary).
    PpOcrv6Config rec_cfg;
    PpOcrv6TensorNames rec_names;
    PpSyntheticWeights rec_weights;
    PpFoldedWeights rec_folded;
};

[[nodiscard]] inline Status generate_opendoc_weights(OpenDocFixtureWeights &out, uint32_t seed = 0x5EEDu) {
    out.layout_cfg = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v2);
    out.layout_names = PpDocLayoutTensorNames{};
    out.layout_weights = generate_layout_weights(out.layout_cfg, seed);
    // UniRec recognizer: PP-OCRv6 tiny recognizer geometry. The detector part of
    // rec_cfg is a placeholder (never run; the CnnExecutor is recognizer-only).
    out.rec_cfg = pp_ocrv6_synthetic_config(PpOcrSize::tiny);
    out.rec_names = PpOcrv6TensorNames{};
    out.rec_weights = generate_pp_synthetic_weights(out.rec_cfg, seed + 1u);
    return fold_pp_weights(out.rec_cfg, out.rec_weights, out.rec_folded);
}

// Write the layout package + the UniRec recognizer package. The recognizer
// package carries ONLY the recognizer tensors (PP-OCRv6 recognizer names); the
// detector tensors are omitted (recognizer-only mode never resolves them).
[[nodiscard]] inline Status write_opendoc_packages(const luisa::filesystem::path &layout_path,
                                                   const luisa::filesystem::path &rec_path,
                                                   const OpenDocFixtureWeights &w) {
    if (auto s = write_layout_package(layout_path, w.layout_cfg, w.layout_names, w.layout_weights); !s.ok()) { return s; }
    // Recognizer package: write only the recognizer component tensors.
    auto writer = PackageWriter::create(rec_path);
    if (!writer.ok()) { return writer.status; }
    auto &wr = *writer.value;
    const auto add = [&wr](const luisa::string &name, luisa::vector<uint64_t> shape,
                           TensorLayout layout, luisa::span<const float> data) -> Status {
        PackageWriter::TensorSpec spec;
        spec.name = name;
        spec.encoding = TensorEncoding::fp32;
        spec.layout = layout;
        spec.shape = std::move(shape);
        spec.payload = p6_f32_bytes(data);
        return wr->add_tensor(spec);
    };
    const auto conv_shape = [](const PpConvStage &s) {
        return luisa::vector<uint64_t>{s.out_c, s.in_c / s.groups, s.kh, s.kw};
    };
    const auto &rec = w.rec_cfg.recognizer;
    const auto &names = w.rec_names;
    for (uint32_t i = 0; i < rec.backbone.size(); i++) {
        const auto &s = rec.backbone[i];
        if (auto st = add(names.rec_backbone_w(i), conv_shape(s), TensorLayout::conv_oihw, w.rec_folded.rec_backbone_w[i]); !st.ok()) { return st; }
        if (auto st = add(names.rec_backbone_b(i), {s.out_c}, TensorLayout::row_major, w.rec_folded.rec_backbone_b[i]); !st.ok()) { return st; }
    }
    uint32_t collapsed = 0;
    if (auto cc = rec.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
    const uint32_t inter = rec.svtr_intermediate;
    const uint32_t classes = rec.num_classes;
    if (auto s = add(names.rec_svtr_fc1_w(), {collapsed, inter}, TensorLayout::kn, w.rec_weights.svtr_fc1_w); !s.ok()) { return s; }
    if (auto s = add(names.rec_svtr_fc1_b(), {inter}, TensorLayout::row_major, w.rec_weights.svtr_fc1_b); !s.ok()) { return s; }
    if (auto s = add(names.rec_svtr_fc2_w(), {inter, collapsed}, TensorLayout::kn, w.rec_weights.svtr_fc2_w); !s.ok()) { return s; }
    if (auto s = add(names.rec_svtr_fc2_b(), {collapsed}, TensorLayout::row_major, w.rec_weights.svtr_fc2_b); !s.ok()) { return s; }
    if (auto s = add(names.rec_ctc_w(), {collapsed, classes}, TensorLayout::kn, w.rec_weights.ctc_w); !s.ok()) { return s; }
    if (auto s = add(names.rec_ctc_b(), {classes}, TensorLayout::row_major, w.rec_weights.ctc_b); !s.ok()) { return s; }
    return wr->finish(p6_manifest("unirec"));
}

// Upload the layout + UniRec recognizer components into a WeightStore.
[[nodiscard]] inline Status upload_opendoc_weights(compute::Device &device, compute::Stream &stream,
                                                   const luisa::filesystem::path &layout_path,
                                                   const luisa::filesystem::path &rec_path,
                                                   WeightStore &store) {
    {
        auto reader = PackageReader::open(layout_path);
        if (!reader.ok()) { return reader.status; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "layout"); !s.ok()) { return s; }
    }
    {
        auto reader = PackageReader::open(rec_path);
        if (!reader.ok()) { return reader.status; }
        if (auto s = store.upload_component_sync(device, stream, **reader.value, "unirec"); !s.ok()) { return s; }
    }
    return Status::success();
}

// Independent UniRec recognizer oracle (UNFOLDED conv->BN graph + SVTR + CTC),
// composed from the PP host oracles. Returns frame logits for a normalized crop.
[[nodiscard]] inline Status opendoc_oracle_recognizer(const PpOcrv6Config &cfg, const PpSyntheticWeights &w,
                                                      luisa::span<const float> crop, float temperature,
                                                      PpOracleRecognizer &out) {
    return pp_oracle_recognizer(cfg, w, crop, temperature, out);
}

}// namespace luisa::ocr::test
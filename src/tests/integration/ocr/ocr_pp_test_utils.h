#pragma once
// Shared synthetic PP-OCRv6 fixture + INDEPENDENT CPU oracle for the Phase 5
// CNN tests. Synthetic only: deterministic weights, the declared per-size
// geometry (models/pp_ocrv6.h pp_ocrv6_synthetic_config), no model download,
// no real-model parity claim.
//
// The oracle is composed ONLY from the Phase 2 host references
// (cpu_reference.h) plus the Phase 5 host oracles (ops_ctc.h ref_*), so it
// shares NO code with the device executor. Folding is exercised explicitly:
// the fixture generates UNFOLDED Conv+BN and RepLK branch weights, folds them
// with pp_fold_conv_bn / pp_fold_replk (the code under test), and the oracle
// evaluates the UNFOLDED two-step graph (conv -> BN; large+small+identity
// branches summed) so the folded device path is compared against the
// unfolded mathematical definition, not against the same folding code.
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

#include "ocr/models/pp_ocrv6.h"
#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_ctc.h"
#include "ocr/package.h"
#include "ocr/runtime/cnn_executor.h"
#include "ocr/runtime/weights.h"

namespace luisa::ocr::test {

// ---------------------------------------------------------------------------
// Temp dir
// ---------------------------------------------------------------------------
struct PpTempDir {
    luisa::filesystem::path path;
    PpTempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_pp_test_" + luisa::format("{}", reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~PpTempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

// ---------------------------------------------------------------------------
// Unfolded synthetic weight set (Conv+BN branches; RepLK 3-branch block).
// ---------------------------------------------------------------------------
struct PpConvBn {
    luisa::vector<float> weight;// [out_c, in_c/groups, kh, kw]
    luisa::vector<float> bias;  // [out_c]
    luisa::vector<float> bn_mean, bn_var, bn_gamma, bn_beta;// [out_c]
};

struct PpRepLkBranches {
    // Depthwise large kernel KxK + BN, small kernel kxk + BN, identity BN.
    PpConvBn large;
    PpConvBn small;
    luisa::vector<float> id_mean, id_var, id_gamma, id_beta;// [C]
};

struct PpSyntheticWeights {
    // Detector.
    luisa::vector<PpConvBn> det_backbone;
    PpConvBn det_lateral;
    PpRepLkBranches det_replk;
    PpConvBn det_db;
    // Recognizer.
    luisa::vector<PpConvBn> rec_backbone;
    luisa::vector<float> svtr_fc1_w, svtr_fc1_b;// [K,N] packed + [N]
    luisa::vector<float> svtr_fc2_w, svtr_fc2_b;
    luisa::vector<float> ctc_w, ctc_b;
};

// BN epsilon used by the fixture (declared synthetic contract value).
inline constexpr float kPpBnEps = 1e-5f;
// RepLK branch kernel sizes (declared): large 7x7 (matches replk_conv), small 3x3.
inline constexpr uint32_t kPpRepLkSmall = 3u;

// ---------------------------------------------------------------------------
// Deterministic generation (LCG; bounded, well-conditioned).
// ---------------------------------------------------------------------------
[[nodiscard]] inline PpSyntheticWeights generate_pp_synthetic_weights(const PpOcrv6Config &cfg,
                                                                      uint32_t seed = 0x5EEDu) {
    PpSyntheticWeights w;
    Lcg lcg;
    lcg.state = seed;
    const auto fill = [&lcg](luisa::vector<float> &v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto &x : v) { x = lcg.uniform(lo, hi); }
    };
    const auto fill_conv_bn = [&](PpConvBn &c, const PpConvStage &s) {
        // Local box-filter init: each output channel averages its input
        // channels over the kernel footprint (weight = 1/(icg*kh*kw)), plus a
        // small deterministic per-element jitter. This is a magnitude-
        // propagating, well-conditioned synthetic init: bright bars drive
        // larger post-ReLU activations than the dark background WITHOUT the
        // random receptive-field blur/explosion of a fully random kernel, so
        // the luminance-calibrated DB head separates them deterministically.
        // The jitter keeps every weight distinct (a real graph, not a stub).
        const uint32_t icg = s.in_c / s.groups;
        const float base = 1.0f / static_cast<float>(icg * s.kh * s.kw);
        c.weight.resize(static_cast<size_t>(s.weight_elements()));
        for (auto &x : c.weight) { x = base + lcg.uniform(-0.002f, 0.002f); }
        fill(c.bias, s.out_c, 0.0f, 0.02f);
        // Near-identity BN (mean~0, var~1, gamma~1, beta~0).
        fill(c.bn_mean, s.out_c, -0.02f, 0.02f);
        fill(c.bn_var, s.out_c, 0.95f, 1.05f);// strictly positive, ~1
        fill(c.bn_gamma, s.out_c, 0.95f, 1.05f);
        fill(c.bn_beta, s.out_c, -0.02f, 0.02f);
    };
    for (const auto &s : cfg.detector.backbone) {
        PpConvBn c;
        fill_conv_bn(c, s);
        w.det_backbone.push_back(std::move(c));
    }
    fill_conv_bn(w.det_lateral, cfg.detector.lateral_conv);
    // RepLK branches: large (KxK) + small (3x3) + identity BN. The declared
    // synthetic contract: all branches are FULL convs over (in_c -> out_c);
    // the identity branch contributes, for each output channel oc, an
    // impulse at the kernel center of input channel (oc % in_c), scaled by
    // that output channel's folded BN gamma. (The depthwise large-kernel
    // exercise lives in the backbone stages; pp_fold_replk's per-channel
    // math is identical for full and depthwise kernels and is additionally
    // unit-tested directly on a depthwise block.)
    {
        const auto &s = cfg.detector.replk_conv;
        PpConvStage large = s;
        fill_conv_bn(w.det_replk.large, large);
        PpConvStage small = s;
        small.kh = small.kw = kPpRepLkSmall;
        small.pad_h = small.pad_w = kPpRepLkSmall / 2u;
        fill_conv_bn(w.det_replk.small, small);
        fill(w.det_replk.id_mean, s.out_c, -0.02f, 0.02f);
        fill(w.det_replk.id_var, s.out_c, 0.95f, 1.05f);
        fill(w.det_replk.id_gamma, s.out_c, 0.95f, 1.05f);
        fill(w.det_replk.id_beta, s.out_c, -0.02f, 0.02f);
    }
    fill_conv_bn(w.det_db, cfg.detector.db_conv);
    for (const auto &s : cfg.recognizer.backbone) {
        PpConvBn c;
        fill_conv_bn(c, s);
        w.rec_backbone.push_back(std::move(c));
    }
    uint32_t collapsed = 0;
    if (auto cc = cfg.recognizer.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
    const uint32_t inter = cfg.recognizer.svtr_intermediate;
    const uint32_t classes = cfg.recognizer.num_classes;
    fill(w.svtr_fc1_w, static_cast<size_t>(collapsed) * inter, -0.25f, 0.25f);
    fill(w.svtr_fc1_b, inter, -0.05f, 0.05f);
    fill(w.svtr_fc2_w, static_cast<size_t>(inter) * collapsed, -0.25f, 0.25f);
    fill(w.svtr_fc2_b, collapsed, -0.05f, 0.05f);
    fill(w.ctc_w, static_cast<size_t>(collapsed) * classes, -0.25f, 0.25f);
    fill(w.ctc_b, classes, -0.05f, 0.05f);
    return w;
}

// ---------------------------------------------------------------------------
// Folding (the code under test) -> folded device weight set.
// ---------------------------------------------------------------------------
struct PpFoldedWeights {
    luisa::vector<luisa::vector<float>> det_backbone_w, det_backbone_b;
    luisa::vector<float> det_lateral_w, det_lateral_b;
    luisa::vector<float> det_replk_w, det_replk_b;
    luisa::vector<float> det_db_w, det_db_b;
    luisa::vector<luisa::vector<float>> rec_backbone_w, rec_backbone_b;
};

[[nodiscard]] inline Status fold_pp_weights(const PpOcrv6Config &cfg, const PpSyntheticWeights &w,
                                            PpFoldedWeights &out) {
    const auto fold_stage = [](const PpConvStage &s, const PpConvBn &c,
                               luisa::vector<float> &fw, luisa::vector<float> &fb) -> Status {
        return pp_fold_conv_bn(c.weight, c.bias, c.bn_mean, c.bn_var, c.bn_gamma, c.bn_beta,
                               s.out_c, s.weight_elements() / s.out_c, kPpBnEps, fw, fb);
    };
    out.det_backbone_w.resize(cfg.detector.backbone.size());
    out.det_backbone_b.resize(cfg.detector.backbone.size());
    for (uint32_t i = 0; i < cfg.detector.backbone.size(); i++) {
        if (auto s = fold_stage(cfg.detector.backbone[i], w.det_backbone[i],
                                out.det_backbone_w[i], out.det_backbone_b[i]); !s.ok()) { return s; }
    }
    if (auto s = fold_stage(cfg.detector.lateral_conv, w.det_lateral, out.det_lateral_w, out.det_lateral_b); !s.ok()) { return s; }
    // RepLK: fold each conv branch with its BN, then merge the three branches
    // through pp_fold_replk (the code under test). The helper is per-channel;
    // a full conv [out_c, in_c, K, K] is merged by treating each (oc, ic)
    // kernel slice as one "channel" (the center-pad/impulse math is identical
    // per slice). The identity branch contributes an impulse ONLY on the
    // (oc, oc % in_c) slice, scaled by oc's folded identity gamma.
    {
        const auto &rs = cfg.detector.replk_conv;
        luisa::vector<float> lw, lb, sw, sb;
        if (auto s = fold_stage(rs, w.det_replk.large, lw, lb); !s.ok()) { return s; }
        PpConvStage small = rs;
        small.kh = small.kw = kPpRepLkSmall;
        if (auto s = fold_stage(small, w.det_replk.small, sw, sb); !s.ok()) { return s; }
        const uint32_t in_c = rs.in_c;
        const uint32_t K = rs.kh;
        const uint32_t slices = rs.out_c * in_c;
        luisa::vector<float> idg(slices, 0.0f), idb(slices, 0.0f);
        luisa::vector<float> id_bias(rs.out_c);
        for (uint32_t oc = 0; oc < rs.out_c; oc++) {
            const double scale = static_cast<double>(w.det_replk.id_gamma[oc]) /
                                 std::sqrt(static_cast<double>(w.det_replk.id_var[oc]) + kPpBnEps);
            idg[oc * in_c + (oc % in_c)] = static_cast<float>(scale);
            id_bias[oc] = static_cast<float>(static_cast<double>(w.det_replk.id_beta[oc]) -
                                             static_cast<double>(w.det_replk.id_mean[oc]) * scale);
        }
        // The helper adds large_bias + small_bias + identity_bias per slice;
        // we pass the LARGE bias only on the ic==0 slice of each oc (so the
        // per-slice sum over ic yields the per-oc bias exactly once), zero
        // elsewhere, and add the identity beta per oc below.
        luisa::vector<float> slice_l(slices, 0.0f), zero_s(slices, 0.0f);
        for (uint32_t oc = 0; oc < rs.out_c; oc++) {
            slice_l[oc * in_c] = lb[oc];
        }
        luisa::vector<float> merged_w, merged_b;
        if (auto s = pp_fold_replk(lw, slice_l, sw, zero_s, idg, idb, slices, K, kPpRepLkSmall,
                                   merged_w, merged_b); !s.ok()) { return s; }
        out.det_replk_w = std::move(merged_w);
        out.det_replk_b.resize(rs.out_c);
        for (uint32_t oc = 0; oc < rs.out_c; oc++) {
            out.det_replk_b[oc] = merged_b[oc * in_c] + sb[oc] + id_bias[oc];
        }
    }
    if (auto s = fold_stage(cfg.detector.db_conv, w.det_db, out.det_db_w, out.det_db_b); !s.ok()) { return s; }
    out.rec_backbone_w.resize(cfg.recognizer.backbone.size());
    out.rec_backbone_b.resize(cfg.recognizer.backbone.size());
    for (uint32_t i = 0; i < cfg.recognizer.backbone.size(); i++) {
        if (auto s = fold_stage(cfg.recognizer.backbone[i], w.rec_backbone[i],
                                out.rec_backbone_w[i], out.rec_backbone_b[i]); !s.ok()) { return s; }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Package writing + weight upload (folded weights, exact tensor map).
// ---------------------------------------------------------------------------
[[nodiscard]] inline luisa::vector<std::byte> pp_f32_bytes(luisa::span<const float> data) {
    luisa::vector<std::byte> bytes(data.size() * sizeof(float));
    if (!data.empty()) { std::memcpy(bytes.data(), data.data(), bytes.size()); }
    return bytes;
}

[[nodiscard]] inline PackageManifest pp_synthetic_manifest(luisa::string_view component) {
    PackageManifest m;
    m.model = "pp_ocrv6:tiny";// selector family; size is a config concern
    m.recipe_version = "0";
    m.checkpoint_revision = "synthetic";
    m.config_fingerprint = "synthetic";
    m.license = "synthetic-test-only";
    m.producer = "ocr_pp_test_utils";
    m.producer_version = "0";
    m.preprocessing_version = "0";
    m.tokenizer_version = "0";
    m.output_parser_version = "0";
    m.component = luisa::string{component};
    return m;
}

[[nodiscard]] inline Status write_pp_synthetic_packages(const luisa::filesystem::path &det_path,
                                                        const luisa::filesystem::path &rec_path,
                                                        const PpOcrv6Config &cfg,
                                                        const PpOcrv6TensorNames &names,
                                                        const PpSyntheticWeights &w,
                                                        const PpFoldedWeights &folded) {
    const auto add = [](PackageWriter &wr, const luisa::string &name, luisa::vector<uint64_t> shape,
                        TensorLayout layout, luisa::span<const float> data) -> Status {
        PackageWriter::TensorSpec spec;
        spec.name = name;
        spec.encoding = TensorEncoding::fp32;
        spec.layout = layout;
        spec.shape = std::move(shape);
        spec.payload = pp_f32_bytes(data);
        return wr.add_tensor(spec);
    };
    const auto conv_shape = [](const PpConvStage &s) {
        return luisa::vector<uint64_t>{s.out_c, s.in_c / s.groups, s.kh, s.kw};
    };
    // Detector package.
    {
        auto writer = PackageWriter::create(det_path);
        if (!writer.ok()) { return writer.status; }
        auto &wr = *writer.value;
        for (uint32_t i = 0; i < cfg.detector.backbone.size(); i++) {
            const auto &s = cfg.detector.backbone[i];
            if (auto st = add(*wr, names.det_backbone_w(i), conv_shape(s), TensorLayout::conv_oihw,
                              folded.det_backbone_w[i]); !st.ok()) { return st; }
            if (auto st = add(*wr, names.det_backbone_b(i), {s.out_c}, TensorLayout::row_major,
                              folded.det_backbone_b[i]); !st.ok()) { return st; }
        }
        if (auto s = add(*wr, names.det_lateral_w(), conv_shape(cfg.detector.lateral_conv), TensorLayout::conv_oihw,
                         folded.det_lateral_w); !s.ok()) { return s; }
        if (auto s = add(*wr, names.det_lateral_b(), {cfg.detector.lateral_conv.out_c}, TensorLayout::row_major,
                         folded.det_lateral_b); !s.ok()) { return s; }
        if (auto s = add(*wr, names.det_replk_w(), conv_shape(cfg.detector.replk_conv), TensorLayout::conv_oihw,
                         folded.det_replk_w); !s.ok()) { return s; }
        if (auto s = add(*wr, names.det_replk_b(), {cfg.detector.replk_conv.out_c}, TensorLayout::row_major,
                         folded.det_replk_b); !s.ok()) { return s; }
        if (auto s = add(*wr, names.det_db_w(), conv_shape(cfg.detector.db_conv), TensorLayout::conv_oihw,
                         folded.det_db_w); !s.ok()) { return s; }
        if (auto s = add(*wr, names.det_db_b(), {cfg.detector.db_conv.out_c}, TensorLayout::row_major,
                         folded.det_db_b); !s.ok()) { return s; }
        if (auto s = wr->finish(pp_synthetic_manifest("detector")); !s.ok()) { return s; }
    }
    // Recognizer package.
    {
        auto writer = PackageWriter::create(rec_path);
        if (!writer.ok()) { return writer.status; }
        auto &wr = *writer.value;
        for (uint32_t i = 0; i < cfg.recognizer.backbone.size(); i++) {
            const auto &s = cfg.recognizer.backbone[i];
            if (auto st = add(*wr, names.rec_backbone_w(i), conv_shape(s), TensorLayout::conv_oihw,
                              folded.rec_backbone_w[i]); !st.ok()) { return st; }
            if (auto st = add(*wr, names.rec_backbone_b(i), {s.out_c}, TensorLayout::row_major,
                              folded.rec_backbone_b[i]); !st.ok()) { return st; }
        }
        uint32_t collapsed = 0;
        if (auto cc = cfg.recognizer.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
        const uint32_t inter = cfg.recognizer.svtr_intermediate;
        const uint32_t classes = cfg.recognizer.num_classes;
        if (auto s = add(*wr, names.rec_svtr_fc1_w(), {collapsed, inter}, TensorLayout::kn, w.svtr_fc1_w); !s.ok()) { return s; }
        if (auto s = add(*wr, names.rec_svtr_fc1_b(), {inter}, TensorLayout::row_major, w.svtr_fc1_b); !s.ok()) { return s; }
        if (auto s = add(*wr, names.rec_svtr_fc2_w(), {inter, collapsed}, TensorLayout::kn, w.svtr_fc2_w); !s.ok()) { return s; }
        if (auto s = add(*wr, names.rec_svtr_fc2_b(), {collapsed}, TensorLayout::row_major, w.svtr_fc2_b); !s.ok()) { return s; }
        if (auto s = add(*wr, names.rec_ctc_w(), {collapsed, classes}, TensorLayout::kn, w.ctc_w); !s.ok()) { return s; }
        if (auto s = add(*wr, names.rec_ctc_b(), {classes}, TensorLayout::row_major, w.ctc_b); !s.ok()) { return s; }
        if (auto s = wr->finish(pp_synthetic_manifest("recognizer")); !s.ok()) { return s; }
    }
    return Status::success();
}

[[nodiscard]] inline Status upload_pp_synthetic_weights(compute::Device &device, compute::Stream &stream,
                                                        const luisa::filesystem::path &det_path,
                                                        const luisa::filesystem::path &rec_path,
                                                        WeightStore &store) {
    auto det = PackageReader::open(det_path);
    if (!det.ok()) { return det.status; }
    if (auto s = store.upload_component_sync(device, stream, *det.value->get(), "detector"); !s.ok()) { return s; }
    auto rec = PackageReader::open(rec_path);
    if (!rec.ok()) { return rec.status; }
    return store.upload_component_sync(device, stream, *rec.value->get(), "recognizer");
}

// ---------------------------------------------------------------------------
// INDEPENDENT CPU oracle: evaluates the UNFOLDED graph.
// ---------------------------------------------------------------------------
struct PpOracleDetector {
    luisa::vector<float> prob;// sigmoid probability map [map_h, map_w]
    uint32_t map_h{0}, map_w{0};
    // Debug stage outputs.
    luisa::vector<float> backbone_out;// final backbone stage output
    uint32_t backbone_h{0}, backbone_w{0};
    luisa::vector<float> fused;// concat(backbone, up(lateral))
    luisa::vector<float> replk;// RepLK branch sum + activation
    luisa::vector<float> replk_large, replk_small, replk_id;// branch outputs
    luisa::vector<float> logits;// DB head pre-sigmoid logits
};

// conv -> BN (two-step, NOT folded) + activation.
[[nodiscard]] inline Status pp_oracle_conv_bn(const PpConvStage &s, const PpConvBn &c,
                                              luisa::span<const float> x, uint32_t in_h, uint32_t in_w,
                                              luisa::vector<float> &out, uint32_t &out_h, uint32_t &out_w) {
    luisa::vector<float> conv_out;
    if (auto st = ref_conv2d(x, c.weight, c.bias, 1u, s.in_c, in_h, in_w, s.out_c, s.kh, s.kw,
                             s.stride_h, s.stride_w, s.pad_h, s.pad_w, s.dil_h, s.dil_w, s.groups,
                             conv_out, out_h, out_w); !st.ok()) { return st; }
    luisa::vector<float> bn_out;
    if (auto st = ref_batch_norm(conv_out, c.bn_mean, c.bn_var, c.bn_gamma, c.bn_beta,
                                 1u, s.out_c, static_cast<uint64_t>(out_h) * out_w, kPpBnEps, bn_out); !st.ok()) { return st; }
    return ref_cnn_act(bn_out, s.act, out);
}

// Full detector oracle on normalized CHW pixels [3, det_h, det_w].
[[nodiscard]] inline Status pp_oracle_detector(const PpOcrv6Config &cfg, const PpSyntheticWeights &w,
                                               luisa::span<const float> pixels, PpOracleDetector &out) {
    const auto &det = cfg.detector;
    luisa::vector<float> cur{pixels.begin(), pixels.end()};
    uint32_t h = det.input_height, wd = det.input_width;
    luisa::vector<float> skip;
    uint32_t skip_h = 0, skip_w = 0;
    for (uint32_t i = 0; i < det.backbone.size(); i++) {
        luisa::vector<float> next;
        uint32_t oh = 0, ow = 0;
        if (auto s = pp_oracle_conv_bn(det.backbone[i], w.det_backbone[i], cur, h, wd, next, oh, ow); !s.ok()) { return s; }
        cur = std::move(next);
        h = oh;
        wd = ow;
        if (i == det.lateral_stage) {
            skip = cur;
            skip_h = h;
            skip_w = wd;
        }
    }
    out.backbone_out = cur;
    out.backbone_h = h;
    out.backbone_w = wd;
    // Lateral conv+BN.
    luisa::vector<float> lateral;
    uint32_t lh = 0, lw = 0;
    if (auto s = pp_oracle_conv_bn(det.lateral_conv, w.det_lateral, skip, skip_h, skip_w, lateral, lh, lw); !s.ok()) { return s; }
    // Upsample to (h, wd).
    luisa::vector<float> up;
    if (auto s = ref_fpn_resize(lateral, 1u, det.lateral_conv.out_c, lh, lw, h, wd, up); !s.ok()) { return s; }
    // Concat cur + up along channels.
    luisa::vector<float> fused;
    if (auto s = ref_concat_channels(cur, up, 1u, det.backbone.back().out_c, det.lateral_conv.out_c,
                                     static_cast<uint64_t>(h) * wd, fused); !s.ok()) { return s; }
    out.fused = fused;
    // RepLK: evaluate the THREE unfolded branches and sum (large conv+BN +
    // small conv+BN + identity BN), then the stage activation.
    const auto &rs = det.replk_conv;
    // RepLK branches are summed BEFORE the stage activation (structural
    // reparameterization): evaluate each branch with act=0, sum, then apply
    // rs.act once. Applying the activation per-branch would be wrong.
    PpConvStage large_noact = rs;
    large_noact.act = 0;
    luisa::vector<float> large_out, small_out;
    uint32_t oh1 = 0, ow1 = 0, oh2 = 0, ow2 = 0;
    if (auto s = pp_oracle_conv_bn(large_noact, w.det_replk.large, fused, h, wd, large_out, oh1, ow1); !s.ok()) { return s; }
    PpConvStage small = rs;
    small.kh = small.kw = kPpRepLkSmall;
    small.pad_h = small.pad_w = kPpRepLkSmall / 2u;
    small.act = 0;
    if (auto s = pp_oracle_conv_bn(small, w.det_replk.small, fused, h, wd, small_out, oh2, ow2); !s.ok()) { return s; }
    // Identity BN branch (declared synthetic contract): for each output
    // channel oc, y[oc] = BN_oc(x[oc % in_c]) -- the identity maps input
    // channel (oc % in_c) to output channel oc through oc's BN.
    luisa::vector<float> id_out(static_cast<size_t>(rs.out_c) * h * wd, 0.0f);
    for (uint32_t oc = 0; oc < rs.out_c; oc++) {
        const uint32_t ic = oc % rs.in_c;
        const double scale = static_cast<double>(w.det_replk.id_gamma[oc]) /
                             std::sqrt(static_cast<double>(w.det_replk.id_var[oc]) + kPpBnEps);
        for (uint64_t p = 0; p < static_cast<uint64_t>(h) * wd; p++) {
            const double v = fused[static_cast<size_t>(ic) * h * wd + p];
            id_out[static_cast<size_t>(oc) * h * wd + p] = static_cast<float>(
                (v - w.det_replk.id_mean[oc]) * scale + w.det_replk.id_beta[oc]);
        }
    }
    out.replk_large = large_out;
    out.replk_small = small_out;
    out.replk_id = id_out;
    luisa::vector<float> replk_sum(large_out.size());
    for (size_t i = 0; i < large_out.size(); i++) {
        replk_sum[i] = large_out[i] + small_out[i] + id_out[i];
    }
    luisa::vector<float> replk_act;
    if (auto s = ref_cnn_act(replk_sum, rs.act, replk_act); !s.ok()) { return s; }
    out.replk = replk_act;
    // DB head conv+BN -> sigmoid.
    luisa::vector<float> logits;
    uint32_t ph = 0, pw = 0;
    if (auto s = pp_oracle_conv_bn(det.db_conv, w.det_db, replk_act, oh1, ow1, logits, ph, pw); !s.ok()) { return s; }
    out.map_h = ph;
    out.map_w = pw;
    out.logits = logits;
    return ref_sigmoid(logits, out.prob);
}

struct PpOracleRecognizer {
    luisa::vector<float> logits;// [frames, classes]
    luisa::vector<float> log_probs;
    luisa::vector<int32_t> ids;
    luisa::vector<float> probs;
    uint32_t frames{0};
};

// Full recognizer oracle on a normalized CHW crop [3, rec_h, rec_w].
[[nodiscard]] inline Status pp_oracle_recognizer(const PpOcrv6Config &cfg, const PpSyntheticWeights &w,
                                                 luisa::span<const float> crop, float temperature,
                                                 PpOracleRecognizer &out) {
    const auto &rec = cfg.recognizer;
    luisa::vector<float> cur{crop.begin(), crop.end()};
    uint32_t h = rec.input_height, wd = rec.input_width;
    for (uint32_t i = 0; i < rec.backbone.size(); i++) {
        luisa::vector<float> next;
        uint32_t oh = 0, ow = 0;
        if (auto s = pp_oracle_conv_bn(rec.backbone[i], w.rec_backbone[i], cur, h, wd, next, oh, ow); !s.ok()) { return s; }
        cur = std::move(next);
        h = oh;
        wd = ow;
    }
    // Height collapse.
    luisa::vector<float> seq;
    if (auto s = ref_height_collapse(cur, 1u, rec.backbone.back().out_c, h, wd, rec.height_collapse, seq); !s.ok()) { return s; }
    uint32_t collapsed = 0;
    if (auto cc = rec.collapsed_channels(); cc.ok()) { collapsed = *cc.value; }
    const uint32_t frames = wd * (h / rec.height_collapse);
    // SVTR fc1 (bias + GELU-erf). The device uses the GEMM epilogue's portable
    // erf approximation; the oracle uses the SAME declared approximation via
    // ref_gemm's bias_gelu_erf epilogue (both are the declared contract).
    luisa::vector<float> svtr1;
    if (auto s = ref_gemm(seq, w.svtr_fc1_w, w.svtr_fc1_b, frames, rec.svtr_intermediate, collapsed,
                          RefGemmEpilogue::bias_gelu_erf, svtr1); !s.ok()) { return s; }
    luisa::vector<float> svtr2;
    if (auto s = ref_gemm(svtr1, w.svtr_fc2_w, w.svtr_fc2_b, frames, collapsed, rec.svtr_intermediate,
                          RefGemmEpilogue::bias, svtr2); !s.ok()) { return s; }
    if (auto s = ref_gemm(svtr2, w.ctc_w, w.ctc_b, frames, rec.num_classes, collapsed,
                          RefGemmEpilogue::bias, out.logits); !s.ok()) { return s; }
    out.frames = frames;
    if (auto s = ref_ctc_log_softmax(out.logits, frames, rec.num_classes, temperature, out.log_probs); !s.ok()) { return s; }
    return ref_ctc_argmax_probs(out.log_probs, frames, rec.num_classes, out.ids, out.probs);
}

// Host preprocessing mirror of the executor's declared convention (bilinear
// half-pixel resize + (pixel/255 - mean)/std, CHW). Independent re-derivation.
[[nodiscard]] inline luisa::vector<float> pp_oracle_preprocess(luisa::span<const uint8_t> pixels,
                                                               uint32_t sw, uint32_t sh,
                                                               uint32_t dw, uint32_t dh,
                                                               luisa::span<const float> mean,
                                                               luisa::span<const float> std_dev) {
    luisa::vector<float> out(static_cast<size_t>(3) * dw * dh);
    const double scale_x = static_cast<double>(sw) / dw;
    const double scale_y = static_cast<double>(sh) / dh;
    for (uint32_t y = 0; y < dh; y++) {
        for (uint32_t x = 0; x < dw; x++) {
            const double sx = (x + 0.5) * scale_x - 0.5;
            const double sy = (y + 0.5) * scale_y - 0.5;
            const double x0f = std::floor(sx), y0f = std::floor(sy);
            const int32_t x0 = static_cast<int32_t>(x0f), y0 = static_cast<int32_t>(y0f);
            const double fx = sx - x0f, fy = sy - y0f;
            for (uint32_t c = 0; c < 3u; c++) {
                const auto at = [&](int32_t px, int32_t py) -> double {
                    const int32_t cx = std::clamp(px, 0, static_cast<int32_t>(sw) - 1);
                    const int32_t cy = std::clamp(py, 0, static_cast<int32_t>(sh) - 1);
                    return pixels[(static_cast<size_t>(cy) * sw + cx) * 3u + c];
                };
                const double v00 = at(x0, y0), v10 = at(x0 + 1, y0);
                const double v01 = at(x0, y0 + 1), v11 = at(x0 + 1, y0 + 1);
                const double top = v00 + (v10 - v00) * fx;
                const double bottom = v01 + (v11 - v01) * fx;
                const double v = (top + (bottom - top) * fy) / 255.0;
                out[static_cast<size_t>(c) * dw * dh + static_cast<size_t>(y) * dw + x] =
                    static_cast<float>((v - mean[c]) / std_dev[c]);
            }
        }
    }
    return out;
}

// Calibrate the folded DB head into a deterministic LUMINANCE detector for the
// synthetic fixture. The random-weight backbone + BN normalizes away absolute
// brightness, so a random DB head does not separate bright bars from a dark
// background. We therefore set the DB head to read a single luminance channel
// (weight 1 on channel 0, 0 elsewhere) with a bias that places the bar
// interiors above the binarize threshold and the background below it. This is
// a DECLARED synthetic fixture transform applied IDENTICALLY to the folded
// device weights and the oracle's conv weight + BN beta, so parity is
// unaffected. It is fixture design (test data), NOT a model claim.
[[nodiscard]] inline Status pp_calibrate_db_head(const PpOcrv6Config &cfg, PpSyntheticWeights &w,
                                                 PpFoldedWeights &folded) {
    const auto &db = cfg.detector.db_conv;
    if (db.out_c != 1u || db.kh != 1u || db.kw != 1u) {
        return Status::error(StatusCode::internal_error, "calibration: DB head must be 1x1 -> 1ch", "db");
    }
    const uint32_t in_c = db.in_c;
    // Luminance head with GAIN: weight[0, ic, 0, 0] = (ic == 0 ? gain : 0).
    // The box-filter backbone compresses the bar/background feature gap, so a
    // gain spreads the logits enough that a bias shift puts bar interiors
    // above the binarize threshold and the background below it. Fix the DB
    // head BN to identity in BOTH arms (no random sign flip).
    constexpr float kLumGain = 12.0f;
    luisa::vector<float> lum_w(static_cast<size_t>(in_c), 0.0f);
    lum_w[0] = kLumGain;
    w.det_db.bn_mean.assign(db.out_c, 0.0f);
    w.det_db.bn_var.assign(db.out_c, 1.0f);
    w.det_db.bn_gamma.assign(db.out_c, 1.0f);
    w.det_db.bn_beta.assign(db.out_c, 0.0f);
    w.det_db.bias.assign(db.out_c, 0.0f);
    folded.det_db_b.assign(db.out_c, 0.0f);
    // Build the canonical test page and compute the oracle features to find
    // the channel-0 activation range on bar vs background map pixels.
    const uint32_t sw = cfg.detector.input_width * 2u, sh = cfg.detector.input_height * 2u;
    luisa::vector<uint8_t> pixels(static_cast<size_t>(sw) * sh * 3u);
    Lcg lcg;
    lcg.state = 7u;
    for (auto &b : pixels) { b = static_cast<uint8_t>(lcg.next() % 40u); }
    const auto bar = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
        for (uint32_t y = y0; y < y1 && y < sh; y++) {
            for (uint32_t x = x0; x < x1 && x < sw; x++) {
                for (uint32_t c = 0; c < 3u; c++) { pixels[(static_cast<size_t>(y) * sw + x) * 3u + c] = 230u; }
            }
        }
    };
    bar(sw / 8u, sh / 6u, sw / 2u, sh / 6u + sh / 8u);
    bar(sw / 8u, sh / 2u, 7u * sw / 8u, sh / 2u + sh / 8u);
    const float mean[3] = {0.5f, 0.5f, 0.5f}, std_dev[3] = {0.5f, 0.5f, 0.5f};
    auto pre = pp_oracle_preprocess(pixels, sw, sh, cfg.detector.input_width, cfg.detector.input_height,
                                    mean, std_dev);
    // Set the luminance weight in BOTH arms, then run the oracle to get the
    // channel-0 (== logit, since weight is 1 on ch0) distribution.
    for (uint32_t ic = 0; ic < in_c; ic++) {
        folded.det_db_w[ic] = lum_w[ic];
        w.det_db.weight[ic] = lum_w[ic];
    }
    PpOracleDetector det;
    if (auto s = pp_oracle_detector(cfg, w, pre, det); !s.ok()) { return s; }
    const uint32_t mw = det.map_w, mh = det.map_h;
    const uint32_t dw = cfg.detector.input_width, dh = cfg.detector.input_height;
    const float bx0 = dw / 8.0f, bx1a = dw / 2.0f, bx1b = 7.0f * dw / 8.0f;
    const float by0a = dh / 6.0f, by1a = dh / 6.0f + dh / 8.0f;
    const float by0b = dh / 2.0f, by1b = dh / 2.0f + dh / 8.0f;
    const auto in_bar = [&](float x, float y) {
        return (x >= bx0 && x < bx1a && y >= by0a && y < by1a) ||
               (x >= bx0 && x < bx1b && y >= by0b && y < by1b);
    };
    luisa::vector<float> bg_vals, fg_vals;
    for (uint32_t my = 0; my < mh; my++) {
        for (uint32_t mx = 0; mx < mw; mx++) {
            const float cx = (mx + 0.5f) * dw / mw;
            const float cy = (my + 0.5f) * dh / mh;
            const float v = det.logits[static_cast<size_t>(my) * mw + mx];
            if (in_bar(cx, cy)) { fg_vals.push_back(v); } else { bg_vals.push_back(v); }
        }
    }
    if (bg_vals.empty() || fg_vals.empty()) {
        return Status::error(StatusCode::internal_error, "calibration: empty class", "db");
    }
    std::sort(bg_vals.begin(), bg_vals.end());
    std::sort(fg_vals.begin(), fg_vals.end());
    // The strided detector map mixes bar/background at bar EDGES, so the two
    // classes overlap there. Use the class MEDIANS (bulk background vs bulk
    // bar interior), which are cleanly separated for the box-filter fixture.
    const float bg_med = bg_vals[bg_vals.size() / 2u];
    const float fg_med = fg_vals[fg_vals.size() / 2u];
    if (fg_med <= bg_med) {
        return Status::error(StatusCode::internal_error,
                             "calibration: luminance classes not separable", "db");
    }
    // Shift so the class-median midpoint maps to logit(0.5). With the gain
    // spreading the classes, bar interiors land well above the 0.3 binarize
    // threshold and the bulk background well below it.
    const auto logit = [](float p) { return std::log(static_cast<double>(p) / (1.0 - p)); };
    const double mid = 0.5 * (static_cast<double>(bg_med) + fg_med);
    const double shift = logit(0.5f) - mid;
    for (auto &b : folded.det_db_b) { b = static_cast<float>(static_cast<double>(b) + shift); }
    for (auto &b : w.det_db.bn_beta) { b = static_cast<float>(static_cast<double>(b) + shift); }
    return Status::success();
}

// ---------------------------------------------------------------------------
// INDEPENDENT crop-orientation/content oracle (M1).
//
// This does NOT consume the executor's downloaded crop. It re-derives the
// expected line crop from the ORIGINAL source page pixels and the reported
// source-coordinate quad, using its own quad->rect math and its own bilinear
// sampler. A reversed quad winding (the M1 bug) swaps width/height and adds
// 90 degrees, which this oracle detects both geometrically and by content.
// ---------------------------------------------------------------------------
struct PpCropExpectation {
    float cx{0}, cy{0}, rw{0}, rh{0}, angle{0};
    luisa::vector<float> pixels;// CHW normalized crop at the recognizer input
};

// Derive the crop rect from a TL,TR,BR,BL quad expressed in DETECTOR-INPUT
// coordinates, independently of the executor's helper.
[[nodiscard]] inline PpCropExpectation pp_oracle_crop_from_source(
    luisa::span<const output::DbPoint> source_quad, uint32_t src_w, uint32_t src_h,
    luisa::span<const uint8_t> source_pixels, uint32_t det_w, uint32_t det_h,
    uint32_t crop_w, uint32_t crop_h, luisa::span<const float> mean, luisa::span<const float> std_dev) {
    PpCropExpectation e;
    if (source_quad.size() < 4u) { return e; }
    // Map the source quad into detector-input coordinates exactly as the
    // executor's preprocessing does (independent x/y stretch scales; the
    // rotated rect uses the average scale for its extents).
    const float sx = static_cast<float>(det_w) / static_cast<float>(src_w);
    const float sy = static_cast<float>(det_h) / static_cast<float>(src_h);
    const float s_avg = 0.5f * (sx + sy);
    output::DbPoint q[4];
    for (int i = 0; i < 4; i++) { q[i] = output::DbPoint{source_quad[i].x * sx, source_quad[i].y * sy}; }
    for (int i = 0; i < 4; i++) {
        e.cx += q[i].x;
        e.cy += q[i].y;
    }
    e.cx *= 0.25f;
    e.cy *= 0.25f;
    const auto len = [](const output::DbPoint &a, const output::DbPoint &b) {
        const float dx = b.x - a.x, dy = b.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    // Edge 0->1 is the TEXT direction (width); 0->3 is the height. The quad
    // is already in detector-input coordinates, so no further scale applies
    // (the synthetic pages use sx == sy; s_avg is kept for clarity).
    (void)s_avg;
    e.rw = 0.5f * (len(q[0], q[1]) + len(q[3], q[2]));
    e.rh = 0.5f * (len(q[0], q[3]) + len(q[1], q[2]));
    e.angle = std::atan2(q[1].y - q[0].y, q[1].x - q[0].x);
    // Sample the preprocessed detector-input image at the rotated rect.
    auto pre = pp_oracle_preprocess(source_pixels, src_w, src_h, det_w, det_h, mean, std_dev);
    e.pixels.assign(static_cast<size_t>(3) * crop_w * crop_h, 0.0f);
    const double cos_a = std::cos(static_cast<double>(e.angle));
    const double sin_a = std::sin(static_cast<double>(e.angle));
    for (uint32_t c = 0; c < 3u; c++) {
        for (uint32_t py = 0; py < crop_h; py++) {
            for (uint32_t px = 0; px < crop_w; px++) {
                const double u = (px + 0.5) / static_cast<double>(crop_w) - 0.5;
                const double v = (py + 0.5) / static_cast<double>(crop_h) - 0.5;
                const double rx = u * e.rw, ry = v * e.rh;
                const double fx = e.cx + rx * cos_a - ry * sin_a;
                const double fy = e.cy + rx * sin_a + ry * cos_a;
                float value = 0.0f;
                if (fx >= 0.0 && fy >= 0.0 && fx <= det_w - 1.0 && fy <= det_h - 1.0) {
                    const double x0f = std::floor(fx), y0f = std::floor(fy);
                    const int32_t x0 = static_cast<int32_t>(x0f), y0 = static_cast<int32_t>(y0f);
                    const double ax = fx - x0f, ay = fy - y0f;
                    const auto at = [&](int32_t xx, int32_t yy) -> double {
                        const int32_t bx = std::clamp(xx, 0, static_cast<int32_t>(det_w) - 1);
                        const int32_t by = std::clamp(yy, 0, static_cast<int32_t>(det_h) - 1);
                        return pre[static_cast<size_t>(c) * det_w * det_h +
                                   static_cast<size_t>(by) * det_w + static_cast<uint32_t>(bx)];
                    };
                    const double t = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * ax;
                    const double b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * ax;
                    value = static_cast<float>(t + (b - t) * ay);
                }
                e.pixels[static_cast<size_t>(c) * crop_w * crop_h + static_cast<size_t>(py) * crop_w + px] = value;
            }
        }
    }
    return e;
}

// Mean brightness of the left vs right half and top vs bottom half of a CHW
// crop. A correct horizontal-text crop of a bright bar is uniformly bright
// along its width; a 90-degree-rotated crop mixes bar and background.
struct PpCropProfile {
    float left{0}, right{0}, top{0}, bottom{0}, overall{0};
};
[[nodiscard]] inline PpCropProfile pp_crop_profile(luisa::span<const float> chw, uint32_t w, uint32_t h) {
    PpCropProfile p;
    if (chw.size() != static_cast<size_t>(3) * w * h || w == 0u || h == 0u) { return p; }
    double sum = 0.0, l = 0.0, r = 0.0, t = 0.0, b = 0.0;
    uint64_t nl = 0, nr = 0, nt = 0, nb = 0;
    for (uint32_t c = 0; c < 3u; c++) {
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                const double v = chw[static_cast<size_t>(c) * w * h + static_cast<size_t>(y) * w + x];
                sum += v;
                if (x < w / 2u) { l += v; nl++; } else { r += v; nr++; }
                if (y < h / 2u) { t += v; nt++; } else { b += v; nb++; }
            }
        }
    }
    const uint64_t n = static_cast<uint64_t>(3) * w * h;
    p.overall = static_cast<float>(sum / static_cast<double>(n));
    p.left = nl ? static_cast<float>(l / static_cast<double>(nl)) : 0.0f;
    p.right = nr ? static_cast<float>(r / static_cast<double>(nr)) : 0.0f;
    p.top = nt ? static_cast<float>(t / static_cast<double>(nt)) : 0.0f;
    p.bottom = nb ? static_cast<float>(b / static_cast<double>(nb)) : 0.0f;
    return p;
}

// The synthetic character dictionary (15 non-blank classes + blank = 16).
// Mixed Latin/CJK/emoji entries exercise multi-byte lookup.
[[nodiscard]] inline output::CtcDictionary pp_synthetic_dictionary() {
    output::CtcDictionary d;
    d.blank_id = 0;
    d.characters = {
        "a", "b", "c", "d", "e", "f", "g",
        "h", "i", "j", "k", "l",
        "\xe4\xb8\xad",                    // 中
        "\xe6\x96\x87",                    // 文
        "\xf0\x9f\x98\x80",                // 😀
    };
    return d;
}

}// namespace luisa::ocr::test
// OCR engine -- PP-DocLayoutV2/V3 contract + CPU decode (Phase 6).
// Host-only algorithm over device-safe types. See the header for the contract.
#include "ocr/models/pp_doc_layout.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <luisa/core/stl/format.h>

#include "ocr/ops/ops_conv.h"
#include "ocr/output/reading_order.h"

namespace luisa::ocr {

[[nodiscard]] luisa::string_view pp_doc_layout_version_name(PpDocLayoutVersion version) noexcept {
    switch (version) {
        case PpDocLayoutVersion::v2: return "v2";
        case PpDocLayoutVersion::v3: return "v3";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpDocLayoutConfig::validate() const {
    if (in_channels == 0u || backbone.empty()) {
        return Status::error(StatusCode::invalid_argument, "layout detector needs channels and a backbone", "backbone");
    }
    if (input_width == 0u || input_height == 0u) {
        return Status::error(StatusCode::invalid_argument, "layout input size must be nonzero", "input");
    }
    if (classes.empty()) {
        return Status::error(StatusCode::invalid_argument, "layout detector needs at least one class", "classes");
    }
    uint32_t c = in_channels;
    for (size_t i = 0; i < backbone.size(); i++) {
        const auto &s = backbone[i];
        if (auto st = s.validate(); !st.ok()) { return st; }
        if (s.in_c != c) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("layout backbone[{}] channel mismatch", i), "backbone");
        }
        c = s.out_c;
    }
    if (auto s = class_head.validate(); !s.ok()) { return s; }
    if (auto s = box_head.validate(); !s.ok()) { return s; }
    if (class_head.in_c != backbone.back().out_c || box_head.in_c != backbone.back().out_c) {
        return Status::error(StatusCode::invalid_argument, "layout heads must consume the backbone output channels", "heads");
    }
    if (class_head.kh != 1u || class_head.kw != 1u || box_head.kh != 1u || box_head.kw != 1u) {
        return Status::error(StatusCode::invalid_argument, "layout heads must be 1x1 convs", "heads");
    }
    if (class_head.out_c != num_classes()) {
        return Status::error(StatusCode::invalid_argument, "class head output must equal the class count", "class_head");
    }
    if (box_head.out_c != 4u) {
        return Status::error(StatusCode::invalid_argument, "box head must output 4 channels", "box_head");
    }
    auto mh = map_height();
    auto mw = map_width();
    if (!mh.ok() || !mw.ok()) { return Status::error(StatusCode::invalid_argument, "layout degenerate geometry", "backbone"); }
    if (*mh.value == 0u || *mw.value == 0u) {
        return Status::error(StatusCode::invalid_argument, "layout map size is zero", "backbone");
    }
    return Status::success();
}

[[nodiscard]] Result<uint32_t> PpDocLayoutConfig::map_height() const {
    uint32_t h = input_height;
    for (const auto &s : backbone) {
        auto oh = conv2d_out_dim(h, s.kh, s.stride_h, s.pad_h, s.dil_h);
        if (!oh.ok()) { return Result<uint32_t>::failure(oh.status); }
        h = *oh.value;
    }
    return Result<uint32_t>::success(h);
}

[[nodiscard]] Result<uint32_t> PpDocLayoutConfig::map_width() const {
    uint32_t w = input_width;
    for (const auto &s : backbone) {
        auto ow = conv2d_out_dim(w, s.kw, s.stride_w, s.pad_w, s.dil_w);
        if (!ow.ok()) { return Result<uint32_t>::failure(ow.status); }
        w = *ow.value;
    }
    return Result<uint32_t>::success(w);
}

[[nodiscard]] PpDocLayoutConfig pp_doc_layout_synthetic_config(PpDocLayoutVersion version) {
    PpDocLayoutConfig cfg;
    cfg.version = version;
    cfg.in_channels = 3;
    // V2 and V3 are DISTINCT contracts: different input sizes, channel widths,
    // and class lists. Nothing is shared blindly between versions.
    const uint32_t base = version == PpDocLayoutVersion::v2 ? 4u : 6u;
    const uint32_t input = version == PpDocLayoutVersion::v2 ? 64u : 96u;
    cfg.input_width = input;
    cfg.input_height = input;
    // Backbone: 3x3 stride-2 conv, then a 3x3 stride-2 depthwise conv. Stride 4.
    PpConvStage s0;
    s0.in_c = 3;
    s0.out_c = base;
    s0.kh = s0.kw = 3;
    s0.stride_h = s0.stride_w = 2;
    s0.pad_h = s0.pad_w = 1;
    s0.act = 1;// relu
    PpConvStage s1;
    s1.in_c = base;
    s1.out_c = base * 2;
    s1.kh = s1.kw = 3;
    s1.stride_h = s1.stride_w = 2;
    s1.pad_h = s1.pad_w = 1;
    s1.groups = base;// depthwise
    s1.act = 1;
    cfg.backbone = {s0, s1};
    // Class head 1x1 -> num_classes (sigmoid applied host-side after download).
    cfg.class_head.in_c = base * 2;
    cfg.class_head.kh = cfg.class_head.kw = 1;
    cfg.class_head.stride_h = cfg.class_head.stride_w = 1;
    cfg.class_head.act = 0;
    // Box head 1x1 -> 4.
    cfg.box_head.in_c = base * 2;
    cfg.box_head.out_c = 4;
    cfg.box_head.kh = cfg.box_head.kw = 1;
    cfg.box_head.stride_h = cfg.box_head.stride_w = 1;
    cfg.box_head.act = 0;
    if (version == PpDocLayoutVersion::v2) {
        cfg.classes = {
            {"text", LayoutBlockType::text},
            {"title", LayoutBlockType::title},
            {"table", LayoutBlockType::table},
            {"formula", LayoutBlockType::formula},
            {"figure", LayoutBlockType::figure},
        };
    } else {
        cfg.classes = {
            {"text", LayoutBlockType::text},
            {"title", LayoutBlockType::title},
            {"table", LayoutBlockType::table},
            {"formula", LayoutBlockType::formula},
            {"figure", LayoutBlockType::figure},
            {"header", LayoutBlockType::header},
            {"footer", LayoutBlockType::footer},
            {"list", LayoutBlockType::list},
        };
    }
    cfg.class_head.out_c = cfg.num_classes();
    return cfg;
}

// ---------------------------------------------------------------------------
// Tensor names + inventory
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpDocLayoutTensorNames::validate() const {
    if (prefix.empty()) {
        return Status::error(StatusCode::invalid_argument, "tensor prefix must not be empty", "prefix");
    }
    return Status::success();
}

[[nodiscard]] luisa::string PpDocLayoutTensorNames::backbone_w(uint32_t i) const {
    return prefix + "backbone." + luisa::format("{}", i) + ".weight";
}
[[nodiscard]] luisa::string PpDocLayoutTensorNames::backbone_b(uint32_t i) const {
    return prefix + "backbone." + luisa::format("{}", i) + ".bias";
}
[[nodiscard]] luisa::string PpDocLayoutTensorNames::class_head_w() const { return prefix + "class_head.weight"; }
[[nodiscard]] luisa::string PpDocLayoutTensorNames::class_head_b() const { return prefix + "class_head.bias"; }
[[nodiscard]] luisa::string PpDocLayoutTensorNames::box_head_w() const { return prefix + "box_head.weight"; }
[[nodiscard]] luisa::string PpDocLayoutTensorNames::box_head_b() const { return prefix + "box_head.bias"; }

namespace {

[[nodiscard]] ExpectedTensor conv_weight(const luisa::string &name, const PpConvStage &s) {
    ExpectedTensor t;
    t.name = name;
    t.shape = {s.out_c, s.in_c / s.groups, s.kh, s.kw};
    t.layout = TensorLayout::conv_oihw;
    return t;
}
[[nodiscard]] ExpectedTensor bias_tensor(const luisa::string &name, uint32_t c) {
    ExpectedTensor t;
    t.name = name;
    t.shape = {c};
    t.layout = TensorLayout::row_major;
    return t;
}

}// namespace

[[nodiscard]] luisa::vector<ExpectedTensor> pp_doc_layout_expected_tensors(
    const PpDocLayoutConfig &config, const PpDocLayoutTensorNames &names) {
    luisa::vector<ExpectedTensor> out;
    for (uint32_t i = 0; i < config.backbone.size(); i++) {
        out.push_back(conv_weight(names.backbone_w(i), config.backbone[i]));
        out.push_back(bias_tensor(names.backbone_b(i), config.backbone[i].out_c));
    }
    out.push_back(conv_weight(names.class_head_w(), config.class_head));
    out.push_back(bias_tensor(names.class_head_b(), config.class_head.out_c));
    out.push_back(conv_weight(names.box_head_w(), config.box_head));
    out.push_back(bias_tensor(names.box_head_b(), config.box_head.out_c));
    return out;
}

[[nodiscard]] luisa::vector<ExpectedTensor> pp_doc_layout_device_expected_tensors(
    const PpDocLayoutConfig &config, const PpDocLayoutTensorNames &names) {
    auto out = pp_doc_layout_expected_tensors(config, names);
    for (auto &e : out) {
        if (e.layout == TensorLayout::conv_oihw || e.layout == TensorLayout::conv_ohwi) {
            e.layout = TensorLayout::row_major;
        }
    }
    return out;
}

[[nodiscard]] Status pp_doc_layout_validate_tensors(luisa::span<const PackageTensor> tensors,
                                                    luisa::span<const ExpectedTensor> expected) {
    for (const auto &exp : expected) {
        const PackageTensor *found = nullptr;
        for (const auto &t : tensors) {
            if (t.name == exp.name) { found = &t; break; }
        }
        if (found == nullptr) {
            return Status::error(StatusCode::not_found,
                                 luisa::format("required pp_doclayout tensor '{}' is missing", exp.name), exp.name);
        }
        if (found->shape != exp.shape) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("pp_doclayout tensor '{}' has the wrong shape", exp.name), exp.name);
        }
        if (found->layout != exp.layout) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("pp_doclayout tensor '{}' must use layout '{}'", exp.name,
                                               tensor_layout_name(exp.layout)), exp.name);
        }
        if (found->encoding != TensorEncoding::fp32) {
            return Status::error(StatusCode::unsupported,
                                 luisa::format("pp_doclayout tensor '{}' must be fp32 in the Phase 6 baseline", exp.name),
                                 exp.name);
        }
    }
    for (const auto &t : tensors) {
        bool known = false;
        for (const auto &exp : expected) {
            if (exp.name == t.name) { known = true; break; }
        }
        if (!known) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("unexpected pp_doclayout tensor '{}'", t.name), t.name);
        }
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Decode options
// ---------------------------------------------------------------------------

[[nodiscard]] Status PpDocLayoutDecodeOptions::validate() const {
    if (!std::isfinite(score_threshold) || score_threshold < 0.0f || score_threshold >= 1.0f) {
        return Status::error(StatusCode::invalid_argument, "score_threshold must be in [0,1)", "score_threshold");
    }
    if (!std::isfinite(nms_iou_threshold) || nms_iou_threshold < 0.0f || nms_iou_threshold > 1.0f) {
        return Status::error(StatusCode::invalid_argument, "nms_iou_threshold must be in [0,1]", "nms_iou_threshold");
    }
    if (!std::isfinite(min_box_side) || min_box_side <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "min_box_side must be > 0", "min_box_side");
    }
    if (!std::isfinite(box_log_cap) || box_log_cap <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "box_log_cap must be > 0", "box_log_cap");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// NMS + decode
// ---------------------------------------------------------------------------

[[nodiscard]] float pp_doc_layout_iou(const output::DbRect &a, const output::DbRect &b) noexcept {
    const float ix0 = std::max(a.x0, b.x0);
    const float iy0 = std::max(a.y0, b.y0);
    const float ix1 = std::min(a.x1, b.x1);
    const float iy1 = std::min(a.y1, b.y1);
    const float iw = std::max(0.0f, ix1 - ix0);
    const float ih = std::max(0.0f, iy1 - iy0);
    const float inter = iw * ih;
    const float area_a = std::max(0.0f, a.x1 - a.x0) * std::max(0.0f, a.y1 - a.y0);
    const float area_b = std::max(0.0f, b.x1 - b.x0) * std::max(0.0f, b.y1 - b.y0);
    const float uni = area_a + area_b - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

[[nodiscard]] luisa::vector<uint32_t> pp_doc_layout_nms(luisa::span<const PpDocLayoutRegion> regions,
                                                        float iou_threshold, bool class_aware) {
    luisa::vector<uint32_t> order(regions.size());
    for (uint32_t i = 0; i < regions.size(); i++) { order[i] = i; }
    // Score descending; stable tie-break on the original index.
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (regions[a].score != regions[b].score) { return regions[a].score > regions[b].score; }
        return a < b;
    });
    luisa::vector<uint32_t> kept;
    luisa::vector<uint8_t> suppressed(regions.size(), 0u);
    for (auto idx : order) {
        if (suppressed[idx]) { continue; }
        kept.push_back(idx);
        if (iou_threshold <= 0.0f) { continue; }// NMS disabled
        for (auto other : order) {
            if (other == idx || suppressed[other]) { continue; }
            if (class_aware && regions[other].class_index != regions[idx].class_index) { continue; }
            if (pp_doc_layout_iou(regions[idx].bounds, regions[other].bounds) > iou_threshold) {
                suppressed[other] = 1u;
            }
        }
    }
    return kept;
}

[[nodiscard]] Result<luisa::vector<PpDocLayoutRegion>> pp_doc_layout_decode(
    luisa::span<const float> class_probs, luisa::span<const float> box_deltas,
    uint32_t map_width, uint32_t map_height, uint32_t num_classes, float stride,
    const output::DbSourceTransform &transform, const PpDocLayoutConfig &config,
    const PpDocLayoutDecodeOptions &options) {
    using R = Result<luisa::vector<PpDocLayoutRegion>>;
    if (auto s = options.validate(); !s.ok()) { return R::failure(s); }
    if (auto s = transform.validate(); !s.ok()) { return R::failure(s); }
    if (map_width == 0u || map_height == 0u || num_classes == 0u) {
        return R::failure(StatusCode::invalid_argument, "degenerate layout map geometry", "map");
    }
    if (!std::isfinite(stride) || stride <= 0.0f) {
        return R::failure(StatusCode::invalid_argument, "layout stride must be > 0", "stride");
    }
    if (num_classes != config.num_classes()) {
        return R::failure(StatusCode::invalid_argument, "class count does not match the config", "num_classes");
    }
    const uint64_t plane = static_cast<uint64_t>(map_width) * map_height;
    if (class_probs.size() != static_cast<size_t>(num_classes) * plane) {
        return R::failure(StatusCode::invalid_argument, "class_probs size mismatch", "class_probs");
    }
    if (box_deltas.size() != static_cast<size_t>(4u) * plane) {
        return R::failure(StatusCode::invalid_argument, "box_deltas size mismatch", "box_deltas");
    }
    // An all-background map yields zero regions (checked below by threshold).
    luisa::vector<PpDocLayoutRegion> candidates;
    const float cap = options.box_log_cap;
    for (uint32_t cy = 0; cy < map_height; cy++) {
        for (uint32_t cx = 0; cx < map_width; cx++) {
            const uint64_t cell = static_cast<uint64_t>(cy) * map_width + cx;
            // Best class at this cell (lowest index wins ties).
            uint32_t best_class = 0u;
            float best_score = -1.0f;
            for (uint32_t k = 0; k < num_classes; k++) {
                const float p = class_probs[static_cast<size_t>(k) * plane + cell];
                if (!std::isfinite(p)) {
                    return R::failure(StatusCode::invalid_argument, "non-finite class probability", "class_probs");
                }
                if (p > best_score) { best_score = p; best_class = k; }
            }
            if (best_score < options.score_threshold) { continue; }
            const float dx = box_deltas[static_cast<size_t>(0u) * plane + cell];
            const float dy = box_deltas[static_cast<size_t>(1u) * plane + cell];
            const float lw = box_deltas[static_cast<size_t>(2u) * plane + cell];
            const float lh = box_deltas[static_cast<size_t>(3u) * plane + cell];
            if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(lw) || !std::isfinite(lh)) {
                return R::failure(StatusCode::invalid_argument, "non-finite box delta", "box_deltas");
            }
            // Center in MAP units; size = stride * exp(clamp(log)).
            const float center_x = (static_cast<float>(cx) + 0.5f + dx) * stride;
            const float center_y = (static_cast<float>(cy) + 0.5f + dy) * stride;
            const float w = stride * std::exp(std::clamp(lw, -cap, cap));
            const float h = stride * std::exp(std::clamp(lh, -cap, cap));
            // Map -> source transform.
            const auto tl = transform.to_source(center_x - 0.5f * w, center_y - 0.5f * h);
            const auto br = transform.to_source(center_x + 0.5f * w, center_y + 0.5f * h);
            PpDocLayoutRegion region;
            region.class_index = best_class;
            region.block_type = config.classes[best_class].block_type;
            region.score = best_score;
            region.bounds = output::DbRect{tl.x, tl.y, br.x, br.y};
            if (region.bounds.x1 - region.bounds.x0 < options.min_box_side ||
                region.bounds.y1 - region.bounds.y0 < options.min_box_side) {
                continue;// degenerate/tiny box dropped
            }
            region.polygon = {
                output::DbPoint{region.bounds.x0, region.bounds.y0},
                output::DbPoint{region.bounds.x1, region.bounds.y0},
                output::DbPoint{region.bounds.x1, region.bounds.y1},
                output::DbPoint{region.bounds.x0, region.bounds.y1},
            };
            candidates.push_back(std::move(region));
        }
    }
    // NMS / class-aware NMS.
    auto kept = pp_doc_layout_nms(candidates, options.nms_iou_threshold, options.class_aware_nms);
    if (options.max_regions != 0u && kept.size() > options.max_regions) {
        kept.resize(options.max_regions);
    }
    luisa::vector<PpDocLayoutRegion> regions;
    regions.reserve(kept.size());
    for (auto idx : kept) { regions.push_back(std::move(candidates[idx])); }
    // Stable reading order.
    return pp_doc_layout_order_regions(regions);
}

[[nodiscard]] Result<luisa::vector<PpDocLayoutRegion>> pp_doc_layout_order_regions(
    luisa::span<const PpDocLayoutRegion> regions) {
    using R = Result<luisa::vector<PpDocLayoutRegion>>;
    if (regions.empty()) { return R::success({}); }
    // Reuse the Phase 5 multi-column reading-order algorithm over plain lines.
    // The original region index is carried in `text` (as a decimal string) so
    // the ordered lines map back to regions unambiguously even when two regions
    // share identical geometry (order_lines is stable on the input index).
    luisa::vector<output::ReadingOrderLine> lines;
    lines.reserve(regions.size());
    for (uint32_t i = 0; i < regions.size(); i++) {
        const auto &region = regions[i];
        output::ReadingOrderLine line;
        line.page_index = 0;
        line.x0 = region.bounds.x0;
        line.y0 = region.bounds.y0;
        line.x1 = region.bounds.x1;
        line.y1 = region.bounds.y1;
        line.text = luisa::format("{}", i);
        lines.push_back(std::move(line));
    }
    auto ordered = output::order_lines(lines);
    if (!ordered.ok()) { return R::failure(ordered.status); }
    luisa::vector<PpDocLayoutRegion> out;
    out.reserve(regions.size());
    for (const auto &line : *ordered.value) {
        // Parse the carried index (decimal, no sign). order_lines never alters
        // text, so this is the exact original region index.
        uint32_t index = 0u;
        bool valid = !line.text.empty();
        for (char c : line.text) {
            if (c < '0' || c > '9') { valid = false; break; }
            index = index * 10u + static_cast<uint32_t>(c - '0');
        }
        if (!valid || index >= regions.size()) {
            return R::failure(StatusCode::internal_error, "reading-order index mapping failed", "reading_order");
        }
        PpDocLayoutRegion region = regions[index];
        region.reading_order = line.reading_order;
        out.push_back(std::move(region));
    }
    return R::success(std::move(out));
}

// ---------------------------------------------------------------------------
// Real-artifact gate
// ---------------------------------------------------------------------------

[[nodiscard]] Status pp_doc_layout_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                          luisa::string_view config_fingerprint) {
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "no official PP-DocLayout checkpoint/config has been qualified; "
                         "real artifacts fail closed (synthetic path only)",
                         "pp_doclayout");
}

}// namespace luisa::ocr
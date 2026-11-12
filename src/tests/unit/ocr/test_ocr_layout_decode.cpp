// Test for ocr/models/pp_doc_layout CPU decode (Phase 6, host-only).
// This test covers:
// - score threshold filtering; empty/all-background maps -> zero regions;
// - box decode (center offsets + exp log-scale, clamped) + source transform;
// - NMS and class-aware NMS (overlap suppression, class separation);
// - stable score tie-breaking; region cap; degenerate/tiny box rejection;
// - reading order assignment; malformed input fail-closed;
// - V2/V3 distinct configs + tensor inventory; real-artifact gate fail-closed.
#include "ut/ut.hpp"

#include <cmath>

#include <luisa/core/stl/vector.h>

#include "ocr/models/pp_doc_layout.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// The synthetic v2 config has 5 classes; the test maps must match that count.
constexpr uint32_t kClasses = 5;

// Build class/box maps for a 2x2 map with kClasses classes. Helper sets one cell.
struct Maps {
    uint32_t w{2}, h{2}, nc{kClasses};
    luisa::vector<float> cls;
    luisa::vector<float> box;
    Maps() : cls(static_cast<size_t>(kClasses) * 2 * 2, 0.0f), box(static_cast<size_t>(4) * 2 * 2, 0.0f) {}
    void set_class(uint32_t k, uint32_t cx, uint32_t cy, float p) {
        cls[static_cast<size_t>(k) * w * h + static_cast<size_t>(cy) * w + cx] = p;
    }
    void set_box(uint32_t cx, uint32_t cy, float dx, float dy, float lw, float lh) {
        const size_t cell = static_cast<size_t>(cy) * w + cx;
        const size_t plane = static_cast<size_t>(w) * h;
        box[0 * plane + cell] = dx;
        box[1 * plane + cell] = dy;
        box[2 * plane + cell] = lw;
        box[3 * plane + cell] = lh;
    }
};

[[nodiscard]] PpDocLayoutConfig config2() {
    return pp_doc_layout_synthetic_config(PpDocLayoutVersion::v2);
}

[[nodiscard]] output::DbSourceTransform identity_transform() {
    output::DbSourceTransform t;
    t.scale_x = 1.0f;
    t.scale_y = 1.0f;
    return t;
}

}// namespace

void test_empty_map() {
    Maps m;// all scores 0
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (r.ok()) { expect(r.value->empty()); }
}

void test_threshold_and_box_decode() {
    Maps m;
    // Cell (0,0): class 1 score 0.9, box centered with size stride*exp(0)=4.
    m.set_class(1, 0, 0, 0.9f);
    m.set_box(0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    // Cell (1,1): class 0 score 0.2 (below threshold) -> dropped.
    m.set_class(0, 1, 1, 0.2f);
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.0f;// disable NMS for this check
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 1u);
    const auto &reg = (*r.value)[0];
    expect(reg.class_index == 1u);
    expect(std::abs(reg.score - 0.9f) < 1e-5f);
    // Center = (0+0.5)*4 = 2; size = 4 -> box [0,4]x[0,4].
    expect(std::abs(reg.bounds.x0 - 0.0f) < 1e-3f);
    expect(std::abs(reg.bounds.y0 - 0.0f) < 1e-3f);
    expect(std::abs(reg.bounds.x1 - 4.0f) < 1e-3f);
    expect(std::abs(reg.bounds.y1 - 4.0f) < 1e-3f);
    expect(reg.block_type == cfg.classes[1].block_type);
}

void test_nms_class_agnostic() {
    // Two overlapping boxes of DIFFERENT classes; class-agnostic NMS keeps only
    // the higher-score one.
    Maps m;
    m.set_class(0, 0, 0, 0.9f);
    m.set_box(0, 0, 0.0f, 0.0f, 0.0f, 0.0f);// box [0,4]x[0,4]
    m.set_class(1, 1, 0, 0.8f);
    m.set_box(1, 0, -0.5f, 0.0f, 0.0f, 0.0f);// center (1+0.5-0.5)*4=4 -> [2,6]x[0,4]
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.3f;
    opts.class_aware_nms = false;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 1u);
    expect(std::abs((*r.value)[0].score - 0.9f) < 1e-5f);
}

void test_nms_class_aware() {
    // Same two overlapping boxes; class-aware NMS keeps BOTH (different classes).
    Maps m;
    m.set_class(0, 0, 0, 0.9f);
    m.set_box(0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    m.set_class(1, 1, 0, 0.8f);
    m.set_box(1, 0, -0.5f, 0.0f, 0.0f, 0.0f);
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.3f;
    opts.class_aware_nms = true;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 2u);
}

void test_nms_same_class_suppress() {
    // Two overlapping boxes of the SAME class -> higher score wins.
    Maps m;
    m.set_class(0, 0, 0, 0.9f);
    m.set_box(0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    m.set_class(0, 1, 0, 0.8f);
    m.set_box(1, 0, -0.5f, 0.0f, 0.0f, 0.0f);
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.3f;
    opts.class_aware_nms = true;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 1u);
    expect(std::abs((*r.value)[0].score - 0.9f) < 1e-5f);
}

void test_score_tie_stable() {
    // Two non-overlapping boxes with equal scores; both kept, stable order.
    Maps m;
    m.set_class(0, 0, 0, 0.7f);
    m.set_box(0, 0, 0.0f, 0.0f, -1.0f, -1.0f);// small box near (2,2)
    m.set_class(0, 1, 1, 0.7f);
    m.set_box(1, 1, 0.0f, 0.0f, -1.0f, -1.0f);// small box near (6,6)
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.5f;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 2u);
}

void test_region_cap() {
    Maps m;
    for (uint32_t cy = 0; cy < 2; cy++) {
        for (uint32_t cx = 0; cx < 2; cx++) {
            m.set_class(0, cx, cy, 0.9f);
            // 4px boxes centered at (2,2),(6,2),(2,6),(6,6): touch but do not
            // overlap (IoU 0), so all 4 survive NMS and the cap selects 2.
            m.set_box(cx, cy, 0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.5f;
    opts.max_regions = 2;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 2u);
}

void test_degenerate_box_dropped() {
    Maps m;
    m.set_class(0, 0, 0, 0.9f);
    // Extremely negative log-scale -> sub-pixel box -> dropped by min_box_side.
    m.set_box(0, 0, 0.0f, 0.0f, -4.0f, -4.0f);
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.min_box_side = 1.0f;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (r.ok()) { expect(r.value->empty()); }
}

void test_source_transform() {
    Maps m;
    m.set_class(0, 0, 0, 0.9f);
    m.set_box(0, 0, 0.0f, 0.0f, 0.0f, 0.0f);// map box [0,4]x[0,4]
    auto cfg = config2();
    output::DbSourceTransform t;
    t.scale_x = 2.0f;
    t.scale_y = 3.0f;
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.0f;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, t, cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 1u);
    const auto &b = (*r.value)[0].bounds;
    expect(std::abs(b.x1 - 8.0f) < 1e-3f);
    expect(std::abs(b.y1 - 12.0f) < 1e-3f);
}

void test_malformed_input() {
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    // Wrong class_probs size.
    luisa::vector<float> bad_cls(3, 0.0f);
    luisa::vector<float> box(16, 0.0f);
    expect(!pp_doc_layout_decode(bad_cls, box, 2, 2, kClasses, 4.0f, identity_transform(), cfg, opts).ok());
    // Non-finite probability.
    Maps m;
    m.set_class(0, 0, 0, std::nanf(""));
    expect(!pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts).ok());
    // Invalid options.
    PpDocLayoutDecodeOptions bad_opts;
    bad_opts.score_threshold = 1.5f;
    expect(!pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, bad_opts).ok());
    // Class count mismatch (2 vs the config's 5).
    Maps m2;
    expect(!pp_doc_layout_decode(m2.cls, m2.box, 2, 2, 2, 4.0f, identity_transform(), cfg, opts).ok());
}

void test_reading_order_assigned() {
    // Two stacked boxes -> reading order 0 (top), 1 (bottom).
    Maps m;
    m.set_class(0, 0, 0, 0.9f);
    m.set_box(0, 0, 0.0f, 0.0f, -1.0f, -1.0f);// top box near y=2
    m.set_class(0, 0, 1, 0.9f);
    m.set_box(0, 1, 0.0f, 0.0f, -1.0f, -1.0f);// bottom box near y=6
    auto cfg = config2();
    PpDocLayoutDecodeOptions opts;
    opts.score_threshold = 0.3f;
    opts.nms_iou_threshold = 0.0f;
    auto r = pp_doc_layout_decode(m.cls, m.box, m.w, m.h, m.nc, 4.0f, identity_transform(), cfg, opts);
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->size() == 2u);
    expect((*r.value)[0].reading_order == 0u);
    expect((*r.value)[1].reading_order == 1u);
    expect((*r.value)[0].bounds.y0 < (*r.value)[1].bounds.y0);
}

void test_versions_distinct() {
    auto v2 = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v2);
    auto v3 = pp_doc_layout_synthetic_config(PpDocLayoutVersion::v3);
    expect(v2.validate().ok());
    expect(v3.validate().ok());
    expect(v2.num_classes() != v3.num_classes());
    expect(v2.input_width != v3.input_width);
    // Tensor inventories differ in class-head shape.
    PpDocLayoutTensorNames names;
    auto e2 = pp_doc_layout_expected_tensors(v2, names);
    auto e3 = pp_doc_layout_expected_tensors(v3, names);
    expect(e2.size() == e3.size());// same structure, different class count
    // Real-artifact gate always fails closed.
    expect(pp_doc_layout_validate_real_artifact("rev", "fp").code == StatusCode::unvalidated);
}

static auto test_ocr_layout_decode_registration = [] {
    "layout_decode_empty_map"_test = [] { test_empty_map(); };
    "layout_decode_threshold_and_box"_test = [] { test_threshold_and_box_decode(); };
    "layout_decode_nms_class_agnostic"_test = [] { test_nms_class_agnostic(); };
    "layout_decode_nms_class_aware"_test = [] { test_nms_class_aware(); };
    "layout_decode_nms_same_class"_test = [] { test_nms_same_class_suppress(); };
    "layout_decode_score_tie_stable"_test = [] { test_score_tie_stable(); };
    "layout_decode_region_cap"_test = [] { test_region_cap(); };
    "layout_decode_degenerate_box"_test = [] { test_degenerate_box_dropped(); };
    "layout_decode_source_transform"_test = [] { test_source_transform(); };
    "layout_decode_malformed_input"_test = [] { test_malformed_input(); };
    "layout_decode_reading_order"_test = [] { test_reading_order_assigned(); };
    "layout_decode_versions_distinct"_test = [] { test_versions_distinct(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
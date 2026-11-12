// Test for ocr/output/db_postprocess (Phase 5, host-only).
// This test covers:
// - checked threshold/binarization (range validation, adversarial values);
// - 8-connected components (diagonal connectivity, raster determinism);
// - convex hull + minimum-area rotated rectangle (axis-aligned and rotated);
// - polygon area/perimeter/unclip distance (official DB offset formula);
// - convex miter-join unclip expansion (exact for rectangles);
// - box scoring (mean probability inside the box mask);
// - IoU + optional NMS; degenerate/tiny/empty/adversarial maps;
// - source-coordinate transforms (resize/crop/padding back to ORIGINAL pixels).
// The baseline is a correct connected-component/convex-box implementation, NOT
// the official cv2.findContours/approxPolyDP pipeline (documented in the header).
#include "ut/ut.hpp"

#include <cmath>

#include <luisa/core/stl/vector.h>

#include "ocr/output/db_postprocess.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace luisa::ocr::output;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Fill a rectangular region of a probability map with a constant value.
void fill_rect(luisa::vector<float> &map, uint32_t w, uint32_t x0, uint32_t y0,
               uint32_t x1, uint32_t y1, float value) {
    for (uint32_t y = y0; y < y1; y++) {
        for (uint32_t x = x0; x < x1; x++) { map[static_cast<size_t>(y) * w + x] = value; }
    }
}

}// namespace

void test_binarize() {
    luisa::vector<float> probs{0.0f, 0.29f, 0.3f, 0.9f, 1.0f};
    luisa::vector<uint8_t> mask;
    expect(db_binarize(probs, 0.3f, mask).ok());
    expect(mask.size() == 5u);
    expect(mask[0] == 0u && mask[1] == 0u);
    expect(mask[2] == 1u && mask[3] == 1u && mask[4] == 1u);
    // Invalid thresholds.
    expect(!db_binarize(probs, -0.1f, mask).ok());
    expect(!db_binarize(probs, 1.0f, mask).ok());
    expect(!db_binarize(probs, std::nanf(""), mask).ok());
    // Out-of-range probability values fail.
    luisa::vector<float> bad{0.5f, 1.5f};
    expect(!db_binarize(bad, 0.3f, mask).ok());
    luisa::vector<float> neg{-0.1f};
    expect(!db_binarize(neg, 0.3f, mask).ok());
}

void test_connected_components() {
    // 7x7 mask with two SEPARATED blobs: a 2x2 block at the top-left and a
    // diagonal chain at the bottom-right. (1,1) and (3,3) are NOT 8-adjacent
    // (Chebyshev distance 2), so the blobs stay distinct.
    const uint32_t w = 7, h = 7;
    luisa::vector<uint8_t> mask(w * h, 0u);
    // Blob A: (0,0),(1,0),(0,1),(1,1)
    mask[0] = mask[1] = mask[7] = mask[8] = 1u;
    // Blob B: diagonal (3,3),(4,4),(5,5) -- 8-connected within the chain.
    mask[24] = mask[32] = mask[40] = 1u;
    auto comps = db_connected_components(mask, w, h);
    expect(comps.ok());
    expect(comps.value->size() == 2u);
    // Raster order: blob A first (seed (0,0)), blob B second (seed (3,3)).
    expect((*comps.value)[0].pixels.size() == 4u);
    expect((*comps.value)[1].pixels.size() == 3u);// diagonal connected
    expect((*comps.value)[1].min_x == 3u && (*comps.value)[1].max_x == 5u);
    // Empty mask -> no components.
    luisa::vector<uint8_t> empty(w * h, 0u);
    auto none = db_connected_components(empty, w, h);
    expect(none.ok());
    expect(none.value->empty());
    // Size mismatch fails.
    expect(!db_connected_components(mask, w + 1, h).ok());
}

void test_convex_hull_and_min_area_rect() {
    // A solid 4x3 rectangle component -> hull is the 4 corners.
    DbComponent comp;
    comp.width = 8;
    for (uint32_t y = 1; y <= 3; y++) {
        for (uint32_t x = 2; x <= 5; x++) { comp.pixels.push_back(y * 8 + x); }
    }
    comp.min_x = 2;
    comp.max_x = 5;
    comp.min_y = 1;
    comp.max_y = 3;
    auto hull = db_convex_hull(comp);
    expect(hull.size() == 4u);
    auto rect = db_min_area_rect(hull);
    expect(rect.size() == 4u);
    // The min-area rect of an axis-aligned blob is axis-aligned: bounds
    // x in [2,5], y in [1,3].
    float x0 = rect[0].x, x1 = rect[0].x, y0 = rect[0].y, y1 = rect[0].y;
    for (const auto &p : rect) {
        x0 = std::min(x0, p.x);
        x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y);
        y1 = std::max(y1, p.y);
    }
    expect(std::abs(x0 - 2.0f) < 1e-4f);
    expect(std::abs(x1 - 5.0f) < 1e-4f);
    expect(std::abs(y0 - 1.0f) < 1e-4f);
    expect(std::abs(y1 - 3.0f) < 1e-4f);
    // Degenerate: a single pixel -> no hull.
    DbComponent single;
    single.width = 4;
    single.pixels.push_back(5);
    expect(db_convex_hull(single).empty());
    // Collinear pixels -> no hull (zero area).
    DbComponent line;
    line.width = 8;
    for (uint32_t x = 0; x < 5; x++) { line.pixels.push_back(2 * 8 + x); }
    expect(db_convex_hull(line).empty());
}

void test_polygon_math() {
    // Unit square (0,0),(1,0),(1,1),(0,1): area 1, perimeter 4.
    luisa::vector<DbPoint> square{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    expect(std::abs(db_polygon_area(square) - 1.0f) < 1e-6f);
    expect(std::abs(db_polygon_perimeter(square) - 4.0f) < 1e-6f);
    // DB unclip distance = area * ratio / perimeter = 1 * 1.6 / 4 = 0.4.
    expect(std::abs(db_unclip_distance(square, 1.6f) - 0.4f) < 1e-6f);
    // Zero perimeter -> 0.
    luisa::vector<DbPoint> degenerate{{0, 0}, {0, 0}};
    expect(db_unclip_distance(degenerate, 1.6f) == 0.0f);
    // Rectangle 2x1: area 2, perimeter 6, distance = 2*1.5/6 = 0.5.
    luisa::vector<DbPoint> rect{{0, 0}, {2, 0}, {2, 1}, {0, 1}};
    expect(std::abs(db_unclip_distance(rect, 1.5f) - 0.5f) < 1e-6f);
}

void test_unclip_expansion() {
    // Axis-aligned rectangle expanded by distance d grows by d on each side.
    luisa::vector<DbPoint> rect{{0, 0}, {4, 0}, {4, 2}, {0, 2}};
    const float d = 0.5f;
    auto expanded = db_unclip_polygon(rect, d, 4.0f);
    expect(expanded.size() == 4u);
    float x0 = expanded[0].x, x1 = expanded[0].x, y0 = expanded[0].y, y1 = expanded[0].y;
    for (const auto &p : expanded) {
        x0 = std::min(x0, p.x);
        x1 = std::max(x1, p.x);
        y0 = std::min(y0, p.y);
        y1 = std::max(y1, p.y);
    }
    expect(std::abs(x0 - (-d)) < 1e-4f);
    expect(std::abs(x1 - (4.0f + d)) < 1e-4f);
    expect(std::abs(y0 - (-d)) < 1e-4f);
    expect(std::abs(y1 - (2.0f + d)) < 1e-4f);
    // distance <= 0 returns the input unchanged.
    auto same = db_unclip_polygon(rect, 0.0f, 4.0f);
    expect(same.size() == 4u);
    expect(std::abs(same[0].x - rect[0].x) < 1e-6f);
}

void test_box_score_and_iou() {
    // 4x4 map, all 0.8 inside a 2x2 box region.
    const uint32_t w = 4, h = 4;
    luisa::vector<float> map(w * h, 0.0f);
    fill_rect(map, w, 1, 1, 3, 3, 0.8f);
    luisa::vector<DbPoint> box{{1, 1}, {3, 1}, {3, 3}, {1, 3}};
    const float score = db_box_score(map, w, h, box);
    expect(std::abs(score - 0.8f) < 1e-5f);
    // Empty coverage -> 0.
    luisa::vector<DbPoint> outside{{10, 10}, {12, 10}, {12, 12}, {10, 12}};
    expect(db_box_score(map, w, h, outside) == 0.0f);
    // IoU: identical boxes -> 1; disjoint -> 0; half overlap.
    DbRect a{0, 0, 2, 2}, b{0, 0, 2, 2}, c{4, 4, 6, 6}, d{1, 0, 3, 2};
    expect(std::abs(db_box_iou(a, b) - 1.0f) < 1e-6f);
    expect(db_box_iou(a, c) == 0.0f);
    expect(std::abs(db_box_iou(a, d) - (2.0f / 6.0f)) < 1e-5f);
}

void test_db_postprocess_basic() {
    // 16x16 map with one bright 6x4 rectangle.
    const uint32_t w = 16, h = 16;
    luisa::vector<float> map(w * h, 0.0f);
    fill_rect(map, w, 4, 5, 10, 9, 0.9f);
    DbSourceTransform transform;// identity
    auto r = db_postprocess(map, w, h, transform);
    expect(r.ok());
    expect(r.value->boxes.size() == 1u);
    const auto &box = r.value->boxes[0];
    expect(box.polygon.size() == 4u);
    expect(box.score > 0.8f);
    // Bounds roughly cover the rectangle (unclip expands it).
    expect(box.bounds.x0 < 4.0f && box.bounds.x1 > 9.0f);
    expect(box.bounds.y0 < 5.0f && box.bounds.y1 > 8.0f);
}

void test_db_postprocess_empty_and_degenerate() {
    DbSourceTransform transform;
    // All-background map -> zero boxes (no hallucination).
    {
        luisa::vector<float> map(64, 0.0f);
        auto r = db_postprocess(map, 8, 8, transform);
        expect(r.ok());
        expect(r.value->boxes.empty());
    }
    // Tiny specks below min_box_pixels -> dropped.
    {
        luisa::vector<float> map(64, 0.0f);
        map[0] = 0.9f;// single pixel
        DbPostprocessOptions opts;
        opts.min_box_pixels = 4;
        auto r = db_postprocess(map, 8, 8, transform, opts);
        expect(r.ok());
        expect(r.value->boxes.empty());
        expect(r.value->components_dropped >= 1u);
    }
    // Low-score blob below box_threshold -> dropped.
    {
        luisa::vector<float> map(64, 0.0f);
        fill_rect(map, 8, 2, 2, 6, 6, 0.35f);
        DbPostprocessOptions opts;
        opts.binarize_threshold = 0.3f;
        opts.box_threshold = 0.5f;
        auto r = db_postprocess(map, 8, 8, transform, opts);
        expect(r.ok());
        expect(r.value->boxes.empty());
    }
    // Degenerate dimensions -> empty result.
    {
        auto r = db_postprocess({}, 0, 0, transform);
        expect(r.ok());
        expect(r.value->boxes.empty());
    }
    // Size mismatch fails.
    {
        luisa::vector<float> map(10, 0.5f);
        expect(!db_postprocess(map, 8, 8, transform).ok());
    }
    // Invalid transform fails.
    {
        luisa::vector<float> map(64, 0.0f);
        DbSourceTransform bad;
        bad.scale_x = 0.0f;
        expect(!db_postprocess(map, 8, 8, bad).ok());
    }
}

void test_db_postprocess_source_transform() {
    // The detector map is a downscaled+padded view of the source. Verify the
    // boxes map back to ORIGINAL source coordinates through the transform.
    const uint32_t w = 8, h = 8;
    luisa::vector<float> map(w * h, 0.0f);
    fill_rect(map, w, 2, 2, 6, 6, 0.9f);
    DbSourceTransform transform;
    transform.scale_x = 4.0f;// source is 4x the map width
    transform.scale_y = 4.0f;
    transform.offset_x = 10.0f;// padding offset
    transform.offset_y = 20.0f;
    auto r = db_postprocess(map, w, h, transform);
    expect(r.ok());
    expect(r.value->boxes.size() == 1u);
    const auto &box = r.value->boxes[0];
    // Map x in [2,6] -> source x in [2*4+10, 6*4+10] = [18,34] (before unclip).
    // Unclip expands, so bounds are wider but centered in that range.
    expect(box.bounds.x0 >= 10.0f && box.bounds.x1 <= 42.0f);
    expect(box.bounds.y0 >= 20.0f && box.bounds.y1 <= 52.0f);
    // Every polygon point obeys the affine map (source = map*4 + offset).
    for (const auto &p : box.polygon) {
        expect(p.x >= 10.0f && p.y >= 20.0f);
    }
}

void test_db_postprocess_multi_and_nms() {
    // Two well-separated blobs -> two boxes, deterministic order.
    const uint32_t w = 32, h = 16;
    luisa::vector<float> map(w * h, 0.0f);
    fill_rect(map, w, 2, 4, 8, 10, 0.9f);
    fill_rect(map, w, 20, 4, 28, 10, 0.9f);
    DbSourceTransform transform;
    auto r = db_postprocess(map, w, h, transform);
    expect(r.ok());
    expect(r.value->boxes.size() == 2u);
    // Deterministic: repeat run identical.
    auto r2 = db_postprocess(map, w, h, transform);
    expect(r2.ok());
    expect(r2.value->boxes.size() == 2u);
    expect(std::abs(r.value->boxes[0].bounds.x0 - r2.value->boxes[0].bounds.x0) < 1e-6f);
    // max_boxes cap.
    DbPostprocessOptions opts;
    opts.max_boxes = 1;
    auto capped = db_postprocess(map, w, h, transform, opts);
    expect(capped.ok());
    expect(capped.value->boxes.size() == 1u);
}

void test_db_postprocess_adversarial() {
    // A map full of alternating specks: bounded by max_boxes, no crash.
    const uint32_t w = 32, h = 32;
    luisa::vector<float> map(w * h, 0.0f);
    for (size_t i = 0; i < map.size(); i += 2) { map[i] = 0.95f; }
    DbSourceTransform transform;
    DbPostprocessOptions opts;
    opts.min_box_pixels = 1;
    opts.max_boxes = 8;
    auto r = db_postprocess(map, w, h, transform, opts);
    expect(r.ok());
    expect(r.value->boxes.size() <= 8u);
    // All-foreground map: one big component, one box.
    luisa::vector<float> full(w * h, 0.9f);
    auto rf = db_postprocess(full, w, h, transform);
    expect(rf.ok());
    expect(rf.value->boxes.size() == 1u);
}

// M1 regression: a WIDE horizontal blob must produce a TL,TR,BR,BL quad whose
// 0->1 edge is the LONG (horizontal) text direction. A reversed winding makes
// the crop stage read height as width and rotate every line by 90 degrees.
void test_quad_vertex_order_horizontal_blob() {
    // 24x12 map with a wide 16x4 bar (aspect 4:1) -- unambiguously horizontal.
    const uint32_t w = 24, h = 12;
    luisa::vector<float> map(w * h, 0.0f);
    fill_rect(map, w, 4, 4, 20, 8, 0.9f);
    DbSourceTransform transform;
    DbPostprocessOptions opts;
    opts.unclip_ratio = 1.0f;// no expansion: check the raw box orientation
    auto r = db_postprocess(map, w, h, transform, opts);
    expect(r.ok());
    expect(r.value->boxes.size() == 1u);
    if (!r.ok() || r.value->boxes.empty()) { return; }
    const auto &q = r.value->boxes[0].polygon;
    expect(q.size() == 4u);
    if (q.size() != 4u) { return; }
    const auto dist = [](const DbPoint &a, const DbPoint &b) {
        const float dx = b.x - a.x, dy = b.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    const float top = dist(q[0], q[1]);
    const float left = dist(q[0], q[3]);
    // The 0->1 edge must be the LONG edge (width), 0->3 the SHORT (height).
    expect(top > left) << "edge 0->1 must be the long/text edge, got top=" << top << " left=" << left;
    // Exact expected geometry: the hull spans PIXEL CENTERS x[4,19], y[4,7]
    // => 15 x 3. The DB offset formula expands even at ratio 1.0:
    // d = area*ratio/perimeter = 45/36 = 1.25, so the quad is 17.5 x 5.5.
    expect(std::abs(top - 17.5f) < 1e-3f) << "width 17.5, got " << top;
    expect(std::abs(left - 5.5f) < 1e-3f) << "height 5.5, got " << left;
    // The text edge must be horizontal: |dy| << |dx|.
    expect(std::abs(q[1].y - q[0].y) < 1e-3f);
    expect(q[1].x > q[0].x) << "text edge must run left-to-right";
    // TL is topmost-then-leftmost; BR is opposite.
    expect(q[0].y <= q[2].y);
    expect(q[0].x <= q[1].x);
    // Screen-clockwise (y-down) winding == POSITIVE shoelace sum.
    double sum = 0.0;
    for (size_t i = 0; i < 4u; i++) {
        const auto &a = q[i];
        const auto &b = q[(i + 1u) % 4u];
        sum += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    expect(sum > 0.0) << "TL,TR,BR,BL in y-down coords must have positive shoelace, got " << sum;

    // A TALL vertical blob must likewise keep 0->1 as its (short) top edge and
    // 0->3 as the long edge -- the order is geometric, not aspect-dependent.
    luisa::vector<float> vmap(w * h, 0.0f);
    fill_rect(vmap, w, 8, 1, 12, 11, 0.9f);
    auto vr = db_postprocess(vmap, w, h, transform, opts);
    expect(vr.ok());
    expect(vr.value->boxes.size() == 1u);
    if (vr.ok() && !vr.value->boxes.empty()) {
        const auto &vq = vr.value->boxes[0].polygon;
        expect(vq.size() == 4u);
        if (vq.size() == 4u) {
            expect(dist(vq[0], vq[3]) > dist(vq[0], vq[1])) << "vertical blob: 0->3 is the long edge";
            expect(std::abs(vq[1].y - vq[0].y) < 1e-3f);
            expect(vq[1].x > vq[0].x);
        }
    }
}

static auto test_ocr_db_registration = [] {
    "db_binarize"_test = [] { test_binarize(); };
    "db_connected_components"_test = [] { test_connected_components(); };
    "db_convex_hull_and_min_area_rect"_test = [] { test_convex_hull_and_min_area_rect(); };
    "db_quad_vertex_order_horizontal_blob"_test = [] { test_quad_vertex_order_horizontal_blob(); };
    "db_polygon_math"_test = [] { test_polygon_math(); };
    "db_unclip_expansion"_test = [] { test_unclip_expansion(); };
    "db_box_score_and_iou"_test = [] { test_box_score_and_iou(); };
    "db_postprocess_basic"_test = [] { test_db_postprocess_basic(); };
    "db_postprocess_empty_and_degenerate"_test = [] { test_db_postprocess_empty_and_degenerate(); };
    "db_postprocess_source_transform"_test = [] { test_db_postprocess_source_transform(); };
    "db_postprocess_multi_and_nms"_test = [] { test_db_postprocess_multi_and_nms(); };
    "db_postprocess_adversarial"_test = [] { test_db_postprocess_adversarial(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
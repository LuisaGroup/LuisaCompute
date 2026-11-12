// OCR engine -- DB detector post-processing implementation (Phase 5).
// Host-only CPU, no OpenCV. See the header for the exact documented baseline.
#include "ocr/output/db_postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <luisa/core/stl/format.h>

namespace luisa::ocr::output {

namespace {

// Signed shoelace area (positive = CCW in standard math coordinates).
[[nodiscard]] double signed_area(luisa::span<const DbPoint> poly) noexcept {
    if (poly.size() < 3u) { return 0.0; }
    double sum = 0.0;
    for (size_t i = 0; i < poly.size(); i++) {
        const auto &a = poly[i];
        const auto &b = poly[(i + 1u) % poly.size()];
        sum += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return 0.5 * sum;
}

// Order 4 rectangle corners as TL,TR,BR,BL in y-DOWN image coordinates
// (clockwise on screen), starting from the topmost-then-leftmost corner.
// Deterministic. The crop stage reads edge 0->1 as the text direction, so
// this order is load-bearing: reversing it swaps width/height and rotates
// every line crop by 90 degrees.
[[nodiscard]] luisa::vector<DbPoint> order_quad(luisa::vector<DbPoint> corners) {
    if (corners.size() != 4u) { return corners; }
    size_t start = 0;
    for (size_t i = 1; i < corners.size(); i++) {
        if (corners[i].y < corners[start].y ||
            (corners[i].y == corners[start].y && corners[i].x < corners[start].x)) {
            start = i;
        }
    }
    luisa::vector<DbPoint> rotated;
    rotated.reserve(4);
    for (size_t k = 0; k < 4; k++) { rotated.push_back(corners[(start + k) % 4]); }
    // Winding: the shoelace sum below uses standard math orientation, but
    // these are y-DOWN image coordinates. A TL,TR,BR,BL quad in image space
    // (e.g. (2,1),(5,1),(5,3),(2,3)) has a POSITIVE shoelace sum (+6), i.e.
    // it reads as CCW in math coords while being CLOCKWISE on screen. So the
    // desired TL,TR,BR,BL order is signed_area > 0; a NEGATIVE sum means the
    // corners run the other way and must be reversed.
    if (signed_area(rotated) < 0.0) { std::swap(rotated[1], rotated[3]); }
    return rotated;
}

[[nodiscard]] DbRect polygon_bounds(luisa::span<const DbPoint> poly) noexcept {
    DbRect box;
    if (poly.empty()) { return box; }
    box.x0 = box.x1 = poly[0].x;
    box.y0 = box.y1 = poly[0].y;
    for (const auto &p : poly) {
        box.x0 = std::min(box.x0, p.x);
        box.y0 = std::min(box.y0, p.y);
        box.x1 = std::max(box.x1, p.x);
        box.y1 = std::max(box.y1, p.y);
    }
    return box;
}

// Even-odd point-in-polygon test (double arithmetic).
[[nodiscard]] bool point_in_polygon(double px, double py, luisa::span<const DbPoint> poly) noexcept {
    bool inside = false;
    const size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const double xi = poly[i].x, yi = poly[i].y;
        const double xj = poly[j].x, yj = poly[j].y;
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

}// namespace

[[nodiscard]] Status DbSourceTransform::validate() const {
    if (!std::isfinite(scale_x) || !std::isfinite(scale_y) ||
        !std::isfinite(offset_x) || !std::isfinite(offset_y)) {
        return Status::error(StatusCode::invalid_argument, "db source transform must be finite", "transform");
    }
    if (scale_x <= 0.0f || scale_y <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "db source transform scale must be positive", "transform");
    }
    return Status::success();
}

[[nodiscard]] Status DbPostprocessOptions::validate() const {
    if (!std::isfinite(binarize_threshold) || binarize_threshold < 0.0f || binarize_threshold >= 1.0f) {
        return Status::error(StatusCode::invalid_argument, "binarize_threshold must be in [0,1)", "binarize_threshold");
    }
    if (!std::isfinite(box_threshold) || box_threshold < 0.0f || box_threshold > 1.0f) {
        return Status::error(StatusCode::invalid_argument, "box_threshold must be in [0,1]", "box_threshold");
    }
    if (!std::isfinite(unclip_ratio) || unclip_ratio < 1.0f) {
        return Status::error(StatusCode::invalid_argument, "unclip_ratio must be >= 1", "unclip_ratio");
    }
    if (min_box_pixels == 0u) {
        return Status::error(StatusCode::invalid_argument, "min_box_pixels must be >= 1", "min_box_pixels");
    }
    if (!std::isfinite(min_box_side) || min_box_side <= 0.0f) {
        return Status::error(StatusCode::invalid_argument, "min_box_side must be > 0", "min_box_side");
    }
    if (!std::isfinite(nms_iou_threshold) || nms_iou_threshold < 0.0f || nms_iou_threshold > 1.0f) {
        return Status::error(StatusCode::invalid_argument, "nms_iou_threshold must be in [0,1]", "nms_iou_threshold");
    }
    if (!std::isfinite(max_miter_ratio) || max_miter_ratio < 1.0f) {
        return Status::error(StatusCode::invalid_argument, "max_miter_ratio must be >= 1", "max_miter_ratio");
    }
    return Status::success();
}

[[nodiscard]] Status db_binarize(luisa::span<const float> probabilities, float threshold,
                                 luisa::vector<uint8_t> &mask) {
    if (!std::isfinite(threshold) || threshold < 0.0f || threshold >= 1.0f) {
        return Status::error(StatusCode::invalid_argument, "binarize threshold must be in [0,1)", "threshold");
    }
    mask.resize(probabilities.size());
    for (size_t i = 0; i < probabilities.size(); i++) {
        const float p = probabilities[i];
        if (!std::isfinite(p) || p < 0.0f || p > 1.0f) {
            return Status::error(StatusCode::invalid_argument,
                                 luisa::format("db probability[{}] = {} outside [0,1]", i, p), "probabilities");
        }
        mask[i] = p >= threshold ? 1u : 0u;
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::vector<DbComponent>> db_connected_components(luisa::span<const uint8_t> mask,
                                                                         uint32_t width, uint32_t height) {
    using R = Result<luisa::vector<DbComponent>>;
    if (width == 0u || height == 0u) { return R::success({}); }
    if (mask.size() != static_cast<size_t>(width) * height) {
        return R::failure(StatusCode::invalid_argument, "db mask span size mismatch", "mask");
    }
    luisa::vector<uint8_t> visited(mask.size(), 0u);
    luisa::vector<DbComponent> components;
    luisa::vector<uint32_t> stack;
    for (uint32_t y0 = 0; y0 < height; y0++) {
        for (uint32_t x0 = 0; x0 < width; x0++) {
            const uint32_t seed = y0 * width + x0;
            if (mask[seed] == 0u || visited[seed] != 0u) { continue; }
            DbComponent comp;
            comp.width = width;
            comp.min_x = comp.max_x = x0;
            comp.min_y = comp.max_y = y0;
            stack.clear();
            stack.push_back(seed);
            visited[seed] = 1u;
            while (!stack.empty()) {
                const uint32_t idx = stack.back();
                stack.pop_back();
                comp.pixels.push_back(idx);
                const uint32_t x = idx % width;
                const uint32_t y = idx / width;
                comp.min_x = std::min(comp.min_x, x);
                comp.max_x = std::max(comp.max_x, x);
                comp.min_y = std::min(comp.min_y, y);
                comp.max_y = std::max(comp.max_y, y);
                for (int32_t dy = -1; dy <= 1; dy++) {
                    for (int32_t dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) { continue; }
                        const int32_t nx = static_cast<int32_t>(x) + dx;
                        const int32_t ny = static_cast<int32_t>(y) + dy;
                        if (nx < 0 || ny < 0 || nx >= static_cast<int32_t>(width) ||
                            ny >= static_cast<int32_t>(height)) {
                            continue;
                        }
                        const uint32_t nidx = static_cast<uint32_t>(ny) * width + static_cast<uint32_t>(nx);
                        if (mask[nidx] != 0u && visited[nidx] == 0u) {
                            visited[nidx] = 1u;
                            stack.push_back(nidx);
                        }
                    }
                }
            }
            std::sort(comp.pixels.begin(), comp.pixels.end());
            components.push_back(std::move(comp));
        }
    }
    return R::success(std::move(components));
}

[[nodiscard]] luisa::vector<DbPoint> db_convex_hull(const DbComponent &component) {
    if (component.width == 0u || component.pixels.empty()) { return {}; }
    // Distinct pixel coordinates, sorted lexicographically (x, then y).
    luisa::vector<DbPoint> pts;
    pts.reserve(component.pixels.size());
    for (auto idx : component.pixels) {
        pts.push_back(DbPoint{static_cast<float>(idx % component.width),
                                  static_cast<float>(idx / component.width)});
    }
    std::sort(pts.begin(), pts.end(), [](const DbPoint &a, const DbPoint &b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(std::unique(pts.begin(), pts.end(), [](const DbPoint &a, const DbPoint &b) {
                  return a.x == b.x && a.y == b.y;
              }),
              pts.end());
    if (pts.size() < 3u) { return {}; }
    const auto cross = [](const DbPoint &o, const DbPoint &a, const DbPoint &b) -> double {
        return (static_cast<double>(a.x) - o.x) * (static_cast<double>(b.y) - o.y) -
               (static_cast<double>(a.y) - o.y) * (static_cast<double>(b.x) - o.x);
    };
    // Monotone chain (standard formulation). Collinear points removed
    // (<= 0 pops), so the result is the strict convex hull in CCW order.
    luisa::vector<DbPoint> lower, upper;
    for (const auto &p : pts) {
        while (lower.size() >= 2u &&
               cross(lower[lower.size() - 2u], lower[lower.size() - 1u], p) <= 0.0) {
            lower.pop_back();
        }
        lower.push_back(p);
    }
    for (size_t i = pts.size(); i-- > 0;) {
        const auto &p = pts[i];
        while (upper.size() >= 2u &&
               cross(upper[upper.size() - 2u], upper[upper.size() - 1u], p) <= 0.0) {
            upper.pop_back();
        }
        upper.push_back(p);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    if (lower.size() < 3u) { return {}; }
    return lower;
}

[[nodiscard]] float db_polygon_area(luisa::span<const DbPoint> polygon) noexcept {
    return static_cast<float>(std::abs(signed_area(polygon)));
}

[[nodiscard]] float db_polygon_perimeter(luisa::span<const DbPoint> polygon) noexcept {
    if (polygon.size() < 2u) { return 0.0f; }
    double sum = 0.0;
    for (size_t i = 0; i < polygon.size(); i++) {
        const auto &a = polygon[i];
        const auto &b = polygon[(i + 1u) % polygon.size()];
        const double dx = static_cast<double>(b.x) - a.x;
        const double dy = static_cast<double>(b.y) - a.y;
        sum += std::sqrt(dx * dx + dy * dy);
    }
    return static_cast<float>(sum);
}

[[nodiscard]] luisa::vector<DbPoint> db_min_area_rect(luisa::span<const DbPoint> hull) {
    if (hull.size() < 3u) { return {}; }
    if (db_polygon_area(hull) <= 0.0f) { return {}; }
    // Rotating calipers: for every hull edge direction, project all hull
    // points onto the edge axis and its normal; keep the minimum-area box.
    double best_area = std::numeric_limits<double>::infinity();
    luisa::vector<DbPoint> best;
    const size_t n = hull.size();
    for (size_t i = 0; i < n; i++) {
        const auto &a = hull[i];
        const auto &b = hull[(i + 1u) % n];
        double ex = static_cast<double>(b.x) - a.x;
        double ey = static_cast<double>(b.y) - a.y;
        const double len = std::sqrt(ex * ex + ey * ey);
        if (len <= 0.0) { continue; }
        ex /= len;
        ey /= len;
        // Normal (left of edge direction).
        const double nx = -ey, ny = ex;
        double min_u = std::numeric_limits<double>::infinity(), max_u = -std::numeric_limits<double>::infinity();
        double min_v = std::numeric_limits<double>::infinity(), max_v = -std::numeric_limits<double>::infinity();
        for (const auto &p : hull) {
            const double dx = static_cast<double>(p.x) - a.x;
            const double dy = static_cast<double>(p.y) - a.y;
            const double u = dx * ex + dy * ey;
            const double v = dx * nx + dy * ny;
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        const double area = (max_u - min_u) * (max_v - min_v);
        if (area < best_area) {
            best_area = area;
            best.clear();
            const auto corner = [&](double u, double v) {
                return DbPoint{static_cast<float>(a.x + u * ex + v * nx),
                                   static_cast<float>(a.y + u * ey + v * ny)};
            };
            best.push_back(corner(min_u, min_v));
            best.push_back(corner(max_u, min_v));
            best.push_back(corner(max_u, max_v));
            best.push_back(corner(min_u, max_v));
        }
    }
    if (best.empty() || best_area <= 0.0) { return {}; }
    return order_quad(std::move(best));
}

[[nodiscard]] float db_unclip_distance(luisa::span<const DbPoint> polygon, float ratio) noexcept {
    const float perimeter = db_polygon_perimeter(polygon);
    if (perimeter <= 0.0f || !std::isfinite(ratio) || ratio <= 0.0f) { return 0.0f; }
    const float area = db_polygon_area(polygon);
    return area * ratio / perimeter;
}

[[nodiscard]] luisa::vector<DbPoint> db_unclip_polygon(luisa::span<const DbPoint> polygon,
                                                           float distance, float max_miter_ratio) {
    luisa::vector<DbPoint> input{polygon.begin(), polygon.end()};
    if (polygon.size() < 3u || !std::isfinite(distance) || distance <= 0.0f) { return input; }
    if (!std::isfinite(max_miter_ratio) || max_miter_ratio < 1.0f) { max_miter_ratio = 4.0f; }
    const double area = signed_area(polygon);
    if (area == 0.0) { return input; }
    // Outward normal sign: for CCW (area>0, math coords) the outward normal
    // of edge (a->b) is (dy, -dx)/len; for CW it is (-dy, dx)/len.
    const double sign = area > 0.0 ? 1.0 : -1.0;
    const size_t n = polygon.size();
    // Offset each edge line outward by `distance`; the new vertex i is the
    // intersection of offset edges (i-1, i) and (i, i+1).
    struct Line {
        double px, py;// a point on the offset line
        double dx, dy;// unit direction
    };
    luisa::vector<Line> lines(n);
    for (size_t i = 0; i < n; i++) {
        const auto &a = polygon[i];
        const auto &b = polygon[(i + 1u) % n];
        double dx = static_cast<double>(b.x) - a.x;
        double dy = static_cast<double>(b.y) - a.y;
        const double len = std::sqrt(dx * dx + dy * dy);
        if (len <= 0.0) { return input; }
        dx /= len;
        dy /= len;
        const double nx = sign * dy, ny = -sign * dx;
        lines[i] = Line{static_cast<double>(a.x) + nx * distance,
                        static_cast<double>(a.y) + ny * distance, dx, dy};
    }
    luisa::vector<DbPoint> out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const auto &l0 = lines[(i + n - 1u) % n];
        const auto &l1 = lines[i];
        // Intersect l0 (p0 + t*d0) with l1 (p1 + s*d1).
        const double denom = l0.dx * l1.dy - l0.dy * l1.dx;
        if (std::abs(denom) < 1e-12) {
            // Parallel edges: keep the offset point of the incoming edge.
            out.push_back(DbPoint{static_cast<float>(l1.px), static_cast<float>(l1.py)});
            continue;
        }
        const double t = ((l1.px - l0.px) * l1.dy - (l1.py - l0.py) * l1.dx) / denom;
        double ix = l0.px + t * l0.dx;
        double iy = l0.py + t * l0.dy;
        // Miter cap: bound the corner displacement from the original vertex.
        const double mx = ix - polygon[i].x;
        const double my = iy - polygon[i].y;
        const double miter = std::sqrt(mx * mx + my * my);
        const double cap = static_cast<double>(distance) * max_miter_ratio;
        if (miter > cap && miter > 0.0) {
            const double s = cap / miter;
            ix = polygon[i].x + mx * s;
            iy = polygon[i].y + my * s;
        }
        out.push_back(DbPoint{static_cast<float>(ix), static_cast<float>(iy)});
    }
    return out;
}

[[nodiscard]] float db_box_score(luisa::span<const float> probabilities, uint32_t width, uint32_t height,
                                 luisa::span<const DbPoint> polygon) noexcept {
    if (polygon.size() < 3u || width == 0u || height == 0u) { return 0.0f; }
    if (probabilities.size() != static_cast<size_t>(width) * height) { return 0.0f; }
    const DbRect bounds = polygon_bounds(polygon);
    const int32_t x0 = std::max(0, static_cast<int32_t>(std::floor(bounds.x0)));
    const int32_t y0 = std::max(0, static_cast<int32_t>(std::floor(bounds.y0)));
    const int32_t x1 = std::min(static_cast<int32_t>(width) - 1, static_cast<int32_t>(std::ceil(bounds.x1)));
    const int32_t y1 = std::min(static_cast<int32_t>(height) - 1, static_cast<int32_t>(std::ceil(bounds.y1)));
    if (x1 < x0 || y1 < y0) { return 0.0f; }
    double sum = 0.0;
    uint64_t count = 0;
    for (int32_t y = y0; y <= y1; y++) {
        for (int32_t x = x0; x <= x1; x++) {
            // Pixel-center sampling of the box mask (official box_score_fast
            // rasterizes the polygon; centers avoid boundary ambiguity).
            if (point_in_polygon(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5, polygon)) {
                sum += probabilities[static_cast<size_t>(y) * width + static_cast<uint32_t>(x)];
                count++;
            }
        }
    }
    return count == 0u ? 0.0f : static_cast<float>(sum / static_cast<double>(count));
}

[[nodiscard]] float db_box_iou(const DbRect &a, const DbRect &b) noexcept {
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

[[nodiscard]] Result<DbPostprocessResult> db_postprocess(luisa::span<const float> probabilities,
                                                         uint32_t width, uint32_t height,
                                                         const DbSourceTransform &transform,
                                                         const DbPostprocessOptions &options) {
    using R = Result<DbPostprocessResult>;
    if (auto s = options.validate(); !s.ok()) { return R::failure(s); }
    if (auto s = transform.validate(); !s.ok()) { return R::failure(s); }
    if (width == 0u || height == 0u) { return R::success(DbPostprocessResult{}); }
    if (probabilities.size() != static_cast<size_t>(width) * height) {
        return R::failure(StatusCode::invalid_argument, "db probability span size mismatch", "probabilities");
    }
    luisa::vector<uint8_t> mask;
    if (auto s = db_binarize(probabilities, options.binarize_threshold, mask); !s.ok()) {
        return R::failure(s);
    }
    auto components = db_connected_components(mask, width, height);
    if (!components.ok()) { return R::failure(components.status); }

    DbPostprocessResult result;
    result.components_found = static_cast<uint32_t>(components.value->size());
    struct Candidate {
        DbBox box;
        float area{0.0f};
    };
    luisa::vector<Candidate> candidates;
    for (const auto &comp : *components.value) {
        if (comp.pixels.size() < options.min_box_pixels) {
            result.components_dropped++;
            continue;
        }
        const auto hull = db_convex_hull(comp);
        const auto rect = db_min_area_rect(hull);
        if (rect.size() != 4u) {
            result.components_dropped++;
            continue;
        }
        // Score on the ORIGINAL (unclipped) box, official box_score_fast.
        const float score = db_box_score(probabilities, width, height, rect);
        if (score < options.box_threshold) {
            result.components_dropped++;
            continue;
        }
        // Unclip/expansion in MAP coordinates.
        const float distance = db_unclip_distance(rect, options.unclip_ratio);
        auto expanded = db_unclip_polygon(rect, distance, options.max_miter_ratio);
        expanded = order_quad(std::move(expanded));
        const DbRect map_bounds = polygon_bounds(expanded);
        const float side_w = map_bounds.x1 - map_bounds.x0;
        const float side_h = map_bounds.y1 - map_bounds.y0;
        if (side_w < options.min_box_side || side_h < options.min_box_side) {
            result.components_dropped++;
            continue;
        }
        DbBox box;
        box.score = score;
        box.polygon.reserve(4);
        for (const auto &p : expanded) {
            box.polygon.push_back(transform.to_source(p.x, p.y));
        }
        box.bounds = polygon_bounds(box.polygon);
        candidates.push_back(Candidate{std::move(box), side_w * side_h});
    }

    // Deterministic order: descending area, then score, then top-left
    // position (stable tie-breaking for identical geometry).
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        if (a.area != b.area) { return a.area > b.area; }
        if (a.box.score != b.box.score) { return a.box.score > b.box.score; }
        if (a.box.bounds.y0 != b.box.bounds.y0) { return a.box.bounds.y0 < b.box.bounds.y0; }
        return a.box.bounds.x0 < b.box.bounds.x0;
    });

    // Optional axis-aligned NMS (off by default).
    luisa::vector<bool> suppressed(candidates.size(), false);
    if (options.nms_iou_threshold > 0.0f) {
        for (size_t i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) { continue; }
            for (size_t j = i + 1u; j < candidates.size(); j++) {
                if (suppressed[j]) { continue; }
                if (db_box_iou(candidates[i].box.bounds, candidates[j].box.bounds) > options.nms_iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) { continue; }
        if (options.max_boxes != 0u && result.boxes.size() >= options.max_boxes) { break; }
        result.boxes.push_back(std::move(candidates[i].box));
    }
    return R::success(std::move(result));
}

}// namespace luisa::ocr::output
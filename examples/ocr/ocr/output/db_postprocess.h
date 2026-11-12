#pragma once
// OCR engine -- DB detector post-processing (Phase 5). Host-only CPU.
// No OpenCV dependency.
//
// WHAT IS IMPLEMENTED (exact, documented baseline):
//   1. Checked threshold/binarization of the DB probability map
//      (probabilities in [0,1]; threshold in [0,1)).
//   2. Connected components on the binary mask with 8-connectivity
//      (iterative flood fill, no recursion; adversarial-size safe).
//   3. Per-component box extraction as the CONVEX HULL of the component's
//      pixels (monotone chain), reduced to the minimum-area enclosing
//      ROTATED RECTANGLE via rotating calipers on the hull edges. This is a
//      correct convex-box baseline; it is NOT the official DB contour
//      pipeline (cv2.findContours + approxPolyDP + minAreaRect on the
//      contour). For convex blobs the boxes agree; for concave blobs the
//      convex hull can be larger than the official contour box. This
//      difference is documented, not hidden (see MODELS.md).
//   4. Polygon unclip/expansion with the official DB offset formula
//      distance = area * unclip_ratio / perimeter (shoelace area, polygon
//      perimeter), implemented as an exact MITER-JOIN polygon offset for
//      convex polygons (each edge moved outward by `distance`, corners
//      intersected). For the convex rectangles produced by step 3 this is
//      exact; a miter-length cap keeps adversarial spikes bounded.
//   5. Box scoring: the mean DB probability inside the ORIGINAL (unclipped)
//      component box, matching the official box_score_fast semantics
//      (mean over the axis-aligned crop of the box mask). Boxes below
//      box_threshold are dropped.
//   6. Optional axis-aligned IoU NMS (suppression threshold configurable;
//      off by default -- the official DB pipeline does not run NMS).
//   7. Degenerate/tiny/empty handling: components smaller than
//      min_box_pixels or with a degenerate (< 3-vertex or zero-area) hull
//      are dropped; an empty/all-background map yields ZERO boxes.
//   8. Source-coordinate transform: detector-map coordinates are mapped
//      back through the detector preprocessing (resize + pad) to ORIGINAL
//      page pixels via an explicit affine (scale + offset per axis).
//
// WHAT IS *NOT* IMPLEMENTED (unqualified vs official DB, see MODELS.md):
//   - cv2.findContours/approxPolyDP contour simplification and the official
//     minAreaRect-on-contour box (we use the convex hull of the mask).
//   - The official pyclipper/Clipper polygon offset library (we use the
//     exact convex miter offset instead).
//   - Official unclip on concave polygons (miter offset is exact only for
//     convex inputs; the baseline only produces convex boxes).
#include <cstdint>

#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr::output {

// Plain geometry types local to the DB module. These are deliberately NOT
// api.h's SourcePoint/SourceBox: api.h pulls model_recipe.h, whose host-only
// AttentionMaskKind clashes with the ops enum in device translation units.
// Keeping DB geometry self-contained lets both host and device TUs include
// this header. The executor maps DbPoint/DbBox to the public API types.
struct DbPoint {
    float x{0.0f};
    float y{0.0f};
};
struct DbRect {
    float x0{0.0f};
    float y0{0.0f};
    float x1{0.0f};
    float y1{0.0f};
    [[nodiscard]] bool empty() const noexcept { return x1 <= x0 || y1 <= y0; }
};

// Affine map from detector probability-map pixels to ORIGINAL source pixels:
// source = map * scale + offset (per axis). The detector preprocessing
// (aspect resize + pad) composes into this single transform.
struct DbSourceTransform {
    float scale_x{1.0f};
    float scale_y{1.0f};
    float offset_x{0.0f};
    float offset_y{0.0f};

    [[nodiscard]] DbPoint to_source(float x, float y) const noexcept {
        return DbPoint{x * scale_x + offset_x, y * scale_y + offset_y};
    }
    [[nodiscard]] Status validate() const;
};

struct DbPostprocessOptions {
    // Binarization threshold for the probability map, in [0, 1).
    float binarize_threshold{0.3f};
    // Minimum mean probability inside the unclipped box, in [0, 1].
    float box_threshold{0.5f};
    // DB unclip expansion ratio (>= 1). 1.0 disables expansion.
    float unclip_ratio{1.6f};
    // Components with fewer pixels than this are dropped (>= 1).
    uint32_t min_box_pixels{4};
    // Minimum box side length in MAP pixels after unclip (> 0).
    float min_box_side{1.0f};
    // Optional axis-aligned NMS IoU threshold in (0, 1]; 0 disables NMS.
    float nms_iou_threshold{0.0f};
    // Maximum number of boxes returned (0 = unlimited). Adversarial maps
    // with thousands of specks are bounded by this cap (largest-area first).
    uint32_t max_boxes{0};
    // Miter-length cap for the convex polygon offset, as a multiple of the
    // offset distance (>= 1). Bounds adversarial near-degenerate corners.
    float max_miter_ratio{4.0f};

    [[nodiscard]] Status validate() const;
};

// One detected text box in ORIGINAL source coordinates.
struct DbBox {
    // Quad polygon (4 points, clockwise in image coordinates: top-left,
    // top-right, bottom-right, bottom-left of the rotated rectangle).
    luisa::vector<DbPoint> polygon;
    // Axis-aligned bounds of the polygon (source coordinates).
    DbRect bounds;
    // Mean DB probability inside the ORIGINAL (unclipped) box mask.
    float score{0.0f};
};

struct DbPostprocessResult {
    luisa::vector<DbBox> boxes;
    // Diagnostics for dropped/degenerate components (non-fatal).
    uint32_t components_found{0};
    uint32_t components_dropped{0};
};

// Run the full DB postprocess on a probability map [height, width] (row
// major, values in [0,1]). Returns boxes in ORIGINAL source coordinates via
// `transform`. An empty or all-background map returns a valid empty result.
[[nodiscard]] Result<DbPostprocessResult> db_postprocess(luisa::span<const float> probabilities,
                                                         uint32_t width, uint32_t height,
                                                         const DbSourceTransform &transform,
                                                         const DbPostprocessOptions &options = {});

// ---------------------------------------------------------------------------
// Exposed primitives (unit-tested independently).
// ---------------------------------------------------------------------------

// Binarize: mask[i] = prob[i] >= threshold ? 1 : 0. Validates sizes/range.
[[nodiscard]] Status db_binarize(luisa::span<const float> probabilities, float threshold,
                                 luisa::vector<uint8_t> &mask);

// One 8-connected component: its pixel indices (row-major over `width`) and
// its bounding box. `width` is the mask row stride the indices refer to.
struct DbComponent {
    luisa::vector<uint32_t> pixels;
    uint32_t width{0};
    uint32_t min_x{0}, min_y{0}, max_x{0}, max_y{0};
};

// Label all 8-connected foreground components of a [h,w] binary mask.
// Components are returned in deterministic raster order of their first
// (topmost-then-leftmost) pixel. Empty mask -> empty list.
[[nodiscard]] Result<luisa::vector<DbComponent>> db_connected_components(luisa::span<const uint8_t> mask,
                                                                         uint32_t width, uint32_t height);

// Convex hull (monotone chain) of a component's pixels, in map coordinates.
// Returns hull vertices in counter-clockwise order (math convention, y up);
// callers treating y-down image coordinates see clockwise order. Duplicate
// collinear points are removed. < 3 distinct points -> empty hull.
[[nodiscard]] luisa::vector<DbPoint> db_convex_hull(const DbComponent &component);

// Minimum-area enclosing rotated rectangle of a convex polygon (rotating
// calipers over hull edges). Returns 4 corners (clockwise in y-down image
// coordinates). Degenerate input (< 3 vertices / zero area) -> empty.
[[nodiscard]] luisa::vector<DbPoint> db_min_area_rect(luisa::span<const DbPoint> hull);

// Shoelace polygon area (absolute) and perimeter.
[[nodiscard]] float db_polygon_area(luisa::span<const DbPoint> polygon) noexcept;
[[nodiscard]] float db_polygon_perimeter(luisa::span<const DbPoint> polygon) noexcept;

// DB unclip distance: area * ratio / perimeter. Zero perimeter -> 0.
[[nodiscard]] float db_unclip_distance(luisa::span<const DbPoint> polygon, float ratio) noexcept;

// Exact convex miter-join offset: move every edge of a convex polygon
// outward by `distance` and intersect adjacent edges. Input must be convex
// and ordered (either winding; the outward side is resolved from the signed
// area). Returns the expanded polygon with the same winding. Degenerate
// input or distance <= 0 returns the input unchanged. `max_miter_ratio`
// caps corner spikes (falls back to a bevel at the cap).
[[nodiscard]] luisa::vector<DbPoint> db_unclip_polygon(luisa::span<const DbPoint> polygon,
                                                           float distance, float max_miter_ratio);

// Mean probability inside the axis-aligned crop of a box mask (official
// box_score_fast semantics): rasterize the polygon into its integer bounds
// and average the probability map over the covered pixels. Empty coverage
// -> 0. The polygon is in MAP coordinates here.
[[nodiscard]] float db_box_score(luisa::span<const float> probabilities, uint32_t width, uint32_t height,
                                 luisa::span<const DbPoint> polygon) noexcept;

// Axis-aligned IoU of two boxes.
[[nodiscard]] float db_box_iou(const DbRect &a, const DbRect &b) noexcept;

}// namespace luisa::ocr::output
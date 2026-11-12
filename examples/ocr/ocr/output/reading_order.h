#pragma once
// OCR engine -- deterministic reading order + canonical joining (Phase 5).
// Host-only ALGORITHM, but the header is device-safe: it uses plain local
// geometry (ReadingOrderLine) instead of api.h's OcrLine, because api.h pulls
// model_recipe.h whose host-only AttentionMaskKind clashes with the ops enum
// in device translation units. The executor maps ReadingOrderLine to/from the
// public OcrLine type.
//
// Ordering contract (deterministic, documented):
//   * Lines are grouped into COLUMNS by horizontal overlap of their source
//     boxes. Two lines share a column when their x-intervals overlap by at
//     least `column_overlap_ratio` of the narrower line's width. Columns are
//     ordered left-to-right by their left edge.
//   * Within a column, lines are ordered top-to-bottom by their vertical
//     center, with a row-tolerance band: lines whose vertical centers differ
//     by less than `row_tolerance * median_line_height` are treated as the
//     SAME row and ordered left-to-right inside that row (handles rotated /
//     skewed / very wide lines whose centers jitter).
//   * Stable tie-breaking: equal keys fall back to (top, left, then the
//     original input index), so the order is a total order and repeat runs
//     are identical.
//   * Source page indices are preserved: lines are ordered WITHIN each page
//     and pages are emitted in ascending page_index order (cross-page
//     composition is a later phase; this module never reorders pages).
//
// Confidence is NEVER synthesized here: it is carried through unchanged from
// the CTC/reference source (has_confidence / confidence).
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr::output {

// Plain reading-order input line. Geometry mirrors the public SourceRegion
// (box + optional polygon + page index); text/confidence are carried through.
struct ReadingOrderLine {
    luisa::string text;
    uint32_t page_index{0};
    // Axis-aligned source box (source pixels). Empty box -> use polygon.
    float x0{0.0f}, y0{0.0f}, x1{0.0f}, y1{0.0f};
    // Optional polygon (source pixels); used when the box is empty.
    luisa::vector<std::pair<float, float>> polygon;
    bool has_confidence{false};
    float confidence{0.0f};
    // CTC kept-token count carried through unchanged (0 = unknown/not a CTC
    // line). Ordering never synthesizes or alters it.
    uint32_t ctc_token_count{0};
    // Assigned by order_lines (0-based, dense, in reading order).
    uint32_t reading_order{0};

    [[nodiscard]] bool box_empty() const noexcept { return x1 <= x0 || y1 <= y0; }
};

struct ReadingOrderOptions {
    // Minimum horizontal overlap (fraction of the narrower width) for two
    // lines to share a column, in [0, 1]. 0 = any overlap; 1 = full containment.
    float column_overlap_ratio{0.3f};
    // Row-tolerance band as a fraction of the median line height (>= 0).
    // Lines within this vertical-center distance are one row.
    float row_tolerance{0.5f};

    [[nodiscard]] Status validate() const;
};

// Assign reading_order to every line (0-based, dense) and return the lines in
// reading order. The input is not mutated; the returned vector owns copies
// with reading_order set. Empty input -> empty output. Lines with an empty
// source box are ordered by their polygon bounds; a fully degenerate region
// (no box and no polygon) sorts last within its page, by input index.
[[nodiscard]] Result<luisa::vector<ReadingOrderLine>> order_lines(luisa::span<const ReadingOrderLine> lines,
                                                                  const ReadingOrderOptions &options = {});

// Canonical page text from ordered lines: join non-empty line texts with
// '\n', no trailing newline.
[[nodiscard]] luisa::string join_ordered_text(luisa::span<const ReadingOrderLine> ordered_lines);

}// namespace luisa::ocr::output
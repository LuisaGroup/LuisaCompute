// Test for ocr/output/reading_order (Phase 5, host-only).
// This test covers:
// - deterministic single-column top-to-bottom order;
// - multi-column left-to-right then top-to-bottom order;
// - rotated/skewed/very-wide lines (row-tolerance banding);
// - stable tie-breaking (identical geometry -> input order);
// - source page indices (pages never reorder across each other);
// - canonical text joining; empty input; degenerate regions;
// - confidence carried through unchanged (never synthesized).
#include "ut/ut.hpp"

#include <cmath>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/output/reading_order.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace luisa::ocr::output;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] ReadingOrderLine line(const char *text, float x0, float y0, float x1, float y1,
                                    uint32_t page = 0) {
    ReadingOrderLine l;
    l.text = text;
    l.page_index = page;
    l.x0 = x0;
    l.y0 = y0;
    l.x1 = x1;
    l.y1 = y1;
    return l;
}

}// namespace

void test_single_column() {
    // Three lines stacked vertically, same column. Input is shuffled.
    luisa::vector<ReadingOrderLine> lines{
        line("bottom", 0, 40, 100, 50),
        line("top", 0, 0, 100, 10),
        line("middle", 0, 20, 100, 30),
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect(r.value->size() == 3u);
    expect((*r.value)[0].text == "top");
    expect((*r.value)[1].text == "middle");
    expect((*r.value)[2].text == "bottom");
    expect((*r.value)[0].reading_order == 0u);
    expect((*r.value)[2].reading_order == 2u);
}

void test_multi_column() {
    // Two columns: left column has 2 lines, right column has 2 lines. Reading
    // order is left column top-to-bottom, then right column top-to-bottom.
    luisa::vector<ReadingOrderLine> lines{
        line("R2", 200, 20, 300, 30),
        line("L2", 0, 20, 100, 30),
        line("R1", 200, 0, 300, 10),
        line("L1", 0, 0, 100, 10),
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect(r.value->size() == 4u);
    expect((*r.value)[0].text == "L1");
    expect((*r.value)[1].text == "L2");
    expect((*r.value)[2].text == "R1");
    expect((*r.value)[3].text == "R2");
}

void test_row_banding_skewed() {
    // Two lines on the "same row" with slightly jittered vertical centers
    // (skew). They should be ordered left-to-right within the row.
    luisa::vector<ReadingOrderLine> lines{
        line("right", 110, 1, 200, 11),  // center y = 6
        line("left", 0, 0, 100, 10),     // center y = 5
    };
    ReadingOrderOptions opts;
    opts.row_tolerance = 0.5f;// band = 0.5 * median height (10) = 5
    auto r = order_lines(lines, opts);
    expect(r.ok());
    expect((*r.value)[0].text == "left");
    expect((*r.value)[1].text == "right");
}

void test_very_wide_line() {
    // A very wide line spanning the full page width and a narrow line below.
    luisa::vector<ReadingOrderLine> lines{
        line("narrow", 0, 30, 50, 40),
        line("wide", 0, 0, 1000, 10),
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect((*r.value)[0].text == "wide");
    expect((*r.value)[1].text == "narrow");
}

void test_stable_tie_break() {
    // Identical geometry: order must fall back to input index (stable).
    luisa::vector<ReadingOrderLine> lines{
        line("first", 0, 0, 100, 10),
        line("second", 0, 0, 100, 10),
        line("third", 0, 0, 100, 10),
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect((*r.value)[0].text == "first");
    expect((*r.value)[1].text == "second");
    expect((*r.value)[2].text == "third");
    // Repeat run identical (determinism).
    auto r2 = order_lines(lines);
    expect(r2.ok());
    for (size_t i = 0; i < r.value->size(); i++) {
        expect((*r.value)[i].text == (*r2.value)[i].text);
    }
}

void test_page_indices() {
    // Lines on two pages: pages are emitted in ascending page_index order,
    // never reordered across each other.
    luisa::vector<ReadingOrderLine> lines{
        line("p1b", 0, 20, 100, 30, 1),
        line("p0a", 0, 0, 100, 10, 0),
        line("p1a", 0, 0, 100, 10, 1),
        line("p0b", 0, 20, 100, 30, 0),
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect(r.value->size() == 4u);
    expect((*r.value)[0].text == "p0a");
    expect((*r.value)[1].text == "p0b");
    expect((*r.value)[2].text == "p1a");
    expect((*r.value)[3].text == "p1b");
    // reading_order is dense across pages.
    expect((*r.value)[3].reading_order == 3u);
}

void test_degenerate_regions() {
    // A line with no box and no polygon sorts last within its page.
    ReadingOrderLine degenerate;
    degenerate.text = "degenerate";
    degenerate.page_index = 0;
    luisa::vector<ReadingOrderLine> lines{
        degenerate,
        line("normal", 0, 0, 100, 10),
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect((*r.value)[0].text == "normal");
    expect((*r.value)[1].text == "degenerate");
}

void test_polygon_bounds_fallback() {
    // A line with an empty box but a valid polygon uses the polygon bounds.
    ReadingOrderLine poly;
    poly.text = "poly";
    poly.page_index = 0;
    poly.polygon = {{0, 0}, {100, 0}, {100, 10}, {0, 10}};
    luisa::vector<ReadingOrderLine> lines{
        line("below", 0, 30, 100, 40),
        poly,
    };
    auto r = order_lines(lines);
    expect(r.ok());
    expect((*r.value)[0].text == "poly");
    expect((*r.value)[1].text == "below");
}

void test_confidence_carry_through() {
    luisa::vector<ReadingOrderLine> lines{
        line("a", 0, 0, 100, 10),
        line("b", 0, 20, 100, 30),
    };
    lines[0].has_confidence = true;
    lines[0].confidence = 0.75f;
    lines[1].has_confidence = false;
    auto r = order_lines(lines);
    expect(r.ok());
    expect((*r.value)[0].has_confidence);
    expect(std::abs((*r.value)[0].confidence - 0.75f) < 1e-6f);
    expect(!(*r.value)[1].has_confidence);
}

void test_empty_and_join() {
    luisa::vector<ReadingOrderLine> none;
    auto r = order_lines(none);
    expect(r.ok());
    expect(r.value->empty());
    expect(join_ordered_text(*r.value).empty());

    luisa::vector<ReadingOrderLine> lines{
        line("hello", 0, 0, 100, 10),
        line("", 0, 20, 100, 30),
        line("world", 0, 40, 100, 50),
    };
    auto ordered = order_lines(lines);
    expect(ordered.ok());
    expect(join_ordered_text(*ordered.value) == "hello\nworld");
}

void test_invalid_options() {
    luisa::vector<ReadingOrderLine> lines{line("a", 0, 0, 100, 10)};
    ReadingOrderOptions bad;
    bad.column_overlap_ratio = 1.5f;
    expect(!order_lines(lines, bad).ok());
    bad.column_overlap_ratio = 0.3f;
    bad.row_tolerance = -1.0f;
    expect(!order_lines(lines, bad).ok());
}

static auto test_ocr_reading_order_registration = [] {
    "reading_order_single_column"_test = [] { test_single_column(); };
    "reading_order_multi_column"_test = [] { test_multi_column(); };
    "reading_order_row_banding_skewed"_test = [] { test_row_banding_skewed(); };
    "reading_order_very_wide_line"_test = [] { test_very_wide_line(); };
    "reading_order_stable_tie_break"_test = [] { test_stable_tie_break(); };
    "reading_order_page_indices"_test = [] { test_page_indices(); };
    "reading_order_degenerate_regions"_test = [] { test_degenerate_regions(); };
    "reading_order_polygon_bounds_fallback"_test = [] { test_polygon_bounds_fallback(); };
    "reading_order_confidence_carry_through"_test = [] { test_confidence_carry_through(); };
    "reading_order_empty_and_join"_test = [] { test_empty_and_join(); };
    "reading_order_invalid_options"_test = [] { test_invalid_options(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
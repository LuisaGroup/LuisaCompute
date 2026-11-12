// Test for ocr/output/table.h/.cpp (Phase 4, host-only).
// Covers: HTML table parsing/validation (basic, header, colspan/rowspan,
// caption, entities, malformed/overlapping/ragged/nested/disallowed tags),
// OTSL parsing/validation (fcel/eced/lcel/ucel/xcel/nl, ragged, dangling,
// unknown tokens), format detection, and canonical HTML re-serialization.
// No device/model required.
#include "ut/ut.hpp"

#include "ocr/output/table.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace luisa::ocr::output;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_html_basic() {
    auto r = parse_html_table("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>");
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->format == TableFormat::html);
    expect(r.value->rows == 2u);
    expect(r.value->cols == 2u);
    expect(r.value->cells.size() == 4u);
    expect(r.value->cells[0].text == "a");
    expect(r.value->cells[3].text == "d");
    expect(r.value->grid.size() == 4u);
    for (auto idx : r.value->grid) { expect(idx >= 0); }
}

void test_html_header_caption_entities() {
    auto r = parse_html_table(
        "<table><caption>Cap &amp; more</caption><tr><th>H1</th><th>H2</th></tr>"
        "<tr><td>&lt;x&gt;</td><td>&quot;q&quot;</td></tr></table>");
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->caption == "Cap & more");
    expect(r.value->cells[0].is_header);
    expect(r.value->cells[1].is_header);
    expect(!r.value->cells[2].is_header);
    expect(r.value->cells[2].text == "<x>");
    expect(r.value->cells[3].text == "\"q\"");
}

void test_html_spans() {
    // colspan=2 on the first cell, rowspan=2 on a later cell.
    auto r = parse_html_table(
        "<table><tr><td colspan=\"2\">wide</td></tr>"
        "<tr><td rowspan=\"2\">tall</td><td>x</td></tr>"
        "<tr><td>y</td></tr></table>");
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->rows == 3u);
    expect(r.value->cols == 2u);
    // wide spans both columns of row 0.
    expect(r.value->cells[0].col_span == 2u);
    expect(r.value->grid[0] == 0 && r.value->grid[1] == 0);
    // tall spans rows 1-2 of column 0.
    expect(r.value->cells[1].row_span == 2u);
    expect(r.value->grid[2] == 1 && r.value->grid[4] == 1);
}

void test_html_malformed() {
    // Unclosed table.
    expect(!parse_html_table("<table><tr><td>a</td></tr>").ok());
    // Unclosed cell.
    expect(!parse_html_table("<table><tr><td>a</tr></table>").ok());
    // Disallowed tag.
    expect(!parse_html_table("<table><tr><td><b>x</b></td></tr></table>").ok());
    // Nested table.
    expect(!parse_html_table("<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>").ok());
    // Text outside a cell.
    expect(!parse_html_table("<table>stray<tr><td>a</td></tr></table>").ok());
    // Overlapping spans: a rowspan cell collides with a later row's cell.
    expect(!parse_html_table(
        "<table><tr><td rowspan=\"2\">a</td><td>b</td></tr>"
        "<tr><td>c</td><td>d</td></tr></table>").ok());
    // Ragged rows (uncovered slot).
    expect(!parse_html_table("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>").ok());
    // Bad colspan.
    expect(!parse_html_table("<table><tr><td colspan=\"0\">a</td></tr></table>").ok());
    expect(!parse_html_table("<table><tr><td colspan=\"x\">a</td></tr></table>").ok());
    // Empty.
    expect(!parse_html_table("").ok());
    expect(!parse_html_table("<table></table>").ok());
    // Comment.
    expect(!parse_html_table("<table><!-- c --><tr><td>a</td></tr></table>").ok());
    // Trailing '<'.
    expect(!parse_html_table("<table><tr><td>a</td></tr></table><").ok());
}

void test_otsl_basic() {
    // 2x2 grid: fcel fcel nl fcel fcel nl
    auto r = parse_otsl_table("fcel fcel nl fcel fcel nl");
    expect(r.ok());
    if (!r.ok()) { return; }
    expect(r.value->format == TableFormat::otsl);
    expect(r.value->rows == 2u);
    expect(r.value->cols == 2u);
    expect(r.value->cells.size() == 4u);
    // eced is an empty cell.
    auto e = parse_otsl_table("fcel eced nl eced fcel");
    expect(e.ok());
    if (e.ok()) { expect(e.value->cells.size() == 4u); }
}

void test_otsl_spans() {
    // lcel extends the left cell's colspan.
    auto r = parse_otsl_table("fcel lcel nl fcel fcel");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->cols == 2u);
        expect(r.value->cells[0].col_span == 2u);
    }
    // ucel extends the upper cell's rowspan.
    auto u = parse_otsl_table("fcel fcel nl ucel fcel");
    expect(u.ok());
    if (u.ok()) {
        expect(u.value->rows == 2u);
        expect(u.value->cells[0].row_span == 2u);
    }
}

void test_otsl_malformed() {
    // Unknown token.
    expect(!parse_otsl_table("fcel bogus nl").ok());
    // Ragged rows.
    expect(!parse_otsl_table("fcel fcel nl fcel").ok());
    // lcel with no left neighbor.
    expect(!parse_otsl_table("lcel fcel").ok());
    // ucel with no upper neighbor.
    expect(!parse_otsl_table("ucel fcel").ok());
    // Empty.
    expect(!parse_otsl_table("").ok());
    expect(!parse_otsl_table("   ").ok());
}

void test_detect_and_serialize() {
    expect(detect_table_format("<table></table>") == TableFormat::html);
    expect(detect_table_format("fcel fcel") == TableFormat::otsl);
    expect(!detect_table_format("").has_value());
    auto r = parse_html_table("<table><tr><th>A</th><td>b&amp;c</td></tr></table>");
    expect(r.ok());
    if (r.ok()) {
        auto html = table_to_html(*r.value);
        expect(html.ok());
        if (html.ok()) {
            expect(html.value->find("<th>A</th>") != luisa::string::npos);
            expect(html.value->find("b&amp;c") != luisa::string::npos);
        }
    }
    // parse_table auto-detects.
    expect(parse_table("<table><tr><td>x</td></tr></table>").ok());
    expect(parse_table("fcel fcel").ok());
    expect(!parse_table("").ok());
}

// M3: canonical 2x2 (C L / U X) and 3x3 (C L L / U X X / U X X) span blocks
// parse losslessly into a single cell with the correct colspan/rowspan, and
// round-trip to HTML with the same geometry.
void test_otsl_canonical_spans() {
    // 2x2: one cell spanning 2 cols x 2 rows.
    auto r2 = parse_otsl_table("fcel lcel nl ucel xcel");
    expect(r2.ok());
    if (r2.ok()) {
        expect(r2.value->rows == 2u);
        expect(r2.value->cols == 2u);
        expect(r2.value->cells.size() == 1u);
        expect(r2.value->cells[0].col_span == 2u);
        expect(r2.value->cells[0].row_span == 2u);
        // All four grid slots map to the single cell.
        for (auto idx : r2.value->grid) { expect(idx == 0); }
        auto html = table_to_html(*r2.value);
        expect(html.ok());
        if (html.ok()) {
            expect(html.value->find("colspan=\"2\"") != luisa::string::npos);
            expect(html.value->find("rowspan=\"2\"") != luisa::string::npos);
        }
    }
    // 3x3: one cell spanning 3 cols x 3 rows.
    auto r3 = parse_otsl_table("fcel lcel lcel nl ucel xcel xcel nl ucel xcel xcel");
    expect(r3.ok());
    if (r3.ok()) {
        expect(r3.value->rows == 3u);
        expect(r3.value->cols == 3u);
        expect(r3.value->cells.size() == 1u);
        expect(r3.value->cells[0].col_span == 3u);
        expect(r3.value->cells[0].row_span == 3u);
        for (auto idx : r3.value->grid) { expect(idx == 0); }
    }
    // Mixed: a 2x2 span block plus a separate single cell column.
    // C L X | F  / U X X | F  (3 cols: 2x2 span + 1 single per row)
    auto rm = parse_otsl_table("fcel lcel fcel nl ucel xcel fcel");
    expect(rm.ok());
    if (rm.ok()) {
        expect(rm.value->rows == 2u);
        expect(rm.value->cols == 3u);
        expect(rm.value->cells.size() == 3u);// span cell + 2 singles
        expect(rm.value->cells[0].col_span == 2u);
        expect(rm.value->cells[0].row_span == 2u);
    }
}

// M3: invalid span geometry is rejected.
void test_otsl_span_invalid() {
    // U under L: a ucel whose upper slot is a horizontal (lcel) continuation,
    // not the cell's origin column. Row0: C L (cell spans cols 0-1). Row1:
    // col0 is a NEW cell, col1 ucel sits under the lcel at (0,1) -> rejected.
    expect(!parse_otsl_table("fcel lcel nl fcel ucel").ok());
    // xcel with mismatched neighbors (up-left != up).
    expect(!parse_otsl_table("fcel fcel nl fcel xcel").ok());
    // xcel with no upper-left neighbor (first row).
    expect(!parse_otsl_table("xcel fcel").ok());
    // xcel with a gap to the cell's right edge (not adjacent).
    expect(!parse_otsl_table("fcel fcel fcel nl fcel fcel xcel").ok());
    // ucel with no upper neighbor (first row).
    expect(!parse_otsl_table("ucel fcel").ok());
    // lcel with no left neighbor (first column).
    expect(!parse_otsl_table("lcel fcel").ok());
    // Ragged rows (inconsistent width).
    expect(!parse_otsl_table("fcel fcel nl fcel").ok());
}

// S1: colspan/rowspan inside an unrelated QUOTED attribute is not matched.
void test_html_attr_quoted_substring() {
    // title contains "colspan=2" but the cell has NO real colspan -> 1x1.
    auto r = parse_html_table("<table><tr><td title=\"use colspan=2 here\">a</td></tr></table>");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->cols == 1u);
        expect(r.value->cells[0].col_span == 1u);
        expect(r.value->cells[0].text == "a");
    }
    // A real colspan AFTER a quoted attribute containing the substring.
    auto r2 = parse_html_table("<table><tr><td title=\"rowspan=9\" colspan=\"2\">a</td></tr><tr><td>b</td><td>c</td></tr></table>");
    expect(r2.ok());
    if (r2.ok()) {
        expect(r2.value->cells[0].col_span == 2u);
        expect(r2.value->cells[0].row_span == 1u);
    }
    // Single-quoted attribute value with the substring.
    auto r3 = parse_html_table("<table><tr><td data-x='colspan=3'>a</td></tr></table>");
    expect(r3.ok());
    if (r3.ok()) { expect(r3.value->cells[0].col_span == 1u); }
}

// S3: OTSL detection requires whole token-only content, not prose.
void test_otsl_detection_prose() {
    expect(detect_table_format("fcel fcel nl fcel fcel") == TableFormat::otsl);
    // Prose starting with a token-like word is NOT a table.
    expect(!detect_table_format("nl is a token").has_value());
    expect(!detect_table_format("fcel means filled cell").has_value());
    expect(!detect_table_format("lcel and ucel are spans").has_value());
    expect(!detect_table_format("just some prose").has_value());
    // A token block with a trailing prose word is NOT a table.
    expect(!detect_table_format("fcel fcel nl hello").has_value());
}

// S5: invalid numeric character references are rejected (no invalid UTF-8).
void test_html_numeric_refs_invalid() {
    // Surrogates.
    expect(!parse_html_table("<table><tr><td>&#xD800;</td></tr></table>").ok());
    expect(!parse_html_table("<table><tr><td>&#xDFFF;</td></tr></table>").ok());
    // Noncharacters.
    expect(!parse_html_table("<table><tr><td>&#xFDD0;</td></tr></table>").ok());
    expect(!parse_html_table("<table><tr><td>&#xFFFE;</td></tr></table>").ok());
    expect(!parse_html_table("<table><tr><td>&#xFFFF;</td></tr></table>").ok());
    expect(!parse_html_table("<table><tr><td>&#x1FFFE;</td></tr></table>").ok());
    // Out of range.
    expect(!parse_html_table("<table><tr><td>&#x110000;</td></tr></table>").ok());
    // Zero.
    expect(!parse_html_table("<table><tr><td>&#0;</td></tr></table>").ok());
    // Valid references still work (ASCII, BMP, astral).
    auto ok = parse_html_table("<table><tr><td>&#65;&#x4E2D;&#x1F600;</td></tr></table>");
    expect(ok.ok());
    if (ok.ok()) {
        const auto &t = ok.value->cells[0].text;
        expect(t.find('A') != luisa::string::npos);
        expect(t.find("\xE4\xB8\xAD") != luisa::string::npos);// 中
        expect(t.find("\xF0\x9F\x98\x80") != luisa::string::npos);// 😀
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "table_html_basic"_test = [] { test_html_basic(); };
    "table_html_header_caption_entities"_test = [] { test_html_header_caption_entities(); };
    "table_html_spans"_test = [] { test_html_spans(); };
    "table_html_malformed"_test = [] { test_html_malformed(); };
    "table_otsl_basic"_test = [] { test_otsl_basic(); };
    "table_otsl_spans"_test = [] { test_otsl_spans(); };
    "table_otsl_malformed"_test = [] { test_otsl_malformed(); };
    "table_detect_and_serialize"_test = [] { test_detect_and_serialize(); };
    "table_otsl_canonical_spans"_test = [] { test_otsl_canonical_spans(); };
    "table_otsl_span_invalid"_test = [] { test_otsl_span_invalid(); };
    "table_html_attr_quoted_substring"_test = [] { test_html_attr_quoted_substring(); };
    "table_otsl_detection_prose"_test = [] { test_otsl_detection_prose(); };
    "table_html_numeric_refs_invalid"_test = [] { test_html_numeric_refs_invalid(); };
    return 0;
}

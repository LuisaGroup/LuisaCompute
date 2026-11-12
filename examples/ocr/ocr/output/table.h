#pragma once
// OCR engine -- HTML / OTSL table parsing + validation (Phase 4). Host-only.
//
// Two DISTINCT table serializations, never auto-mixed:
//   * HTML: <table>...</table> with <tr>/<td>/<th>, optional colspan/rowspan.
//   * OTSL: the compact token language (fcel/eced/lcel/ucel/xcel/nl) used by
//     table-structure models.
//
// Both are parsed into a common TableCell grid and STRICTLY validated:
//   * HTML: balanced tags, only table/tr/td/th (and caption) allowed, no
//     nested tables, colspan/rowspan positive integers, rectangular grid
//     after span expansion, no overlapping spans.
//   * OTSL: only the six known tokens, a rectangular grid, lcel/ucel must
//     reference an existing left/upper cell, no dangling spans.
// Malformed input is an ERROR (invalid_argument), never a silent text
// fallback. No confidence is produced.
#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr::output {

enum class TableFormat : uint32_t {
    html = 0,
    otsl = 1,
};

[[nodiscard]] luisa::string_view table_format_name(TableFormat format) noexcept;

// One logical cell after span expansion. `text` is the cell content; the
// span fields describe the ORIGINAL span (1 = no span). `is_header` marks
// <th> cells. `row`/`col` are the TOP-LEFT grid coordinates of the cell.
struct TableCell {
    luisa::string text;
    uint32_t row{0};
    uint32_t col{0};
    uint32_t row_span{1};
    uint32_t col_span{1};
    bool is_header{false};
};

struct ParsedTable {
    TableFormat format{TableFormat::html};
    uint32_t rows{0};
    uint32_t cols{0};
    // Logical cells (one per original <td>/<th> or OTSL fcel/eced), in
    // row-major order of their top-left coordinates.
    luisa::vector<TableCell> cells;
    // Expanded grid: grid[r*cols+c] = index into cells, or -1 when the slot
    // is covered by a span but is not a cell origin. Always rows*cols.
    luisa::vector<int32_t> grid;
    luisa::string caption;// HTML <caption> text ("" when absent)
};

// Parse + validate an HTML table. Fails closed on any structural violation.
[[nodiscard]] Result<ParsedTable> parse_html_table(luisa::string_view html);

// Parse + validate an OTSL table. The input is the token sequence (whitespace
// separated); `nl` ends a row. Fails closed on unknown tokens / ragged rows /
// dangling span references.
[[nodiscard]] Result<ParsedTable> parse_otsl_table(luisa::string_view otsl);

// Detect the format of a table-looking string: leading '<' (after optional
// whitespace) => html; otherwise otsl. Returns nullopt for an empty string.
[[nodiscard]] luisa::optional<TableFormat> detect_table_format(luisa::string_view text);

// Parse + validate with automatic format detection.
[[nodiscard]] Result<ParsedTable> parse_table(luisa::string_view text);

// Serialize a parsed table back to canonical HTML (used by compose_markdown
// when a canonical form is requested). The grid is re-emitted with explicit
// colspan/rowspan; header cells use <th>.
[[nodiscard]] Result<luisa::string> table_to_html(const ParsedTable &table);

}// namespace luisa::ocr::output

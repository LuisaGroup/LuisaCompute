// OCR engine -- HTML / OTSL table parsing + validation (Phase 4). Host-only.
// Strict structural validation; malformed input is an error. See table.h.
#include "ocr/output/table.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include <luisa/core/stl/format.h>

namespace luisa::ocr::output {

[[nodiscard]] luisa::string_view table_format_name(TableFormat format) noexcept {
    switch (format) {
        case TableFormat::html: return "html";
        case TableFormat::otsl: return "otsl";
    }
    return "unknown";
}

namespace {

[[nodiscard]] bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

[[nodiscard]] luisa::string_view trim(luisa::string_view s) noexcept {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) { b++; }
    while (e > b && is_space(s[e - 1])) { e--; }
    return s.substr(b, e - b);
}

[[nodiscard]] luisa::string to_lower(luisa::string_view s) {
    luisa::string out;
    out.reserve(s.size());
    for (char c : s) {
        out += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    }
    return out;
}

// S5: a numeric character reference must denote a valid Unicode scalar value.
// Surrogates (D800-DFFF), noncharacters (FDD0-FDEF and the last two code points
// of every plane), zero, and out-of-range values are rejected so the decoder
// never emits invalid UTF-8.
[[nodiscard]] bool valid_unicode_scalar(uint32_t code) noexcept {
    if (code == 0u || code > 0x10FFFFu) { return false; }
    if (code >= 0xD800u && code <= 0xDFFFu) { return false; }// surrogates
    if (code >= 0xFDD0u && code <= 0xFDEFu) { return false; }// noncharacters
    if ((code & 0xFFFEu) == 0xFFFEu) { return false; }// xFFFE / xFFFF noncharacters
    return true;
}

// Decode the five core XML entities (and numeric &#..; / &#x..;). Unknown
// named entities are left verbatim (HTML tolerance). An INVALID numeric
// reference (surrogate / noncharacter / out-of-range / zero) is an error so
// no invalid UTF-8 is ever emitted (S5).
[[nodiscard]] Result<luisa::string> decode_entities(luisa::string_view s) {
    using R = Result<luisa::string>;
    luisa::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '&') {
            out += s[i];
            continue;
        }
        const auto semi = s.find(';', i + 1);
        if (semi == luisa::string_view::npos || semi - i > 10) {
            out += s[i];
            continue;
        }
        const auto ent = s.substr(i + 1, semi - i - 1);
        if (ent == "amp") { out += '&'; }
        else if (ent == "lt") { out += '<'; }
        else if (ent == "gt") { out += '>'; }
        else if (ent == "quot") { out += '"'; }
        else if (ent == "apos") { out += '\''; }
        else if (!ent.empty() && ent[0] == '#') {
            // Numeric character reference.
            uint32_t code = 0;
            bool ok = true;
            if (ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X')) {
                if (ent.size() == 2) { ok = false; }
                for (size_t k = 2; k < ent.size() && ok; k++) {
                    const char c = ent[k];
                    uint32_t d = 0;
                    if (c >= '0' && c <= '9') { d = c - '0'; }
                    else if (c >= 'a' && c <= 'f') { d = c - 'a' + 10; }
                    else if (c >= 'A' && c <= 'F') { d = c - 'A' + 10; }
                    else { ok = false; break; }
                    code = code * 16 + d;
                    if (code > 0x10FFFFu) { ok = false; break; }
                }
            } else {
                if (ent.size() == 1) { ok = false; }
                for (size_t k = 1; k < ent.size() && ok; k++) {
                    const char c = ent[k];
                    if (c < '0' || c > '9') { ok = false; break; }
                    code = code * 10 + (c - '0');
                    if (code > 0x10FFFFu) { ok = false; break; }
                }
            }
            if (!ok || !valid_unicode_scalar(code)) {
                return R::failure(StatusCode::invalid_argument,
                                  "HTML numeric character reference is not a valid Unicode scalar value", "html");
            }
            // Encode as UTF-8 (code is a valid scalar value).
            if (code < 0x80) {
                out += static_cast<char>(code);
            } else if (code < 0x800) {
                out += static_cast<char>(0xC0 | (code >> 6));
                out += static_cast<char>(0x80 | (code & 0x3F));
            } else if (code < 0x10000) {
                out += static_cast<char>(0xE0 | (code >> 12));
                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (code & 0x3F));
            } else {
                out += static_cast<char>(0xF0 | (code >> 18));
                out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (code & 0x3F));
            }
        } else {
            out += s.substr(i, semi - i + 1);
        }
        i = semi;
    }
    return R::success(std::move(out));
}

// Encode the core XML special characters for canonical HTML output.
[[nodiscard]] luisa::string encode_entities(luisa::string_view s) {
    luisa::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c; break;
        }
    }
    return out;
}

// One HTML tag token: name (lowercase), is_close, self_close, raw attribute
// text, and the byte span in the source.
struct HtmlTag {
    luisa::string name;
    bool is_close{false};
    bool self_close{false};
    luisa::string attrs;
    uint64_t begin{0};
    uint64_t end{0};
};

// Tokenize HTML into tags + text. Fails closed on an unterminated tag or a
// '<' that does not start a valid tag. Comments are rejected (a table must
// not hide content in comments).
[[nodiscard]] Status tokenize_html(luisa::string_view html, luisa::vector<HtmlTag> &tags,
                                   luisa::vector<luisa::string> &texts, luisa::vector<uint64_t> &text_pos) {
    uint64_t i = 0;
    const uint64_t n = html.size();
    luisa::string pending_text;
    uint64_t pending_pos = 0;
    const auto flush_text = [&]() -> Status {
        if (!pending_text.empty()) {
            auto decoded = decode_entities(pending_text);
            if (!decoded.ok()) { return decoded.status; }
            texts.push_back(std::move(*decoded.value));
            text_pos.push_back(pending_pos);
            pending_text.clear();
        }
        return Status::success();
    };
    while (i < n) {
        if (html[i] != '<') {
            if (pending_text.empty()) { pending_pos = i; }
            pending_text += html[i];
            i++;
            continue;
        }
        // A '<' must start a tag.
        if (i + 1 >= n) {
            return Status::error(StatusCode::invalid_argument, "HTML table has a trailing '<'", "html");
        }
        if (html.compare(i, 4, "<!--") == 0) {
            return Status::error(StatusCode::invalid_argument, "HTML comments are not allowed in a table", "html");
        }
        if (auto s = flush_text(); !s.ok()) { return s; }
        const uint64_t begin = i;
        i++;
        HtmlTag tag;
        tag.begin = begin;
        if (html[i] == '/') {
            tag.is_close = true;
            i++;
        }
        const uint64_t name_start = i;
        while (i < n && (std::isalnum(static_cast<unsigned char>(html[i])) || html[i] == '-' || html[i] == '_')) { i++; }
        if (i == name_start) {
            return Status::error(StatusCode::invalid_argument, "HTML tag has an empty name", "html");
        }
        tag.name = to_lower(html.substr(name_start, i - name_start));
        // Attributes until '>' (quoted values may contain '>').
        const uint64_t attr_start = i;
        bool in_quote = false;
        char quote = 0;
        while (i < n) {
            const char c = html[i];
            if (in_quote) {
                if (c == quote) { in_quote = false; }
                i++;
                continue;
            }
            if (c == '"' || c == '\'') {
                in_quote = true;
                quote = c;
                i++;
                continue;
            }
            if (c == '>') { break; }
            i++;
        }
        if (i >= n) {
            return Status::error(StatusCode::invalid_argument, "HTML tag is not closed with '>'", "html");
        }
        tag.attrs = luisa::string{html.substr(attr_start, i - attr_start)};
        if (!tag.attrs.empty() && tag.attrs.back() == '/') {
            tag.self_close = true;
            tag.attrs.pop_back();
        }
        tag.end = i + 1;
        i++;
        tags.push_back(std::move(tag));
    }
    if (auto s = flush_text(); !s.ok()) { return s; }
    return Status::success();
}

// Extract a positive integer attribute (colspan/rowspan). Absent => 1.
// Malformed (non-integer, zero, negative, huge) => error.
// S1: the attribute string is TOKENIZED into name[=value] pairs, skipping
// quoted values entirely, so a substring like "colspan" inside an unrelated
// quoted attribute (e.g. title="use colspan=2 here") is never matched.
[[nodiscard]] Result<uint32_t> span_attr(luisa::string_view attrs, luisa::string_view name) {
    using R = Result<uint32_t>;
    const luisa::string lower = to_lower(attrs);
    const size_t n = lower.size();
    size_t i = 0;
    while (i < n) {
        // Skip whitespace between attributes.
        while (i < n && is_space(lower[i])) { i++; }
        if (i >= n) { break; }
        // Read the attribute name.
        const size_t name_start = i;
        while (i < n && !is_space(lower[i]) && lower[i] != '=') { i++; }
        const auto attr_name = luisa::string_view{lower}.substr(name_start, i - name_start);
        // Skip whitespace before '='.
        size_t k = i;
        while (k < n && is_space(lower[k])) { k++; }
        luisa::string value;
        bool has_value = false;
        if (k < n && lower[k] == '=') {
            has_value = true;
            k++;
            while (k < n && is_space(lower[k])) { k++; }
            if (k < n && (lower[k] == '"' || lower[k] == '\'')) {
                const char quote = lower[k];
                k++;
                const size_t vstart = k;
                while (k < n && lower[k] != quote) { k++; }
                value = luisa::string{attrs.substr(vstart, k - vstart)};
                if (k < n) { k++; }// closing quote
            } else {
                const size_t vstart = k;
                while (k < n && !is_space(lower[k])) { k++; }
                value = luisa::string{attrs.substr(vstart, k - vstart)};
            }
            i = k;
        } else {
            i = k;
        }
        if (attr_name != name) { continue; }
        // This is the target attribute.
        if (!has_value) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("table attribute '{}' has no value", name), luisa::string{name});
        }
        const auto trimmed = trim(value);
        if (trimmed.empty()) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("table attribute '{}' is empty", name), luisa::string{name});
        }
        uint64_t v = 0;
        for (char c : trimmed) {
            if (c < '0' || c > '9') {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::format("table attribute '{}' must be a positive integer", name),
                                  luisa::string{name});
            }
            v = v * 10 + (c - '0');
            if (v > 1000u) {
                return R::failure(StatusCode::invalid_argument,
                                  luisa::format("table attribute '{}' is unreasonably large", name),
                                  luisa::string{name});
            }
        }
        if (v == 0u) {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("table attribute '{}' must be >= 1", name), luisa::string{name});
        }
        return R::success(static_cast<uint32_t>(v));
    }
    return R::success(1u);
}

}// namespace

// ---------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------

[[nodiscard]] Result<ParsedTable> parse_html_table(luisa::string_view html) {
    using R = Result<ParsedTable>;
    const auto src = trim(html);
    if (src.empty()) {
        return R::failure(StatusCode::invalid_argument, "HTML table is empty", "html");
    }
    luisa::vector<HtmlTag> tags;
    luisa::vector<luisa::string> texts;
    luisa::vector<uint64_t> text_pos;
    if (auto s = tokenize_html(src, tags, texts, text_pos); !s.ok()) { return R::failure(s); }

    ParsedTable table;
    table.format = TableFormat::html;
    // Grid occupancy during parsing: expanded rows of cell indices (-1 free).
    luisa::vector<luisa::vector<int32_t>> grid_rows;
    uint32_t cur_row = 0;
    bool in_table = false, in_row = false, in_cell = false, in_caption = false;
    bool table_closed = false;
    luisa::string cur_text;
    luisa::string caption_text;
    uint32_t cell_col = 0;
    int32_t cur_cell_idx = -1;
    size_t text_idx = 0;
    // Collect text chunks whose source offset is < pos (texts are in order).
    const auto text_before = [&](uint64_t pos) -> luisa::string {
        luisa::string out;
        while (text_idx < texts.size() && text_pos[text_idx] < pos) {
            out += texts[text_idx];
            text_idx++;
        }
        return out;
    };

    const auto ensure_row = [&](uint32_t r) {
        while (grid_rows.size() <= r) { grid_rows.emplace_back(); }
    };
    const auto next_free_col = [&](uint32_t r, uint32_t c) -> uint32_t {
        ensure_row(r);
        while (c < grid_rows[r].size() && grid_rows[r][c] != -1) { c++; }
        return c;
    };
    const auto occupy = [&](uint32_t r, uint32_t c, uint32_t rs, uint32_t cs, int32_t idx) -> Status {
        for (uint32_t dr = 0; dr < rs; dr++) {
            ensure_row(r + dr);
            auto &row = grid_rows[r + dr];
            if (row.size() < c + cs) { row.resize(c + cs, -1); }
            for (uint32_t dc = 0; dc < cs; dc++) {
                if (row[c + dc] != -1) {
                    return Status::error(StatusCode::invalid_argument,
                                         "HTML table has overlapping colspan/rowspan cells", "html");
                }
                row[c + dc] = idx;
            }
        }
        return Status::success();
    };

    for (const auto &tag : tags) {
        const auto chunk = text_before(tag.begin);
        if (in_cell) { cur_text += chunk; }
        else if (in_caption) { caption_text += chunk; }
        else if (!trim(chunk).empty()) {
            return R::failure(StatusCode::invalid_argument, "HTML table has text outside a cell", "html");
        }

        // Cell close: finalize the open cell.
        if (tag.is_close && (tag.name == "td" || tag.name == "th")) {
            if (!in_cell) {
                return R::failure(StatusCode::invalid_argument, "HTML closing cell tag without an open cell", "html");
            }
            table.cells[static_cast<size_t>(cur_cell_idx)].text = trim(cur_text);
            in_cell = false;
            cur_cell_idx = -1;
            continue;
        }

        if (tag.name == "table") {
            if (tag.is_close) {
                if (!in_table || in_row || in_cell || in_caption) {
                    return R::failure(StatusCode::invalid_argument, "HTML </table> is misplaced", "html");
                }
                in_table = false;
                table_closed = true;
            } else {
                if (in_table || table_closed) {
                    return R::failure(StatusCode::invalid_argument, "HTML tables must not nest or repeat", "html");
                }
                if (tag.self_close) {
                    return R::failure(StatusCode::invalid_argument, "HTML <table/> is invalid", "html");
                }
                in_table = true;
            }
        } else if (tag.name == "caption") {
            if (!in_table || in_row || in_cell) {
                return R::failure(StatusCode::invalid_argument, "HTML <caption> must be inside <table> before rows", "html");
            }
            if (tag.is_close) {
                if (!in_caption) { return R::failure(StatusCode::invalid_argument, "HTML </caption> without <caption>", "html"); }
                in_caption = false;
            } else {
                if (in_caption || !grid_rows.empty()) {
                    return R::failure(StatusCode::invalid_argument, "HTML <caption> must precede rows and not nest", "html");
                }
                in_caption = true;
            }
        } else if (tag.name == "tr") {
            if (!in_table || in_cell || in_caption) {
                return R::failure(StatusCode::invalid_argument, "HTML <tr> must be inside <table> outside cells", "html");
            }
            if (tag.is_close) {
                if (!in_row) { return R::failure(StatusCode::invalid_argument, "HTML </tr> without <tr>", "html"); }
                in_row = false;
                cur_row++;
            } else {
                if (in_row) { return R::failure(StatusCode::invalid_argument, "HTML <tr> must not nest", "html"); }
                if (tag.self_close) {
                    // An empty row is allowed (self-closing <tr/>).
                    ensure_row(cur_row);
                    cur_row++;
                } else {
                    in_row = true;
                    cell_col = 0;
                    ensure_row(cur_row);
                }
            }
        } else if (tag.name == "td" || tag.name == "th") {
            if (!in_table || !in_row || in_cell || in_caption) {
                return R::failure(StatusCode::invalid_argument, "HTML cell must be inside an open <tr>", "html");
            }
            if (tag.is_close) {
                return R::failure(StatusCode::invalid_argument, "HTML closing cell tag without an open cell", "html");
            }
            auto cs = span_attr(tag.attrs, "colspan");
            if (!cs.ok()) { return R::failure(cs.status); }
            auto rs = span_attr(tag.attrs, "rowspan");
            if (!rs.ok()) { return R::failure(rs.status); }
            TableCell cell;
            cell.is_header = tag.name == "th";
            cell.col_span = *cs.value;
            cell.row_span = *rs.value;
            cell_col = next_free_col(cur_row, cell_col);
            cell.row = cur_row;
            cell.col = cell_col;
            const int32_t idx = static_cast<int32_t>(table.cells.size());
            if (auto s = occupy(cur_row, cell_col, cell.row_span, cell.col_span, idx); !s.ok()) {
                return R::failure(s);
            }
            table.cells.push_back(std::move(cell));
            cur_cell_idx = idx;
            cur_text.clear();
            in_cell = true;
            cell_col += *cs.value;
            if (tag.self_close) {
                // Self-closing cell: empty content, immediately closed.
                table.cells[static_cast<size_t>(idx)].text = "";
                in_cell = false;
                cur_cell_idx = -1;
            }
        } else {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("HTML table contains a disallowed tag <{}>", tag.name), "html");
        }
    }
    // Trailing text after the last tag.
    const auto tail = text_before(src.size());
    if (in_cell) { cur_text += tail; }
    else if (in_caption) { caption_text += tail; }
    else if (!trim(tail).empty()) {
        return R::failure(StatusCode::invalid_argument, "HTML table has trailing text outside a cell", "html");
    }
    if (!table_closed || in_table || in_row || in_cell || in_caption) {
        return R::failure(StatusCode::invalid_argument, "HTML table is not properly closed", "html");
    }
    if (table.cells.empty()) {
        return R::failure(StatusCode::invalid_argument, "HTML table has no cells", "html");
    }
    table.caption = luisa::string{trim(caption_text)};

    // Finalize the rectangular grid.
    uint32_t rows = static_cast<uint32_t>(grid_rows.size());
    uint32_t cols = 0;
    for (const auto &row : grid_rows) { cols = std::max(cols, static_cast<uint32_t>(row.size())); }
    if (rows == 0 || cols == 0) {
        return R::failure(StatusCode::invalid_argument, "HTML table has an empty grid", "html");
    }
    table.rows = rows;
    table.cols = cols;
    table.grid.assign(static_cast<size_t>(rows) * cols, -1);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < grid_rows[r].size(); c++) {
            table.grid[static_cast<size_t>(r) * cols + c] = grid_rows[r][c];
        }
    }
    for (auto idx : table.grid) {
        if (idx == -1) {
            return R::failure(StatusCode::invalid_argument, "HTML table grid has an uncovered slot (ragged rows)", "html");
        }
    }
    return R::success(std::move(table));
}

// ---------------------------------------------------------------------------
// OTSL
// ---------------------------------------------------------------------------

[[nodiscard]] Result<ParsedTable> parse_otsl_table(luisa::string_view otsl) {
    using R = Result<ParsedTable>;
    ParsedTable table;
    table.format = TableFormat::otsl;
    // Tokenize on whitespace.
    luisa::vector<luisa::string> tokens;
    {
        luisa::string cur;
        for (char c : otsl) {
            if (is_space(c)) {
                if (!cur.empty()) { tokens.push_back(std::move(cur)); cur.clear(); }
            } else {
                cur += c;
            }
        }
        if (!cur.empty()) { tokens.push_back(std::move(cur)); }
    }
    if (tokens.empty()) {
        return R::failure(StatusCode::invalid_argument, "OTSL table is empty", "otsl");
    }
    // Grid of cell indices (-1 free), built row by row.
    luisa::vector<luisa::vector<int32_t>> grid_rows;
    uint32_t row = 0, col = 0;
    uint32_t row_width = 0;
    bool have_width = false;
    const auto ensure_row = [&](uint32_t r) {
        while (grid_rows.size() <= r) { grid_rows.emplace_back(); }
    };
    const auto next_free = [&](uint32_t r, uint32_t c) -> uint32_t {
        ensure_row(r);
        while (c < grid_rows[r].size() && grid_rows[r][c] != -1) { c++; }
        return c;
    };
    for (const auto &tok : tokens) {
        if (tok == "nl") {
            if (!have_width) {
                row_width = col;
                have_width = true;
            } else if (col != row_width) {
                return R::failure(StatusCode::invalid_argument,
                                  "OTSL rows have inconsistent widths (ragged grid)", "otsl");
            }
            row++;
            col = 0;
            continue;
        }
        if (tok == "fcel" || tok == "eced") {
            const uint32_t c = next_free(row, col);
            ensure_row(row);
            if (grid_rows[row].size() <= c) { grid_rows[row].resize(c + 1, -1); }
            TableCell cell;
            cell.row = row;
            cell.col = c;
            cell.is_header = false;
            const int32_t idx = static_cast<int32_t>(table.cells.size());
            grid_rows[row][c] = idx;
            table.cells.push_back(std::move(cell));
            col = c + 1;
        } else if (tok == "lcel") {
            // Horizontal continuation: claim THIS slot for the cell to the
            // left and extend its colspan by exactly one. The left slot must
            // be in the same row (a cell origin or an earlier lcel of it).
            if (col == 0) {
                return R::failure(StatusCode::invalid_argument, "OTSL lcel has no left neighbor", "otsl");
            }
            ensure_row(row);
            const uint32_t c = col - 1;
            if (c >= grid_rows[row].size() || grid_rows[row][c] == -1) {
                return R::failure(StatusCode::invalid_argument, "OTSL lcel references an empty left slot", "otsl");
            }
            const int32_t idx = grid_rows[row][c];
            auto &cell = table.cells[static_cast<size_t>(idx)];
            if (cell.row != row) {
                return R::failure(StatusCode::invalid_argument, "OTSL lcel left slot is a rowspan continuation", "otsl");
            }
            // The left slot must be the right edge of the cell's current span.
            if (cell.col + cell.col_span != col) {
                return R::failure(StatusCode::invalid_argument, "OTSL lcel is not adjacent to the cell's right edge", "otsl");
            }
            if (grid_rows[row].size() <= col) { grid_rows[row].resize(col + 1, -1); }
            if (grid_rows[row][col] != -1) {
                return R::failure(StatusCode::invalid_argument, "OTSL lcel target slot is already occupied", "otsl");
            }
            grid_rows[row][col] = idx;
            cell.col_span++;
            col++;
        } else if (tok == "ucel") {
            // Vertical continuation: claim THIS slot for the cell whose origin
            // is in the SAME column above, and extend its rowspan by exactly
            // one. The slot directly above must belong to that same cell
            // (origin or an earlier ucel) — never a horizontal-only (lcel)
            // continuation, and never a different cell.
            if (row == 0) {
                return R::failure(StatusCode::invalid_argument, "OTSL ucel has no upper neighbor", "otsl");
            }
            ensure_row(row - 1);
            if (col >= grid_rows[row - 1].size() || grid_rows[row - 1][col] == -1) {
                return R::failure(StatusCode::invalid_argument, "OTSL ucel references an empty upper slot", "otsl");
            }
            const int32_t idx = grid_rows[row - 1][col];
            const auto &cell = table.cells[static_cast<size_t>(idx)];
            // Reject "U under L": the upper slot must be in the cell's ORIGIN
            // column. If it is a horizontal continuation (col != origin col),
            // a vertical span there is invalid.
            if (cell.col != col) {
                return R::failure(StatusCode::invalid_argument,
                                  "OTSL ucel is under a horizontal (lcel) continuation, not the cell's origin column", "otsl");
            }
            // The cell's span must reach exactly this row (no gap / overlap).
            if (cell.row + cell.row_span != row) {
                return R::failure(StatusCode::invalid_argument, "OTSL ucel is not adjacent to the cell's bottom edge", "otsl");
            }
            ensure_row(row);
            if (grid_rows[row].size() <= col) { grid_rows[row].resize(col + 1, -1); }
            if (grid_rows[row][col] != -1) {
                return R::failure(StatusCode::invalid_argument, "OTSL ucel target slot is already occupied", "otsl");
            }
            grid_rows[row][col] = idx;
            table.cells[static_cast<size_t>(idx)].row_span++;
            col++;
        } else if (tok == "xcel") {
            // Diagonal continuation: claim ONLY this slot (row, col) for the
            // cell whose origin is up-and-left, extending its colspan AND
            // rowspan by exactly one each. The three neighbors (left, up,
            // up-left) must all already belong to that same cell; xcel never
            // fills any other slot itself (they are claimed by their own
            // lcel/ucel/xcel tokens). This makes the canonical 2x2 (C L / U X)
            // and 3x3 (C L L / U X X / U X X) blocks parse losslessly without
            // double span increments.
            if (row == 0 || col == 0) {
                return R::failure(StatusCode::invalid_argument, "OTSL xcel needs an upper-left neighbor", "otsl");
            }
            ensure_row(row - 1);
            ensure_row(row);
            const auto slot = [&](uint32_t r, uint32_t c) -> int32_t {
                if (r >= grid_rows.size() || c >= grid_rows[r].size()) { return -1; }
                return grid_rows[r][c];
            };
            const int32_t up_left = slot(row - 1, col - 1);
            const int32_t up = slot(row - 1, col);
            const int32_t left = slot(row, col - 1);
            if (up_left == -1 || up == -1 || left == -1) {
                return R::failure(StatusCode::invalid_argument, "OTSL xcel has an empty neighbor slot", "otsl");
            }
            if (up_left != up || up_left != left) {
                return R::failure(StatusCode::invalid_argument, "OTSL xcel neighbors belong to different cells", "otsl");
            }
            const int32_t idx = up_left;
            auto &cell = table.cells[static_cast<size_t>(idx)];
            // xcel claims this diagonal slot. The preceding lcel/ucel tokens
            // may have ALREADY grown the span to cover it (canonical 2x2:
            // C L / U X). If the slot is already inside the cell's rectangle,
            // xcel only fills it (NO re-increment — that would double-count).
            // Otherwise xcel extends the span by one in the dimension(s) that
            // do not yet reach this slot (canonical 3x3 interior diagonals).
            const bool col_covered = cell.col + cell.col_span > col;
            const bool row_covered = cell.row + cell.row_span > row;
            if (!col_covered) {
                if (cell.col + cell.col_span != col) {
                    return R::failure(StatusCode::invalid_argument, "OTSL xcel is not adjacent to the cell's right edge", "otsl");
                }
                cell.col_span++;
            }
            if (!row_covered) {
                if (cell.row + cell.row_span != row) {
                    return R::failure(StatusCode::invalid_argument, "OTSL xcel is not adjacent to the cell's bottom edge", "otsl");
                }
                cell.row_span++;
            }
            if (grid_rows[row].size() <= col) { grid_rows[row].resize(col + 1, -1); }
            if (grid_rows[row][col] != -1) {
                return R::failure(StatusCode::invalid_argument, "OTSL xcel target slot is already occupied", "otsl");
            }
            grid_rows[row][col] = idx;
            col++;
        } else {
            return R::failure(StatusCode::invalid_argument,
                              luisa::format("OTSL contains an unknown token '{}'", tok), "otsl");
        }
    }
    // Final row (no trailing nl). The highest populated grid row determines
    // the row count: a span continuation (ucel/xcel) on the last line still
    // populates that row even though `col` may have advanced past it.
    if (!have_width) {
        row_width = col;
    } else if (col != 0 && col != row_width) {
        return R::failure(StatusCode::invalid_argument, "OTSL final row has an inconsistent width", "otsl");
    }
    // Compute the row count from the grid (number of populated rows), which is
    // robust to a final row without a trailing nl.
    uint32_t grid_row_count = 0;
    for (uint32_t r = 0; r < grid_rows.size(); r++) {
        bool populated = false;
        for (auto idx : grid_rows[r]) {
            if (idx != -1) { populated = true; break; }
        }
        if (populated) { grid_row_count = r + 1; }
    }
    if (col != 0 && grid_row_count < row + 1) { grid_row_count = row + 1; }
    row = grid_row_count;
    if (row == 0 || row_width == 0) {
        return R::failure(StatusCode::invalid_argument, "OTSL table has no cells", "otsl");
    }
    table.rows = row;
    table.cols = row_width;
    table.grid.assign(static_cast<size_t>(row) * row_width, -1);
    for (uint32_t r = 0; r < row; r++) {
        if (r >= grid_rows.size()) { continue; }
        for (uint32_t c = 0; c < row_width && c < grid_rows[r].size(); c++) {
            table.grid[static_cast<size_t>(r) * row_width + c] = grid_rows[r][c];
        }
    }
    // Every grid slot must be covered (rectangular, no holes).
    for (auto idx : table.grid) {
        if (idx == -1) {
            return R::failure(StatusCode::invalid_argument, "OTSL grid has an uncovered slot", "otsl");
        }
    }
    // Span consistency: each cell's declared rectangle must fit the table and
    // match EXACTLY the slots assigned to it (no span exceeding the table, no
    // double-counted slot). This is the lossless round-trip guarantee.
    for (size_t i = 0; i < table.cells.size(); i++) {
        const auto &cell = table.cells[i];
        if (cell.row_span == 0u || cell.col_span == 0u ||
            cell.row + cell.row_span > row || cell.col + cell.col_span > row_width) {
            return R::failure(StatusCode::invalid_argument, "OTSL cell span exceeds the table bounds", "otsl");
        }
        uint64_t covered = 0;
        for (uint32_t r = cell.row; r < cell.row + cell.row_span; r++) {
            for (uint32_t c = cell.col; c < cell.col + cell.col_span; c++) {
                if (table.grid[static_cast<size_t>(r) * row_width + c] != static_cast<int32_t>(i)) {
                    return R::failure(StatusCode::invalid_argument, "OTSL cell span does not match its assigned slots", "otsl");
                }
                covered++;
            }
        }
        if (covered != static_cast<uint64_t>(cell.row_span) * cell.col_span) {
            return R::failure(StatusCode::invalid_argument, "OTSL cell span area mismatch", "otsl");
        }
    }
    return R::success(std::move(table));
}

[[nodiscard]] luisa::optional<TableFormat> detect_table_format(luisa::string_view text) {
    const auto t = trim(text);
    if (t.empty()) { return luisa::nullopt; }
    if (t.front() == '<') { return TableFormat::html; }
    // S3: OTSL is detected ONLY when the WHOLE block is OTSL tokens (every
    // whitespace-separated token is one of the six known tokens). Prose that
    // merely starts with a token-like word ("nl is...", "fcel means...") is
    // NOT a table.
    luisa::string cur;
    bool any = false;
    const auto is_token = [](luisa::string_view s) noexcept {
        return s == "fcel" || s == "eced" || s == "lcel" || s == "ucel" || s == "xcel" || s == "nl";
    };
    for (char c : t) {
        if (is_space(c)) {
            if (!cur.empty()) {
                if (!is_token(cur)) { return luisa::nullopt; }
                any = true;
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        if (!is_token(cur)) { return luisa::nullopt; }
        any = true;
    }
    if (!any) { return luisa::nullopt; }
    return TableFormat::otsl;
}

[[nodiscard]] Result<ParsedTable> parse_table(luisa::string_view text) {
    using R = Result<ParsedTable>;
    const auto fmt = detect_table_format(text);
    if (!fmt.has_value()) {
        return R::failure(StatusCode::invalid_argument, "table text is empty", "table");
    }
    if (*fmt == TableFormat::html) { return parse_html_table(text); }
    return parse_otsl_table(text);
}

[[nodiscard]] Result<luisa::string> table_to_html(const ParsedTable &table) {
    using R = Result<luisa::string>;
    if (table.rows == 0 || table.cols == 0) {
        return R::failure(StatusCode::invalid_argument, "cannot serialize an empty table", "table");
    }
    luisa::string out = "<table>";
    if (!table.caption.empty()) {
        out += "<caption>" + encode_entities(table.caption) + "</caption>";
    }
    // Emit cells row by row; a cell is emitted at its origin only.
    for (uint32_t r = 0; r < table.rows; r++) {
        out += "<tr>";
        for (uint32_t c = 0; c < table.cols; c++) {
            const int32_t idx = table.grid[static_cast<size_t>(r) * table.cols + c];
            if (idx < 0) {
                return R::failure(StatusCode::invalid_argument, "table grid has an uncovered slot", "table");
            }
            const auto &cell = table.cells[static_cast<size_t>(idx)];
            if (cell.row != r || cell.col != c) { continue; }// span continuation
            const char *tag = cell.is_header ? "th" : "td";
            out += "<";
            out += tag;
            if (cell.col_span > 1) { out += luisa::format(" colspan=\"{}\"", cell.col_span); }
            if (cell.row_span > 1) { out += luisa::format(" rowspan=\"{}\"", cell.row_span); }
            out += ">";
            out += encode_entities(cell.text);
            out += "</";
            out += tag;
            out += ">";
        }
        out += "</tr>";
    }
    out += "</table>";
    return R::success(std::move(out));
}

}// namespace luisa::ocr::output

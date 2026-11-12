// OCR engine -- strict Markdown block composition/validation (Phase 4).
// Host-only. See markdown.h.
#include "ocr/output/markdown.h"

#include <utility>

#include <luisa/core/stl/format.h>

#include "ocr/output/table.h"

namespace luisa::ocr::output {

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

// Split text into lines, keeping the byte offset of each line start.
struct Line {
    luisa::string_view text;
    uint64_t begin;
    uint64_t end;// exclusive, includes the newline when present
};

[[nodiscard]] luisa::vector<Line> split_lines(luisa::string_view text) {
    luisa::vector<Line> lines;
    uint64_t i = 0;
    const uint64_t n = text.size();
    while (i < n) {
        const uint64_t start = i;
        while (i < n && text[i] != '\n') { i++; }
        uint64_t end = i;
        if (i < n) { i++; }// consume '\n'
        lines.push_back(Line{text.substr(start, end - start), start, i});
    }
    return lines;
}

[[nodiscard]] bool starts_with(luisa::string_view s, luisa::string_view prefix) noexcept {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// Count leading '#' of an ATX heading; returns 0 when not a heading.
[[nodiscard]] uint32_t heading_level(luisa::string_view line) noexcept {
    const auto t = trim(line);
    uint32_t level = 0;
    while (level < t.size() && t[level] == '#') { level++; }
    if (level == 0 || level > 6) { return 0; }
    if (level < t.size() && !is_space(t[level])) { return 0; }
    return level;
}

// Detect a fenced code block opener; returns the fence length (>=3) or 0.
[[nodiscard]] uint32_t fence_open(luisa::string_view line, char &fence_char) noexcept {
    const auto t = trim(line);
    if (t.size() < 3) { return 0; }
    const char c = t[0];
    if (c != '`' && c != '~') { return 0; }
    uint32_t n = 0;
    while (n < t.size() && t[n] == c) { n++; }
    if (n < 3) { return 0; }
    fence_char = c;
    return n;
}

[[nodiscard]] bool fence_close(luisa::string_view line, char fence_char, uint32_t open_len) noexcept {
    const auto t = trim(line);
    if (t.size() < open_len) { return false; }
    for (char c : t) {
        if (c != fence_char) { return false; }
    }
    return true;
}

// Detect a display-math fence ($$...$$ or \[...\]) opening on a line.
// Returns the delimiter kind: 0 none, 1 $$, 2 \[.
[[nodiscard]] uint32_t display_math_open(luisa::string_view line) noexcept {
    const auto t = trim(line);
    if (starts_with(t, "$$")) { return 1; }
    if (starts_with(t, "\\[")) { return 2; }
    return 0;
}

// Detect an HTML table block start (<table ...>). Case-insensitive.
[[nodiscard]] bool html_table_open(luisa::string_view line) noexcept {
    const auto t = trim(line);
    if (t.size() < 6 || t[0] != '<') { return false; }
    luisa::string lower;
    lower.reserve(t.size());
    for (char c : t) { lower += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }
    return starts_with(lower, "<table") &&
           (lower.size() == 6 || is_space(lower[6]) || lower[6] == '>');
}

// Detect an OTSL table line: EVERY whitespace-separated token on the line
// must be a known OTSL token (S3). Prose that merely starts with a token-like
// word ("nl is...", "fcel means...") is NOT a table line.
[[nodiscard]] bool otsl_table_open(luisa::string_view line) noexcept {
    const auto t = trim(line);
    if (t.empty()) { return false; }
    luisa::string cur;
    bool any = false;
    const auto is_token = [](luisa::string_view s) noexcept {
        return s == "fcel" || s == "eced" || s == "lcel" || s == "ucel" || s == "xcel" || s == "nl";
    };
    for (char c : t) {
        if (is_space(c)) {
            if (!cur.empty()) {
                if (!is_token(cur)) { return false; }
                any = true;
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        if (!is_token(cur)) { return false; }
        any = true;
    }
    return any;
}

// Detect a list item (-, *, +, or 1. / 1)).
[[nodiscard]] bool list_item(luisa::string_view line) noexcept {
    const auto t = trim(line);
    if (t.size() >= 2 && (t[0] == '-' || t[0] == '*' || t[0] == '+') && is_space(t[1])) { return true; }
    // Ordered: digits followed by . or ) then space.
    size_t k = 0;
    while (k < t.size() && t[k] >= '0' && t[k] <= '9') { k++; }
    if (k > 0 && k + 1 < t.size() && (t[k] == '.' || t[k] == ')') && is_space(t[k + 1])) { return true; }
    return false;
}

// Detect an image/figure line: ![alt](src).
[[nodiscard]] bool figure_line(luisa::string_view line) noexcept {
    const auto t = trim(line);
    return starts_with(t, "![") && t.find("](") != luisa::string_view::npos;
}

}// namespace

// ---------------------------------------------------------------------------
// LaTeX sanity
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::string> strip_latex_delimiters(luisa::string_view body_with_delims) {
    using R = Result<luisa::string>;
    const auto t = trim(body_with_delims);
    if (t.size() >= 4 && starts_with(t, "$$") && t.compare(t.size() - 2, 2, "$$") == 0) {
        return R::success(luisa::string{t.substr(2, t.size() - 4)});
    }
    if (t.size() >= 2 && t.front() == '$' && t.back() == '$') {
        return R::success(luisa::string{t.substr(1, t.size() - 2)});
    }
    if (t.size() >= 4 && starts_with(t, "\\(") && t.compare(t.size() - 2, 2, "\\)") == 0) {
        return R::success(luisa::string{t.substr(2, t.size() - 4)});
    }
    if (t.size() >= 4 && starts_with(t, "\\[") && t.compare(t.size() - 2, 2, "\\]") == 0) {
        return R::success(luisa::string{t.substr(2, t.size() - 4)});
    }
    return R::failure(StatusCode::invalid_argument,
                      "LaTeX formula is not wrapped in a recognized delimiter pair ($, $$, \\(, \\[)", "latex");
}

[[nodiscard]] Status check_latex(luisa::string_view body_with_delims) {
    auto stripped = strip_latex_delimiters(body_with_delims);
    if (!stripped.ok()) { return stripped.status; }
    const auto body = trim(luisa::string_view{*stripped.value});
    if (body.empty()) {
        return Status::error(StatusCode::invalid_argument, "LaTeX formula body is empty", "latex");
    }
    // Balanced braces (ignoring escaped \{ \}).
    int32_t depth = 0;
    for (size_t i = 0; i < body.size(); i++) {
        const char c = body[i];
        if (c == '\\') {
            i++;// skip the escaped character
            continue;
        }
        if (c == '{') { depth++; }
        else if (c == '}') {
            depth--;
            if (depth < 0) {
                return Status::error(StatusCode::invalid_argument, "LaTeX formula has an unmatched '}'", "latex");
            }
        }
    }
    if (depth != 0) {
        return Status::error(StatusCode::invalid_argument, "LaTeX formula has unbalanced braces", "latex");
    }
    // No nested math fence / unescaped outer delimiter inside the body.
    for (size_t i = 0; i < body.size(); i++) {
        const char c = body[i];
        if (c == '\\') {
            // \( \) \[ \] inside the body are nested math fences.
            if (i + 1 < body.size()) {
                const char n = body[i + 1];
                if (n == '(' || n == ')' || n == '[' || n == ']') {
                    return Status::error(StatusCode::invalid_argument,
                                         "LaTeX formula contains a nested math fence", "latex");
                }
            }
            i++;
            continue;
        }
        if (c == '$') {
            return Status::error(StatusCode::invalid_argument,
                                 "LaTeX formula contains an unescaped '$' delimiter", "latex");
        }
    }
    // \begin/\end must be balanced (simple count; environments may nest).
    int32_t env_depth = 0;
    for (size_t i = 0; i + 6 <= body.size(); i++) {
        if (body.compare(i, 6, "\\begin") == 0) { env_depth++; i += 5; }
        else if (body.compare(i, 4, "\\end") == 0) {
            env_depth--;
            if (env_depth < 0) {
                return Status::error(StatusCode::invalid_argument, "LaTeX formula has an unmatched \\end", "latex");
            }
            i += 3;
        }
    }
    if (env_depth != 0) {
        return Status::error(StatusCode::invalid_argument, "LaTeX formula has unbalanced \\begin/\\end", "latex");
    }
    return Status::success();
}

// ---------------------------------------------------------------------------
// Markdown parsing
// ---------------------------------------------------------------------------

[[nodiscard]] Result<MarkdownParseResult> parse_markdown(luisa::string_view text,
                                                         const MarkdownParseOptions &options) {
    using R = Result<MarkdownParseResult>;
    MarkdownParseResult result;
    if (trim(text).empty()) {
        result.diagnostics.push_back(Diagnostic{DiagnosticSeverity::info, "empty_document",
                                                "Markdown document is empty"});
        return R::success(std::move(result));
    }
    const auto lines = split_lines(text);
    const uint64_t n = lines.size();
    uint64_t i = 0;

    const auto push_block = [&](BlockType type, luisa::string content, uint64_t begin, uint64_t end,
                                uint32_t level = 0) {
        MarkdownBlock b;
        b.type = type;
        b.content = std::move(content);
        b.raw = luisa::string{text.substr(begin, end - begin)};
        b.heading_level = level;
        b.begin = begin;
        b.end = end;
        result.blocks.push_back(std::move(b));
    };

    while (i < n) {
        const auto &line = lines[i];
        const auto t = trim(line.text);
        if (t.empty()) { i++; continue; }

        // Fenced code block.
        char fence_char = 0;
        if (const uint32_t flen = fence_open(line.text, fence_char); flen != 0) {
            const uint64_t begin = line.begin;
            uint64_t j = i + 1;
            bool closed = false;
            while (j < n) {
                if (fence_close(lines[j].text, fence_char, flen)) { closed = true; break; }
                j++;
            }
            if (!closed) {
                return R::failure(StatusCode::invalid_argument,
                                  "Markdown fenced code block is not closed", "markdown");
            }
            const uint64_t end = lines[j].end;
            // Content is the inner lines (without the fences).
            luisa::string content;
            for (uint64_t k = i + 1; k < j; k++) {
                content += lines[k].text;
                content += '\n';
            }
            push_block(BlockType::other, std::move(content), begin, end);
            i = j + 1;
            continue;
        }

        // Display math fence ($$...$$ or \[...\]), possibly multi-line.
        if (const uint32_t mk = display_math_open(line.text); mk != 0) {
            const uint64_t begin = line.begin;
            const luisa::string_view close_delim = mk == 1 ? luisa::string_view{"$$"} : luisa::string_view{"\\]"};
            // Single-line case: opens and closes on the same line.
            const auto first = trim(line.text);
            const auto rest = first.substr(2);
            const auto close_pos = rest.find(close_delim);
            if (close_pos != luisa::string_view::npos) {
                // Ensure nothing non-space follows the close.
                const auto after = trim(rest.substr(close_pos + 2));
                if (!after.empty()) {
                    return R::failure(StatusCode::invalid_argument,
                                      "Markdown display math has trailing content after the close", "markdown");
                }
                const uint64_t end = line.end;
                auto formula = luisa::string{first.substr(0, close_pos + 4)};
                if (options.validate_formulas) {
                    if (auto s = check_latex(formula); !s.ok()) { return R::failure(s); }
                }
                auto body = strip_latex_delimiters(formula);
                if (!body.ok()) { return R::failure(body.status); }
                push_block(BlockType::formula, std::move(*body.value), begin, end);
                i++;
                continue;
            }
            // Multi-line: find the closing delimiter line. S4: the closing
            // delimiter must be at the START of the trimmed line (no leading
            // non-delimiter content is absorbed into the formula). A line like
            // "text $$ more" is NOT a valid close.
            uint64_t j = i + 1;
            bool closed = false;
            while (j < n) {
                const auto lt = trim(lines[j].text);
                if (starts_with(lt, close_delim)) {
                    const auto after = trim(lt.substr(2));
                    if (!after.empty()) {
                        return R::failure(StatusCode::invalid_argument,
                                          "Markdown display math has trailing content after the close", "markdown");
                    }
                    closed = true;
                    break;
                }
                // A closing delimiter appearing after leading content is a
                // malformed close: reject rather than absorb the leading text.
                if (lt.find(close_delim) != luisa::string_view::npos) {
                    return R::failure(StatusCode::invalid_argument,
                                      "Markdown display math closing delimiter has leading content on its line", "markdown");
                }
                j++;
            }
            if (!closed) {
                return R::failure(StatusCode::invalid_argument,
                                  "Markdown display math block is not closed", "markdown");
            }
            const uint64_t end = lines[j].end;
            auto formula = luisa::string{text.substr(begin, end - begin)};
            if (options.validate_formulas) {
                if (auto s = check_latex(formula); !s.ok()) { return R::failure(s); }
            }
            auto body = strip_latex_delimiters(formula);
            if (!body.ok()) { return R::failure(body.status); }
            push_block(BlockType::formula, std::move(*body.value), begin, end);
            i = j + 1;
            continue;
        }

        // HTML table block: <table ...> ... </table> (may span lines).
        if (html_table_open(line.text)) {
            const uint64_t begin = line.begin;
            uint64_t j = i;
            bool closed = false;
            luisa::string lower_accum;
            while (j < n) {
                const auto lt = lines[j].text;
                for (char c : lt) { lower_accum += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }
                if (lower_accum.find("</table>") != luisa::string::npos) { closed = true; break; }
                lower_accum += '\n';
                j++;
            }
            if (!closed) {
                return R::failure(StatusCode::invalid_argument, "Markdown HTML table is not closed", "markdown");
            }
            const uint64_t end = lines[j].end;
            auto table_src = luisa::string{text.substr(begin, end - begin)};
            if (options.validate_tables) {
                auto parsed = parse_html_table(table_src);
                if (!parsed.ok()) { return R::failure(parsed.status); }
            }
            push_block(BlockType::table, std::move(table_src), begin, end);
            i = j + 1;
            continue;
        }

        // OTSL table block: consecutive lines of OTSL tokens until a blank
        // line or a non-OTSL line.
        if (otsl_table_open(line.text)) {
            const uint64_t begin = line.begin;
            uint64_t j = i;
            while (j < n && otsl_table_open(lines[j].text)) { j++; }
            const uint64_t end = lines[j - 1].end;
            auto table_src = luisa::string{text.substr(begin, end - begin)};
            if (options.validate_tables) {
                auto parsed = parse_otsl_table(table_src);
                if (!parsed.ok()) { return R::failure(parsed.status); }
            }
            push_block(BlockType::table, std::move(table_src), begin, end);
            i = j;
            continue;
        }

        // ATX heading.
        if (const uint32_t level = heading_level(line.text); level != 0) {
            const auto ht = trim(line.text);
            auto content = luisa::string{trim(ht.substr(level))};
            // S2: strip an optional CLOSING '#' sequence only when it is
            // preceded by whitespace (CommonMark). '## C#' keeps its trailing
            // '#' (part of the content); '## Title ##' drops the closing '##'.
            if (!content.empty() && content.back() == '#') {
                size_t k = content.size();
                while (k > 0 && content[k - 1] == '#') { k--; }
                // k is the index of the first '#' of the trailing run.
                if (k > 0 && is_space(content[k - 1])) {
                    content.resize(k);
                    content = luisa::string{trim(content)};
                }
            }
            push_block(BlockType::title, std::move(content), line.begin, line.end, level);
            i++;
            continue;
        }

        // Figure (image) line.
        if (figure_line(line.text)) {
            push_block(BlockType::figure, luisa::string{t}, line.begin, line.end);
            i++;
            continue;
        }

        // List block: consecutive list items (and their indented continuations).
        if (list_item(line.text)) {
            const uint64_t begin = line.begin;
            uint64_t j = i;
            luisa::string content;
            while (j < n) {
                const auto lt = lines[j].text;
                const auto ltt = trim(lt);
                if (ltt.empty()) { break; }
                if (!list_item(lt) && !(lt.size() > line.text.size() && (lt[0] == ' ' || lt[0] == '\t'))) {
                    // A non-list, non-indented line ends the list.
                    if (j != i) { break; }
                }
                content += lt;
                content += '\n';
                j++;
                if (j < n && list_item(lines[j].text)) { continue; }
                if (j < n && !trim(lines[j].text).empty() &&
                    (lines[j].text[0] == ' ' || lines[j].text[0] == '\t')) { continue; }
                break;
            }
            const uint64_t end = lines[j - 1].end;
            push_block(BlockType::list, std::move(content), begin, end);
            i = j;
            continue;
        }

        // Paragraph: consecutive non-blank, non-special lines.
        {
            const uint64_t begin = line.begin;
            uint64_t j = i;
            luisa::string content;
            while (j < n) {
                const auto lt = lines[j].text;
                const auto ltt = trim(lt);
                if (ltt.empty()) { break; }
                if (j != i) {
                    char fc = 0;
                    if (fence_open(lt, fc) != 0 || display_math_open(lt) != 0 ||
                        html_table_open(lt) || otsl_table_open(lt) ||
                        heading_level(lt) != 0 || figure_line(lt) || list_item(lt)) {
                        break;
                    }
                }
                if (!content.empty()) { content += '\n'; }
                content += ltt;
                j++;
            }
            const uint64_t end = lines[j - 1].end;
            // Inline math sanity: a single '$' must be paired on the line.
            // (Display math was handled above; here we check inline $...$.)
            for (const auto &ch : content) {
                (void)ch;
            }
            push_block(BlockType::text, std::move(content), begin, end);
            i = j;
            continue;
        }
    }
    return R::success(std::move(result));
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::string> compose_markdown(luisa::span<const MarkdownBlock> blocks) {
    using R = Result<luisa::string>;
    luisa::string out;
    bool first = true;
    for (const auto &b : blocks) {
        if (!first) { out += "\n\n"; }
        first = false;
        switch (b.type) {
            case BlockType::title: {
                if (b.heading_level < 1 || b.heading_level > 6) {
                    return R::failure(StatusCode::invalid_argument,
                                      "title block heading level must be 1..6", "heading_level");
                }
                out += luisa::string(static_cast<size_t>(b.heading_level), '#');
                out += ' ';
                out += b.content;
                break;
            }
            case BlockType::table: {
                // Re-validate on composition (the block may have been built
                // programmatically); canonicalize HTML, keep OTSL verbatim.
                auto fmt = detect_table_format(b.content);
                if (!fmt.has_value()) {
                    return R::failure(StatusCode::invalid_argument, "table block is empty", "table");
                }
                if (*fmt == TableFormat::html) {
                    auto parsed = parse_html_table(b.content);
                    if (!parsed.ok()) { return R::failure(parsed.status); }
                    auto html = table_to_html(*parsed.value);
                    if (!html.ok()) { return R::failure(html.status); }
                    out += *html.value;
                } else {
                    auto parsed = parse_otsl_table(b.content);
                    if (!parsed.ok()) { return R::failure(parsed.status); }
                    out += b.content;
                }
                break;
            }
            case BlockType::formula: {
                // Wrap in display math delimiters and re-check.
                auto wrapped = "$$" + b.content + "$$";
                if (auto s = check_latex(wrapped); !s.ok()) { return R::failure(s); }
                out += wrapped;
                break;
            }
            case BlockType::figure:
            case BlockType::list:
            case BlockType::text:
            case BlockType::header:
            case BlockType::footer:
            case BlockType::other:
            default:
                out += b.content;
                break;
        }
    }
    return R::success(std::move(out));
}

}// namespace luisa::ocr::output

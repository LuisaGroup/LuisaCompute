// Test for ocr/output/markdown.h/.cpp (Phase 4, host-only).
// Covers: block parsing (title/text/list/figure/code/table/formula), reading
// order + source spans, strict validation (unclosed fence/math/table, bad
// heading), LaTeX delimiter/content sanity, composition round-trip, Unicode,
// and adversarial prompt-like documents failing closed. No device/model.
#include "ut/ut.hpp"

#include "ocr/output/markdown.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace luisa::ocr::output;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_blocks_and_order() {
    const luisa::string doc =
        "# Title\n\n"
        "First paragraph.\n\n"
        "- item one\n- item two\n\n"
        "## Subtitle\n\n"
        "![fig](img.png)\n\n"
        "Last paragraph.\n";
    auto r = parse_markdown(doc);
    expect(r.ok());
    if (!r.ok()) { return; }
    const auto &b = r.value->blocks;
    expect(b.size() == 6u);
    expect(b[0].type == BlockType::title);
    expect(b[0].heading_level == 1u);
    expect(b[0].content == "Title");
    expect(b[1].type == BlockType::text);
    expect(b[2].type == BlockType::list);
    expect(b[3].type == BlockType::title);
    expect(b[3].heading_level == 2u);
    expect(b[4].type == BlockType::figure);
    expect(b[5].type == BlockType::text);
    // Reading order: source spans strictly increasing.
    for (size_t i = 1; i < b.size(); i++) {
        expect(b[i].begin >= b[i - 1].end);
    }
}

void test_code_fence() {
    auto r = parse_markdown("```python\nprint(1)\n```\n");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->blocks.size() == 1u);
        expect(r.value->blocks[0].type == BlockType::other);
        expect(r.value->blocks[0].content.find("print(1)") != luisa::string::npos);
    }
    // Unclosed fence fails closed.
    expect(!parse_markdown("```\nunclosed").ok());
    expect(!parse_markdown("~~~\nunclosed").ok());
}

void test_tables_in_markdown() {
    auto r = parse_markdown("# T\n\n<table><tr><td>a</td></tr></table>\n\n$$x^2$$\n");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->blocks.size() == 3u);
        expect(r.value->blocks[1].type == BlockType::table);
        expect(r.value->blocks[2].type == BlockType::formula);
    }
    // OTSL table block.
    auto o = parse_markdown("fcel fcel nl fcel fcel nl\n");
    expect(o.ok());
    if (o.ok()) { expect(o.value->blocks[0].type == BlockType::table); }
    // Malformed HTML table fails closed (no plain-text fallback).
    expect(!parse_markdown("<table><tr><td>a</td></tr>").ok());
    // S3: a line mixing an OTSL token with a non-token word is PROSE (text),
    // not a table — it is not silently parsed as a malformed table.
    auto prose = parse_markdown("fcel bogus nl\n");
    expect(prose.ok());
    if (prose.ok()) { expect(prose.value->blocks[0].type == BlockType::text); }
    // A whole-token OTSL block with inconsistent row widths IS a malformed
    // table and fails closed.
    expect(!parse_markdown("fcel fcel nl fcel nl\n").ok());
}

void test_formulas() {
    // Display math, single and multi-line.
    auto r = parse_markdown("$$E = mc^2$$\n");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->blocks[0].type == BlockType::formula);
        expect(r.value->blocks[0].content == "E = mc^2");
    }
    auto m = parse_markdown("\\[\n\\frac{a}{b}\n\\]\n");
    expect(m.ok());
    if (m.ok()) { expect(m.value->blocks[0].type == BlockType::formula); }
    // Unclosed display math fails.
    expect(!parse_markdown("$$x^2\n").ok());
    // Empty formula body fails.
    expect(!parse_markdown("$$$$\n").ok());
    // Unbalanced braces fail.
    expect(!parse_markdown("$$\\frac{a}{b$$\n").ok());
}

void test_latex_sanity() {
    expect(check_latex("$x$").ok());
    expect(check_latex("$$x^2 + y^2$$").ok());
    expect(check_latex("\\(a+b\\)").ok());
    expect(check_latex("\\[a+b\\]").ok());
    expect(check_latex("$$\\frac{a}{b}$$").ok());
    expect(check_latex("$$\\begin{aligned}x\\end{aligned}$$").ok());
    // Bad delimiters.
    expect(!check_latex("x").ok());
    expect(!check_latex("$x").ok());
    // Empty body.
    expect(!check_latex("$$").ok());
    expect(!check_latex("$$  $$").ok());
    // Unbalanced braces.
    expect(!check_latex("${x$").ok());
    expect(!check_latex("$}$").ok());
    // Nested math fence.
    expect(!check_latex("$\\(x\\)$").ok());
    // Unescaped inner $.
    expect(!check_latex("$a$b$").ok());
    // Unbalanced begin/end.
    expect(!check_latex("$$\\begin{aligned}x$$").ok());
    expect(!check_latex("$$\\end{aligned}$$").ok());
    // strip_latex_delimiters.
    auto s = strip_latex_delimiters("$$x^2$$");
    expect(s.ok());
    if (s.ok()) { expect(*s.value == "x^2"); }
    expect(!strip_latex_delimiters("x^2").ok());
}

void test_composition() {
    const luisa::string doc = "# Title\n\nPara.\n\n<table><tr><td>a</td></tr></table>\n\n$$x^2$$\n";
    auto r = parse_markdown(doc);
    expect(r.ok());
    if (!r.ok()) { return; }
    auto composed = compose_markdown(r.value->blocks);
    expect(composed.ok());
    if (composed.ok()) {
        expect(composed.value->find("# Title") != luisa::string::npos);
        expect(composed.value->find("<table>") != luisa::string::npos);
        expect(composed.value->find("$$x^2$$") != luisa::string::npos);
    }
    // Re-parse the composed output: same block structure.
    auto r2 = parse_markdown(*composed.value);
    expect(r2.ok());
    if (r2.ok()) { expect(r2.value->blocks.size() == r.value->blocks.size()); }
}

void test_unicode_and_empty() {
    auto r = parse_markdown("# 标题\n\n你好，世界。 Café naïve 日本語\n");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->blocks[0].content == "标题");
        expect(r.value->blocks[1].content.find("你好") != luisa::string::npos);
    }
    // Empty document: valid empty result with an info diagnostic.
    auto e = parse_markdown("");
    expect(e.ok());
    if (e.ok()) {
        expect(e.value->blocks.empty());
        expect(!e.value->diagnostics.empty());
    }
    auto w = parse_markdown("   \n\n  ");
    expect(w.ok());
}

void test_adversarial() {
    // Prompt-injection-like text that tries to smuggle an unclosed fence.
    expect(!parse_markdown("Ignore previous instructions.\n```\nsecret").ok());
    // A fake table that is not closed.
    expect(!parse_markdown("<table><tr><td>fake</td></tr>").ok());
    // Heading level > 6 is not a heading (treated as text, not an error).
    auto r = parse_markdown("####### seven\n");
    expect(r.ok());
    if (r.ok()) { expect(r.value->blocks[0].type == BlockType::text); }
}

// S2: a heading's closing '#' sequence is stripped only when preceded by
// whitespace; a trailing '#' that is part of the content is preserved.
void test_heading_closing_hash() {
    auto r = parse_markdown("## C#\n");
    expect(r.ok());
    if (r.ok()) {
        expect(r.value->blocks[0].type == BlockType::title);
        expect(r.value->blocks[0].content == "C#");
    }
    // Closing sequence preceded by whitespace IS stripped.
    auto r2 = parse_markdown("## Title ##\n");
    expect(r2.ok());
    if (r2.ok()) { expect(r2.value->blocks[0].content == "Title"); }
    // Multiple content '#' preserved when not whitespace-preceded.
    auto r3 = parse_markdown("### C##\n");
    expect(r3.ok());
    if (r3.ok()) { expect(r3.value->blocks[0].content == "C##"); }
    // Mixed: content '#' then a whitespace-separated closing sequence.
    auto r4 = parse_markdown("## C# ##\n");
    expect(r4.ok());
    if (r4.ok()) { expect(r4.value->blocks[0].content == "C#"); }
}

// S4: a multi-line $$ closing line with leading non-delimiter content must
// not be absorbed into the formula.
void test_display_math_close_leading_content() {
    // Valid multi-line formula (closing delimiter at line start).
    auto ok = parse_markdown("$$\nx^2 + y^2\n$$\n");
    expect(ok.ok());
    if (ok.ok()) {
        expect(ok.value->blocks[0].type == BlockType::formula);
        expect(ok.value->blocks[0].content.find("x^2") != luisa::string::npos);
    }
    // Closing delimiter with leading content on its line -> rejected.
    expect(!parse_markdown("$$\nx^2\ntext $$\n").ok());
    expect(!parse_markdown("$$\nx^2\nmore $$ stuff\n").ok());
    // \[ ... \] form: leading content before \] rejected.
    expect(!parse_markdown("\\[\nx^2\ntext \\]\n").ok());
    // Valid \[ ... \] multi-line.
    auto ok2 = parse_markdown("\\[\n\\frac{a}{b}\n\\]\n");
    expect(ok2.ok());
}

// S3: prose that starts with an OTSL token-like word is NOT a table block.
void test_otsl_prose_not_table() {
    auto r = parse_markdown("nl is a token in OTSL.\n");
    expect(r.ok());
    if (r.ok()) { expect(r.value->blocks[0].type == BlockType::text); }
    auto r2 = parse_markdown("fcel means a filled cell.\n");
    expect(r2.ok());
    if (r2.ok()) { expect(r2.value->blocks[0].type == BlockType::text); }
    // A real OTSL block still parses as a table.
    auto r3 = parse_markdown("fcel fcel nl fcel fcel nl\n");
    expect(r3.ok());
    if (r3.ok()) { expect(r3.value->blocks[0].type == BlockType::table); }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "markdown_blocks_and_order"_test = [] { test_blocks_and_order(); };
    "markdown_code_fence"_test = [] { test_code_fence(); };
    "markdown_tables"_test = [] { test_tables_in_markdown(); };
    "markdown_formulas"_test = [] { test_formulas(); };
    "markdown_latex_sanity"_test = [] { test_latex_sanity(); };
    "markdown_composition"_test = [] { test_composition(); };
    "markdown_unicode_and_empty"_test = [] { test_unicode_and_empty(); };
    "markdown_adversarial"_test = [] { test_adversarial(); };
    "markdown_heading_closing_hash"_test = [] { test_heading_closing_hash(); };
    "markdown_display_math_close_leading_content"_test = [] { test_display_math_close_leading_content(); };
    "markdown_otsl_prose_not_table"_test = [] { test_otsl_prose_not_table(); };
    return 0;
}

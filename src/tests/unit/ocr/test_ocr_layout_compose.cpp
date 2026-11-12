// Test for the Phase 6 host composition + public bridges (host-only):
// ocr/models/paddle_ocr_vl_result, ocr/models/open_doc_result,
// ocr/runtime/layout_result. Covers: type-specific adapters (text/formula/
// table) and invalid-output fail-closed; Markdown/JSON/plain/spotting
// composition; strict structured-output errors (no plain-text fallback);
// public RecognizeResult schema (lines/blocks/source page/box/polygon/type/
// content/reading order); confidence only from validated detector/CTC scores
// labeled internal/test (no pseudo-probability for generated content); empty
// pages; cross-page ordering.
#include "ut/ut.hpp"

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/open_doc_result.h"
#include "ocr/models/paddle_ocr_vl_result.h"
#include "ocr/output/json_output.h"
#include "ocr/runtime/layout_result.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Build a plain layout run with the given regions (one page).
[[nodiscard]] LayoutRunResult make_run(luisa::vector<LayoutRegionResult> regions, uint32_t page = 0) {
    LayoutRunResult run;
    LayoutPageResult p;
    p.page_index = page;
    p.source_width = 100;
    p.source_height = 100;
    p.regions = std::move(regions);
    run.pages.push_back(std::move(p));
    return run;
}

[[nodiscard]] LayoutRegionResult region(uint32_t id, LayoutBlockType type, const char *content,
                                        uint32_t order, float x0 = 0, float y0 = 0, float x1 = 10, float y1 = 10) {
    LayoutRegionResult r;
    r.region_id = id;
    r.page_index = 0;
    r.reading_order = order;
    r.block_type = type;
    r.content = content;
    r.source_box = output::DbRect{x0, y0, x1, y1};
    r.polygon = {output::DbPoint{x0, y0}, output::DbPoint{x1, y0},
                 output::DbPoint{x1, y1}, output::DbPoint{x0, y1}};
    return r;
}

[[nodiscard]] ModelMetadata metadata() {
    ModelMetadata m;
    m.model_id = ModelId::paddle_ocr_vl_1_6;
    m.target_name = "paddle_ocr_vl_1_6";
    m.backend = "vk";
    m.support_state = SupportState::planned;
    m.validated = false;
    return m;
}

}// namespace

// ---------------------------------------------------------------------------
// OpenDoc type-specific adapters.
// ---------------------------------------------------------------------------
void test_opendoc_adapters() {
    // Text: verbatim trimmed.
    auto text = open_doc_adapt_region(LayoutBlockType::text, "  hello world  ");
    expect(text.ok());
    if (text.ok()) { expect(*text.value == "hello world"); }
    // Empty text: empty content (blank region, not an error).
    auto empty = open_doc_adapt_region(LayoutBlockType::text, "   ");
    expect(empty.ok());
    if (empty.ok()) { expect(empty.value->empty()); }
    // Formula: valid LaTeX -> body without delimiters.
    auto formula = open_doc_adapt_region(LayoutBlockType::formula, "$$x^2 + y^2 = z^2$$");
    expect(formula.ok());
    if (formula.ok()) { expect(*formula.value == "x^2 + y^2 = z^2"); }
    // Invalid formula: unbalanced braces -> ERROR (no plain-text fallback).
    auto bad_formula = open_doc_adapt_region(LayoutBlockType::formula, "$$x^2 + {y^2$$");
    expect(!bad_formula.ok());
    expect(bad_formula.status.code == StatusCode::invalid_argument);
    // Formula without delimiters -> ERROR.
    auto no_delim = open_doc_adapt_region(LayoutBlockType::formula, "x^2 + y^2");
    expect(!no_delim.ok());
    // Table: valid OTSL -> canonical source.
    auto table = open_doc_adapt_region(LayoutBlockType::table, "fcel fcel nl fcel fcel nl");
    expect(table.ok());
    // Invalid table: ragged OTSL -> ERROR.
    auto bad_table = open_doc_adapt_region(LayoutBlockType::table, "fcel fcel nl fcel nl");
    expect(!bad_table.ok());
    // Invalid HTML table -> ERROR.
    auto bad_html = open_doc_adapt_region(LayoutBlockType::table, "<table><tr><td>a</td></table>");
    expect(!bad_html.ok());
}

// ---------------------------------------------------------------------------
// OpenDoc composition: plain / markdown / extract-refused.
// ---------------------------------------------------------------------------
void test_opendoc_compose() {
    auto run = make_run({
        region(0, LayoutBlockType::title, "Title", 0),
        region(1, LayoutBlockType::text, "Body text", 1),
        region(2, LayoutBlockType::formula, "$$a+b$$", 2),
    });
    // Plain: joined in reading order.
    auto plain = open_doc_compose(run, LayoutOutputMode::plain);
    expect(plain.ok());
    if (plain.ok()) {
        expect(plain.value->text == "Title\nBody text\na+b");
        expect(plain.value->blocks.size() == 3u);
        expect(plain.value->lines.size() == 3u);
    }
    // Markdown: title -> heading, formula -> display math.
    auto md = open_doc_compose(run, LayoutOutputMode::markdown);
    expect(md.ok());
    if (md.ok()) {
        expect(md.value->text.find("# Title") != luisa::string::npos);
        expect(md.value->text.find("$$a+b$$") != luisa::string::npos);
    }
    // Extract: OpenDoc does not declare extract -> unsupported.
    auto extract = open_doc_compose(run, LayoutOutputMode::extract);
    expect(!extract.ok());
    expect(extract.status.code == StatusCode::unsupported);
    // Malformed region content fails closed (no plain-text fallback).
    auto bad_run = make_run({region(0, LayoutBlockType::formula, "$$unbalanced {$$", 0)});
    auto bad = open_doc_compose(bad_run, LayoutOutputMode::markdown);
    expect(!bad.ok());
}

// ---------------------------------------------------------------------------
// Paddle composition: plain / markdown / spotting / extract.
// ---------------------------------------------------------------------------
void test_paddle_compose() {
    auto run = make_run({
        region(0, LayoutBlockType::title, "Heading", 0),
        region(1, LayoutBlockType::text, "Paragraph", 1),
        region(2, LayoutBlockType::table, "fcel fcel nl fcel fcel nl", 2),
    });
    auto plain = paddle_ocr_vl_compose(run, LayoutOutputMode::plain, "");
    expect(plain.ok());
    if (plain.ok()) { expect(plain.value->blocks.size() == 3u); }
    auto md = paddle_ocr_vl_compose(run, LayoutOutputMode::markdown, "");
    expect(md.ok());
    if (md.ok()) { expect(md.value->text.find("# Heading") != luisa::string::npos); }
    // Spotting: lines/blocks carry source coordinates.
    auto spot = paddle_ocr_vl_compose(run, LayoutOutputMode::spotting, "");
    expect(spot.ok());
    if (spot.ok()) {
        expect(spot.value->lines.size() == 3u);
        expect(!spot.value->lines[0].source_box.empty());
    }
    // Extract: valid JSON conforming to a schema.
    auto json_run = make_run({region(0, LayoutBlockType::text, "{\"name\":\"test\",\"value\":3}", 0)});
    const char *schema = R"({"type":"object","properties":{"name":{"type":"string"},"value":{"type":"integer"}},"required":["name","value"]})";
    auto extract = paddle_ocr_vl_compose(json_run, LayoutOutputMode::extract, schema);
    expect(extract.ok());
    if (extract.ok()) { expect(extract.value->text.find("\"name\"") != luisa::string::npos); }
    // Extract without schema -> ERROR.
    auto no_schema = paddle_ocr_vl_compose(json_run, LayoutOutputMode::extract, "");
    expect(!no_schema.ok());
    // Extract with non-conforming JSON -> ERROR (no plain fallback).
    auto bad_json_run = make_run({region(0, LayoutBlockType::text, "{\"name\":123}", 0)});
    auto bad_extract = paddle_ocr_vl_compose(bad_json_run, LayoutOutputMode::extract, schema);
    expect(!bad_extract.ok());
    // Malformed table -> ERROR.
    auto bad_table_run = make_run({region(0, LayoutBlockType::table, "<table><tr><td>x</td></table>", 0)});
    auto bad_md = paddle_ocr_vl_compose(bad_table_run, LayoutOutputMode::markdown, "");
    expect(!bad_md.ok());
}

// ---------------------------------------------------------------------------
// Public bridge: schema + confidence provenance.
// ---------------------------------------------------------------------------
void test_public_bridge() {
    // Generative content: NO confidence.
    auto run = make_run({region(0, LayoutBlockType::text, "generated text", 0)});
    auto doc = paddle_ocr_vl_compose(run, LayoutOutputMode::spotting, "");
    expect(doc.ok());
    if (!doc.ok()) { return; }
    auto result = paddle_ocr_vl_to_public(*doc.value, run, metadata());
    expect(result.blocks.size() == 1u);
    expect(result.lines.size() == 1u);
    expect(!result.blocks[0].has_confidence);
    expect(!result.lines[0].has_confidence);
    expect(result.blocks[0].source.page_index == 0u);
    expect(!result.blocks[0].source.box.empty());
    expect(result.blocks[0].source.polygon.size() == 4u);
    expect(result.metadata.model_id == ModelId::paddle_ocr_vl_1_6);
    expect(!result.metadata.validated);
    // JSON serialization of the full schema succeeds.
    auto json = output::serialize_result(result);
    expect(json.ok());
    if (json.ok()) {
        expect(json.value->find("\"blocks\"") != luisa::string::npos);
        expect(json.value->find("\"lines\"") != luisa::string::npos);
    }
    // CTC confidence (OpenDoc): carried through + labeled internal/test.
    auto ctc_run = make_run({region(0, LayoutBlockType::text, "ctc text", 0)});
    ctc_run.pages[0].regions[0].has_confidence = true;
    ctc_run.pages[0].regions[0].confidence = 0.87f;
    auto ctc_doc = open_doc_compose(ctc_run, LayoutOutputMode::plain);
    expect(ctc_doc.ok());
    if (!ctc_doc.ok()) { return; }
    auto ctc_result = open_doc_to_public(*ctc_doc.value, ctc_run, metadata());
    expect(ctc_result.blocks.size() == 1u);
    expect(ctc_result.blocks[0].has_confidence);
    expect(std::abs(ctc_result.blocks[0].confidence - 0.87f) < 1e-5f);
    bool labeled = false;
    for (const auto &d : ctc_result.diagnostics) {
        if (d.code == "confidence_internal_test") { labeled = true; }
    }
    expect(labeled) << "CTC confidence must be labeled internal/test";
}

// ---------------------------------------------------------------------------
// Empty page + cross-page ordering.
// ---------------------------------------------------------------------------
void test_empty_and_cross_page() {
    // Empty page: valid empty document.
    LayoutRunResult empty_run;
    LayoutPageResult p;
    p.page_index = 0;
    empty_run.pages.push_back(p);
    auto empty = paddle_ocr_vl_compose(empty_run, LayoutOutputMode::markdown, "");
    expect(empty.ok());
    if (empty.ok()) {
        expect(empty.value->text.empty());
        expect(empty.value->blocks.empty());
    }
    // Cross-page: pages preserved ascending, reading order within each page.
    LayoutRunResult multi;
    LayoutPageResult p0;
    p0.page_index = 0;
    p0.regions = {region(0, LayoutBlockType::text, "page0", 0)};
    LayoutPageResult p1;
    p1.page_index = 1;
    p1.regions = {region(1, LayoutBlockType::text, "page1", 0)};
    p1.regions[0].page_index = 1;
    multi.pages = {p0, p1};
    auto doc = paddle_ocr_vl_compose(multi, LayoutOutputMode::plain, "");
    expect(doc.ok());
    if (doc.ok()) {
        expect(doc.value->text == "page0\npage1");
        expect(doc.value->blocks.size() == 2u);
        expect(doc.value->blocks[0].page_index == 0u);
        expect(doc.value->blocks[1].page_index == 1u);
    }
}

static auto test_ocr_layout_compose_registration = [] {
    "layout_compose_opendoc_adapters"_test = [] { test_opendoc_adapters(); };
    "layout_compose_opendoc_compose"_test = [] { test_opendoc_compose(); };
    "layout_compose_paddle_compose"_test = [] { test_paddle_compose(); };
    "layout_compose_public_bridge"_test = [] { test_public_bridge(); };
    "layout_compose_empty_and_cross_page"_test = [] { test_empty_and_cross_page(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
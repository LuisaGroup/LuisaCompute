// OCR engine -- PaddleOCR-VL host composition + public bridge (Phase 6).
// HOST-ONLY TU. See the header for the contract.
#include "ocr/models/paddle_ocr_vl_result.h"

#include <algorithm>
#include <utility>

#include <luisa/core/stl/format.h>

#include "ocr/output/extract.h"
#include "ocr/output/markdown.h"
#include "ocr/output/table.h"
#include "ocr/runtime/layout_result.h"

namespace luisa::ocr {

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

// Adapt one region's raw generative text into type-specific canonical content.
// STRICT: a malformed table/formula is an ERROR, never plain text. An empty
// decode yields empty content (a blank region, not an error).
[[nodiscard]] Result<luisa::string> adapt_region(LayoutBlockType type, luisa::string_view text) {
    using R = Result<luisa::string>;
    const auto t = trim(text);
    switch (type) {
        case LayoutBlockType::table: {
            if (t.empty()) { return R::success(luisa::string{}); }
            auto parsed = output::parse_table(t);
            if (!parsed.ok()) { return R::failure(parsed.status); }
            // Canonical HTML for HTML tables; OTSL kept verbatim.
            if (parsed.value->format == output::TableFormat::html) {
                return output::table_to_html(*parsed.value);
            }
            return R::success(luisa::string{t});
        }
        case LayoutBlockType::formula: {
            if (t.empty()) { return R::success(luisa::string{}); }
            // The generative formula must be wrapped in a recognized delimiter
            // pair; validate and strip to the canonical body.
            if (auto s = output::check_latex(t); !s.ok()) { return R::failure(s); }
            return output::strip_latex_delimiters(t);
        }
        case LayoutBlockType::title:
        case LayoutBlockType::text:
        case LayoutBlockType::list:
        case LayoutBlockType::header:
        case LayoutBlockType::footer:
        case LayoutBlockType::figure:
        case LayoutBlockType::other:
        default:
            return R::success(luisa::string{t});
    }
}

// Build the Markdown block for one region (type-specific rendering).
[[nodiscard]] Result<output::MarkdownBlock> to_markdown_block(LayoutBlockType type, luisa::string_view content,
                                                              uint32_t heading_level) {
    using R = Result<output::MarkdownBlock>;
    output::MarkdownBlock block;
    block.content = luisa::string{content};
    switch (type) {
        case LayoutBlockType::title:
            block.type = BlockType::title;
            block.heading_level = heading_level == 0u ? 1u : heading_level;
            break;
        case LayoutBlockType::table:
            block.type = BlockType::table;
            break;
        case LayoutBlockType::formula:
            block.type = BlockType::formula;
            break;
        case LayoutBlockType::list:
            block.type = BlockType::list;
            break;
        case LayoutBlockType::figure:
            block.type = BlockType::figure;
            break;
        case LayoutBlockType::header:
            block.type = BlockType::header;
            break;
        case LayoutBlockType::footer:
            block.type = BlockType::footer;
            break;
        case LayoutBlockType::text:
        case LayoutBlockType::other:
        default:
            block.type = BlockType::text;
            break;
    }
    return R::success(std::move(block));
}

}// namespace

[[nodiscard]] Result<LayoutDocument> paddle_ocr_vl_compose(const LayoutRunResult &run,
                                                           LayoutOutputMode mode,
                                                           luisa::string_view extract_schema) {
    using R = Result<LayoutDocument>;
    LayoutDocument doc;
    doc.diagnostics = run.diagnostics;

    // Collect adapted regions in reading order (pages ascending, then order).
    struct Adapted {
        const LayoutRegionResult *region;
        luisa::string content;
    };
    luisa::vector<Adapted> adapted;
    for (const auto &page : run.pages) {
        for (const auto &region : page.regions) {
            auto content = adapt_region(region.block_type, region.content);
            if (!content.ok()) { return R::failure(content.status); }
            adapted.push_back(Adapted{&region, std::move(*content.value)});
        }
    }

    // Extract mode: the concatenated region content must be valid JSON conforming
    // to the schema. The canonical text is the serialized JSON.
    if (mode == LayoutOutputMode::extract) {
        if (extract_schema.empty()) {
            return R::failure(StatusCode::invalid_argument, "extract mode requires a schema", "extract_schema");
        }
        luisa::string joined;
        for (const auto &a : adapted) { joined += a.content; }
        auto extracted = output::extract_json(joined, extract_schema);
        if (!extracted.ok()) { return R::failure(extracted.status); }
        auto serialized = output::serialize_json(extracted.value->value);
        if (!serialized.ok()) { return R::failure(serialized.status); }
        doc.text = std::move(*serialized.value);
        for (const auto &d : extracted.value->diagnostics) { doc.diagnostics.push_back(d); }
        // Blocks/lines still carry the per-region geometry + raw content.
        for (const auto &a : adapted) {
            LayoutBlock block;
            block.type = a.region->block_type;
            block.content = a.content;
            block.page_index = a.region->page_index;
            block.reading_order = a.region->reading_order;
            block.source_box = a.region->source_box;
            block.polygon = a.region->polygon;
            block.has_confidence = false;// generative: no confidence
            doc.blocks.push_back(std::move(block));
        }
        return R::success(std::move(doc));
    }

    // Markdown mode: compose per-type Markdown blocks.
    if (mode == LayoutOutputMode::markdown) {
        luisa::vector<output::MarkdownBlock> blocks;
        blocks.reserve(adapted.size());
        for (const auto &a : adapted) {
            if (a.content.empty()) { continue; }// blank region: no block
            auto block = to_markdown_block(a.region->block_type, a.content, a.region->heading_level);
            if (!block.ok()) { return R::failure(block.status); }
            blocks.push_back(std::move(*block.value));
        }
        auto composed = output::compose_markdown(blocks);
        if (!composed.ok()) { return R::failure(composed.status); }
        doc.text = std::move(*composed.value);
    } else {
        // plain / spotting: join non-empty region texts in reading order.
        luisa::string joined;
        bool first = true;
        for (const auto &a : adapted) {
            if (a.content.empty()) { continue; }
            if (!first) { joined += '\n'; }
            first = false;
            joined += a.content;
        }
        doc.text = std::move(joined);
    }

    // Lines + blocks carry source geometry for every mode (spotting explicitly
    // requires coordinates; the other modes carry them too for provenance).
    uint32_t line_order = 0;
    for (const auto &a : adapted) {
        LayoutBlock block;
        block.type = a.region->block_type;
        block.content = a.content;
        block.page_index = a.region->page_index;
        block.reading_order = a.region->reading_order;
        block.source_box = a.region->source_box;
        block.polygon = a.region->polygon;
        block.has_confidence = false;// generative content: NO confidence
        block.confidence = 0.0f;
        doc.blocks.push_back(block);
        if (!a.content.empty()) {
            LayoutLine line;
            line.text = a.content;
            line.page_index = a.region->page_index;
            line.reading_order = line_order++;
            line.source_box = a.region->source_box;
            line.polygon = a.region->polygon;
            line.has_confidence = false;
            doc.lines.push_back(std::move(line));
        }
    }
    return R::success(std::move(doc));
}

[[nodiscard]] RecognizeResult paddle_ocr_vl_to_public(const LayoutDocument &doc, const LayoutRunResult &run,
                                                      const ModelMetadata &metadata) {
    return layout_document_to_public(doc, run, metadata);
}

}// namespace luisa::ocr
// OCR engine -- layout document -> public RecognizeResult bridge (Phase 6).
// HOST-ONLY TU. See the header for the contract.
#include "ocr/runtime/layout_result.h"

#include <utility>

#include <luisa/core/stl/format.h>

#include "ocr/output/extract.h"
#include "ocr/output/markdown.h"
#include "ocr/output/table.h"

namespace luisa::ocr {

namespace {

[[nodiscard]] SourceRegion to_source_region(uint32_t page_index, const output::DbRect &box,
                                            luisa::span<const output::DbPoint> polygon) {
    SourceRegion region;
    region.page_index = page_index;
    region.box = SourceBox{box.x0, box.y0, box.x1, box.y1};
    region.polygon.reserve(polygon.size());
    for (const auto &p : polygon) { region.polygon.push_back(SourcePoint{p.x, p.y}); }
    return region;
}

}// namespace

[[nodiscard]] RecognizeResult layout_document_to_public(const LayoutDocument &doc, const LayoutRunResult &run,
                                                        const ModelMetadata &metadata) {
    RecognizeResult result;
    result.text = doc.text;
    result.metadata = metadata;
    result.diagnostics = doc.diagnostics;
    for (const auto &d : run.diagnostics) { result.diagnostics.push_back(d); }

    bool any_confidence = false;
    result.lines.reserve(doc.lines.size());
    for (const auto &line : doc.lines) {
        OcrLine out;
        out.text = line.text;
        out.source = to_source_region(line.page_index, line.source_box, line.polygon);
        out.reading_order = line.reading_order;
        out.has_confidence = line.has_confidence;
        out.confidence = line.confidence;
        any_confidence = any_confidence || line.has_confidence;
        result.lines.push_back(std::move(out));
    }
    result.blocks.reserve(doc.blocks.size());
    for (const auto &block : doc.blocks) {
        OcrBlock out;
        out.type = static_cast<BlockType>(block.type);
        out.content = block.content;
        out.source = to_source_region(block.page_index, block.source_box, block.polygon);
        out.reading_order = block.reading_order;
        out.has_confidence = block.has_confidence;
        out.confidence = block.confidence;
        any_confidence = any_confidence || block.has_confidence;
        result.blocks.push_back(std::move(out));
    }
    // Confidence provenance: any carried score is an internal/test detector/CTC
    // source, never a calibrated generative probability. Make that explicit.
    if (any_confidence) {
        result.diagnostics.push_back(Diagnostic{
            DiagnosticSeverity::info, "confidence_internal_test",
            "confidence values are internal/test detector or CTC scores, not calibrated "
            "generative probabilities"});
    }
    result.counters.pages_processed = run.pages.size();
    result.counters.regions_processed = run.regions_processed;
    result.counters.patches_processed = run.patches_processed;
    result.counters.output_tokens = run.output_tokens;
    result.counters.resident_weight_bytes = run.weight_bytes;
    // Honest peak: weights + fixed workspace + peak per-region KV cache (the
    // orchestrator computes this; it previously reported workspace alone).
    result.counters.peak_device_bytes = run.peak_device_bytes;
    return result;
}

// ---------------------------------------------------------------------------
// Shared strict layout composition (Phase 7)
// ---------------------------------------------------------------------------

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
            if (parsed.value->format == output::TableFormat::html) {
                return output::table_to_html(*parsed.value);
            }
            return R::success(luisa::string{t});
        }
        case LayoutBlockType::formula: {
            if (t.empty()) { return R::success(luisa::string{}); }
            if (auto s = output::check_latex(t); !s.ok()) { return R::failure(s); }
            return output::strip_latex_delimiters(t);
        }
        default:
            return R::success(luisa::string{t});
    }
}

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
        case LayoutBlockType::table: block.type = BlockType::table; break;
        case LayoutBlockType::formula: block.type = BlockType::formula; break;
        case LayoutBlockType::list: block.type = BlockType::list; break;
        case LayoutBlockType::figure: block.type = BlockType::figure; break;
        case LayoutBlockType::header: block.type = BlockType::header; break;
        case LayoutBlockType::footer: block.type = BlockType::footer; break;
        default: block.type = BlockType::text; break;
    }
    return R::success(std::move(block));
}

}// namespace

[[nodiscard]] Result<LayoutDocument> layout_compose_strict(const LayoutRunResult &run,
                                                           LayoutOutputMode mode,
                                                           luisa::string_view extract_schema) {
    using R = Result<LayoutDocument>;
    LayoutDocument doc;
    doc.diagnostics = run.diagnostics;
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
        for (const auto &a : adapted) {
            LayoutBlock block;
            block.type = a.region->block_type;
            block.content = a.content;
            block.page_index = a.region->page_index;
            block.reading_order = a.region->reading_order;
            block.source_box = a.region->source_box;
            block.polygon = a.region->polygon;
            block.has_confidence = false;
            doc.blocks.push_back(std::move(block));
        }
        return R::success(std::move(doc));
    }
    if (mode == LayoutOutputMode::markdown) {
        luisa::vector<output::MarkdownBlock> blocks;
        blocks.reserve(adapted.size());
        for (const auto &a : adapted) {
            if (a.content.empty()) { continue; }
            auto block = to_markdown_block(a.region->block_type, a.content, a.region->heading_level);
            if (!block.ok()) { return R::failure(block.status); }
            blocks.push_back(std::move(*block.value));
        }
        auto composed = output::compose_markdown(blocks);
        if (!composed.ok()) { return R::failure(composed.status); }
        doc.text = std::move(*composed.value);
    } else {
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
    uint32_t line_order = 0;
    for (const auto &a : adapted) {
        LayoutBlock block;
        block.type = a.region->block_type;
        block.content = a.content;
        block.page_index = a.region->page_index;
        block.reading_order = a.region->reading_order;
        block.source_box = a.region->source_box;
        block.polygon = a.region->polygon;
        block.has_confidence = false;
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

}// namespace luisa::ocr
// OCR engine -- OpenDoc host adapters + composition + public bridge (Phase 6).
// HOST-ONLY TU. See the header for the contract.
#include "ocr/models/open_doc_result.h"

#include <utility>

#include <luisa/core/stl/format.h>

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

}// namespace

[[nodiscard]] Result<luisa::string> open_doc_adapt_region(LayoutBlockType block_type,
                                                          luisa::string_view decoded_text) {
    using R = Result<luisa::string>;
    const auto t = trim(decoded_text);
    switch (block_type) {
        case LayoutBlockType::formula: {
            if (t.empty()) { return R::success(luisa::string{}); }
            if (auto s = output::check_latex(t); !s.ok()) { return R::failure(s); }
            return output::strip_latex_delimiters(t);
        }
        case LayoutBlockType::table: {
            if (t.empty()) { return R::success(luisa::string{}); }
            auto parsed = output::parse_table(t);
            if (!parsed.ok()) { return R::failure(parsed.status); }
            if (parsed.value->format == output::TableFormat::html) {
                return output::table_to_html(*parsed.value);
            }
            return R::success(luisa::string{t});
        }
        case LayoutBlockType::text:
        case LayoutBlockType::title:
        case LayoutBlockType::list:
        case LayoutBlockType::header:
        case LayoutBlockType::footer:
        case LayoutBlockType::figure:
        case LayoutBlockType::other:
        default:
            return R::success(luisa::string{t});
    }
}

[[nodiscard]] Result<LayoutDocument> open_doc_compose(const LayoutRunResult &run, LayoutOutputMode mode) {
    using R = Result<LayoutDocument>;
    if (mode == LayoutOutputMode::extract) {
        return R::failure(StatusCode::unsupported,
                          "OpenDoc does not declare extract mode (two-stage text/formula/table recognizer)", "mode");
    }
    LayoutDocument doc;
    doc.diagnostics = run.diagnostics;

    struct Adapted {
        const LayoutRegionResult *region;
        luisa::string content;
    };
    luisa::vector<Adapted> adapted;
    for (const auto &page : run.pages) {
        for (const auto &region : page.regions) {
            auto content = open_doc_adapt_region(region.block_type, region.content);
            if (!content.ok()) { return R::failure(content.status); }
            adapted.push_back(Adapted{&region, std::move(*content.value)});
        }
    }

    if (mode == LayoutOutputMode::markdown) {
        luisa::vector<output::MarkdownBlock> blocks;
        blocks.reserve(adapted.size());
        for (const auto &a : adapted) {
            if (a.content.empty()) { continue; }
            output::MarkdownBlock block;
            block.content = a.content;
            switch (a.region->block_type) {
                case LayoutBlockType::title:
                    block.type = BlockType::title;
                    block.heading_level = a.region->heading_level == 0u ? 1u : a.region->heading_level;
                    break;
                case LayoutBlockType::table: block.type = BlockType::table; break;
                case LayoutBlockType::formula: block.type = BlockType::formula; break;
                case LayoutBlockType::list: block.type = BlockType::list; break;
                case LayoutBlockType::figure: block.type = BlockType::figure; break;
                case LayoutBlockType::header: block.type = BlockType::header; break;
                case LayoutBlockType::footer: block.type = BlockType::footer; break;
                default: block.type = BlockType::text; break;
            }
            blocks.push_back(std::move(block));
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
        // CTC confidence is an explicitly validated score source (internal/test).
        block.has_confidence = a.region->has_confidence;
        block.confidence = a.region->confidence;
        doc.blocks.push_back(block);
        if (!a.content.empty()) {
            LayoutLine line;
            line.text = a.content;
            line.page_index = a.region->page_index;
            line.reading_order = line_order++;
            line.source_box = a.region->source_box;
            line.polygon = a.region->polygon;
            line.has_confidence = a.region->has_confidence;
            line.confidence = a.region->confidence;
            doc.lines.push_back(std::move(line));
        }
    }
    return R::success(std::move(doc));
}

[[nodiscard]] RecognizeResult open_doc_to_public(const LayoutDocument &doc, const LayoutRunResult &run,
                                                 const ModelMetadata &metadata) {
    return layout_document_to_public(doc, run, metadata);
}

}// namespace luisa::ocr
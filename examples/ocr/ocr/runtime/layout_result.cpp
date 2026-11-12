// OCR engine -- layout document -> public RecognizeResult bridge (Phase 6).
// HOST-ONLY TU. See the header for the contract.
#include "ocr/runtime/layout_result.h"

#include <utility>

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
    result.counters.peak_device_bytes = run.workspace_bytes;
    return result;
}

}// namespace luisa::ocr
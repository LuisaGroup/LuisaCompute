// OCR engine -- CNN executor public-API bridge implementation (Phase 5).
// Host-only. See cnn_executor_result.h.
#include "ocr/runtime/cnn_executor_result.h"

#include <utility>

namespace luisa::ocr {

[[nodiscard]] RecognizeResult cnn_result_to_public(const CnnRunResult &run, const ModelMetadata &metadata) {
    RecognizeResult result;
    result.metadata = metadata;
    uint64_t regions = 0;
    uint64_t output_tokens = 0;
    luisa::vector<luisa::string> page_texts;
    for (const auto &page : run.pages) {
        for (const auto &line : page.lines) {
            OcrLine pub;
            pub.text = line.text;
            pub.reading_order = line.reading_order;
            pub.has_confidence = line.has_confidence;
            pub.confidence = line.confidence;
            pub.source.page_index = line.page_index;
            pub.source.box.x0 = line.bounds.x0;
            pub.source.box.y0 = line.bounds.y0;
            pub.source.box.x1 = line.bounds.x1;
            pub.source.box.y1 = line.bounds.y1;
            pub.source.polygon.reserve(line.polygon.size());
            for (const auto &p : line.polygon) {
                pub.source.polygon.push_back(SourcePoint{p.x, p.y});
            }
            // CTC kept-class-token count (NOT text.size(), which counts
            // UTF-8 BYTES and would over-count CJK/emoji by 3-4x). Lines
            // without a CTC count contribute 0 (unknown), never a guess.
            output_tokens += line.ctc_token_count;
            regions++;
            result.lines.push_back(std::move(pub));
        }
        if (!page.text.empty()) { page_texts.push_back(page.text); }
    }
    // Canonical multi-page text: join non-empty page texts with '\n'.
    for (size_t i = 0; i < page_texts.size(); i++) {
        if (i != 0) { result.text += '\n'; }
        result.text += page_texts[i];
    }
    result.counters.pages_processed = run.pages.size();
    result.counters.regions_processed = regions;
    result.counters.output_tokens = output_tokens;
    result.counters.resident_weight_bytes = run.weight_bytes;
    result.counters.peak_device_bytes = run.weight_bytes + run.workspace_bytes;
    return result;
}

}// namespace luisa::ocr
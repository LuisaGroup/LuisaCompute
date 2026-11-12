// Test for ocr/runtime/cnn_executor_result (Phase 5 review S4, host-only).
// This test covers the plain-CNN-result -> public RecognizeResult bridge:
// - lines/polygons/boxes/page indices/reading order mapped to the PUBLIC types;
// - confidence carried through unchanged (present only where CTC provided it);
// - output_tokens is the CTC KEPT-TOKEN count, NOT a UTF-8 byte count
//   (CJK/emoji are one token but several bytes);
// - unknown token counts contribute 0, never a guessed value;
// - canonical multi-page text joining and empty results.
//
// Host-only TU: includes api.h (host AttentionMaskKind) and therefore must NOT
// include the ops DSL headers.
#include "ut/ut.hpp"

#include <cmath>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/runtime/cnn_executor_result.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] CnnLine make_line(const char *text, uint32_t page, uint32_t order,
                                uint32_t tokens, bool has_conf, float conf) {
    CnnLine l;
    l.text = text;
    l.page_index = page;
    l.reading_order = order;
    l.bounds = output::DbRect{1.0f, 2.0f, 30.0f, 12.0f};
    l.polygon = {output::DbPoint{1.0f, 2.0f}, output::DbPoint{30.0f, 2.0f},
                 output::DbPoint{30.0f, 12.0f}, output::DbPoint{1.0f, 12.0f}};
    l.has_confidence = has_conf;
    l.confidence = conf;
    l.ctc_token_count = tokens;
    return l;
}

[[nodiscard]] ModelMetadata make_metadata() {
    ModelMetadata m;
    m.model_id = ModelId::pp_ocrv6;
    m.pp_size = PpOcrSize::tiny;
    m.target_name = "pp_ocrv6:tiny";
    m.backend = "vk";
    m.support_state = SupportState::planned;
    m.validated = false;
    return m;
}

}// namespace

void test_bridge_maps_public_types() {
    CnnRunResult run;
    CnnPageResult page;
    page.page_index = 3;
    page.text = "hello";
    page.lines.push_back(make_line("hello", 3, 0, 5, true, 0.75f));
    run.pages.push_back(page);
    run.weight_bytes = 1024;
    run.workspace_bytes = 2048;

    auto pub = cnn_result_to_public(run, make_metadata());
    expect(pub.lines.size() == 1u);
    if (pub.lines.empty()) { return; }
    const auto &l = pub.lines[0];
    expect(l.text == "hello");
    expect(l.source.page_index == 3u);
    expect(l.reading_order == 0u);
    expect(l.has_confidence);
    expect(std::abs(l.confidence - 0.75f) < 1e-6f);
    // Public geometry types carry the source box AND polygon.
    expect(std::abs(l.source.box.x0 - 1.0f) < 1e-6f);
    expect(std::abs(l.source.box.x1 - 30.0f) < 1e-6f);
    expect(l.source.polygon.size() == 4u);
    if (l.source.polygon.size() == 4u) {
        expect(std::abs(l.source.polygon[1].x - 30.0f) < 1e-6f);
        expect(std::abs(l.source.polygon[2].y - 12.0f) < 1e-6f);
    }
    expect(pub.text == "hello");
    // Metadata is passed through (identity/support never invented).
    expect(pub.metadata.model_id == ModelId::pp_ocrv6);
    expect(pub.metadata.support_state == SupportState::planned);
    expect(!pub.metadata.validated);
    // Counters.
    expect(pub.counters.pages_processed == 1u);
    expect(pub.counters.regions_processed == 1u);
    expect(pub.counters.resident_weight_bytes == 1024u);
    expect(pub.counters.peak_device_bytes == 3072u);
}

void test_bridge_output_tokens_not_bytes() {
    // "中文😀" = 3 CTC tokens but 3+3+4 = 10 UTF-8 bytes. The counter must be 3.
    const char *cjk_emoji = "\xe4\xb8\xad\xe6\x96\x87\xf0\x9f\x98\x80";
    CnnRunResult run;
    CnnPageResult page;
    page.page_index = 0;
    page.text = cjk_emoji;
    page.lines.push_back(make_line(cjk_emoji, 0, 0, 3, true, 0.9f));
    run.pages.push_back(page);
    auto pub = cnn_result_to_public(run, make_metadata());
    expect(pub.lines.size() == 1u);
    if (!pub.lines.empty()) {
        expect(pub.lines[0].text == cjk_emoji);
        expect(pub.lines[0].text.size() == 10u);// bytes, for contrast
    }
    expect(pub.counters.output_tokens == 3u)
        << "output_tokens must be the CTC token count (3), not bytes (10); got "
        << pub.counters.output_tokens;

    // Mixed Latin + CJK across two lines: 4 + 2 = 6 tokens (not 4 + 6 bytes).
    CnnRunResult mixed;
    CnnPageResult p2;
    p2.page_index = 0;
    p2.lines.push_back(make_line("abcd", 0, 0, 4, true, 0.8f));
    p2.lines.push_back(make_line("\xe4\xb8\xad\xe6\x96\x87", 0, 1, 2, true, 0.7f));
    p2.text = "abcd\n\xe4\xb8\xad\xe6\x96\x87";
    mixed.pages.push_back(p2);
    auto pub2 = cnn_result_to_public(mixed, make_metadata());
    expect(pub2.counters.output_tokens == 6u) << "got " << pub2.counters.output_tokens;
    expect(pub2.counters.regions_processed == 2u);
}

void test_bridge_unknown_tokens_and_confidence() {
    // A line with no CTC token count contributes 0 (unknown), never a guess
    // derived from the text length. Confidence absence is preserved.
    CnnRunResult run;
    CnnPageResult page;
    page.page_index = 0;
    page.text = "abcd";
    page.lines.push_back(make_line("abcd", 0, 0, 0, false, 0.0f));
    run.pages.push_back(page);
    auto pub = cnn_result_to_public(run, make_metadata());
    expect(pub.counters.output_tokens == 0u) << "unknown token count must stay 0";
    expect(pub.lines.size() == 1u);
    if (!pub.lines.empty()) {
        expect(!pub.lines[0].has_confidence);
        expect(pub.lines[0].confidence == 0.0f);
    }
}

void test_bridge_multipage_and_empty() {
    // Two pages: text joined with '\n' in page order; empty page skipped.
    CnnRunResult run;
    CnnPageResult a;
    a.page_index = 0;
    a.text = "first";
    a.lines.push_back(make_line("first", 0, 0, 5, true, 0.9f));
    CnnPageResult b;
    b.page_index = 1;
    b.text = "";// blank page: no lines, no text
    CnnPageResult c;
    c.page_index = 2;
    c.text = "third";
    c.lines.push_back(make_line("third", 2, 0, 5, true, 0.8f));
    run.pages = {a, b, c};
    auto pub = cnn_result_to_public(run, make_metadata());
    expect(pub.text == "first\nthird");
    expect(pub.counters.pages_processed == 3u);
    expect(pub.counters.regions_processed == 2u);
    expect(pub.counters.output_tokens == 10u);
    expect(pub.lines.size() == 2u);

    // Fully empty run -> valid empty public result.
    CnnRunResult empty;
    auto pe = cnn_result_to_public(empty, make_metadata());
    expect(pe.text.empty());
    expect(pe.lines.empty());
    expect(pe.counters.pages_processed == 0u);
    expect(pe.counters.output_tokens == 0u);
}

static auto test_ocr_cnn_result_registration = [] {
    "cnn_result_maps_public_types"_test = [] { test_bridge_maps_public_types(); };
    "cnn_result_output_tokens_not_bytes"_test = [] { test_bridge_output_tokens_not_bytes(); };
    "cnn_result_unknown_tokens_and_confidence"_test = [] { test_bridge_unknown_tokens_and_confidence(); };
    "cnn_result_multipage_and_empty"_test = [] { test_bridge_multipage_and_empty(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
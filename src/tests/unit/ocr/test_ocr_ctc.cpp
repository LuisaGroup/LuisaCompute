// Test for ocr/output/ctc_decode (Phase 5, host-only).
// This test covers:
// - blank=0 and nonzero blank ids; consecutive duplicate collapse; blank-
//   separated repeats; CJK/emoji multi-byte dictionary entries; empty frames;
//   invalid class ids / probabilities; temperature + log-softmax numerical
//   stability; confidence edge cases (arithmetic mean, empty -> no confidence).
// - the declared PP-OCR reference collapse/confidence formula, verified against
//   hand-computed expectations.
#include "ut/ut.hpp"

#include <cmath>
#include <limits>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/output/ctc_decode.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace luisa::ocr::output;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Build a dictionary from a list of UTF-8 entries (blank excluded).
[[nodiscard]] CtcDictionary make_dict(std::initializer_list<const char *> entries, uint32_t blank = 0) {
    CtcDictionary d;
    d.blank_id = blank;
    for (const auto *e : entries) { d.characters.emplace_back(e); }
    return d;
}

// One-hot-ish logits: make class `id` the clear argmax of frame t.
void set_argmax(luisa::vector<float> &logits, uint32_t frames, uint32_t classes,
                uint32_t frame, uint32_t id, float peak = 8.0f) {
    const size_t base = static_cast<size_t>(frame) * classes;
    for (uint32_t c = 0; c < classes; c++) { logits[base + c] = 0.0f; }
    logits[base + id] = peak;
}

}// namespace

void test_dictionary_contract() {
    auto d = make_dict({"a", "b", "c"});
    expect(d.validate().ok());
    expect(d.num_classes() == 4u);// 3 chars + blank
    expect(d.entry(0) == nullptr);// blank
    expect(d.entry(1) != nullptr && *d.entry(1) == "a");
    expect(d.entry(3) != nullptr && *d.entry(3) == "c");
    expect(d.entry(4) == nullptr);// out of range
    // Nonzero blank: blank=2, chars [a,b,c] -> ids 0->a,1->b,3->c.
    auto d2 = make_dict({"a", "b", "c"}, 2);
    expect(d2.validate().ok());
    expect(d2.entry(2) == nullptr);
    expect(d2.entry(0) != nullptr && *d2.entry(0) == "a");
    expect(d2.entry(1) != nullptr && *d2.entry(1) == "b");
    expect(d2.entry(3) != nullptr && *d2.entry(3) == "c");
    // Empty dictionary fails.
    CtcDictionary empty;
    expect(!empty.validate().ok());
    // Empty entry fails.
    auto bad = make_dict({"a", ""});
    expect(!bad.validate().ok());
}

void test_blank_and_collapse() {
    auto d = make_dict({"a", "b"});// classes: 0=blank,1=a,2=b
    // [a, a] -> "a" (consecutive duplicate collapse).
    {
        luisa::vector<int32_t> ids{1, 1};
        luisa::vector<float> probs{0.9f, 0.8f};
        auto r = ctc_decode_greedy(ids, probs, d);
        expect(r.ok());
        expect(r.value->text == "a");
        expect(r.value->ids.size() == 1u);
        expect(r.value->has_confidence);
        expect(std::abs(r.value->confidence - 0.9f) < 1e-6f);// only first kept
    }
    // [a, blank, a] -> "aa" (blank-separated repeats are NOT collapsed).
    {
        luisa::vector<int32_t> ids{1, 0, 1};
        luisa::vector<float> probs{0.9f, 0.99f, 0.7f};
        auto r = ctc_decode_greedy(ids, probs, d);
        expect(r.ok());
        expect(r.value->text == "aa");
        expect(r.value->ids.size() == 2u);
        // confidence = mean(0.9, 0.7) = 0.8
        expect(std::abs(r.value->confidence - 0.8f) < 1e-6f);
    }
    // [blank, blank] -> empty, no confidence.
    {
        luisa::vector<int32_t> ids{0, 0};
        luisa::vector<float> probs{0.99f, 0.99f};
        auto r = ctc_decode_greedy(ids, probs, d);
        expect(r.ok());
        expect(r.value->empty());
        expect(r.value->text.empty());
        expect(!r.value->has_confidence);
        expect(r.value->confidence == 0.0f);
    }
    // [a, b, b, a] -> "aba".
    {
        luisa::vector<int32_t> ids{1, 2, 2, 1};
        luisa::vector<float> probs{0.6f, 0.7f, 0.7f, 0.9f};
        auto r = ctc_decode_greedy(ids, probs, d);
        expect(r.ok());
        expect(r.value->text == "aba");
        // kept probs: 0.6 (a), 0.7 (b), 0.9 (a) -> mean = 0.7333...
        expect(std::abs(r.value->confidence - (0.6f + 0.7f + 0.9f) / 3.0f) < 1e-5f);
    }
}

void test_nonzero_blank() {
    auto d = make_dict({"a", "b"}, 2);// classes: 0=a,1=b,2=blank
    // [a, blank, a] -> "aa".
    luisa::vector<int32_t> ids{0, 2, 0};
    luisa::vector<float> probs{0.8f, 0.99f, 0.6f};
    auto r = ctc_decode_greedy(ids, probs, d);
    expect(r.ok());
    expect(r.value->text == "aa");
    expect(std::abs(r.value->confidence - 0.7f) < 1e-6f);
    // [b, b] -> "b".
    luisa::vector<int32_t> ids2{1, 1};
    luisa::vector<float> probs2{0.5f, 0.9f};
    auto r2 = ctc_decode_greedy(ids2, probs2, d);
    expect(r2.ok());
    expect(r2.value->text == "b");
}

void test_cjk_emoji() {
    // Multi-byte UTF-8 entries: CJK ideographs + an emoji.
    auto d = make_dict({"\xe4\xb8\xad", "\xe6\x96\x87", "\xf0\x9f\x98\x80"});// 中 文 😀
    expect(d.validate().ok());
    // ids: 1=中 2=文 1=中 3=😀 -> "中文中😀"
    luisa::vector<int32_t> ids{1, 2, 1, 3};
    luisa::vector<float> probs{0.9f, 0.9f, 0.9f, 0.9f};
    auto r = ctc_decode_greedy(ids, probs, d);
    expect(r.ok());
    expect(r.value->text == "\xe4\xb8\xad\xe6\x96\x87\xe4\xb8\xad\xf0\x9f\x98\x80");
    expect(r.value->ids.size() == 4u);
    expect(std::abs(r.value->confidence - 0.9f) < 1e-6f);
}

void test_empty_and_invalid() {
    auto d = make_dict({"a", "b"});
    // Zero frames -> valid empty result.
    {
        auto r = ctc_decode_greedy({}, {}, d);
        expect(r.ok());
        expect(r.value->empty());
        expect(!r.value->has_confidence);
    }
    // Mismatched sizes fail.
    {
        luisa::vector<int32_t> ids{1};
        luisa::vector<float> probs{0.5f, 0.5f};
        expect(!ctc_decode_greedy(ids, probs, d).ok());
    }
    // Invalid class id (>= num_classes) fails.
    {
        luisa::vector<int32_t> ids{3};
        luisa::vector<float> probs{0.5f};
        expect(!ctc_decode_greedy(ids, probs, d).ok());
    }
    // Negative id fails.
    {
        luisa::vector<int32_t> ids{-1};
        luisa::vector<float> probs{0.5f};
        expect(!ctc_decode_greedy(ids, probs, d).ok());
    }
    // Invalid probability (NaN / >1 / <0) fails.
    {
        luisa::vector<int32_t> ids{1};
        luisa::vector<float> nan{std::nanf("")};
        expect(!ctc_decode_greedy(ids, nan, d).ok());
        luisa::vector<float> over{1.5f};
        expect(!ctc_decode_greedy(ids, over, d).ok());
        luisa::vector<float> under{-0.1f};
        expect(!ctc_decode_greedy(ids, under, d).ok());
    }
}

void test_logits_decode_and_temperature() {
    auto d = make_dict({"a", "b"});// 3 classes
    const uint32_t frames = 4, classes = 3;
    luisa::vector<float> logits(static_cast<size_t>(frames) * classes, 0.0f);
    // frame0 -> a, frame1 -> a (dup), frame2 -> blank, frame3 -> b
    set_argmax(logits, frames, classes, 0, 1);
    set_argmax(logits, frames, classes, 1, 1);
    set_argmax(logits, frames, classes, 2, 0);
    set_argmax(logits, frames, classes, 3, 2);
    auto r = ctc_decode_logits(logits, frames, d);
    expect(r.ok());
    expect(r.value->text == "ab");
    expect(r.value->has_confidence);
    // With a high peak (8.0) the softmax prob of the argmax is ~1; confidence
    // should be close to 1 but strictly < 1 (other classes have mass).
    expect(r.value->confidence > 0.9f && r.value->confidence < 1.0f);

    // Temperature scaling changes confidence but NOT the argmax/collapse.
    CtcDecodeOptions hot;
    hot.temperature = 4.0f;
    auto r_hot = ctc_decode_logits(logits, frames, d, hot);
    expect(r_hot.ok());
    expect(r_hot.value->text == "ab");
    // Higher temperature flattens the distribution -> lower confidence.
    expect(r_hot.value->confidence < r.value->confidence);

    // Invalid temperature fails.
    CtcDecodeOptions bad;
    bad.temperature = 0.0f;
    expect(!ctc_decode_logits(logits, frames, d, bad).ok());
    bad.temperature = -1.0f;
    expect(!ctc_decode_logits(logits, frames, d, bad).ok());
}

void test_log_softmax_stability() {
    auto d = make_dict({"a", "b"});
    const uint32_t frames = 2, classes = 3;
    // Large-magnitude logits: max-subtraction must prevent overflow.
    luisa::vector<float> logits(static_cast<size_t>(frames) * classes, 0.0f);
    logits[0] = 1000.0f;
    logits[1] = 999.0f;
    logits[2] = 998.0f;// frame0 -> a (id1)? no: argmax is class0=blank
    // Make frame0 argmax class1 (a) with huge values.
    logits[0] = 998.0f;
    logits[1] = 1000.0f;
    logits[2] = 999.0f;
    // frame1 -> b with huge negative values.
    logits[3] = -1000.0f;
    logits[4] = -1002.0f;
    logits[5] = -999.0f;
    auto r = ctc_decode_logits(logits, frames, d);
    expect(r.ok());
    expect(r.value->text == "ab");
    expect(std::isfinite(r.value->confidence));
    expect(r.value->confidence > 0.5f);

    // Non-finite logits fail closed.
    logits[0] = std::numeric_limits<float>::infinity();
    expect(!ctc_decode_logits(logits, frames, d).ok());
}

void test_confidence_reference_formula() {
    // Hand-computed reference: 3 kept frames with probs 0.2, 0.4, 0.6.
    // Arithmetic mean = 0.4. NOT geometric (0.363), NOT min (0.2).
    auto d = make_dict({"a", "b", "c"});
    luisa::vector<int32_t> ids{1, 2, 3};
    luisa::vector<float> probs{0.2f, 0.4f, 0.6f};
    auto r = ctc_decode_greedy(ids, probs, d);
    expect(r.ok());
    expect(r.value->text == "abc");
    expect(std::abs(r.value->confidence - 0.4f) < 1e-6f);
    // Verify kept-frame provenance.
    expect(r.value->kept_frames.size() == 3u);
    expect(r.value->kept_frames[0].frame == 0u);
    expect(r.value->kept_frames[2].id == 3u);
}

void test_join_line_texts() {
    luisa::vector<luisa::string> lines{"hello", "", "world", ""};
    expect(join_line_texts(lines) == "hello\nworld");
    luisa::vector<luisa::string> empty_lines{"", ""};
    expect(join_line_texts(empty_lines).empty());
    luisa::vector<luisa::string> none;
    expect(join_line_texts(none).empty());
    luisa::vector<luisa::string> one{"only"};
    expect(join_line_texts(one) == "only");
}

static auto test_ocr_ctc_registration = [] {
    "ctc_dictionary_contract"_test = [] { test_dictionary_contract(); };
    "ctc_blank_and_collapse"_test = [] { test_blank_and_collapse(); };
    "ctc_nonzero_blank"_test = [] { test_nonzero_blank(); };
    "ctc_cjk_emoji"_test = [] { test_cjk_emoji(); };
    "ctc_empty_and_invalid"_test = [] { test_empty_and_invalid(); };
    "ctc_logits_decode_and_temperature"_test = [] { test_logits_decode_and_temperature(); };
    "ctc_log_softmax_stability"_test = [] { test_log_softmax_stability(); };
    "ctc_confidence_reference_formula"_test = [] { test_confidence_reference_formula(); };
    "ctc_join_line_texts"_test = [] { test_join_line_texts(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
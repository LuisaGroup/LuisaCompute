#pragma once
// OCR engine -- CTC greedy decode (Phase 5). Host-only.
//
// DECLARED PP-OCR REFERENCE SEMANTICS (PaddleOCR CTCLabelDecode greedy arm).
// This is the reference definition this engine implements and tests against;
// it is NOT a claim of bit-parity with any official checkpoint output:
//   1. Per frame t: id[t] = argmax over classes of the frame logits
//      (temperature-scaled log-softmax is monotone, so the argmax is taken on
//      the raw or log-softmaxed logits identically; lowest index wins ties).
//   2. Collapse: scan frames left to right; DROP every blank frame; DROP a
//      frame whose id equals the PREVIOUS RAW frame id (consecutive duplicate
//      collapse BEFORE blank removal -- so [A, blank, A] emits "AA" while
//      [A, A] emits "A"). This is the standard CTC greedy collapse.
//   3. Character lookup: each kept id maps to ONE character-dictionary entry
//      (an entry may be a multi-byte UTF-8 string, e.g. a CJK ideograph or an
//      emoji). The dictionary EXCLUDES the blank; class id -> entry index is
//      `id - 1` when blank == 0 (PaddleOCR prepends 'blank' at index 0), or
//      `id` for ids below blank and `id - 1` for ids above blank when the
//      blank sits elsewhere. An id outside [0, num_classes) is INVALID and
//      fails the decode with invalid_argument (never silently dropped).
//   4. Confidence: the ARITHMETIC MEAN of the per-frame softmax probabilities
//      of the KEPT frames (the frames that produced characters after
//      collapse). NOT a geometric mean, NOT a minimum, NOT a beam score.
//      An empty decode (all blank / no frames) has has_confidence == false
//      and confidence == 0 -- never a fabricated 1.0.
//
// Probabilities come from the temperature-scaled softmax of the frame logits
// (ops_ctc log-softmax + exp). With temperature == 1 this is exactly the
// PaddleOCR reference; other temperatures are an explicit caller choice and
// change the confidence values accordingly (documented, never implicit).
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr::output {

// Character dictionary + blank contract for one recognizer. `characters`
// holds one UTF-8 entry per NON-BLANK class, in class order (blank removed).
// num_classes == characters.size() + 1. blank_id must be < num_classes.
struct CtcDictionary {
    luisa::vector<luisa::string> characters;
    uint32_t blank_id{0};

    [[nodiscard]] uint32_t num_classes() const noexcept {
        return static_cast<uint32_t>(characters.size()) + 1u;
    }
    [[nodiscard]] Status validate() const;
    // Map a class id to its dictionary entry; nullptr for blank/invalid.
    [[nodiscard]] const luisa::string *entry(uint32_t id) const noexcept;
};

struct CtcDecodeOptions {
    // Softmax temperature used for the per-frame probabilities. Must be
    // finite and > 0. 1.0 == the declared PP-OCR reference.
    float temperature{1.0f};

    [[nodiscard]] Status validate() const;
};

// One kept (collapsed) frame: the source frame index, its class id, and its
// softmax probability. Exposed so tests can verify the collapse itself.
struct CtcKeptFrame {
    uint32_t frame{0};
    uint32_t id{0};
    float probability{0.0f};
};

struct CtcDecodeResult {
    luisa::string text;
    // Kept class ids in emission order (blank/duplicates already removed).
    luisa::vector<uint32_t> ids;
    luisa::vector<CtcKeptFrame> kept_frames;
    // Confidence: arithmetic mean over kept frame probabilities. Present
    // ONLY when at least one character was emitted.
    bool has_confidence{false};
    float confidence{0.0f};

    [[nodiscard]] bool empty() const noexcept { return ids.empty(); }
};

// Decode from per-frame greedy ids + per-frame probabilities of the selected
// ids (the ops_ctc device outputs). `ids.size() == probs.size() == frames`.
// Every id must be < dict.num_classes(); probabilities must be finite and in
// [0, 1]. Invalid input fails with invalid_argument; an all-blank/zero-frame
// input returns a valid EMPTY result (no text, no confidence).
[[nodiscard]] Result<CtcDecodeResult> ctc_decode_greedy(luisa::span<const int32_t> ids,
                                                        luisa::span<const float> probs,
                                                        const CtcDictionary &dict);

// Decode directly from frame logits [frames, num_classes] (row-major). The
// probabilities are the temperature-scaled softmax of each row (max-
// subtracted, double accumulation); the argmax/collapse/lookup/confidence
// follow the declared reference above. Zero frames -> valid empty result.
[[nodiscard]] Result<CtcDecodeResult> ctc_decode_logits(luisa::span<const float> logits,
                                                        uint32_t frames,
                                                        const CtcDictionary &dict,
                                                        const CtcDecodeOptions &options = {});

// Canonical joining of decoded line texts into one page text: lines are
// joined with '\n' in the given (reading) order; empty lines are skipped; no
// trailing newline. Used by the CNN executor for RecognizeResult::text.
[[nodiscard]] luisa::string join_line_texts(luisa::span<const luisa::string> lines);

}// namespace luisa::ocr::output
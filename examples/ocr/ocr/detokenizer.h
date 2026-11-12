#pragma once
// OCR engine -- incremental detokenizer (Phase 1). Host-only.
//
// Converts a stream of token IDs into text incrementally, holding back any
// incomplete multi-byte UTF-8 sequence so a character split across tokens is
// never emitted half-formed. Handles EOS and stop strings: when a stop string
// is matched, the text up to (but NOT including) the stop is emitted and the
// stream is marked stopped; stop text never leaks into the output. Special
// tokens are suppressed per the recipe policy.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"
#include "ocr/tokenizer.h"

namespace luisa::ocr {

struct DetokenizerOptions {
    bool skip_special_tokens{true};
    luisa::vector<luisa::string> stop_strings;
    // EOS token ids (in addition to the tokenizer's own eos). Empty = use the
    // tokenizer's eos only.
    luisa::vector<uint32_t> eos_ids;
};

class IncrementalDetokenizer {
public:
    // The tokenizer must outlive the detokenizer.
    IncrementalDetokenizer(const Tokenizer &tokenizer, DetokenizerOptions options);

    // Feed one token. Returns the newly-committed text (may be empty while a
    // multi-byte character or a potential stop-string prefix is pending).
    // On EOS/stop the stream stops; further pushes return empty text.
    [[nodiscard]] Result<luisa::string> push(uint32_t token_id);

    // Flush any pending complete text at end of stream (does not emit an
    // incomplete trailing multi-byte sequence). Returns the final committed
    // text and marks the stream finished.
    [[nodiscard]] Result<luisa::string> finish();

    [[nodiscard]] bool stopped() const noexcept { return _stopped; }
    [[nodiscard]] bool finished() const noexcept { return _finished; }
    // The full committed text so far.
    [[nodiscard]] const luisa::string &text() const noexcept { return _committed; }

private:
    [[nodiscard]] Result<luisa::string> commit_pending(bool final);

    const Tokenizer *_tokenizer;
    DetokenizerOptions _options;
    luisa::string _committed;  // text already returned to the caller
    luisa::string _pending;    // decoded bytes not yet safe to emit
    bool _stopped{false};
    bool _finished{false};
    // Unigram/Metaspace: exactly ONE leading structural U+2581 is stripped at
    // stream level (from the first decoded content), never per token. This
    // matches whole-sequence decode's single leading strip.
    bool _stripped_leading{false};
};

}// namespace luisa::ocr
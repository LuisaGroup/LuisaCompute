// OCR engine -- incremental detokenizer (Phase 1).
// Holds back incomplete UTF-8 and stop-string prefixes; never leaks stop text.
#include "ocr/detokenizer.h"

#include <algorithm>
#include <utility>

namespace luisa::ocr {

IncrementalDetokenizer::IncrementalDetokenizer(const Tokenizer &tokenizer, DetokenizerOptions options)
    : _tokenizer{&tokenizer}, _options{std::move(options)} {
    if (_tokenizer->eos_id().has_value()) {
        _options.eos_ids.emplace_back(*_tokenizer->eos_id());
    }
}

[[nodiscard]] Result<luisa::string> IncrementalDetokenizer::commit_pending(bool final) {
    using R = Result<luisa::string>;
    // Determine how much of _pending is safe to emit:
    // 1. Only complete UTF-8 sequences.
    // 2. Hold back any suffix that is a proper prefix of a stop string (it may
    //    still grow into a full stop on the next token).
    size_t safe = ocr_utf8_complete_prefix(_pending);
    if (!final) {
        // Hold back the longest suffix that is a proper prefix of any stop.
        for (const auto &stop : _options.stop_strings) {
            if (stop.empty()) { continue; }
            const size_t max_len = std::min(stop.size() - 1, safe);
            for (size_t len = max_len; len > 0; len--) {
                if (_pending.compare(safe - len, len, stop.data(), len) == 0) {
                    safe = std::min(safe, safe - len);
                    break;
                }
            }
        }
    }
    // Check for a full stop within the emittable region.
    size_t emit_end = safe;
    bool hit_stop = false;
    for (const auto &stop : _options.stop_strings) {
        if (stop.empty()) { continue; }
        const auto pos = _pending.find(stop);
        if (pos != luisa::string::npos && pos + stop.size() <= safe) {
            if (pos < emit_end) {
                emit_end = pos;
                hit_stop = true;
            }
        }
    }
    luisa::string out = _pending.substr(0, emit_end);
    _committed += out;
    if (hit_stop) {
        _pending.clear();
        _stopped = true;
    } else {
        _pending.erase(0, emit_end);
    }
    return R::success(std::move(out));
}

[[nodiscard]] Result<luisa::string> IncrementalDetokenizer::push(uint32_t token_id) {
    using R = Result<luisa::string>;
    if (_stopped || _finished) { return R::success(luisa::string{}); }
    // EOS handling.
    for (auto eos : _options.eos_ids) {
        if (token_id == eos) {
            auto flushed = commit_pending(true);
            if (!flushed.ok()) { return flushed; }
            _stopped = true;
            return flushed;
        }
    }
    // Decode this single token WITHOUT any per-token metaspace strip and
    // allowing a partial trailing UTF-8 sequence (buffered across tokens).
    auto piece = _tokenizer->decode_partial(luisa::span<const uint32_t>{&token_id, 1},
                                            _options.skip_special_tokens);
    if (!piece.ok()) { return R::failure(piece.status); }
    _pending += *piece.value;
    // Official SentencePiece/HF Metaspace: strip exactly ONE leading structural
    // U+2581 at STREAM level, never per token. decode_partial maps U+2581 to a
    // space, so the structural prefix arrives as a leading ' ' in the decoded
    // bytes. A second leading metaspace decodes to another space and is KEPT
    // ([▁,▁hello] -> ' hello'). Whole-sequence decode applies the same single
    // strip, so batch/incremental agree.
    if (!_stripped_leading && _tokenizer->model_kind() == TokenizerModelKind::unigram && !_pending.empty()) {
        if (_pending.front() == ' ') {
            _pending.erase(0, 1);
        }
        // Only the first content decides; even when it does not start with a
        // space, no later strip may happen.
        _stripped_leading = true;
    }
    return commit_pending(false);
}

[[nodiscard]] Result<luisa::string> IncrementalDetokenizer::finish() {
    using R = Result<luisa::string>;
    if (_finished) { return R::success(luisa::string{}); }
    auto flushed = commit_pending(true);
    if (!flushed.ok()) { return flushed; }
    _finished = true;
    return flushed;
}

}// namespace luisa::ocr
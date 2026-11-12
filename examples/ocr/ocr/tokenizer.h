#pragma once
// OCR engine -- checked HF tokenizer.json loader (Phase 1). Host-only.
//
// Supported, explicitly validated subsets (everything else fails closed with
// StatusCode::unsupported -- no silent approximation):
//
// 1. Byte-level BPE ("Model.type == BPE" with a byte_encoder/byte_decoder of
//    the GPT-2 style). Normalizer: null or NFC. Pre-tokenizer: null,
//    ByteLevel (add_prefix_space true/false), or a Sequence of
//    [Split(Regex GPT-2 pattern), ByteLevel]. Post-processor: null,
//    ByteLevel, or TemplateProcessing whose template is exactly a fixed
//    special-token prefix/suffix around the pair (no Jinja, no general
//    template interpreter). Added/special tokens: longest-match first,
//    special tokens are not split and are suppressed on decode by default.
//
// 2. SentencePiece-style Unigram ("Model.type == Unigram") with byte fallback
//    and the Metaspace replacement convention (U+2581 for space). Viterbi
//    segmentation with the exact HF scores; unknown characters fall back to
//    "<0xHH>" byte tokens when byte_fallback is enabled, else to the unk
//    token when one exists, else the input is rejected.
//
// Invalid UTF-8 policy (explicit): encode() REJECTS input that is not valid
// UTF-8 with StatusCode::invalid_argument. There is no lossy replacement.
//
// Token IDs are preserved exactly as stored; no vocabulary re-indexing.
#include <cstdint>

#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "ocr/status.h"

namespace luisa::ocr {

enum class TokenizerModelKind : uint32_t {
    byte_level_bpe = 0,
    unigram = 1,
};

[[nodiscard]] luisa::string_view tokenizer_model_kind_name(TokenizerModelKind kind) noexcept;

struct AddedToken {
    uint32_t id{0};
    luisa::string content;
    bool special{false};
    bool lstrip{false};
    bool rstrip{false};
    bool single_word{false};
};

struct TokenizerConfig {
    // Limits applied while parsing (checked).
    uint64_t max_document_bytes{64ull * 1024ull * 1024ull};
    uint32_t max_vocab_entries{1u << 20u};
    uint32_t max_added_tokens{4096u};
};

class Tokenizer;
struct TokenizerDeleter {
    void operator()(Tokenizer *tokenizer) const noexcept;
};

class Tokenizer {
public:
    [[nodiscard]] static Result<luisa::unique_ptr<Tokenizer, TokenizerDeleter>> load(
        const luisa::filesystem::path &tokenizer_json, const TokenizerConfig &config = {});
    // Parse from an in-memory JSON document (same validation).
    [[nodiscard]] static Result<luisa::unique_ptr<Tokenizer, TokenizerDeleter>> parse(
        luisa::string_view json, const TokenizerConfig &config = {});

    ~Tokenizer() noexcept;
    Tokenizer(const Tokenizer &) = delete;
    Tokenizer &operator=(const Tokenizer &) = delete;

    [[nodiscard]] TokenizerModelKind model_kind() const noexcept { return _kind; }
    [[nodiscard]] uint32_t vocab_size() const noexcept { return _vocab_size; }
    [[nodiscard]] luisa::optional<uint32_t> token_to_id(luisa::string_view piece) const;
    [[nodiscard]] luisa::optional<luisa::string> id_to_token(uint32_t id) const;
    [[nodiscard]] luisa::span<const AddedToken> added_tokens() const noexcept { return _added; }
    [[nodiscard]] luisa::optional<uint32_t> bos_id() const noexcept { return _bos; }
    [[nodiscard]] luisa::optional<uint32_t> eos_id() const noexcept { return _eos; }
    [[nodiscard]] luisa::optional<uint32_t> unk_id() const noexcept { return _unk; }
    [[nodiscard]] luisa::optional<uint32_t> pad_id() const noexcept { return _pad; }

    // Encode UTF-8 text to exact token IDs. `add_special_tokens` applies the
    // validated post-processor prefix/suffix (BPE) or BOS/EOS (Unigram).
    // Invalid UTF-8 is rejected. Added/special tokens are matched longest
    // first and never split.
    [[nodiscard]] Result<luisa::vector<uint32_t>> encode(
        luisa::string_view text, bool add_special_tokens = true) const;

    // Decode token IDs back to text. Special tokens are skipped when
    // `skip_special_tokens` is true. Unknown IDs are rejected (no silent
    // substitution). Byte-level BPE output is validated as UTF-8; an incomplete
    // multi-byte sequence at the end is returned as-is only when
    // `allow_partial_utf8` is set (used by the incremental detokenizer).
    [[nodiscard]] Result<luisa::string> decode(
        luisa::span<const uint32_t> ids, bool skip_special_tokens = true,
        bool allow_partial_utf8 = false) const;

    // Decode a token subsequence for incremental detokenization. Identical to
    // decode() except that (a) a trailing incomplete UTF-8 sequence is allowed
    // (the detokenizer buffers it across token boundaries) and (b) the ONE
    // leading structural metaspace is NOT stripped here — the detokenizer
    // strips it exactly once at stream start, never per token, so batch and
    // incremental output agree. Special tokens are suppressed per the flag.
    [[nodiscard]] Result<luisa::string> decode_partial(
        luisa::span<const uint32_t> ids, bool skip_special_tokens = true) const;

private:
    Tokenizer() noexcept = default;

    [[nodiscard]] Result<luisa::vector<uint32_t>> encode_bpe(luisa::string_view text, bool allow_prefix_space = true) const;
    [[nodiscard]] Result<luisa::vector<uint32_t>> encode_unigram(luisa::string_view text, bool allow_prefix_space = true) const;
    [[nodiscard]] Result<luisa::vector<uint32_t>> encode_added_aware(
        luisa::string_view text, bool add_special_tokens) const;
    [[nodiscard]] Result<luisa::string> decode_impl(
        luisa::span<const uint32_t> ids, bool skip_special_tokens,
        bool allow_partial_utf8, bool strip_leading_metaspace) const;

    TokenizerModelKind _kind{TokenizerModelKind::byte_level_bpe};
    uint32_t _vocab_size{0};
    // BPE state.
    luisa::unordered_map<luisa::string, uint32_t> _bpe_vocab;
    luisa::unordered_map<luisa::string, uint32_t> _bpe_ranks;// "a\0b" -> rank
    uint32_t _byte_tokens[256]{};
    bool _has_byte_tokens{false};
    bool _byte_level_add_prefix_space{false};
    bool _bpe_nfc_normalize{false};
    // Unigram state.
    struct UnigramPiece {
        luisa::string piece;
        float score{0.0f};
    };
    luisa::vector<UnigramPiece> _unigram_pieces;// indexed by token id
    luisa::unordered_map<luisa::string, uint32_t> _unigram_lookup;
    // IDs of <0xHH> byte-fallback pieces. These are EXCLUDED from ordinary
    // Viterbi literal matching (they are only reachable through byte fallback),
    // so literal text like "<0x41>" is never silently turned into a byte.
    luisa::unordered_map<uint32_t, bool> _unigram_byte_token_ids;
    bool _unigram_byte_fallback{false};
    uint32_t _unigram_unk_id{0};
    bool _unigram_has_unk{false};
    bool _unigram_add_bos{false};
    bool _unigram_add_eos{false};
    // Whether add_bos_token/add_eos_token were explicitly present in the model
    // JSON. When absent, the <s>/</s> presence heuristic decides insertion.
    bool _unigram_bos_flag_present{false};
    bool _unigram_eos_flag_present{false};
    // Metaspace convention: whether encode prepends the structural prefix
    // metaspace (SentencePiece add_prefix_space / add_dummy_prefix). Honored
    // from the normalizer; defaults to true. Only U+2581 replacement is
    // supported (any other replacement fails closed at parse time).
    bool _unigram_add_prefix_space{true};
    // Shared.
    luisa::unordered_map<uint32_t, luisa::string> _id_to_piece;
    luisa::vector<AddedToken> _added;// sorted longest-first for matching
    luisa::unordered_map<uint32_t, bool> _special_ids;
    luisa::optional<uint32_t> _bos, _eos, _unk, _pad;
    luisa::vector<uint32_t> _post_prefix, _post_suffix;// validated template
    uint32_t _byte_decoder[512]{};// GPT-2 piece code point (0..511) -> byte
    bool _byte_dec_present[512]{};
    bool _has_byte_decoder{false};
};

// UTF-8 helpers shared by tokenizer/detokenizer/prompt (host-only).
[[nodiscard]] bool ocr_is_valid_utf8(luisa::string_view text) noexcept;
// Length of the UTF-8 sequence starting at text[0], or 0 when invalid.
[[nodiscard]] uint32_t ocr_utf8_sequence_length(uint8_t lead) noexcept;
// Byte index just past the last COMPLETE UTF-8 sequence in `text`
// (== text.size() when the whole string is complete/empty).
[[nodiscard]] size_t ocr_utf8_complete_prefix(luisa::string_view text) noexcept;

}// namespace luisa::ocr
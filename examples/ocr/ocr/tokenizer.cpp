// OCR engine -- checked HF tokenizer.json loader (Phase 1).
// Only explicitly validated subsets are accepted; everything else fails
// closed. No exceptions, no RTTI, no silent approximation.
#include "ocr/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

#include <yyjson.h>

namespace luisa::ocr {

// ---------------------------------------------------------------------------
// UTF-8 helpers (shared)
// ---------------------------------------------------------------------------

[[nodiscard]] uint32_t ocr_utf8_sequence_length(uint8_t lead) noexcept {
    if (lead < 0x80u) { return 1; }
    if ((lead & 0xe0u) == 0xc0u) { return lead >= 0xc2u ? 2u : 0u; }
    if ((lead & 0xf0u) == 0xe0u) { return 3; }
    if ((lead & 0xf8u) == 0xf0u) { return lead <= 0xf4u ? 4u : 0u; }
    return 0;
}

[[nodiscard]] bool ocr_is_valid_utf8(luisa::string_view text) noexcept {
    size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<uint8_t>(text[i]);
        const auto len = ocr_utf8_sequence_length(lead);
        if (len == 0 || i + len > text.size()) { return false; }
        uint32_t cp = 0;
        if (len == 1) {
            i += 1;
            continue;
        }
        cp = lead & (0x7fu >> len);
        for (uint32_t k = 1; k < len; k++) {
            const auto b = static_cast<uint8_t>(text[i + k]);
            if ((b & 0xc0u) != 0x80u) { return false; }
            cp = (cp << 6u) | (b & 0x3fu);
        }
        if (len == 3 && cp < 0x800u) { return false; }
        if (len == 4 && cp < 0x10000u) { return false; }
        if (cp > 0x10ffffu || (cp >= 0xd800u && cp <= 0xdfffu)) { return false; }
        i += len;
    }
    return true;
}

[[nodiscard]] size_t ocr_utf8_complete_prefix(luisa::string_view text) noexcept {
    size_t i = 0;
    size_t complete = 0;
    while (i < text.size()) {
        const auto lead = static_cast<uint8_t>(text[i]);
        const auto len = ocr_utf8_sequence_length(lead);
        if (len == 0) { return complete; }
        if (i + len > text.size()) { return complete; }
        bool ok = true;
        uint32_t cp = len == 1 ? lead : (lead & (0x7fu >> len));
        for (uint32_t k = 1; k < len; k++) {
            const auto b = static_cast<uint8_t>(text[i + k]);
            if ((b & 0xc0u) != 0x80u) {
                ok = false;
                break;
            }
            cp = (cp << 6u) | (b & 0x3fu);
        }
        if (!ok) { return complete; }
        if (len == 3 && cp < 0x800u) { return complete; }
        if (len == 4 && cp < 0x10000u) { return complete; }
        if (cp > 0x10ffffu || (cp >= 0xd800u && cp <= 0xdfffu)) { return complete; }
        i += len;
        complete = i;
    }
    return complete;
}

[[nodiscard]] luisa::string_view tokenizer_model_kind_name(TokenizerModelKind kind) noexcept {
    switch (kind) {
        case TokenizerModelKind::byte_level_bpe: return "byte_level_bpe";
        case TokenizerModelKind::unigram: return "unigram";
        default: return "unknown";
    }
}

namespace {

struct JsonDoc {
    yyjson_doc *doc{nullptr};
    ~JsonDoc() noexcept {
        if (doc != nullptr) { yyjson_doc_free(doc); }
    }
};

[[nodiscard]] Status unsupported(luisa::string_view what, luisa::string detail = {}) {
    return Status::error(StatusCode::unsupported,
                         luisa::string{"unsupported tokenizer semantics: "} + luisa::string{what},
                         std::move(detail));
}
[[nodiscard]] Status malformed(luisa::string_view what, luisa::string detail = {}) {
    return Status::error(StatusCode::invalid_argument,
                         luisa::string{"malformed tokenizer JSON: "} + luisa::string{what},
                         std::move(detail));
}

[[nodiscard]] luisa::string_view json_str(yyjson_val *v) noexcept {
    if (v == nullptr || !yyjson_is_str(v)) { return {}; }
    return {yyjson_get_str(v), yyjson_get_len(v)};
}

// Read a JSON unsigned integer that must fit in uint32. yyjson stores uints as
// 64-bit; narrowing without this check would silently wrap oversized ids.
[[nodiscard]] bool json_u32(yyjson_val *v, uint32_t &out) noexcept {
    if (v == nullptr || !yyjson_is_uint(v)) { return false; }
    const auto value = yyjson_get_uint(v);
    if (value > 0xffffffffull) { return false; }
    out = static_cast<uint32_t>(value);
    return true;
}

// True when a unigram piece is a <0xHH> byte-fallback token (HH = two hex
// digits, upper or lower case). Such pieces are reserved for byte fallback.
[[nodiscard]] bool is_byte_fallback_piece(luisa::string_view piece) noexcept {
    if (piece.size() != 6) { return false; }
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') { return false; }
    const auto is_hex = [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    };
    return is_hex(piece[3]) && is_hex(piece[4]);
}

// Append one code point as UTF-8.
void append_utf8(luisa::string &out, uint32_t cp) {
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xc0u | (cp >> 6u)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xe0u | (cp >> 12u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xf0u | (cp >> 18u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    }
}

// Decode one code point at text[pos]; returns bytes consumed (0 on invalid).
[[nodiscard]] uint32_t decode_utf8(luisa::string_view text, size_t pos, uint32_t &cp) noexcept {
    const auto lead = static_cast<uint8_t>(text[pos]);
    const auto len = ocr_utf8_sequence_length(lead);
    if (len == 0 || pos + len > text.size()) { return 0; }
    if (len == 1) {
        cp = lead;
        return 1;
    }
    uint32_t value = lead & (0x7fu >> len);
    for (uint32_t k = 1; k < len; k++) {
        const auto b = static_cast<uint8_t>(text[pos + k]);
        if ((b & 0xc0u) != 0x80u) { return 0; }
        value = (value << 6u) | (b & 0x3fu);
    }
    if (len == 3 && value < 0x800u) { return 0; }
    if (len == 4 && value < 0x10000u) { return 0; }
    if (value > 0x10ffffu || (value >= 0xd800u && value <= 0xdfffu)) { return 0; }
    cp = value;
    return len;
}

// Complete set of code points that are NOT NFC-invariant as single characters
// (Unicode 15.1 / Python unicodedata: NFC(c) != c), compressed to 73 ranges.
// This includes ALL singleton mappings (U+2126 OHM SIGN, U+212A KELVIN SIGN,
// U+212B ANGSTROM SIGN, U+037E GREEK QUESTION MARK, U+0374 GREEK NUMERAL SIGN,
// U+0387 GREEK ANO TELEIA, the CJK compatibility ideographs, ...) and every
// character with a canonical decomposition. Full NFC is NOT implemented, so
// BPE encode REJECTS any input containing one of these (fail closed) instead
// of silently passing raw bytes. Precomposed NFC-invariant text passes.
[[nodiscard]] bool is_nfc_noninvariant_cp(uint32_t cp) noexcept {
    if (cp >= 0x0340u && cp <= 0x0341u) { return true; }
    if (cp >= 0x0343u && cp <= 0x0344u) { return true; }
    if (cp == 0x0374u) { return true; }
    if (cp == 0x037Eu) { return true; }
    if (cp == 0x0387u) { return true; }
    if (cp >= 0x0958u && cp <= 0x095Fu) { return true; }
    if (cp >= 0x09DCu && cp <= 0x09DDu) { return true; }
    if (cp == 0x09DFu) { return true; }
    if (cp == 0x0A33u) { return true; }
    if (cp == 0x0A36u) { return true; }
    if (cp >= 0x0A59u && cp <= 0x0A5Bu) { return true; }
    if (cp == 0x0A5Eu) { return true; }
    if (cp >= 0x0B5Cu && cp <= 0x0B5Du) { return true; }
    if (cp == 0x0F43u) { return true; }
    if (cp == 0x0F4Du) { return true; }
    if (cp == 0x0F52u) { return true; }
    if (cp == 0x0F57u) { return true; }
    if (cp == 0x0F5Cu) { return true; }
    if (cp == 0x0F69u) { return true; }
    if (cp == 0x0F73u) { return true; }
    if (cp >= 0x0F75u && cp <= 0x0F76u) { return true; }
    if (cp == 0x0F78u) { return true; }
    if (cp == 0x0F81u) { return true; }
    if (cp == 0x0F93u) { return true; }
    if (cp == 0x0F9Du) { return true; }
    if (cp == 0x0FA2u) { return true; }
    if (cp == 0x0FA7u) { return true; }
    if (cp == 0x0FACu) { return true; }
    if (cp == 0x0FB9u) { return true; }
    if (cp == 0x1F71u) { return true; }
    if (cp == 0x1F73u) { return true; }
    if (cp == 0x1F75u) { return true; }
    if (cp == 0x1F77u) { return true; }
    if (cp == 0x1F79u) { return true; }
    if (cp == 0x1F7Bu) { return true; }
    if (cp == 0x1F7Du) { return true; }
    if (cp == 0x1FBBu) { return true; }
    if (cp == 0x1FBEu) { return true; }
    if (cp == 0x1FC9u) { return true; }
    if (cp == 0x1FCBu) { return true; }
    if (cp == 0x1FD3u) { return true; }
    if (cp == 0x1FDBu) { return true; }
    if (cp == 0x1FE3u) { return true; }
    if (cp == 0x1FEBu) { return true; }
    if (cp >= 0x1FEEu && cp <= 0x1FEFu) { return true; }
    if (cp == 0x1FF9u) { return true; }
    if (cp == 0x1FFBu) { return true; }
    if (cp == 0x1FFDu) { return true; }
    if (cp >= 0x2000u && cp <= 0x2001u) { return true; }
    if (cp == 0x2126u) { return true; }
    if (cp >= 0x212Au && cp <= 0x212Bu) { return true; }
    if (cp >= 0x2329u && cp <= 0x232Au) { return true; }
    if (cp == 0x2ADCu) { return true; }
    if (cp >= 0xF900u && cp <= 0xFA0Du) { return true; }
    if (cp == 0xFA10u) { return true; }
    if (cp == 0xFA12u) { return true; }
    if (cp >= 0xFA15u && cp <= 0xFA1Eu) { return true; }
    if (cp == 0xFA20u) { return true; }
    if (cp == 0xFA22u) { return true; }
    if (cp >= 0xFA25u && cp <= 0xFA26u) { return true; }
    if (cp >= 0xFA2Au && cp <= 0xFA6Du) { return true; }
    if (cp >= 0xFA70u && cp <= 0xFAD9u) { return true; }
    if (cp == 0xFB1Du) { return true; }
    if (cp == 0xFB1Fu) { return true; }
    if (cp >= 0xFB2Au && cp <= 0xFB36u) { return true; }
    if (cp >= 0xFB38u && cp <= 0xFB3Cu) { return true; }
    if (cp == 0xFB3Eu) { return true; }
    if (cp >= 0xFB40u && cp <= 0xFB41u) { return true; }
    if (cp >= 0xFB43u && cp <= 0xFB44u) { return true; }
    if (cp >= 0xFB46u && cp <= 0xFB4Eu) { return true; }
    if (cp >= 0x1D15Eu && cp <= 0x1D164u) { return true; }
    if (cp >= 0x1D1BBu && cp <= 0x1D1C0u) { return true; }
    if (cp >= 0x2F800u && cp <= 0x2FA1Du) { return true; }
    return false;
}

[[nodiscard]] bool is_gpt2_number(uint32_t cp) noexcept;

// True for code points in the common combining-mark (Mn/Mc) blocks. Full NFC
// is NOT implemented; any input containing a combining mark could be recomposed
// by NFC, so it is rejected (fail closed) rather than silently passed through.
// This is a documented superset of the common combining ranges, not the exact
// Unicode Mn/Mc category.
[[nodiscard]] bool is_combining_mark_cp(uint32_t cp) noexcept {
    if (cp >= 0x0300u && cp <= 0x036fu) { return true; }// Combining Diacritical Marks
    if (cp >= 0x0483u && cp <= 0x0489u) { return true; }// Cyrillic combining
    if (cp >= 0x0591u && cp <= 0x05bdu) { return true; }// Hebrew points
    if (cp >= 0x05bfu && cp <= 0x05c7u) { return true; }
    if (cp >= 0x0610u && cp <= 0x061au) { return true; }// Arabic
    if (cp >= 0x064bu && cp <= 0x065fu) { return true; }// Arabic harakat
    if (cp >= 0x0670u && cp <= 0x0670u) { return true; }
    if (cp >= 0x06d6u && cp <= 0x06dcu) { return true; }
    if (cp >= 0x06dfu && cp <= 0x06e4u) { return true; }
    if (cp >= 0x06e7u && cp <= 0x06e8u) { return true; }
    if (cp >= 0x06eau && cp <= 0x06edu) { return true; }
    if (cp >= 0x0711u && cp <= 0x0711u) { return true; }// Syriac
    if (cp >= 0x0730u && cp <= 0x074au) { return true; }
    if (cp >= 0x07ebu && cp <= 0x07f3u) { return true; }// Thaana
    if (cp >= 0x0816u && cp <= 0x0819u) { return true; }// Samaritan
    if (cp >= 0x081bu && cp <= 0x0823u) { return true; }
    if (cp >= 0x0825u && cp <= 0x0827u) { return true; }
    if (cp >= 0x0829u && cp <= 0x082du) { return true; }
    if (cp >= 0x0859u && cp <= 0x085bu) { return true; }// Mandaic
    if (cp >= 0x08d3u && cp <= 0x08e1u) { return true; }// Arabic ext
    if (cp >= 0x08e3u && cp <= 0x0903u) { return true; }
    if (cp >= 0x093au && cp <= 0x093cu) { return true; }// Devanagari
    if (cp >= 0x0941u && cp <= 0x0948u) { return true; }
    if (cp >= 0x094du && cp <= 0x094du) { return true; }
    if (cp >= 0x0951u && cp <= 0x0957u) { return true; }
    if (cp >= 0x0962u && cp <= 0x0963u) { return true; }
    if (cp >= 0x0981u && cp <= 0x0981u) { return true; }// Bengali
    if (cp >= 0x09bcu && cp <= 0x09bcu) { return true; }
    if (cp >= 0x09c1u && cp <= 0x09c4u) { return true; }
    if (cp >= 0x09cdu && cp <= 0x09cdu) { return true; }
    if (cp >= 0x09e2u && cp <= 0x09e3u) { return true; }
    if (cp >= 0x0a01u && cp <= 0x0a02u) { return true; }// Gurmukhi
    if (cp >= 0x0a3cu && cp <= 0x0a3cu) { return true; }
    if (cp >= 0x0a41u && cp <= 0x0a42u) { return true; }
    if (cp >= 0x0a47u && cp <= 0x0a48u) { return true; }
    if (cp >= 0x0a4bu && cp <= 0x0a4du) { return true; }
    if (cp >= 0x0a70u && cp <= 0x0a71u) { return true; }
    if (cp >= 0x0a81u && cp <= 0x0a82u) { return true; }// Gujarati
    if (cp >= 0x0abcu && cp <= 0x0abcu) { return true; }
    if (cp >= 0x0ac1u && cp <= 0x0ac5u) { return true; }
    if (cp >= 0x0ac7u && cp <= 0x0ac8u) { return true; }
    if (cp >= 0x0acdu && cp <= 0x0acdu) { return true; }
    if (cp >= 0x0b01u && cp <= 0x0b01u) { return true; }// Oriya
    if (cp >= 0x0b3cu && cp <= 0x0b3cu) { return true; }
    if (cp >= 0x0b3fu && cp <= 0x0b3fu) { return true; }
    if (cp >= 0x0b41u && cp <= 0x0b44u) { return true; }
    if (cp >= 0x0b4du && cp <= 0x0b4du) { return true; }
    if (cp >= 0x0b82u && cp <= 0x0b82u) { return true; }// Tamil
    if (cp >= 0x0bc0u && cp <= 0x0bc0u) { return true; }
    if (cp >= 0x0bcdu && cp <= 0x0bcdu) { return true; }
    if (cp >= 0x0c00u && cp <= 0x0c00u) { return true; }// Telugu
    if (cp >= 0x0c3eu && cp <= 0x0c40u) { return true; }
    if (cp >= 0x0c46u && cp <= 0x0c48u) { return true; }
    if (cp >= 0x0c4au && cp <= 0x0c4du) { return true; }
    if (cp >= 0x0c55u && cp <= 0x0c56u) { return true; }
    if (cp >= 0x0c81u && cp <= 0x0c81u) { return true; }// Kannada
    if (cp >= 0x0cbcu && cp <= 0x0cbcu) { return true; }
    if (cp >= 0x0cbfu && cp <= 0x0cbfu) { return true; }
    if (cp >= 0x0cc6u && cp <= 0x0cc6u) { return true; }
    if (cp >= 0x0cccu && cp <= 0x0ccdu) { return true; }
    if (cp >= 0x0d01u && cp <= 0x0d01u) { return true; }// Malayalam
    if (cp >= 0x0d41u && cp <= 0x0d44u) { return true; }
    if (cp >= 0x0d4du && cp <= 0x0d4du) { return true; }
    if (cp >= 0x0dcau && cp <= 0x0dcau) { return true; }// Sinhala
    if (cp >= 0x0dd2u && cp <= 0x0dd4u) { return true; }
    if (cp >= 0x0dd6u && cp <= 0x0dd6u) { return true; }
    if (cp >= 0x0e31u && cp <= 0x0e31u) { return true; }// Thai
    if (cp >= 0x0e34u && cp <= 0x0e3au) { return true; }
    if (cp >= 0x0e47u && cp <= 0x0e4eu) { return true; }
    if (cp >= 0x0eb1u && cp <= 0x0eb1u) { return true; }// Lao
    if (cp >= 0x0eb4u && cp <= 0x0eb9u) { return true; }
    if (cp >= 0x0ebbu && cp <= 0x0ebcu) { return true; }
    if (cp >= 0x0ec8u && cp <= 0x0ecdu) { return true; }
    if (cp >= 0x0f18u && cp <= 0x0f19u) { return true; }// Tibetan
    if (cp >= 0x0f35u && cp <= 0x0f35u) { return true; }
    if (cp >= 0x0f37u && cp <= 0x0f37u) { return true; }
    if (cp >= 0x0f39u && cp <= 0x0f39u) { return true; }
    if (cp >= 0x0f71u && cp <= 0x0f7eu) { return true; }
    if (cp >= 0x0f80u && cp <= 0x0f84u) { return true; }
    if (cp >= 0x0f86u && cp <= 0x0f87u) { return true; }
    if (cp >= 0x0f8du && cp <= 0x0f97u) { return true; }
    if (cp >= 0x0f99u && cp <= 0x0fbcu) { return true; }
    if (cp >= 0x0fc6u && cp <= 0x0fc6u) { return true; }
    if (cp >= 0x102du && cp <= 0x1030u) { return true; }// Myanmar
    if (cp >= 0x1032u && cp <= 0x1037u) { return true; }
    if (cp >= 0x1039u && cp <= 0x103au) { return true; }
    if (cp >= 0x103du && cp <= 0x103eu) { return true; }
    if (cp >= 0x1058u && cp <= 0x1059u) { return true; }
    if (cp >= 0x105eu && cp <= 0x1060u) { return true; }
    if (cp >= 0x1071u && cp <= 0x1074u) { return true; }
    if (cp >= 0x1082u && cp <= 0x1082u) { return true; }
    if (cp >= 0x1085u && cp <= 0x1086u) { return true; }
    if (cp >= 0x108du && cp <= 0x108du) { return true; }
    if (cp >= 0x109du && cp <= 0x109du) { return true; }
    if (cp >= 0x135du && cp <= 0x135fu) { return true; }// Ethiopic
    if (cp >= 0x1712u && cp <= 0x1714u) { return true; }// Tagalog
    if (cp >= 0x1732u && cp <= 0x1734u) { return true; }// Hanunoo
    if (cp >= 0x1752u && cp <= 0x1753u) { return true; }// Buhid
    if (cp >= 0x1772u && cp <= 0x1773u) { return true; }// Tagbanwa
    if (cp >= 0x17b4u && cp <= 0x17b5u) { return true; }// Khmer
    if (cp >= 0x17b7u && cp <= 0x17bdu) { return true; }
    if (cp >= 0x17c6u && cp <= 0x17c6u) { return true; }
    if (cp >= 0x17c9u && cp <= 0x17d3u) { return true; }
    if (cp >= 0x17ddu && cp <= 0x17ddu) { return true; }
    if (cp >= 0x180bu && cp <= 0x180du) { return true; }// Mongolian FVS
    if (cp >= 0x1885u && cp <= 0x1886u) { return true; }
    if (cp >= 0x18a9u && cp <= 0x18a9u) { return true; }
    if (cp >= 0x1920u && cp <= 0x1922u) { return true; }// Limbu
    if (cp >= 0x1927u && cp <= 0x1928u) { return true; }
    if (cp >= 0x1932u && cp <= 0x1932u) { return true; }
    if (cp >= 0x1939u && cp <= 0x193bu) { return true; }
    if (cp >= 0x1a17u && cp <= 0x1a18u) { return true; }// Buginese
    if (cp >= 0x1a1bu && cp <= 0x1a1bu) { return true; }
    if (cp >= 0x1a56u && cp <= 0x1a56u) { return true; }// Tai Tham
    if (cp >= 0x1a58u && cp <= 0x1a5eu) { return true; }
    if (cp >= 0x1a60u && cp <= 0x1a60u) { return true; }
    if (cp >= 0x1a62u && cp <= 0x1a62u) { return true; }
    if (cp >= 0x1a65u && cp <= 0x1a6cu) { return true; }
    if (cp >= 0x1a73u && cp <= 0x1a7cu) { return true; }
    if (cp >= 0x1a7fu && cp <= 0x1a7fu) { return true; }
    if (cp >= 0x1ab0u && cp <= 0x1affu) { return true; }// Combining Diacritical Marks Extended
    if (cp >= 0x1b00u && cp <= 0x1b03u) { return true; }// Balinese
    if (cp >= 0x1b34u && cp <= 0x1b34u) { return true; }
    if (cp >= 0x1b36u && cp <= 0x1b3au) { return true; }
    if (cp >= 0x1b3cu && cp <= 0x1b3cu) { return true; }
    if (cp >= 0x1b42u && cp <= 0x1b42u) { return true; }
    if (cp >= 0x1b6bu && cp <= 0x1b73u) { return true; }
    if (cp >= 0x1b80u && cp <= 0x1b81u) { return true; }// Sundanese
    if (cp >= 0x1ba2u && cp <= 0x1ba5u) { return true; }
    if (cp >= 0x1ba8u && cp <= 0x1ba9u) { return true; }
    if (cp >= 0x1babu && cp <= 0x1badu) { return true; }
    if (cp >= 0x1be6u && cp <= 0x1be6u) { return true; }// Batak
    if (cp >= 0x1be8u && cp <= 0x1be9u) { return true; }
    if (cp >= 0x1bedu && cp <= 0x1bedu) { return true; }
    if (cp >= 0x1befu && cp <= 0x1bf1u) { return true; }
    if (cp >= 0x1c2cu && cp <= 0x1c33u) { return true; }// Lepcha
    if (cp >= 0x1c36u && cp <= 0x1c37u) { return true; }
    if (cp >= 0x1cd0u && cp <= 0x1cd2u) { return true; }// Vedic extensions
    if (cp >= 0x1cd4u && cp <= 0x1ce0u) { return true; }
    if (cp >= 0x1ce2u && cp <= 0x1ce8u) { return true; }
    if (cp >= 0x1cedu && cp <= 0x1cedu) { return true; }
    if (cp >= 0x1cf4u && cp <= 0x1cf4u) { return true; }
    if (cp >= 0x1cf8u && cp <= 0x1cf9u) { return true; }
    if (cp >= 0x1dc0u && cp <= 0x1dffu) { return true; }// Combining Diacritical Marks Supplement
    if (cp >= 0x20d0u && cp <= 0x20ffu) { return true; }// Combining Diacritical Marks for Symbols
    if (cp >= 0x2cefu && cp <= 0x2cf1u) { return true; }// Coptic
    if (cp >= 0x2d7fu && cp <= 0x2d7fu) { return true; }// Tifinagh
    if (cp >= 0x2de0u && cp <= 0x2deau) { return true; }// Cyrillic ext
    if (cp >= 0x302au && cp <= 0x302fu) { return true; }// CJK ideographic tone marks
    if (cp >= 0x3099u && cp <= 0x309au) { return true; }// kana voiced marks
    if (cp >= 0xa66fu && cp <= 0xa672u) { return true; }// Cyrillic ext
    if (cp >= 0xa674u && cp <= 0xa67du) { return true; }
    if (cp >= 0xa69eu && cp <= 0xa69fu) { return true; }
    if (cp >= 0xa6f0u && cp <= 0xa6f1u) { return true; }// Bamum
    if (cp >= 0xa802u && cp <= 0xa802u) { return true; }// Syloti Nagri
    if (cp >= 0xa806u && cp <= 0xa806u) { return true; }
    if (cp >= 0xa80bu && cp <= 0xa80bu) { return true; }
    if (cp >= 0xa825u && cp <= 0xa826u) { return true; }
    if (cp >= 0xa8c4u && cp <= 0xa8c5u) { return true; }// Saurashtra
    if (cp >= 0xa8e0u && cp <= 0xa8f1u) { return true; }// Devanagari ext
    if (cp >= 0xa926u && cp <= 0xa92du) { return true; }// Kayah Li
    if (cp >= 0xa947u && cp <= 0xa951u) { return true; }// Rejang
    if (cp >= 0xa980u && cp <= 0xa982u) { return true; }// Javanese
    if (cp >= 0xa9b3u && cp <= 0xa9b3u) { return true; }
    if (cp >= 0xa9b6u && cp <= 0xa9b9u) { return true; }
    if (cp >= 0xa9bcu && cp <= 0xa9bcu) { return true; }
    if (cp >= 0xa9e5u && cp <= 0xa9e5u) { return true; }// Myanmar ext
    if (cp >= 0xaa29u && cp <= 0xaa2eu) { return true; }// Cham
    if (cp >= 0xaa31u && cp <= 0xaa32u) { return true; }
    if (cp >= 0xaa35u && cp <= 0xaa36u) { return true; }
    if (cp >= 0xaa43u && cp <= 0xaa43u) { return true; }
    if (cp >= 0xaa4cu && cp <= 0xaa4cu) { return true; }
    if (cp >= 0xaa7cu && cp <= 0xaa7cu) { return true; }// Myanmar ext
    if (cp >= 0xaab0u && cp <= 0xaab0u) { return true; }// Tai Viet
    if (cp >= 0xaab2u && cp <= 0xaab4u) { return true; }
    if (cp >= 0xaab7u && cp <= 0xaab8u) { return true; }
    if (cp >= 0xaabeu && cp <= 0xaabfu) { return true; }
    if (cp >= 0xaac1u && cp <= 0xaac1u) { return true; }
    if (cp >= 0xaaecu && cp <= 0xaaedu) { return true; }// Meetei Mayek
    if (cp >= 0xaaf6u && cp <= 0xaaf6u) { return true; }
    if (cp >= 0xabe5u && cp <= 0xabe5u) { return true; }// Meetei Mayek
    if (cp >= 0xabe8u && cp <= 0xabe8u) { return true; }
    if (cp >= 0xabedu && cp <= 0xabedu) { return true; }
    if (cp >= 0xfb1eu && cp <= 0xfb1eu) { return true; }// Hebrew judeo-spanish
    if (cp >= 0xfe00u && cp <= 0xfe0fu) { return true; }// Variation Selectors
    if (cp >= 0xfe20u && cp <= 0xfe2fu) { return true; }// Combining Half Marks
    if (cp >= 0x101fdu && cp <= 0x101fdu) { return true; }// Phaistos
    if (cp >= 0x102e0u && cp <= 0x102e0u) { return true; }// Coptic epact
    if (cp >= 0x10376u && cp <= 0x1037au) { return true; }// Old Permic
    if (cp >= 0x10a01u && cp <= 0x10a03u) { return true; }// Kharoshthi
    if (cp >= 0x10a05u && cp <= 0x10a06u) { return true; }
    if (cp >= 0x10a0cu && cp <= 0x10a0fu) { return true; }
    if (cp >= 0x10a38u && cp <= 0x10a3au) { return true; }
    if (cp >= 0x10a3fu && cp <= 0x10a3fu) { return true; }
    if (cp >= 0x10ae5u && cp <= 0x10ae6u) { return true; }// Manichaean
    if (cp >= 0x11001u && cp <= 0x11001u) { return true; }// Brahmi
    if (cp >= 0x11038u && cp <= 0x11046u) { return true; }
    if (cp >= 0x1107fu && cp <= 0x11081u) { return true; }
    if (cp >= 0x110b3u && cp <= 0x110b6u) { return true; }
    if (cp >= 0x110b9u && cp <= 0x110bau) { return true; }
    if (cp >= 0x11100u && cp <= 0x11102u) { return true; }// Chakma
    if (cp >= 0x11127u && cp <= 0x1112bu) { return true; }
    if (cp >= 0x1112du && cp <= 0x11134u) { return true; }
    if (cp >= 0x11173u && cp <= 0x11173u) { return true; }// Mahajani
    if (cp >= 0x11180u && cp <= 0x11181u) { return true; }// Sharada
    if (cp >= 0x111b6u && cp <= 0x111beu) { return true; }
    if (cp >= 0x1122fu && cp <= 0x11231u) { return true; }// Khojki
    if (cp >= 0x11234u && cp <= 0x11234u) { return true; }
    if (cp >= 0x11236u && cp <= 0x11237u) { return true; }
    if (cp >= 0x1123eu && cp <= 0x1123eu) { return true; }
    if (cp >= 0x112dfu && cp <= 0x112dfu) { return true; }// Khudawadi
    if (cp >= 0x112e3u && cp <= 0x112eau) { return true; }
    if (cp >= 0x11300u && cp <= 0x11301u) { return true; }// Grantha
    if (cp >= 0x1133cu && cp <= 0x1133cu) { return true; }
    if (cp >= 0x11340u && cp <= 0x11340u) { return true; }
    if (cp >= 0x11366u && cp <= 0x1136cu) { return true; }
    if (cp >= 0x11370u && cp <= 0x11374u) { return true; }
    if (cp >= 0x11438u && cp <= 0x1143fu) { return true; }// Newa
    if (cp >= 0x11442u && cp <= 0x11444u) { return true; }
    if (cp >= 0x11446u && cp <= 0x11446u) { return true; }
    if (cp >= 0x114b3u && cp <= 0x114b8u) { return true; }// Tirhuta
    if (cp >= 0x114bau && cp <= 0x114bau) { return true; }
    if (cp >= 0x114bfu && cp <= 0x114c0u) { return true; }
    if (cp >= 0x114c2u && cp <= 0x114c3u) { return true; }
    if (cp >= 0x115b2u && cp <= 0x115b5u) { return true; }// Siddham
    if (cp >= 0x115bcu && cp <= 0x115bdu) { return true; }
    if (cp >= 0x115bfu && cp <= 0x115c0u) { return true; }
    if (cp >= 0x115dcu && cp <= 0x115ddu) { return true; }
    if (cp >= 0x11633u && cp <= 0x1163au) { return true; }// Modi
    if (cp >= 0x1163du && cp <= 0x1163du) { return true; }
    if (cp >= 0x1163fu && cp <= 0x11640u) { return true; }
    if (cp >= 0x116abu && cp <= 0x116abu) { return true; }// Takri
    if (cp >= 0x116adu && cp <= 0x116adu) { return true; }
    if (cp >= 0x116b0u && cp <= 0x116b5u) { return true; }
    if (cp >= 0x116b7u && cp <= 0x116b7u) { return true; }
    if (cp >= 0x1171du && cp <= 0x1171fu) { return true; }// Ahom
    if (cp >= 0x11722u && cp <= 0x11725u) { return true; }
    if (cp >= 0x11727u && cp <= 0x1172bu) { return true; }
    if (cp >= 0x11c30u && cp <= 0x11c36u) { return true; }// Bhaiksuki
    if (cp >= 0x11c38u && cp <= 0x11c3du) { return true; }
    if (cp >= 0x11c3fu && cp <= 0x11c3fu) { return true; }
    if (cp >= 0x11c92u && cp <= 0x11ca7u) { return true; }// Marchen
    if (cp >= 0x11caau && cp <= 0x11cb0u) { return true; }
    if (cp >= 0x11cb2u && cp <= 0x11cb3u) { return true; }
    if (cp >= 0x11cb5u && cp <= 0x11cb6u) { return true; }
    if (cp >= 0x16af0u && cp <= 0x16af4u) { return true; }// Bassa Vah
    if (cp >= 0x16b30u && cp <= 0x16b36u) { return true; }// Pahawh Hmong
    if (cp >= 0x16f8fu && cp <= 0x16f92u) { return true; }// Miao
    if (cp >= 0x1bc9du && cp <= 0x1bc9eu) { return true; }// Duployan
    if (cp >= 0x1d167u && cp <= 0x1d169u) { return true; }// Musical symbols
    if (cp >= 0x1d17bu && cp <= 0x1d182u) { return true; }
    if (cp >= 0x1d185u && cp <= 0x1d18bu) { return true; }
    if (cp >= 0x1d1aau && cp <= 0x1d1adu) { return true; }
    if (cp >= 0x1d242u && cp <= 0x1d244u) { return true; }// Combining Greek musical
    if (cp >= 0x1da00u && cp <= 0x1da36u) { return true; }// SignWriting
    if (cp >= 0x1da3bu && cp <= 0x1da6cu) { return true; }
    if (cp >= 0x1da75u && cp <= 0x1da75u) { return true; }
    if (cp >= 0x1da84u && cp <= 0x1da84u) { return true; }
    if (cp >= 0x1da9bu && cp <= 0x1da9fu) { return true; }
    if (cp >= 0x1daa1u && cp <= 0x1daafu) { return true; }
    if (cp >= 0x1e000u && cp <= 0x1e006u) { return true; }// Glagolitic supplement
    if (cp >= 0x1e008u && cp <= 0x1e018u) { return true; }
    if (cp >= 0x1e01bu && cp <= 0x1e021u) { return true; }
    if (cp >= 0x1e023u && cp <= 0x1e024u) { return true; }
    if (cp >= 0x1e026u && cp <= 0x1e02au) { return true; }
    if (cp >= 0x1e8d0u && cp <= 0x1e8d6u) { return true; }// Mende Kikakui
    if (cp >= 0x1e944u && cp <= 0x1e94au) { return true; }// Adlam
    if (cp >= 0xe0100u && cp <= 0xe01efu) { return true; }// Variation Selectors Supplement
    return false;
}

[[nodiscard]] bool is_letter_cp(uint32_t cp) noexcept {
    // Unicode L* approximation for the scripts OCR targets need. Used for the
    // GPT-2 pre-tokenizer's \p{L} runs AND added-token single_word boundaries.
    // It MUST NOT overlap the \p{N} digit blocks (Arabic-Indic, Devanagari,
    // fullwidth, ...): those are numbers, and classifying them as letters would
    // change the GPT-2 " ?\p{L}+| ?\p{N}+" split. This is a documented subset,
    // not the full Unicode L* category; code points outside it are treated as
    // symbols (never silently as letters).
    if (is_gpt2_number(cp)) { return false; }// digits are never letters
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) { return true; }
    if (cp >= 0x00c0u && cp <= 0x024fu && cp != 0x00d7u && cp != 0x00f7u) { return true; }
    // Greek/Coptic, Cyrillic, Armenian, Hebrew, Arabic LETTERS, Syriac, Thaana,
    // Devanagari letters, etc. The explicit digit blocks inside 0x0370..0x1fff
    // are excluded by the is_gpt2_number guard above; the remaining Nd blocks
    // are excluded here so they are not treated as letters.
    if (cp >= 0x0370u && cp <= 0x1fffu) {
        if (cp >= 0x0660u && cp <= 0x0669u) { return false; }// Arabic-Indic
        if (cp >= 0x06f0u && cp <= 0x06f9u) { return false; }// Extended Arabic-Indic
        if (cp >= 0x0966u && cp <= 0x096fu) { return false; }// Devanagari
        if (cp >= 0x09e6u && cp <= 0x09efu) { return false; }// Bengali
        if (cp >= 0x0a66u && cp <= 0x0a6fu) { return false; }// Gurmukhi
        if (cp >= 0x0ae6u && cp <= 0x0aefu) { return false; }// Gujarati
        if (cp >= 0x0b66u && cp <= 0x0b6fu) { return false; }// Oriya
        if (cp >= 0x0be6u && cp <= 0x0befu) { return false; }// Tamil
        if (cp >= 0x0c66u && cp <= 0x0c6fu) { return false; }// Telugu
        if (cp >= 0x0ce6u && cp <= 0x0cefu) { return false; }// Kannada
        if (cp >= 0x0d66u && cp <= 0x0d6fu) { return false; }// Malayalam
        if (cp >= 0x0e50u && cp <= 0x0e59u) { return false; }// Thai
        if (cp >= 0x0ed0u && cp <= 0x0ed9u) { return false; }// Lao
        if (cp >= 0x0f20u && cp <= 0x0f29u) { return false; }// Tibetan
        if (cp >= 0x1040u && cp <= 0x1049u) { return false; }// Myanmar
        if (cp >= 0x17e0u && cp <= 0x17e9u) { return false; }// Khmer
        if (cp >= 0x1810u && cp <= 0x1819u) { return false; }// Mongolian
        return true;
    }
    if (cp >= 0x3040u && cp <= 0x30ffu) { return true; }// kana
    if (cp >= 0x3400u && cp <= 0x4dbfu) { return true; }// CJK ext A
    if (cp >= 0x4e00u && cp <= 0x9fffu) { return true; }// CJK
    if (cp >= 0xac00u && cp <= 0xd7afu) { return true; }// Hangul
    if (cp >= 0xf900u && cp <= 0xfaffu) { return true; }// CJK compat
    return false;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level pre-tokenization regex, implemented as an explicit matcher
// over code points (no std::regex: the project builds without exceptions and
// the pattern must be exact). Pattern (HF ByteLevel):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// ---------------------------------------------------------------------------

[[nodiscard]] bool is_gpt2_whitespace(uint32_t cp) noexcept {
    // The GPT-2 pattern's \s class: [ \t\n\r\f\v] plus the Unicode spaces the
    // reference pattern includes via \s under Python re (Unicode mode).
    switch (cp) {
        case ' ': case '\t': case '\n': case '\r': case '\f': case '\v':
        case 0x85u: case 0xa0u: case 0x1680u:
        case 0x2028u: case 0x2029u: case 0x202fu: case 0x205fu: case 0x3000u:
            return true;
        default:
            return cp >= 0x2000u && cp <= 0x200au;
    }
}

[[nodiscard]] bool is_gpt2_number(uint32_t cp) noexcept {
    // \p{N}: decimal digits plus the common number blocks OCR needs.
    if (cp >= '0' && cp <= '9') { return true; }
    if (cp >= 0x0660u && cp <= 0x0669u) { return true; }// Arabic-Indic
    if (cp >= 0x06f0u && cp <= 0x06f9u) { return true; }// Extended Arabic-Indic
    if (cp >= 0x0966u && cp <= 0x096fu) { return true; }// Devanagari
    if (cp >= 0xff10u && cp <= 0xff19u) { return true; }// fullwidth
    if (cp >= 0x2160u && cp <= 0x2188u) { return true; }// Roman numerals
    return false;
}

struct CpSpan {
    size_t begin;// byte offset
    size_t end;  // byte offset (exclusive)
};

// Tokenize one normal (non-added) text chunk into GPT-2 pre-token byte spans.
[[nodiscard]] luisa::vector<CpSpan> gpt2_pretokenize(luisa::string_view text) {
    luisa::vector<CpSpan> spans;
    // Decode to code points with byte offsets.
    struct Cp {
        uint32_t value;
        size_t begin;
        size_t end;
    };
    luisa::vector<Cp> cps;
    size_t pos = 0;
    while (pos < text.size()) {
        uint32_t cp = 0;
        const auto len = decode_utf8(text, pos, cp);
        if (len == 0) {
            // Caller validated UTF-8; treat a stray byte as its own symbol.
            cps.emplace_back(Cp{static_cast<uint8_t>(text[pos]), pos, pos + 1});
            pos += 1;
        } else {
            cps.emplace_back(Cp{cp, pos, pos + len});
            pos += len;
        }
    }
    const auto n = cps.size();
    const auto is_letter = [&](size_t i) { return is_letter_cp(cps[i].value); };
    const auto is_number = [&](size_t i) { return is_gpt2_number(cps[i].value); };
    const auto is_space = [&](size_t i) { return is_gpt2_whitespace(cps[i].value); };

    size_t i = 0;
    while (i < n) {
        const size_t start = i;
        // Contractions: 's 't 're 've 'm 'll 'd (case-sensitive, ASCII ').
        if (cps[i].value == '\'' && i + 1 < n) {
            const auto next = cps[i + 1].value;
            size_t consumed = 0;
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                consumed = 2;
            } else if (i + 2 < n && cps[i + 2].value == 'e' && (next == 'r' || next == 'v')) {
                consumed = 3;
            } else if (i + 2 < n && next == 'l' && cps[i + 2].value == 'l') {
                consumed = 3;
            }
            if (consumed > 0) {
                spans.emplace_back(CpSpan{cps[start].begin, cps[start + consumed - 1].end});
                i += consumed;
                continue;
            }
        }
        // Optional single leading space, then a run of letters / numbers /
        // other non-space non-letter non-number symbols.
        size_t j = i;
        bool has_prefix_space = false;
        if (cps[j].value == ' ') {
            has_prefix_space = true;
            j += 1;
        }
        if (j < n && !is_space(j)) {
            if (is_letter(j)) {
                while (j < n && is_letter(j)) { j += 1; }
                spans.emplace_back(CpSpan{cps[start].begin, cps[j - 1].end});
                i = j;
                continue;
            }
            if (is_number(j)) {
                while (j < n && is_number(j)) { j += 1; }
                spans.emplace_back(CpSpan{cps[start].begin, cps[j - 1].end});
                i = j;
                continue;
            }
            // Symbol run: [^\s\p{L}\p{N}]+
            while (j < n && !is_space(j) && !is_letter(j) && !is_number(j)) { j += 1; }
            spans.emplace_back(CpSpan{cps[start].begin, cps[j - 1].end});
            i = j;
            continue;
        }
        // Whitespace run. \s+(?!\S): all but the last whitespace when the run
        // is followed by a non-space; otherwise the whole run.
        size_t k = i;
        while (k < n && is_space(k)) { k += 1; }
        size_t end = k;
        if (k < n && k > i + 1) { end = k - 1; }// leave one space for the next token
        spans.emplace_back(CpSpan{cps[start].begin, cps[end - 1].end});
        i = end;
        (void)has_prefix_space;
    }
    return spans;
}

// The GPT-2 bytes_to_unicode mapping: byte -> code point used in the vocab.
[[nodiscard]] uint32_t gpt2_byte_to_codepoint(uint8_t b) noexcept {
    const bool printable = (b >= 0x21u && b <= 0x7eu) || (b >= 0xa1u && b <= 0xacu) || (b >= 0xaeu && b <= 0xffu);
    if (printable) { return b; }
    // Non-printable bytes map to 256.. in order of first appearance.
    uint32_t offset = 0;
    for (uint32_t k = 0; k < b; k++) {
        const auto kb = static_cast<uint8_t>(k);
        const bool kp = (kb >= 0x21u && kb <= 0x7eu) || (kb >= 0xa1u && kb <= 0xacu) || (kb >= 0xaeu && kb <= 0xffu);
        if (!kp) { offset += 1; }
    }
    return 256u + offset;
}

}// namespace

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

Tokenizer::~Tokenizer() noexcept = default;

void TokenizerDeleter::operator()(Tokenizer *tokenizer) const noexcept { delete tokenizer; }

[[nodiscard]] Result<luisa::unique_ptr<Tokenizer, TokenizerDeleter>> Tokenizer::load(
    const luisa::filesystem::path &tokenizer_json, const TokenizerConfig &config) {
    using R = Result<luisa::unique_ptr<Tokenizer, TokenizerDeleter>>;
    const auto path_text = luisa::to_string(tokenizer_json);
    std::FILE *file = std::fopen(path_text.c_str(), "rb");
    if (file == nullptr) {
        return R::failure(StatusCode::not_found,
                          luisa::string{"cannot open tokenizer file '"} + path_text + "'", "tokenizer");
    }
    luisa::string bytes;
    char buffer[65536];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        if (bytes.size() + got > config.max_document_bytes) {
            std::fclose(file);
            return R::failure(StatusCode::invalid_argument, "tokenizer document exceeds the size limit", "tokenizer");
        }
        bytes.append(buffer, got);
    }
    const bool read_ok = std::ferror(file) == 0;
    std::fclose(file);
    if (!read_ok) {
        return R::failure(StatusCode::io_error, "failed to read tokenizer file", "tokenizer");
    }
    return parse(bytes, config);
}

[[nodiscard]] Result<luisa::unique_ptr<Tokenizer, TokenizerDeleter>> Tokenizer::parse(
    luisa::string_view json, const TokenizerConfig &config) {
    using R = Result<luisa::unique_ptr<Tokenizer, TokenizerDeleter>>;
    if (json.size() > config.max_document_bytes) {
        return R::failure(StatusCode::invalid_argument, "tokenizer document exceeds the size limit", "tokenizer");
    }
    if (!ocr_is_valid_utf8(json)) {
        return R::failure(malformed("tokenizer document is not valid UTF-8", "tokenizer"));
    }
    JsonDoc holder;
    yyjson_read_err err;
    holder.doc = yyjson_read_opts(const_cast<char *>(json.data()), json.size(), YYJSON_READ_NOFLAG, nullptr, &err);
    if (holder.doc == nullptr) {
        return R::failure(malformed(err.msg != nullptr ? err.msg : "JSON parse error", "tokenizer"));
    }
    auto *root = yyjson_doc_get_root(holder.doc);
    if (!yyjson_is_obj(root)) { return R::failure(malformed("root must be an object", "tokenizer")); }

    auto tokenizer = luisa::unique_ptr<Tokenizer, TokenizerDeleter>{new Tokenizer{}};

    // --- added_tokens (before the model, so special ids are known) ---
    if (auto *added = yyjson_obj_get(root, "added_tokens"); added != nullptr && !yyjson_is_null(added)) {
        if (!yyjson_is_arr(added)) { return R::failure(malformed("added_tokens must be an array", "added_tokens")); }
        if (yyjson_arr_size(added) > config.max_added_tokens) {
            return R::failure(malformed("too many added tokens", "added_tokens"));
        }
        yyjson_val *item;
        size_t ai, am;
        yyjson_arr_foreach(added, ai, am, item) {
            if (!yyjson_is_obj(item)) { return R::failure(malformed("added token entry must be an object", "added_tokens")); }
            auto *id_val = yyjson_obj_get(item, "id");
            auto *content_val = yyjson_obj_get(item, "content");
            uint32_t added_id = 0;
            if (!json_u32(id_val, added_id) || !yyjson_is_str(content_val)) {
                return R::failure(malformed("added token needs a uint32 id and string content", "added_tokens"));
            }
            AddedToken token;
            token.id = added_id;
            token.content = luisa::string{json_str(content_val)};
            if (token.content.empty()) { return R::failure(malformed("added token content is empty", "added_tokens")); }
            if (!ocr_is_valid_utf8(token.content)) { return R::failure(malformed("added token content is not UTF-8", "added_tokens")); }
            const auto read_bool = [&](const char *key) -> bool {
                auto *v = yyjson_obj_get(item, key);
                return v != nullptr && yyjson_is_true(v);
            };
            token.special = read_bool("special");
            token.lstrip = read_bool("lstrip");
            token.rstrip = read_bool("rstrip");
            token.single_word = read_bool("single_word");
            if (tokenizer->_special_ids.count(token.id) != 0) {
                return R::failure(malformed("duplicate added token id", "added_tokens"));
            }
            tokenizer->_special_ids[token.id] = token.special;
            tokenizer->_added.emplace_back(std::move(token));
        }
    }

    // --- model ---
    auto *model = yyjson_obj_get(root, "model");
    if (model == nullptr || !yyjson_is_obj(model)) { return R::failure(malformed("missing model object", "model")); }
    const auto model_type = json_str(yyjson_obj_get(model, "type"));
    if (model_type == "BPE") {
        tokenizer->_kind = TokenizerModelKind::byte_level_bpe;
        // Byte-level BPE requires the GPT-2 byte encoder/decoder. A BPE model
        // without them (e.g. a raw char-level BPE) is not supported.
        auto *byte_encoder = yyjson_obj_get(model, "byte_encoder");
        auto *byte_decoder = yyjson_obj_get(model, "byte_decoder");
        if (byte_encoder == nullptr || !yyjson_is_obj(byte_encoder) ||
            byte_decoder == nullptr || !yyjson_is_obj(byte_decoder)) {
            return R::failure(unsupported("BPE without a byte_encoder/byte_decoder (only byte-level BPE is supported)", "model"));
        }
        // Validate the byte decoder maps every byte piece to a unique byte.
        bool seen_byte[256] = {};
        yyjson_val *bk;
        yyjson_val *bv;
        size_t bi, bm;
        yyjson_obj_foreach(byte_decoder, bi, bm, bk, bv) {
            const auto piece = json_str(bk);
            if (!yyjson_is_uint(bv) || piece.empty()) {
                return R::failure(malformed("byte_decoder entries must map a piece to a uint byte", "model.byte_decoder"));
            }
            const auto byte_value = yyjson_get_uint(bv);
            if (byte_value > 255u) { return R::failure(malformed("byte_decoder value out of range", "model.byte_decoder")); }
            // The piece is a single code point in the GPT-2 mapping (0..511).
            uint32_t cp = 0;
            if (decode_utf8(piece, 0, cp) != piece.size()) {
                return R::failure(malformed("byte_decoder piece is not one code point", "model.byte_decoder"));
            }
            if (cp >= 512u) {
                return R::failure(malformed("byte_decoder code point out of the GPT-2 range", "model.byte_decoder"));
            }
            if (tokenizer->_byte_dec_present[cp]) {
                return R::failure(malformed("duplicate code point in byte_decoder", "model.byte_decoder"));
            }
            tokenizer->_byte_decoder[cp] = static_cast<uint32_t>(byte_value);
            tokenizer->_byte_dec_present[cp] = true;
            if (seen_byte[byte_value]) { return R::failure(malformed("duplicate byte in byte_decoder", "model.byte_decoder")); }
            seen_byte[byte_value] = true;
        }
        for (uint32_t b = 0; b < 256u; b++) {
            if (!seen_byte[b]) { return R::failure(malformed("byte_decoder does not cover all 256 bytes", "model.byte_decoder")); }
        }
        // Every GPT-2 piece code point must be present (the mapping is a
        // bijection over 0..511 minus the unused printable gaps).
        for (uint32_t b = 0; b < 256u; b++) {
            const auto cp = gpt2_byte_to_codepoint(static_cast<uint8_t>(b));
            if (!tokenizer->_byte_dec_present[cp] || tokenizer->_byte_decoder[cp] != b) {
                return R::failure(malformed("byte_decoder is not the GPT-2 bytes_to_unicode mapping", "model.byte_decoder"));
            }
        }
        tokenizer->_has_byte_decoder = true;

        // vocab: piece -> id
        auto *vocab = yyjson_obj_get(model, "vocab");
        if (vocab == nullptr || !yyjson_is_obj(vocab)) { return R::failure(malformed("BPE model needs a vocab object", "model.vocab")); }
        if (yyjson_obj_size(vocab) > config.max_vocab_entries) {
            return R::failure(malformed("vocab exceeds the entry limit", "model.vocab"));
        }
        yyjson_val *vk;
        yyjson_val *vv;
        size_t vi, vm;
        yyjson_obj_foreach(vocab, vi, vm, vk, vv) {
            const auto piece = json_str(vk);
            uint32_t vocab_id = 0;
            if (!json_u32(vv, vocab_id) || piece.empty()) {
                return R::failure(malformed("vocab entries must map a non-empty piece to a uint32 id", "model.vocab"));
            }
            const auto id = vocab_id;
            if (tokenizer->_bpe_vocab.count(luisa::string{piece}) != 0) {
                return R::failure(malformed("duplicate vocab piece", "model.vocab"));
            }
            tokenizer->_bpe_vocab[luisa::string{piece}] = id;
            tokenizer->_id_to_piece[id] = luisa::string{piece};
        }
        // byte tokens: the single-byte pieces of the GPT-2 mapping.
        for (uint32_t b = 0; b < 256u; b++) {
            luisa::string piece;
            append_utf8(piece, gpt2_byte_to_codepoint(static_cast<uint8_t>(b)));
            const auto it = tokenizer->_bpe_vocab.find(piece);
            if (it == tokenizer->_bpe_vocab.end()) {
                return R::failure(malformed("vocab is missing a GPT-2 byte token", "model.vocab"));
            }
            tokenizer->_byte_tokens[b] = it->second;
        }
        tokenizer->_has_byte_tokens = true;

        // merges: list of "a b" strings (or [a, b] pairs).
        auto *merges = yyjson_obj_get(model, "merges");
        if (merges == nullptr || !yyjson_is_arr(merges)) {
            return R::failure(malformed("BPE model needs a merges array", "model.merges"));
        }
        yyjson_val *merge;
        size_t mi, mm;
        uint32_t rank = 0;
        yyjson_arr_foreach(merges, mi, mm, merge) {
            luisa::string a, b;
            if (yyjson_is_str(merge)) {
                const auto text = json_str(merge);
                const auto space = text.find(' ');
                if (space == luisa::string_view::npos || space == 0 || space + 1 >= text.size()) {
                    return R::failure(malformed("merge string must be 'a b'", "model.merges"));
                }
                a = luisa::string{text.substr(0, space)};
                b = luisa::string{text.substr(space + 1)};
            } else if (yyjson_is_arr(merge) && yyjson_arr_size(merge) == 2) {
                a = luisa::string{json_str(yyjson_arr_get(merge, 0))};
                b = luisa::string{json_str(yyjson_arr_get(merge, 1))};
            } else {
                return R::failure(malformed("merge entry must be a string or a pair", "model.merges"));
            }
            if (a.empty() || b.empty() || a.find(' ') != luisa::string::npos || b.find(' ') != luisa::string::npos) {
                return R::failure(malformed("merge pieces must be non-empty and space-free", "model.merges"));
            }
            luisa::string key = a;
            key.push_back('\0');
            key += b;
            if (tokenizer->_bpe_ranks.count(key) == 0) {
                tokenizer->_bpe_ranks[std::move(key)] = rank;
            }
            rank += 1;
        }
    } else if (model_type == "Unigram") {
        tokenizer->_kind = TokenizerModelKind::unigram;
        auto *vocab = yyjson_obj_get(model, "vocab");
        if (vocab == nullptr || !yyjson_is_arr(vocab)) {
            return R::failure(malformed("Unigram model needs a vocab array", "model.vocab"));
        }
        if (yyjson_arr_size(vocab) > config.max_vocab_entries) {
            return R::failure(malformed("vocab exceeds the entry limit", "model.vocab"));
        }
        auto *unk_id_val = yyjson_obj_get(model, "unk_id");
        if (unk_id_val != nullptr && !yyjson_is_null(unk_id_val)) {
            uint32_t unk_id = 0;
            if (!json_u32(unk_id_val, unk_id)) {
                return R::failure(malformed("unk_id must be a uint32", "model.unk_id"));
            }
            tokenizer->_unigram_unk_id = unk_id;
            tokenizer->_unigram_has_unk = true;
            tokenizer->_unk = tokenizer->_unigram_unk_id;
        }
        auto *byte_fallback = yyjson_obj_get(model, "byte_fallback");
        tokenizer->_unigram_byte_fallback = byte_fallback != nullptr && yyjson_is_true(byte_fallback);
        // Explicit BOS/EOS insertion flags (SentencePiece add_bos/add_eos).
        // Default false; the <s>/</s> presence heuristic below only applies
        // when the flags are absent.
        bool bos_flag_present = false, eos_flag_present = false;
        if (auto *v = yyjson_obj_get(model, "add_bos_token"); v != nullptr && !yyjson_is_null(v)) {
            if (!yyjson_is_bool(v)) { return R::failure(malformed("add_bos_token must be a bool", "model.add_bos_token")); }
            tokenizer->_unigram_add_bos = yyjson_is_true(v);
            bos_flag_present = true;
        }
        if (auto *v = yyjson_obj_get(model, "add_eos_token"); v != nullptr && !yyjson_is_null(v)) {
            if (!yyjson_is_bool(v)) { return R::failure(malformed("add_eos_token must be a bool", "model.add_eos_token")); }
            tokenizer->_unigram_add_eos = yyjson_is_true(v);
            eos_flag_present = true;
        }
        yyjson_val *entry;
        size_t ui, um;
        uint32_t id = 0;
        yyjson_arr_foreach(vocab, ui, um, entry) {
            if (!yyjson_is_arr(entry) || yyjson_arr_size(entry) != 2) {
                return R::failure(malformed("unigram vocab entry must be [piece, score]", "model.vocab"));
            }
            auto *piece_val = yyjson_arr_get(entry, 0);
            auto *score_val = yyjson_arr_get(entry, 1);
            if (!yyjson_is_str(piece_val) || !yyjson_is_num(score_val)) {
                return R::failure(malformed("unigram vocab entry needs a string piece and a numeric score", "model.vocab"));
            }
            const auto piece = luisa::string{json_str(piece_val)};
            const auto score = static_cast<float>(yyjson_get_num(score_val));
            if (!std::isfinite(score)) { return R::failure(malformed("unigram score is not finite", "model.vocab")); }
            if (piece.empty()) { return R::failure(malformed("unigram piece is empty", "model.vocab")); }
            if (tokenizer->_unigram_lookup.count(piece) != 0) {
                return R::failure(malformed("duplicate unigram piece", "model.vocab"));
            }
            tokenizer->_unigram_lookup[piece] = id;
            tokenizer->_unigram_pieces.emplace_back(UnigramPiece{piece, score});
            tokenizer->_id_to_piece[id] = piece;
            if (is_byte_fallback_piece(piece)) {
                tokenizer->_unigram_byte_token_ids[id] = true;
            }
            id += 1;
        }
        if (tokenizer->_unigram_has_unk && tokenizer->_unigram_unk_id >= id) {
            return R::failure(malformed("unk_id out of range", "model.unk_id"));
        }
        // When the explicit flags are absent, fall back to the <s>/</s>
        // presence heuristic (applied after special ids are resolved below).
        tokenizer->_unigram_bos_flag_present = bos_flag_present;
        tokenizer->_unigram_eos_flag_present = eos_flag_present;
    } else {
        return R::failure(unsupported(luisa::string{"model type '"} + luisa::string{model_type} + "' (only byte-level BPE and Unigram are supported)", "model.type"));
    }
    tokenizer->_vocab_size = static_cast<uint32_t>(tokenizer->_id_to_piece.size());

    // --- normalizer ---
    if (auto *normalizer = yyjson_obj_get(root, "normalizer"); normalizer != nullptr && !yyjson_is_null(normalizer)) {
        if (!yyjson_is_obj(normalizer)) { return R::failure(malformed("normalizer must be an object", "normalizer")); }
        const auto type = json_str(yyjson_obj_get(normalizer, "type"));
        if (tokenizer->_kind == TokenizerModelKind::byte_level_bpe) {
            if (type != "NFC") {
                return R::failure(unsupported(luisa::string{"normalizer '"} + luisa::string{type} + "' (only null/NFC for byte-level BPE)", "normalizer"));
            }
            // NFC is accepted as a contract only. Full composition is NOT
            // implemented: encode rejects any input containing a combining
            // mark OR any NFC non-invariant character (singleton mappings and
            // canonical decompositions; fail closed) rather than silently
            // passing raw bytes through. Precomposed NFC text passes.
            tokenizer->_bpe_nfc_normalize = true;
        } else {
            // Unigram: the SentencePiece Metaspace/Replace convention is
            // applied as the space replacement; other normalizers fail closed.
            if (type != "Metaspace" && type != "Replace" && type != "NFC" && type != "Identity" && type != "Precompiled") {
                return R::failure(unsupported(luisa::string{"normalizer '"} + luisa::string{type} + "' for Unigram", "normalizer"));
            }
            if (type == "Precompiled") {
                // A precompiled charsmap cannot be interpreted without the
                // SentencePiece runtime; reject rather than approximate.
                return R::failure(unsupported("Precompiled normalizer requires the SentencePiece runtime", "normalizer"));
            }
            if (type == "NFC") {
                // Full NFC composition is not implemented for Unigram; reject
                // rather than silently pass decomposed input through.
                return R::failure(unsupported("NFC normalizer for Unigram is not implemented (use Metaspace or null)", "normalizer"));
            }
            if (type == "Metaspace" || type == "Replace") {
                // Only the U+2581 metaspace replacement is supported. Any other
                // replacement character would change the space convention, so
                // fail closed rather than silently substitute U+2581.
                if (auto *rep = yyjson_obj_get(normalizer, "replacement"); rep != nullptr && !yyjson_is_null(rep)) {
                    if (!yyjson_is_str(rep)) {
                        return R::failure(malformed("normalizer.replacement must be a string", "normalizer"));
                    }
                    if (json_str(rep) != "\xe2\x96\x81") {
                        return R::failure(unsupported("only the U+2581 metaspace replacement is supported", "normalizer.replacement"));
                    }
                }
                // Honor add_prefix_space (default true). An explicit false means
                // encode must NOT prepend the structural metaspace.
                if (auto *aps = yyjson_obj_get(normalizer, "add_prefix_space"); aps != nullptr && !yyjson_is_null(aps)) {
                    if (!yyjson_is_bool(aps)) {
                        return R::failure(malformed("normalizer.add_prefix_space must be a bool", "normalizer"));
                    }
                    tokenizer->_unigram_add_prefix_space = yyjson_is_true(aps);
                }
            }
        }
    }

    // --- pre_tokenizer (BPE only) ---
    if (tokenizer->_kind == TokenizerModelKind::byte_level_bpe) {
        auto *pre = yyjson_obj_get(root, "pre_tokenizer");
        bool saw_byte_level = false;
        const auto parse_byte_level = [&](yyjson_val *node) -> Status {
            const auto type = json_str(yyjson_obj_get(node, "type"));
            if (type != "ByteLevel") { return unsupported(luisa::string{"pre-tokenizer '"} + luisa::string{type} + "'", "pre_tokenizer"); }
            auto *aps = yyjson_obj_get(node, "add_prefix_space");
            tokenizer->_byte_level_add_prefix_space = aps != nullptr && yyjson_is_true(aps);
            // trim_offsets=true would require offset tracking we do not expose;
            // the default ByteLevel used by OCR checkpoints is false.
            auto *trim = yyjson_obj_get(node, "trim_offsets");
            if (trim != nullptr && yyjson_is_true(trim)) {
                return unsupported("ByteLevel trim_offsets=true", "pre_tokenizer");
            }
            auto *use_regex = yyjson_obj_get(node, "use_regex");
            if (use_regex != nullptr && yyjson_is_false(use_regex)) {
                return unsupported("ByteLevel use_regex=false", "pre_tokenizer");
            }
            saw_byte_level = true;
            return Status::success();
        };
        if (pre != nullptr && !yyjson_is_null(pre)) {
            if (!yyjson_is_obj(pre)) { return R::failure(malformed("pre_tokenizer must be an object", "pre_tokenizer")); }
            const auto type = json_str(yyjson_obj_get(pre, "type"));
            if (type == "Sequence") {
                auto *pretoks = yyjson_obj_get(pre, "pretokenizers");
                if (pretoks == nullptr || !yyjson_is_arr(pretoks)) {
                    return R::failure(malformed("Sequence pre_tokenizer needs pretokenizers", "pre_tokenizer"));
                }
                yyjson_val *node;
                size_t pi, pm;
                yyjson_arr_foreach(pretoks, pi, pm, node) {
                    if (!yyjson_is_obj(node)) { return R::failure(malformed("pretokenizer node must be an object", "pre_tokenizer")); }
                    const auto ntype = json_str(yyjson_obj_get(node, "type"));
                    if (ntype == "ByteLevel") {
                        auto s = parse_byte_level(node);
                        if (!s.ok()) { return R::failure(std::move(s)); }
                    } else if (ntype == "Split") {
                        // Only the exact GPT-2 pattern is supported.
                        auto *pattern = yyjson_obj_get(node, "pattern");
                        auto *regex = pattern != nullptr ? yyjson_obj_get(pattern, "Regex") : nullptr;
                        const auto regex_text = json_str(regex);
                        static constexpr luisa::string_view kGpt2Pattern =
                            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
                        if (regex_text != kGpt2Pattern) {
                            return R::failure(unsupported("Split pattern is not the supported GPT-2 pattern", "pre_tokenizer"));
                        }
                        auto *behavior = yyjson_obj_get(node, "behavior");
                        const auto behavior_text = json_str(behavior);
                        if (!behavior_text.empty() && behavior_text != "Isolated" && behavior_text != "Removed") {
                            return R::failure(unsupported(luisa::string{"Split behavior '"} + luisa::string{behavior_text} + "'", "pre_tokenizer"));
                        }
                    } else {
                        return R::failure(unsupported(luisa::string{"pre-tokenizer '"} + luisa::string{ntype} + "'", "pre_tokenizer"));
                    }
                }
            } else if (type == "ByteLevel") {
                auto s = parse_byte_level(pre);
                if (!s.ok()) { return R::failure(std::move(s)); }
            } else {
                return R::failure(unsupported(luisa::string{"pre-tokenizer '"} + luisa::string{type} + "'", "pre_tokenizer"));
            }
        }
        if (!saw_byte_level) {
            // A byte-level BPE without a ByteLevel pre-tokenizer would encode
            // raw text differently; require it explicitly.
            return R::failure(unsupported("byte-level BPE requires a ByteLevel pre-tokenizer", "pre_tokenizer"));
        }
    }

    // --- post_processor ---
    if (auto *post = yyjson_obj_get(root, "post_processor"); post != nullptr && !yyjson_is_null(post)) {
        if (!yyjson_is_obj(post)) { return R::failure(malformed("post_processor must be an object", "post_processor")); }
        const auto type = json_str(yyjson_obj_get(post, "type"));
        if (type == "ByteLevel") {
            // No token insertion; nothing to validate beyond the type.
        } else if (type == "TemplateProcessing") {
            // Only a fixed special-token prefix/suffix around the single
            // sequence is supported (no general Jinja/template interpreter).
            const auto parse_template = [&](yyjson_val *tmpl, luisa::vector<uint32_t> &prefix,
                                            luisa::vector<uint32_t> &suffix) -> Status {
                if (!yyjson_is_arr(tmpl)) { return malformed("template must be an array", "post_processor"); }
                bool seen_sequence = false;
                bool after_sequence = false;
                yyjson_val *node;
                size_t ti, tm;
                yyjson_arr_foreach(tmpl, ti, tm, node) {
                    if (!yyjson_is_obj(node)) { return malformed("template node must be an object", "post_processor"); }
                    if (auto *special = yyjson_obj_get(node, "SpecialToken"); special != nullptr) {
                        auto *sid = yyjson_obj_get(special, "id");
                        if (!yyjson_is_str(sid)) { return malformed("SpecialToken needs a string id", "post_processor"); }
                        const auto piece = json_str(sid);
                        const auto it = tokenizer->_id_to_piece.end();
                        (void)it;
                        // Resolve the special token piece to its id.
                        luisa::optional<uint32_t> id;
                        if (tokenizer->_kind == TokenizerModelKind::byte_level_bpe) {
                            const auto vit = tokenizer->_bpe_vocab.find(luisa::string{piece});
                            if (vit != tokenizer->_bpe_vocab.end()) { id = vit->second; }
                        } else {
                            const auto vit = tokenizer->_unigram_lookup.find(luisa::string{piece});
                            if (vit != tokenizer->_unigram_lookup.end()) { id = vit->second; }
                        }
                        if (!id.has_value()) {
                            return malformed(luisa::string{"SpecialToken '"} + luisa::string{piece} + "' is not in the vocab", "post_processor");
                        }
                        if (after_sequence) {
                            suffix.emplace_back(*id);
                        } else {
                            prefix.emplace_back(*id);
                        }
                    } else if (yyjson_obj_get(node, "Sequence") != nullptr) {
                        if (seen_sequence) { return unsupported("template with more than one Sequence", "post_processor"); }
                        seen_sequence = true;
                        after_sequence = true;
                    } else {
                        return malformed("template node must be SpecialToken or Sequence", "post_processor");
                    }
                }
                if (!seen_sequence) { return malformed("template has no Sequence", "post_processor"); }
                return Status::success();
            };
            auto *single = yyjson_obj_get(post, "single");
            if (single == nullptr) { return R::failure(malformed("TemplateProcessing needs a single template", "post_processor")); }
            auto s = parse_template(single, tokenizer->_post_prefix, tokenizer->_post_suffix);
            if (!s.ok()) { return R::failure(std::move(s)); }
            // A pair template, when present, must be the same fixed shape; we
            // do not support pair encoding, so reject a differing pair.
            if (auto *pair = yyjson_obj_get(post, "pair"); pair != nullptr && !yyjson_is_null(pair)) {
                luisa::vector<uint32_t> pp, ps;
                auto ps_status = parse_template(pair, pp, ps);
                if (!ps_status.ok()) { return R::failure(std::move(ps_status)); }
                if (pp != tokenizer->_post_prefix || ps != tokenizer->_post_suffix) {
                    return R::failure(unsupported("pair template differs from single (pair encoding not supported)", "post_processor"));
                }
            }
        } else if (type == "BertProcessing" || type == "RobertaProcessing") {
            return R::failure(unsupported(luisa::string{"post-processor '"} + luisa::string{type} + "'", "post_processor"));
        } else {
            return R::failure(unsupported(luisa::string{"post-processor '"} + luisa::string{type} + "'", "post_processor"));
        }
    }

    // --- special token ids from added_tokens content conventions ---
    const auto find_special = [&](std::initializer_list<luisa::string_view> names) -> luisa::optional<uint32_t> {
        for (const auto &token : tokenizer->_added) {
            for (auto name : names) {
                if (token.content == name) { return token.id; }
            }
        }
        return luisa::nullopt;
    };
    tokenizer->_bos = find_special({"<s>", "<|begin_of_text|>", "[CLS]", "<bos>"});
    tokenizer->_eos = find_special({"</s>", "<|end_of_text|>", "[SEP]", "<eos>", "<|endoftext|>"});
    tokenizer->_pad = find_special({"<pad>", "[PAD]"});
    if (!tokenizer->_unk.has_value()) {
        tokenizer->_unk = find_special({"<unk>", "[UNK]"});
    }
    if (tokenizer->_kind == TokenizerModelKind::unigram) {
        // SentencePiece convention: BOS/EOS are the <s>/</s> pieces. The
        // heuristic only applies when the explicit add_bos_token/add_eos_token
        // flags were absent; an explicit false is always honored.
        if (!tokenizer->_unigram_bos_flag_present && tokenizer->_bos.has_value()) {
            tokenizer->_unigram_add_bos = true;
        }
        if (!tokenizer->_unigram_eos_flag_present && tokenizer->_eos.has_value()) {
            tokenizer->_unigram_add_eos = true;
        }
    }

    // Sort added tokens longest-first (stable by id for equal lengths).
    std::sort(tokenizer->_added.begin(), tokenizer->_added.end(),
              [](const AddedToken &a, const AddedToken &b) {
                  if (a.content.size() != b.content.size()) { return a.content.size() > b.content.size(); }
                  return a.id < b.id;
              });
    return R::success(std::move(tokenizer));
}

[[nodiscard]] luisa::optional<uint32_t> Tokenizer::token_to_id(luisa::string_view piece) const {
    if (_kind == TokenizerModelKind::byte_level_bpe) {
        const auto it = _bpe_vocab.find(luisa::string{piece});
        if (it != _bpe_vocab.end()) { return it->second; }
    } else {
        const auto it = _unigram_lookup.find(luisa::string{piece});
        if (it != _unigram_lookup.end()) { return it->second; }
    }
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<luisa::string> Tokenizer::id_to_token(uint32_t id) const {
    const auto it = _id_to_piece.find(id);
    if (it == _id_to_piece.end()) { return luisa::nullopt; }
    return it->second;
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

[[nodiscard]] Result<luisa::vector<uint32_t>> Tokenizer::encode_bpe(luisa::string_view text, bool allow_prefix_space) const {
    using R = Result<luisa::vector<uint32_t>>;
    // NFC is accepted as a contract, but full composition is NOT implemented.
    // Any input containing a combining mark OR an NFC non-invariant character
    // (e.g. U+2126/U+212A/U+212B/U+037E/U+0374/U+0387 singleton mappings or a
    // canonical decomposition) is rejected (fail closed) rather than silently
    // passed through unchanged. Precomposed NFC text passes.
    if (_bpe_nfc_normalize) {
        size_t pos = 0;
        while (pos < text.size()) {
            uint32_t cp = 0;
            const auto len = decode_utf8(text, pos, cp);
            if (len == 0) { return R::failure(malformed("invalid UTF-8 in encode input", "text")); }
            if (is_combining_mark_cp(cp) || is_nfc_noninvariant_cp(cp)) {
                return R::failure(unsupported("NFC normalization of non-NFC-invariant input is not implemented (provide precomposed NFC text)", "normalizer"));
            }
            pos += len;
        }
    }
    luisa::string working;
    // The structural prefix space is added only to the FIRST pre-tokenized
    // chunk of the whole input (HF ByteLevel). Added-token splitting produces
    // multiple chunks; later chunks must NOT get a spurious leading space.
    if (allow_prefix_space && _byte_level_add_prefix_space && !text.empty() && text.front() != ' ') {
        working.push_back(' ');
    }
    working.append(text.data(), text.size());

    luisa::vector<uint32_t> ids;
    const auto spans = gpt2_pretokenize(working);
    for (const auto &span : spans) {
        const auto piece_text = working.substr(span.begin, span.end - span.begin);
        // Map each byte to its GPT-2 code point piece, then BPE-merge.
        luisa::vector<luisa::string> symbols;
        symbols.reserve(piece_text.size());
        for (auto ch : piece_text) {
            luisa::string symbol;
            append_utf8(symbol, gpt2_byte_to_codepoint(static_cast<uint8_t>(ch)));
            symbols.emplace_back(std::move(symbol));
        }
        // Greedy lowest-rank merge (standard BPE).
        while (symbols.size() > 1) {
            uint32_t best_rank = UINT32_MAX;
            size_t best_index = 0;
            bool found = false;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                luisa::string key = symbols[i];
                key.push_back('\0');
                key += symbols[i + 1];
                const auto it = _bpe_ranks.find(key);
                if (it != _bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_index = i;
                    found = true;
                }
            }
            if (!found) { break; }
            symbols[best_index] += symbols[best_index + 1];
            symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_index + 1));
        }
        for (const auto &symbol : symbols) {
            const auto it = _bpe_vocab.find(symbol);
            if (it == _bpe_vocab.end()) {
                return R::failure(Status::error(StatusCode::internal_error,
                                                "BPE symbol missing from vocab (byte tokens should cover it)", "encode"));
            }
            ids.emplace_back(it->second);
        }
    }
    return R::success(std::move(ids));
}

[[nodiscard]] Result<luisa::vector<uint32_t>> Tokenizer::encode_unigram(luisa::string_view text, bool allow_prefix_space) const {
    using R = Result<luisa::vector<uint32_t>>;
    // Metaspace convention: inner spaces become U+2581; a leading structural
    // metaspace is prepended only when add_prefix_space is set AND this is the
    // first chunk of the whole input (added-token splitting yields later chunks
    // that must not receive a spurious prefix).
    luisa::string working;
    if (allow_prefix_space && _unigram_add_prefix_space) { append_utf8(working, 0x2581u); }
    for (auto ch : text) {
        if (ch == ' ') {
            append_utf8(working, 0x2581u);
        } else {
            working.push_back(ch);
        }
    }
    // Viterbi over code-point boundaries.
    struct Cp {
        size_t begin;
        size_t end;
    };
    luisa::vector<Cp> cps;
    size_t pos = 0;
    while (pos < working.size()) {
        uint32_t cp = 0;
        const auto len = decode_utf8(working, pos, cp);
        if (len == 0) { return R::failure(malformed("invalid UTF-8 in encode input", "text")); }
        cps.emplace_back(Cp{pos, pos + len});
        pos += len;
    }
    const auto n = cps.size();
    // best[i] = best score to cover cps[0..i); back[i] = (start_cp, token_id).
    luisa::vector<float> best(n + 1, -INFINITY);
    luisa::vector<std::pair<size_t, uint32_t>> back(n + 1, {0, 0});
    best[0] = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(best[i])) { continue; }
        // Try every piece starting at cp i. <0xHH> byte-fallback pieces are
        // EXCLUDED from literal matching: they are only reachable through the
        // byte-fallback path below, so literal text like "<0x41>" is never
        // silently turned into a single byte.
        for (size_t j = i + 1; j <= n; j++) {
            const auto piece = working.substr(cps[i].begin, cps[j - 1].end - cps[i].begin);
            const auto it = _unigram_lookup.find(luisa::string{piece});
            if (it == _unigram_lookup.end()) { continue; }
            const auto id = it->second;
            if (_unigram_byte_token_ids.count(id) != 0) { continue; }
            const auto score = best[i] + _unigram_pieces[id].score;
            if (score > best[j]) {
                best[j] = score;
                back[j] = {i, id};
            }
        }
        // Single-character fallback: byte fallback or unk.
        const auto single = working.substr(cps[i].begin, cps[i].end - cps[i].begin);
        const auto single_it = _unigram_lookup.find(luisa::string{single});
        const bool single_is_byte_token =
            single_it != _unigram_lookup.end() && _unigram_byte_token_ids.count(single_it->second) != 0;
        if (single_it == _unigram_lookup.end() || single_is_byte_token) {
            if (_unigram_byte_fallback) {
                // Each raw byte of the character becomes a <0xHH> token.
                bool ok = true;
                float score = best[i];
                for (size_t b = cps[i].begin; b < cps[i].end; b++) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned int>(static_cast<uint8_t>(working[b])));
                    const auto bit = _unigram_lookup.find(luisa::string{buf});
                    if (bit == _unigram_lookup.end()) {
                        ok = false;
                        break;
                    }
                    score += _unigram_pieces[bit->second].score;
                }
                if (ok && score > best[i + 1]) {
                    // Sentinel: re-expand the byte chain during backtracking.
                    best[i + 1] = score;
                    back[i + 1] = {i, UINT32_MAX};
                }
            } else if (_unigram_has_unk) {
                const auto score = best[i] + _unigram_pieces[_unigram_unk_id].score;
                if (score > best[i + 1]) {
                    best[i + 1] = score;
                    back[i + 1] = {i, _unigram_unk_id};
                }
            }
        }
    }
    if (!std::isfinite(best[n])) {
        return R::failure(Status::error(StatusCode::invalid_argument,
                                        "text cannot be tokenized by the unigram model (no byte fallback / unk)", "text"));
    }
    // Reconstruct the path. Backtracking walks segments from the END to the
    // START, so each segment's ids are appended to a per-segment buffer and the
    // SEGMENTS are reversed at the end. A multi-byte byte-fallback chain must
    // keep its byte order (reversing the flat id list would reverse the bytes
    // inside each character).
    luisa::vector<luisa::vector<uint32_t>> segments;
    size_t i = n;
    while (i > 0) {
        const auto [start, id] = back[i];
        luisa::vector<uint32_t> segment;
        if (id == UINT32_MAX) {
            // Byte fallback for cps[start], in forward byte order.
            for (size_t b = cps[start].begin; b < cps[start].end; b++) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned int>(static_cast<uint8_t>(working[b])));
                const auto bit = _unigram_lookup.find(luisa::string{buf});
                segment.emplace_back(bit->second);
            }
        } else {
            segment.emplace_back(id);
        }
        segments.emplace_back(std::move(segment));
        i = start;
    }
    luisa::vector<uint32_t> ids;
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
        for (auto id : *it) { ids.emplace_back(id); }
    }
    return R::success(std::move(ids));
}

[[nodiscard]] Result<luisa::vector<uint32_t>> Tokenizer::encode_added_aware(
    luisa::string_view text, bool add_special_tokens) const {
    using R = Result<luisa::vector<uint32_t>>;
    if (!ocr_is_valid_utf8(text)) {
        return R::failure(Status::error(StatusCode::invalid_argument,
                                        "encode input is not valid UTF-8 (no lossy replacement)", "text"));
    }
    luisa::vector<uint32_t> ids;
    // Split around added/special tokens (longest match first).
    size_t pos = 0;
    luisa::string normal;
    bool first_chunk = true;
    const auto flush_normal = [&]() -> Status {
        if (normal.empty()) { return Status::success(); }
        // Only the first chunk may receive the structural prefix space.
        auto chunk = _kind == TokenizerModelKind::byte_level_bpe
                         ? encode_bpe(normal, first_chunk)
                         : encode_unigram(normal, first_chunk);
        first_chunk = false;
        if (!chunk.ok()) { return chunk.status; }
        for (auto id : *chunk.value) { ids.emplace_back(id); }
        normal.clear();
        return Status::success();
    };
    while (pos < text.size()) {
        bool matched = false;
        for (const auto &token : _added) {
            const auto &content = token.content;
            if (content.size() > text.size() - pos) { continue; }
            if (text.compare(pos, content.size(), content) != 0) { continue; }
            if (token.single_word) {
                // Boundary check: not embedded inside a word.
                bool left_ok = pos == 0;
                if (!left_ok) {
                    uint32_t cp = 0;
                    // Decode the code point immediately before pos.
                    size_t back = pos - 1;
                    while (back > 0 && (static_cast<uint8_t>(text[back]) & 0xc0u) == 0x80u) { back -= 1; }
                    const auto len = decode_utf8(text, back, cp);
                    left_ok = len == 0 || !is_letter_cp(cp);
                }
                bool right_ok = pos + content.size() >= text.size();
                if (!right_ok) {
                    uint32_t cp = 0;
                    const auto len = decode_utf8(text, pos + content.size(), cp);
                    right_ok = len == 0 || !is_letter_cp(cp);
                }
                if (!left_ok || !right_ok) { continue; }
            }
            // lstrip trims trailing whitespace from the pending normal text
            // BEFORE it is encoded.
            if (token.lstrip) {
                while (!normal.empty() && (normal.back() == ' ' || normal.back() == '\t' || normal.back() == '\n' || normal.back() == '\r')) {
                    normal.pop_back();
                }
            }
            auto s = flush_normal();
            if (!s.ok()) { return R::failure(std::move(s)); }
            ids.emplace_back(token.id);
            pos += content.size();
            if (token.rstrip) {
                while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')) {
                    pos += 1;
                }
            }
            matched = true;
            break;
        }
        if (!matched) {
            normal.push_back(text[pos]);
            pos += 1;
        }
    }
    auto s = flush_normal();
    if (!s.ok()) { return R::failure(std::move(s)); }

    if (add_special_tokens) {
        if (_kind == TokenizerModelKind::byte_level_bpe) {
            luisa::vector<uint32_t> wrapped;
            for (auto id : _post_prefix) { wrapped.emplace_back(id); }
            for (auto id : ids) { wrapped.emplace_back(id); }
            for (auto id : _post_suffix) { wrapped.emplace_back(id); }
            ids = std::move(wrapped);
        } else {
            luisa::vector<uint32_t> wrapped;
            if (_unigram_add_bos && _bos.has_value()) { wrapped.emplace_back(*_bos); }
            for (auto id : ids) { wrapped.emplace_back(id); }
            if (_unigram_add_eos && _eos.has_value()) { wrapped.emplace_back(*_eos); }
            ids = std::move(wrapped);
        }
    }
    return R::success(std::move(ids));
}

[[nodiscard]] Result<luisa::vector<uint32_t>> Tokenizer::encode(
    luisa::string_view text, bool add_special_tokens) const {
    return encode_added_aware(text, add_special_tokens);
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

// Shared decoding core. Concatenates the (non-suppressed) pieces and converts
// them to text. For unigram, exactly ONE leading structural U+2581 is stripped
// from the joined piece text (official SentencePiece/HF Metaspace semantics:
// decode([▁hello])='hello', [▁hello,▁world]='hello world', [▁,▁hello]=' hello').
// `strip_leading_metaspace` must be true only for a WHOLE sequence; the
// incremental detokenizer strips once at stream level instead (never per
// token), preserving batch/incremental equivalence.
[[nodiscard]] Result<luisa::string> Tokenizer::decode_impl(
    luisa::span<const uint32_t> ids, bool skip_special_tokens,
    bool allow_partial_utf8, bool strip_leading_metaspace) const {
    using R = Result<luisa::string>;
    luisa::string out;
    if (_kind == TokenizerModelKind::byte_level_bpe) {
        // Concatenate the byte pieces, then map GPT-2 code points back to bytes.
        luisa::string raw;
        for (auto id : ids) {
            const auto special_it = _special_ids.find(id);
            if (special_it != _special_ids.end() && special_it->second && skip_special_tokens) { continue; }
            const auto piece = id_to_token(id);
            if (!piece.has_value()) {
                return R::failure(Status::error(StatusCode::invalid_argument, "unknown token id in decode", "ids"));
            }
            raw += *piece;
        }
        // Map each code point back to its original byte.
        size_t pos = 0;
        while (pos < raw.size()) {
            uint32_t cp = 0;
            const auto len = decode_utf8(raw, pos, cp);
            if (len == 0) {
                return R::failure(Status::error(StatusCode::invalid_argument, "BPE token pieces are not valid UTF-8", "decode"));
            }
            if (!_byte_dec_present[cp]) {
                return R::failure(Status::error(StatusCode::invalid_argument, "BPE piece code point outside the byte mapping", "decode"));
            }
            out.push_back(static_cast<char>(_byte_decoder[cp]));
            pos += len;
        }
        if (!allow_partial_utf8 && !ocr_is_valid_utf8(out)) {
            return R::failure(Status::error(StatusCode::invalid_argument,
                                            "decoded bytes are not valid UTF-8 (partial multi-byte sequence)", "decode"));
        }
    } else {
        // Unigram: concatenate pieces, reverse Metaspace, resolve byte tokens.
        luisa::string joined;
        for (auto id : ids) {
            const auto special_it = _special_ids.find(id);
            if (special_it != _special_ids.end() && special_it->second && skip_special_tokens) { continue; }
            const auto piece = id_to_token(id);
            if (!piece.has_value()) {
                return R::failure(Status::error(StatusCode::invalid_argument, "unknown token id in decode", "ids"));
            }
            joined += *piece;
        }
        // Official SentencePiece/HF Metaspace: strip exactly ONE leading
        // structural U+2581 from the whole joined sequence. A second leading
        // metaspace (e.g. [▁,▁hello]) decodes to a real space.
        if (strip_leading_metaspace && joined.compare(0, 3, "\xe2\x96\x81") == 0) {
            joined.erase(0, 3);
        }
        size_t pos = 0;
        while (pos < joined.size()) {
            // Byte fallback token <0xHH> (6 characters: '<','0','x',H,H,'>').
            if (joined.compare(pos, 3, "<0x") == 0 && pos + 6 <= joined.size() && joined[pos + 5] == '>') {
                const auto hex = joined.substr(pos + 3, 2);
                uint32_t value = 0;
                bool ok = true;
                for (auto c : hex) {
                    value <<= 4u;
                    if (c >= '0' && c <= '9') { value |= static_cast<uint32_t>(c - '0'); }
                    else if (c >= 'A' && c <= 'F') { value |= static_cast<uint32_t>(c - 'A' + 10); }
                    else if (c >= 'a' && c <= 'f') { value |= static_cast<uint32_t>(c - 'a' + 10); }
                    else { ok = false; break; }
                }
                if (ok) {
                    out.push_back(static_cast<char>(value));
                    pos += 6;
                    continue;
                }
            }
            uint32_t cp = 0;
            const auto len = decode_utf8(joined, pos, cp);
            if (len == 0) {
                return R::failure(Status::error(StatusCode::invalid_argument, "unigram pieces are not valid UTF-8", "decode"));
            }
            if (cp == 0x2581u) {
                out.push_back(' ');
            } else {
                out.append(joined.data() + pos, len);
            }
            pos += len;
        }
        if (!allow_partial_utf8 && !ocr_is_valid_utf8(out)) {
            return R::failure(Status::error(StatusCode::invalid_argument,
                                            "decoded bytes are not valid UTF-8 (partial multi-byte sequence)", "decode"));
        }
    }
    return R::success(std::move(out));
}

[[nodiscard]] Result<luisa::string> Tokenizer::decode(
    luisa::span<const uint32_t> ids, bool skip_special_tokens, bool allow_partial_utf8) const {
    return decode_impl(ids, skip_special_tokens, allow_partial_utf8, /*strip_leading_metaspace=*/true);
}

[[nodiscard]] Result<luisa::string> Tokenizer::decode_partial(
    luisa::span<const uint32_t> ids, bool skip_special_tokens) const {
    // Incremental subsequence decode: allow a trailing partial UTF-8 sequence
    // so it can be buffered across token boundaries. The leading structural
    // metaspace is NOT stripped here — the detokenizer strips exactly once at
    // stream start (never per token), matching whole-sequence decode.
    return decode_impl(ids, skip_special_tokens, /*allow_partial_utf8=*/true, /*strip_leading_metaspace=*/false);
}

}// namespace luisa::ocr
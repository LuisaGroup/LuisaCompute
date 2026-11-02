// Chunker implementation (report.md §2). See chunker.h for the contract.

#include "chunker.h"

#include <algorithm>
#include <cctype>

#include <luisa/core/stl/algorithm.h>

namespace luisa::embed {
namespace {

// A segmentation unit: a half-open byte range [begin, end) of the document.
struct Unit {
    size_t begin = 0u;
    size_t end = 0u;
};

// Advances past a whole UTF-8 code point starting at `i` (never lands inside a
// multi-byte sequence; invalid bytes advance by one).
[[nodiscard]] size_t advance_utf8(luisa::string_view text, size_t i) noexcept {
    const auto first = static_cast<uint8_t>(text[i]);
    size_t length = 1u;
    if (first >= 0xc2u && first <= 0xdfu) { length = 2u; }
    else if (first >= 0xe0u && first <= 0xefu) { length = 3u; }
    else if (first >= 0xf0u && first <= 0xf4u) { length = 4u; }
    return i + ((i + length <= text.size()) ? length : 1u);
}

[[nodiscard]] bool ascii_space(const char c) noexcept {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

// Sentence-terminal punctuation (ASCII + the common full-width CJK marks),
// decoded as code points. `；` and `…` are terminals in CJK typography (and a
// Chinese document uses `；` exactly where an English one uses '.'), so they are
// listed here; `sentence_end_is_cjk` below decides that they need no following
// whitespace.
[[nodiscard]] bool sentence_end(uint32_t cp) noexcept {
    switch (cp) {
        case 0x2eu: // '.'
        case 0x21u: // '!'
        case 0x3fu: // '?'
        case 0x3002u:// '。'
        case 0xff01u:// '！'
        case 0xff1fu:// '？'
        case 0xff1bu:// '；'
        case 0x2026u:// '…'
            return true;
        default: return false;
    }
}

// True for the terminals a CJK sentence ends on. Chinese and Japanese text runs
// sentences together with NO space after the terminal ("第一句。第二句。"), so
// these end a sentence unconditionally; the ASCII trio still needs following
// whitespace, which is what keeps "3.14" and "e.g." in one piece.
[[nodiscard]] bool sentence_end_is_cjk(uint32_t cp) noexcept {
    switch (cp) {
        case 0x3002u:// '。'
        case 0xff01u:// '！'
        case 0xff1fu:// '？'
        case 0xff1bu:// '；'
        case 0x2026u:// '…'
            return true;
        default: return false;
    }
}

// Closing punctuation that belongs to the preceding sentence.
[[nodiscard]] bool sentence_close(uint32_t cp) noexcept {
    switch (cp) {
        case 0x22u: // '"'
        case 0x27u: // '\''
        case 0x29u: // ')'
        case 0x5du: // ']'
        case 0x7du: // '}'
        case 0x300du:// '」'
        case 0x300fu:// '』'
        case 0x201du:// '"'
        case 0x2019u:// '\''
            return true;
        default: return false;
    }
}

[[nodiscard]] uint32_t decode_cp(luisa::string_view text, size_t i, size_t *next) noexcept {
    const auto first = static_cast<uint8_t>(text[i]);
    if (first <= 0x7fu) { *next = i + 1u; return first; }
    uint32_t cp = 0u;
    size_t count = 0u;
    if (first >= 0xc2u && first <= 0xdfu) { cp = first & 0x1fu; count = 1u; }
    else if (first >= 0xe0u && first <= 0xefu) { cp = first & 0x0fu; count = 2u; }
    else if (first >= 0xf0u && first <= 0xf4u) { cp = first & 0x07u; count = 3u; }
    else { *next = i + 1u; return 0xfffeu; }
    for (size_t j = 1u; j <= count; j++) {
        const auto next_byte = static_cast<uint8_t>(text[i + j]);
        if ((next_byte & 0xc0u) != 0x80u) { *next = i + 1u; return 0xfffeu; }
        cp = (cp << 6u) | (next_byte & 0x3fu);
    }
    *next = i + count + 1u;
    return cp;
}

// Whitespace at byte `i`, decoded as a code point: ASCII whitespace plus U+3000
// IDEOGRAPHIC SPACE, which CJK text uses where ASCII text uses U+0020 (a Chinese
// document is routinely written with full-width spaces, and token mode must not
// treat "数据库　连接池" as one word). Returns the byte index just past the
// whitespace, or `i` itself when `text[i]` starts no whitespace code point.
[[nodiscard]] size_t skip_space(luisa::string_view text, size_t i) noexcept {
    if (i >= text.size()) { return i; }
    if (ascii_space(text[i])) { return i + 1u; }
    size_t next = i;
    const auto cp = decode_cp(text, i, &next);
    return cp == 0x3000u ? next : i;
}

// Units split at blank lines: a run of TWO OR MORE newlines (whitespace
// allowed between them) terminates the current paragraph; a single newline
// keeps the paragraph running.
[[nodiscard]] luisa::vector<Unit> split_paragraphs(luisa::string_view text) {
    luisa::vector<Unit> units;
    size_t unit_begin = 0u;
    size_t i = 0u;
    while (i < text.size()) {
        if (text[i] == '\n' || text[i] == '\r') {
            const auto break_at = i;
            // Consume the whitespace run and count the newlines inside it.
            size_t newlines = 0u;
            auto j = i;
            while (j < text.size() && ascii_space(text[j])) {
                if (text[j] == '\n' || text[j] == '\r') { newlines++; }
                j++;
            }
            if (newlines >= 2u) {
                if (break_at > unit_begin) {
                    units.push_back(Unit{unit_begin, break_at});
                }
                unit_begin = j;// skip the whole blank region
            }
            i = j;
            continue;
        }
        i = advance_utf8(text, i);
    }
    if (text.size() > unit_begin) {
        units.push_back(Unit{unit_begin, text.size()});
    }
    return units;
}

// Units split after sentence-ending punctuation (plus closing quotes) when the
// terminator is followed by whitespace or the end of the text.
[[nodiscard]] luisa::vector<Unit> split_sentences(luisa::string_view text) {
    luisa::vector<Unit> units;
    size_t unit_begin = 0u;
    size_t i = 0u;
    while (i < text.size()) {
        size_t next = i;
        const auto cp = decode_cp(text, i, &next);
        if (sentence_end(cp)) {
            // Attach any closing punctuation to this sentence.
            auto end = next;
            while (end < text.size()) {
                size_t close_next = end;
                const auto close_cp = decode_cp(text, end, &close_next);
                if (!sentence_close(close_cp)) { break; }
                end = close_next;
            }
            // A CJK terminal ends the sentence by itself; an ASCII one only when a
            // whitespace (ASCII or U+3000) or the end of the text follows it.
            if (end >= text.size() || sentence_end_is_cjk(cp) || skip_space(text, end) > end) {
                if (end > unit_begin) {
                    units.push_back(Unit{unit_begin, end});
                }
                unit_begin = end;
            }
            i = end;
            continue;
        }
        i = next;
    }
    if (text.size() > unit_begin) {
        units.push_back(Unit{unit_begin, text.size()});
    }
    return units;
}

// Units split at whitespace runs (ASCII plus U+3000): the segmentation the
// bge-m3 Metaspace pre-tokenizer applies (one U+2581 per ASCII space, split at
// each U+2581). U+3000 is treated as a word separator too, because a CJK
// document separates words with it far more often than with U+0020.
[[nodiscard]] luisa::vector<Unit> split_words(luisa::string_view text) {
    luisa::vector<Unit> units;
    size_t i = 0u;
    while (i < text.size()) {
        size_t after_space = skip_space(text, i);
        if (after_space > i) {
            i = after_space;
            continue;
        }
        const auto begin = i;
        while (i < text.size()) {
            after_space = skip_space(text, i);
            if (after_space > i) { break; }
            i = advance_utf8(text, i);
        }
        units.push_back(Unit{begin, i});
    }
    return units;
}

// A unit longer than `chunk_size` is split at code-point boundaries into pieces
// of at most `chunk_size` bytes, so one oversized unit cannot become one
// oversized chunk. Needed because CJK text has no ASCII word boundaries to fall
// back on: a 512-byte unit is only ~170 Chinese characters, and a single
// paragraph often exceeds it.
//
// The pieces step by `chunk_size - overlap`, not by `chunk_size`, so the overlap
// tail survives INSIDE a split unit: once the unit is in pieces, each piece is
// already one whole chunk and the unit-level overlap step in pack_units() cannot
// re-include part of a piece (a piece is never smaller than the overlap budget).
// Piece boundaries never land inside a multi-byte code point.
[[nodiscard]] luisa::vector<Unit> explode_oversized_units(
    luisa::string_view text,
    const luisa::vector<Unit> &units,
    uint32_t chunk_size,
    uint32_t chunk_overlap) {
    luisa::vector<Unit> out;
    out.reserve(units.size());
    const auto limit = static_cast<size_t>(chunk_size);
    // A stride of at least one byte keeps the loop finite when the caller asks
    // for an overlap as large as the chunk.
    const auto stride = static_cast<size_t>(chunk_overlap) < limit
                            ? limit - static_cast<size_t>(chunk_overlap)
                            : limit;
    for (const auto &unit : units) {
        if (unit.end <= unit.begin || unit.end - unit.begin <= limit) {
            out.push_back(unit);
            continue;
        }
        size_t begin = unit.begin;
        while (begin < unit.end) {
            size_t end = std::min(begin + limit, unit.end);
            // Back up to a code-point boundary (never before `begin`).
            while (end < unit.end && end > begin &&
                   (static_cast<uint8_t>(text[end]) & 0xc0u) == 0x80u) {
                end--;
            }
            if (end == begin) {// chunk_size smaller than one code point: take it whole
                end = std::min(advance_utf8(text, begin), unit.end);
            }
            out.push_back(Unit{begin, end});
            if (end >= unit.end) { break; }
            // The next piece starts inside this one by the overlap, never after
            // `end` (that would drop text) and never at or before `begin` (that
            // would not advance). The start is snapped FORWARD to a code-point
            // boundary, so a chunk can never begin inside a multi-byte sequence.
            size_t next_begin = begin + stride;
            while (next_begin < unit.end && next_begin > begin &&
                   (static_cast<uint8_t>(text[next_begin]) & 0xc0u) == 0x80u) {
                next_begin++;
            }
            if (next_begin > end) { next_begin = end; }
            if (next_begin <= begin) { next_begin = end; }
            begin = next_begin;
        }
    }
    return out;
}

// Greedy packing: accumulate units until the next one would exceed
// `chunk_size`, then start the next chunk by re-including trailing units of
// the previous chunk up to `overlap` bytes. Every unit is at most `chunk_size`
// bytes (explode_oversized_units ran first), so no chunk is oversized unless
// `chunk_overlap >= chunk_size`. `begin`/`end` are validated UTF-8 boundaries by
// construction (unit boundaries).
[[nodiscard]] luisa::vector<Chunk> pack_units(
    luisa::string_view document,
    const luisa::vector<Unit> &units,
    uint32_t chunk_size,
    uint32_t chunk_overlap) {
    luisa::vector<Chunk> chunks;
    if (units.empty()) {
        chunks.push_back(Chunk{0u, 0u});
        return chunks;
    }
    const auto overlap = static_cast<size_t>(chunk_overlap < chunk_size ? chunk_overlap : chunk_size);
    size_t cursor = 0u;// next unit index not yet covered by a chunk
    while (cursor < units.size()) {
        const auto chunk_begin = units[cursor].begin;
        size_t chunk_end = units[cursor].end;
        size_t next = cursor + 1u;
        while (next < units.size() &&
               units[next].end - chunk_begin <= chunk_size) {
            chunk_end = units[next].end;
            next++;
        }
        chunks.push_back(Chunk{chunk_begin, chunk_end - chunk_begin});
        if (next >= units.size()) { break; }
        // Overlap: step back over the tail of this chunk while the previous
        // unit still fits inside the overlap budget and does not repeat the
        // whole chunk.
        size_t overlap_begin = next;
        while (overlap_begin > cursor &&
               chunk_end - units[overlap_begin - 1u].begin <= overlap) {
            overlap_begin--;
        }
        if (overlap_begin <= cursor) {
            // No unit fits the overlap budget (or the chunk was a single
            // oversized unit): the next chunk simply starts where this one
            // ended. `next > cursor` always holds, so progress is guaranteed.
            overlap_begin = next;
        }
        cursor = overlap_begin;
    }
    return chunks;
}

}// namespace

luisa::vector<Chunk> chunk_text(luisa::string_view document, const ChunkOptions &options) {
    if (options.chunk_size == 0u || options.by == ChunkOptions::By::off ||
        document.empty()) {
        return {Chunk{0u, static_cast<uint64_t>(document.size())}};
    }
    luisa::vector<Unit> units;
    switch (options.by) {
        case ChunkOptions::By::off: break;
        case ChunkOptions::By::paragraph:
            units = split_paragraphs(document);
            break;
        case ChunkOptions::By::sentence:
            units = split_sentences(document);
            break;
        case ChunkOptions::By::token:
            units = split_words(document);
            break;
    }
    if (units.empty()) {
        // Whitespace-only document: one empty chunk keeps counts well defined.
        return {Chunk{0u, 0u}};
    }

    return pack_units(document,
                      explode_oversized_units(document, units, options.chunk_size,
                                              options.chunk_overlap),
                      options.chunk_size, options.chunk_overlap);
}

}// namespace luisa::embed

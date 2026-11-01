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
// decoded as code points.
[[nodiscard]] bool sentence_end(uint32_t cp) noexcept {
    switch (cp) {
        case 0x2eu: // '.'
        case 0x21u: // '!'
        case 0x3fu: // '?'
        case 0x3002u:// '。'
        case 0xff01u:// '！'
        case 0xff1fu:// '？'
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
            if (end >= text.size() || ascii_space(text[end])) {
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

// Units split at ASCII whitespace runs: the segmentation the bge-m3 Metaspace
// pre-tokenizer applies (one U+2581 per ASCII space, split at each U+2581).
[[nodiscard]] luisa::vector<Unit> split_words(luisa::string_view text) {
    luisa::vector<Unit> units;
    size_t i = 0u;
    while (i < text.size()) {
        while (i < text.size() && ascii_space(text[i])) { i++; }
        if (i >= text.size()) { break; }
        const auto begin = i;
        while (i < text.size() && !ascii_space(text[i])) { i = advance_utf8(text, i); }
        units.push_back(Unit{begin, i});
    }
    return units;
}

// Greedy packing: accumulate units until the next one would exceed
// `chunk_size`, then start the next chunk by re-including trailing units of
// the previous chunk up to `overlap` bytes. An individual unit larger than
// `chunk_size` becomes its own (oversized) chunk. `begin`/`end` are validated
// UTF-8 boundaries by construction (unit boundaries).
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
    return pack_units(document, units, options.chunk_size, options.chunk_overlap);
}

}// namespace luisa::embed

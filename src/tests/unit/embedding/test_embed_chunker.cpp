// Test for embedding/embed/chunker.h — the RAG chunker contract (report.md §2).
// Host-only test (LUISA_EMBED_HOST_ONLY=1): no device, no backend, no model.
//
// What it pins, one test per behaviour:
// - CJK sentence mode: "第一句。第二句。" is TWO units. Chinese runs sentences
//   together with no space after `。`, so the ASCII "terminator + whitespace" rule
//   returned ONE unit for the whole paragraph (the defect this test was written
//   for); `；` and `…` are sentence terminals too, and a closing quote (`」`)
//   belongs to the sentence it follows;
// - ASCII sentence mode keeps its whitespace requirement, so "3.14" and "e.g."
//   stay inside their sentence;
// - U+3000 IDEOGRAPHIC SPACE is a word separator in token mode (a CJK document
//   separates words with it where English text uses U+0020);
// - a unit bigger than chunk_size is hard-split at code-point boundaries into
//   chunk_size-byte pieces that step by (chunk_size - overlap), so one long CJK
//   paragraph still yields bounded chunks WITH the overlap tail and no boundary
//   lands inside a multi-byte code point;
// - paragraph mode (the default), whole-file mode and the blank-document case are
//   unchanged, and every chunk's [byte_offset, byte_length) is a view of the
//   source (no re-encoding, no normalization).
//
// The chunk sizes in the tests are deliberately small (12..21 bytes): the units
// are packed greedily, so a size that fits two units only demonstrates packing.
// Run: xmake run test_embed_chunker
#include "ut/ut.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "embedding/embed/chunker.h"

using namespace luisa;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A chunk's text as an owned std::string (the test compares it with literals).
std::string text_of(const Chunk &chunk, std::string_view document) {
    return std::string{document.substr(static_cast<size_t>(chunk.byte_offset),
                                      static_cast<size_t>(chunk.byte_length))};
}

// True when `offset` is a UTF-8 code-point start (so a range whose two ends are
// code-point starts can be sliced without splitting a scalar).
bool is_boundary(std::string_view document, size_t offset) {
    if (offset >= document.size()) { return true; }
    return (static_cast<unsigned char>(document[offset]) & 0xc0u) != 0x80u;
}

// Every chunk is inside the document, both ends are code-point boundaries, the
// chunks never rewind, and no chunk is empty while others carry text. Gaps ARE
// legitimate — the whitespace a unit ends on is dropped (a paragraph break, the
// U+3000 between two words) — and so is the overlap of a chunked run.
bool chunks_are_sane(std::string_view document, const luisa::vector<Chunk> &chunks) {
    if (chunks.empty()) { return false; }
    std::uint64_t previous_begin = 0u;
    for (size_t i = 0; i < chunks.size(); i++) {
        const auto &chunk = chunks[i];
        if (chunk.byte_offset + chunk.byte_length > document.size()) { return false; }
        if (!is_boundary(document, static_cast<size_t>(chunk.byte_offset))) { return false; }
        if (!is_boundary(document, static_cast<size_t>(chunk.byte_offset + chunk.byte_length))) { return false; }
        if (i > 0u && chunk.byte_offset < previous_begin) { return false; }
        if (chunks.size() > 1u && chunk.byte_length == 0u) { return false; }
        previous_begin = chunk.byte_offset;
    }
    return true;
}

ChunkOptions by(ChunkOptions::By mode, std::uint32_t size, std::uint32_t overlap = 0u) {
    ChunkOptions options;
    options.by = mode;
    options.chunk_size = size;
    options.chunk_overlap = overlap;
    return options;
}

// Asserts the exact chunk sequence: count, byte ranges and texts.
bool chunks_equal(std::string_view document, const luisa::vector<Chunk> &chunks,
                  const std::vector<std::pair<std::uint64_t, std::uint64_t>> &expected) {
    if (chunks.size() != expected.size()) { return false; }
    for (size_t i = 0; i < chunks.size(); i++) {
        if (chunks[i].byte_offset != expected[i].first) { return false; }
        if (chunks[i].byte_length != expected[i].second) { return false; }
        if (text_of(chunks[i], document).size() != expected[i].second) { return false; }
    }
    return true;
}

void reg_cjk_sentence_mode() {
    "chunker_cjk_sentence_split"_test = [] {
        // 第一句。第二句。 — 24 bytes of CJK and not one ASCII whitespace byte.
        const std::string document = "第一句。第二句。";
        const auto chunks = chunk_text(document, by(ChunkOptions::By::sentence, 12u));
        expect(chunks.size() == 2_u);
        if (chunks.size() == 2u) {
            expect(text_of(chunks[0], document) == "第一句。");
            expect(text_of(chunks[1], document) == "第二句。");
        }
        expect(chunks_equal(document, chunks, {{0u, 12u}, {12u, 12u}}));
        expect(chunks_are_sane(document, chunks));
    };

    "chunker_cjk_terminals_and_closing_quotes"_test = [] {
        const std::string document = "你好！真的吗？";
        const auto chunks = chunk_text(document, by(ChunkOptions::By::sentence, 12u));
        expect(chunks.size() == 2_u);
        if (chunks.size() == 2u) {
            expect(text_of(chunks[0], document) == "你好！");
            expect(text_of(chunks[1], document) == "真的吗？");
        }

        // 好的；| 然后……| 「结语。」 — the `；` and each `…` are terminals, and the
        // closing `」` stays with the sentence it closes.
        const std::string punctuation = "好的；然后……「结语。」";
        const auto more = chunk_text(punctuation, by(ChunkOptions::By::sentence, 15u));
        expect(more.size() == 3_u);
        if (more.size() == 3u) {
            expect(text_of(more[0], punctuation) == "好的；");
            expect(text_of(more[1], punctuation) == "然后……");
            expect(text_of(more[2], punctuation) == "「结语。」");
        }
        expect(chunks_are_sane(punctuation, more));
    };

    "chunker_ascii_sentence_needs_whitespace"_test = [] {
        // "3.14" must not split: the dot is followed by a digit, not by whitespace.
        const std::string document = "Pi is 3.14 today. Next one!";
        const auto chunks = chunk_text(document, by(ChunkOptions::By::sentence, 18u));
        expect(chunks.size() == 2_u);
        if (chunks.size() == 2u) {
            expect(text_of(chunks[0], document) == "Pi is 3.14 today.");
            expect(text_of(chunks[1], document) == " Next one!");
        }
        expect(chunks_are_sane(document, chunks));
    };
}

void reg_cjk_whitespace_and_oversize() {
    "chunker_u3000_is_a_word_separator"_test = [] {
        // 数据库　连接池 — 9 + 3 (U+3000) + 9 bytes.
        const std::string document = "数据库　连接池";
        const auto chunks = chunk_text(document, by(ChunkOptions::By::token, 9u));
        expect(chunks.size() == 2_u);
        if (chunks.size() == 2u) {
            expect(text_of(chunks[0], document) == "数据库");
            expect(text_of(chunks[1], document) == "连接池");
        }
        expect(chunks_equal(document, chunks, {{0u, 9u}, {12u, 9u}}));
        expect(chunks_are_sane(document, chunks));
    };

    "chunker_oversized_unit_hard_splits_with_overlap"_test = [] {
        // One CJK paragraph, 600 characters = 1800 bytes, no blank line and no
        // sentence terminal: nothing to split on but code points.
        std::string document;
        for (int i = 0; i < 600; i++) { document += "字"; }
        const auto chunks = chunk_text(document, by(ChunkOptions::By::paragraph, 512u, 64u));
        // 512-byte pieces stepping by 512 - 64 = 448, snapped to the 3-byte grid:
        // [0,510) [450,510) [900,510) [1350,450).
        expect(chunks.size() == 4_u);
        expect(chunks_equal(document, chunks,
                            {{0u, 510u}, {450u, 510u}, {900u, 510u}, {1350u, 450u}}));
        for (const auto &chunk : chunks) {
            expect(chunk.byte_length <= 512u);
            // 3-byte characters: every offset and length is a multiple of 3.
            expect(chunk.byte_offset % 3u == 0u);
            expect(chunk.byte_length % 3u == 0u);
        }
        expect(chunks_are_sane(document, chunks));
        if (chunks.size() == 4u) {
            // The overlap tail repeats: the next chunk starts before the previous ended.
            expect(chunks[1].byte_offset < chunks[0].byte_offset + chunks[0].byte_length);
            // Nothing is lost: the last chunk ends at the end of the document.
            expect(chunks.back().byte_offset + chunks.back().byte_length == document.size());
        }
    };

    "chunker_cjk_paragraph_mode_and_whole_file"_test = [] {
        const std::string document = "第一段第一句。\n\n第二段文字。";
        const auto chunks = chunk_text(document, by(ChunkOptions::By::paragraph, 21u));
        expect(chunks.size() == 2_u);
        if (chunks.size() == 2u) {
            expect(text_of(chunks[0], document) == "第一段第一句。");
            expect(text_of(chunks[1], document) == "第二段文字。");
        }
        expect(chunks_are_sane(document, chunks));

        const auto whole = chunk_text(document, ChunkOptions::whole_file());
        expect(whole.size() == 1_u);
        if (whole.size() == 1u) {
            expect(whole[0].byte_offset == 0u);
            expect(whole[0].byte_length == document.size());
        }

        // An EMPTY document is the one zero-length chunk; a whitespace-only one is a
        // single chunk covering it (there is nothing to split on).
        const std::string blank = "   \n\t ";
        const auto empties = chunk_text(blank, by(ChunkOptions::By::paragraph, 512u));
        expect(empties.size() == 1_u);
        if (empties.size() == 1u) {
            expect(empties[0].byte_offset == 0u);
            expect(empties[0].byte_length == blank.size());
        }
        const auto nothing = chunk_text("", by(ChunkOptions::By::paragraph, 512u));
        expect(nothing.size() == 1_u);
        if (nothing.size() == 1u) { expect(nothing[0].byte_length == 0u); }
    };
}

void reg_byte_exact_views() {
    "chunker_chunks_are_byte_exact_views"_test = [] {
        // Mixed scripts: every chunk must slice back to its own bytes.
        const std::string document =
            "介绍向量检索。Embedding retrieval maps a query to a vector. "
            "第二段：数据库连接池。";
        const auto chunks = chunk_text(document, by(ChunkOptions::By::sentence, 40u, 8u));
        expect(chunks_are_sane(document, chunks));
        std::string rebuilt;
        for (const auto &chunk : chunks) {
            const auto piece = text_of(chunk, document);
            expect(!piece.empty());
            expect(document.compare(static_cast<size_t>(chunk.byte_offset),
                                    piece.size(), piece) == 0);
            rebuilt += piece;
        }
        // The pieces cover the document (overlap may repeat bytes, never drop them).
        expect(rebuilt.size() >= document.size());
        const auto &last = chunks.back();
        expect(last.byte_offset + last.byte_length == document.size());
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_cjk_sentence_mode();
    reg_cjk_whitespace_and_oversize();
    reg_byte_exact_views();
    return 0;
}

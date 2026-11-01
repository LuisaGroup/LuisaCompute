// RAG chunker for the retrieval index (report.md §2). Splits one document into
// byte-range chunks along paragraph / sentence / token boundaries and packs the
// units greedily into chunks of at most `chunk_size` bytes with an overlap tail.
// Everything is byte-exact and UTF-8 aware: a chunk boundary never lands inside
// a multi-byte code point, and every chunk records [byte_offset, byte_length)
// into the source text so the RAG layer can quote the exact passage.
//
// `By::token` segments at ASCII whitespace, which is exactly the bge-m3
// Metaspace pre-tokenizer segmentation (split at U+2581 = one per ASCII space),
// so engine mode and callback mode produce identical chunk boundaries without
// touching the tokenizer.
#pragma once

#include <cstdint>
#include <utility>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::embed {

// Chunking request. The in-class defaults (512-byte paragraph chunks) are the
// 1.2 RAG behavior; the historical whole-file behavior is one chunk covering
// the entire document, expressed by ChunkOptions::whole_file() (chunk_size == 0
// / by == off). Callers porting from 1.1 must switch to whole_file() — a plain
// `ChunkOptions{}` now chunks.
struct ChunkOptions {
    uint32_t chunk_size = 512u;    // target maximum chunk payload in bytes
    uint32_t chunk_overlap = 64u;  // bytes repeated from the tail of the previous chunk
    uint32_t max_seq_len = 0u;     // 0 = engine default (512); passed to the encoder
    enum class By : uint32_t {
        off = 0u,      // whole document (today's behavior)
        paragraph = 1u,// blank-line separated blocks
        sentence = 2u, // punctuation-terminated sentences (UTF-8 aware)
        token = 3u,    // whitespace word boundaries (Metaspace segmentation)
    };
    By by = By::paragraph;

    [[nodiscard]] bool operator==(const ChunkOptions &other) const noexcept {
        return chunk_size == other.chunk_size &&
               chunk_overlap == other.chunk_overlap &&
               max_seq_len == other.max_seq_len && by == other.by;
    }

    // The historical whole-file behavior: one chunk covering the entire
    // document. This is what a default-constructed value MEANT before 1.2,
    // so every NULL-options / legacy overload path must use it explicitly
    // (a plain `ChunkOptions{}` now means paragraph chunking at 512 bytes).
    [[nodiscard]] static ChunkOptions whole_file() noexcept {
        ChunkOptions options;
        options.chunk_size = 0u;
        options.chunk_overlap = 0u;
        options.max_seq_len = 0u;
        options.by = By::off;
        return options;
    }
};

// One chunk of a document: its byte range in the source text.
struct Chunk {
    uint64_t byte_offset = 0u;
    uint64_t byte_length = 0u;

    [[nodiscard]] luisa::string_view text(luisa::string_view document) const noexcept {
        return document.substr(static_cast<size_t>(byte_offset),
                               static_cast<size_t>(byte_length));
    }
};

// Splits `document` into chunks. Empty / blank documents yield a single
// zero-length chunk at offset 0 (the index still stores one record per
// document, so counts stay well defined). `options.chunk_size == 0` or
// `options.by == By::off` yields exactly one chunk covering the whole text.
[[nodiscard]] luisa::vector<Chunk> chunk_text(
    luisa::string_view document,
    const ChunkOptions &options);

}// namespace luisa::embed

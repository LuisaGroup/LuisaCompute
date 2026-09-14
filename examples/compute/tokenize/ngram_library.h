#pragma once

#include "ngram_tokenizer.h"

#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>

namespace tokenize {

// Sentinel written into draft slots that hold no token.
inline constexpr uint32_t ngram_invalid_id = 0xFFFFFFFFu;

// Host-side n-gram library: a corpus of documents packed into one flat
// token-ID stream plus per-document offsets/lengths. Retrieval treats the
// corpus as a single flat stream (cross-document matches are allowed, like
// vLLM matches within one request stream); "earliest match in corpus wins".
//
// The class doubles as its own builder: documents are tokenized and packed
// incrementally with add_document(), then sealed in place with finalize().
// Token IDs are dense uint32 starting at 0, assigned in first-seen order
// and kept stable across add_document() calls. After finalize() the library
// is immutable: no more documents may be added. The device buffers are
// created and uploaded by NgramRetriever.
class NgramLibrary {
public:
    NgramLibrary() noexcept = default;

    // Tokenize one document and append its token IDs to the library.
    void add_document(luisa::string_view text);

    // Seal the accumulated documents in place: append the trailing
    // doc_offsets entry (end of the last document) and mark the library
    // finalized. Called exactly once, after the last add_document().
    void finalize();

    [[nodiscard]] bool finalized() const noexcept { return _finalized; }
    [[nodiscard]] uint32_t num_docs() const noexcept {
        return static_cast<uint32_t>(doc_lengths.size());
    }
    [[nodiscard]] uint32_t size() const noexcept {
        return static_cast<uint32_t>(tokens.size());
    }

    // Host-side corpus, complete after finalize().
    luisa::vector<uint32_t> tokens;       // concatenated corpus token IDs
    luisa::vector<uint32_t> doc_offsets;  // size = num_docs + 1 (starts + end)
    luisa::vector<uint32_t> doc_lengths;  // size = num_docs
    uint32_t vocab_size = 0;              // number of distinct token IDs

    // device-side copies, valid after NgramRetriever uploads the library
    luisa::compute::Buffer<uint32_t> tokens_buf;
    luisa::compute::Buffer<uint32_t> offsets_buf;
    luisa::compute::Buffer<uint32_t> lengths_buf;

private:
    // luisa::unordered_map hashes/compares string CONTENT (vstd::HashMap
    // would hash and compare the string objects bytewise instead).
    luisa::unordered_map<luisa::string, uint32_t> _vocab;
    uint32_t _next_id = 0;
    bool _finalized = false;
};

// Host-built n-gram hash index for the `hash` kernel variant: an open
// addressing table (linear probing, multiply-shift hashing) mapping every
// distinct corpus n-gram of length n in [min_n, max_n] to its EARLIEST
// corpus position. Slot key 0 marks an empty slot; FNV collisions between
// different n-grams are disambiguated by content verification on device
// (and on host during insertion). Building is O(corpus * (max_n - min_n))
// and happens once per library, amortized over all retrieve calls.
struct NgramHashIndex {
    uint32_t cap_log2 = 0;               // table capacity is 1 << cap_log2
    luisa::vector<uint64_t> keys;        // 0 = empty slot
    luisa::vector<uint32_t> pos;         // earliest corpus position per key
    luisa::compute::Buffer<uint64_t> keys_buf;
    luisa::compute::Buffer<uint32_t> pos_buf;
};

// Build the n-gram hash index for `corpus`. Every position p with
// p + n < lib_len for some n in [min_n, max_n] is indexed.
[[nodiscard]] NgramHashIndex build_ngram_hash_index(
    luisa::span<const uint32_t> corpus,
    uint32_t min_n, uint32_t max_n);

}// namespace tokenize

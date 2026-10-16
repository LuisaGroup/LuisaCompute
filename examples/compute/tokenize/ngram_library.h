#pragma once

#include "ngram_tokenizer.h"

#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>

namespace tokenize {

// Sentinel written into draft slots that hold no token.
inline constexpr uint32_t ngram_invalid_id = 0xFFFFFFFFu;

// Smoothing mode of the n-gram language model (LM mode only; the retrieval
// path never smooths). Values are baked into kernel arguments as uint32_t.
enum class NgramSmoothing : uint32_t {
    add_k = 0, // P(w|ctx) = (count(ctx,w) + k) / (count(ctx) + k * V)
    backoff = 1,// simple backoff: highest order with count(ctx,w) > 0 wins,
                // floored at the unigram probability count(w) / total_tokens
};

namespace detail {

// FNV-1a 64 over the raw little-endian bytes of n uint32 tokens. THE single
// hash implementation of the whole example: every host index builder and
// every DSL kernel reproduces this exact byte stream, so keys are directly
// comparable across the host/device boundary. Do not "improve" this loop.
[[nodiscard]] inline uint64_t fnv1a64_tokens(const uint32_t *tokens, uint32_t n) noexcept {
    uint64_t h = 14695981039346656037ull;
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t w = tokens[i];
        for (auto b = 0; b < 4; ++b) {
            h ^= (w >> (b * 8)) & 0xFFu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

}// namespace detail

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

    // Look up the dense token ID of a word (normalized like add_document
    // does). Returns ngram_invalid_id when the word is not in the vocabulary.
    [[nodiscard]] uint32_t find_token(luisa::string_view word) const noexcept;

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

// Host-built n-gram COUNT index: same open-addressing table layout, keys and
// probe order as NgramHashIndex. It indexes, each exactly once:
//   - the retrievable prefix windows: length n in [min_n, max_n] at every
//     position p with p + n < lib_len (identical window set to
//     build_ngram_hash_index), and
//   - the continuation windows: length max_n + 1 at every position p with
//     p + max_n < lib_len (the continuation token exists; the window may end
//     exactly at the corpus end).
//
// Because the length-n entries use the identical window rule and hash as the
// retrieval index, the existing `hash` retrieval kernel works on this table
// unchanged, and the counts give the MLE probabilities of exactly the
// continuations the inference logic proposes:
//   P(w | matched n-gram ctx) = count(ctx, w) / count(ctx)
// The denominator identity count(ctx) == sum_w count(ctx, w) is EXACT for
// |ctx| == max_n (every match's continuation window is indexed). For
// |ctx| < max_n a match whose continuation is the corpus's FINAL token
// (position lib_len - |ctx| - 1) has no indexed continuation window, so the
// identity holds up to that occurrence; the device kernels and the host
// oracles share the same table, so all consumers stay consistent.
// `pos` keeps the earliest representative position per slot,
// used for on-device content verification of FNV collisions and by tests to
// rebuild the (n-gram -> count) map. `counts_host` mirrors counts_buf after
// NgramTrainer::download_counts().
struct NgramCountIndex {
    uint32_t cap_log2 = 0;               // table capacity is 1 << cap_log2
    luisa::vector<uint64_t> keys;        // 0 = empty slot
    luisa::vector<uint32_t> pos;         // representative corpus position per slot
    luisa::vector<uint32_t> counts_host; // mirror of counts_buf (after download)
    luisa::compute::Buffer<uint64_t> keys_buf;
    luisa::compute::Buffer<uint32_t> pos_buf;
    luisa::compute::Buffer<uint32_t> counts_buf;
};

// Build the count index for `corpus`: for every n in [min_n, max_n] and
// every position p with p + n < lib_len, index the length-n prefix window;
// additionally index the length-(max_n+1) continuation window at every
// position with p + max_n < lib_len. Capacity keeps load <= 1/2.
[[nodiscard]] NgramCountIndex build_ngram_count_index(
    luisa::span<const uint32_t> corpus,
    uint32_t min_n, uint32_t max_n);

}// namespace tokenize

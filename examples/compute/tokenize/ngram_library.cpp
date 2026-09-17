#include "ngram_library.h"

#include <luisa/core/logging.h>

namespace tokenize {

void NgramLibrary::add_document(luisa::string_view text) {
    LUISA_ASSERT(!_finalized, "cannot add documents to a finalized library");
    auto words = NgramTokenizer::split(text);
    doc_offsets.emplace_back(static_cast<uint32_t>(tokens.size()));
    doc_lengths.emplace_back(static_cast<uint32_t>(words.size()));
    tokens.reserve(tokens.size() + words.size());
    for (auto &w : words) {
        auto it = _vocab.find(w);
        uint32_t id;
        if (it != _vocab.end()) {
            id = it->second;
        } else {
            id = _next_id++;
            _vocab.emplace(luisa::string{w}, id);
        }
        tokens.emplace_back(id);
    }
    vocab_size = _next_id;
}

void NgramLibrary::finalize() {
    LUISA_ASSERT(!_finalized, "library already finalized");
    // doc_offsets must have num_docs + 1 entries: the start of every
    // document plus the end of the last one.
    doc_offsets.emplace_back(static_cast<uint32_t>(tokens.size()));
    _finalized = true;
}

uint32_t NgramLibrary::find_token(luisa::string_view word) const noexcept {
    auto it = _vocab.find(NgramTokenizer::normalize(word));
    return it == _vocab.end() ? ngram_invalid_id : it->second;
}

NgramHashIndex build_ngram_hash_index(luisa::span<const uint32_t> corpus,
                                      uint32_t min_n, uint32_t max_n) {
    const uint32_t lib_len = static_cast<uint32_t>(corpus.size());
    LUISA_ASSERT(min_n >= 1u && min_n <= max_n, "invalid n-gram range");
    LUISA_ASSERT(lib_len > 0, "corpus must not be empty");
    uint64_t entries_max = 0;
    for (uint32_t n = min_n; n <= max_n; ++n) {
        if (n < lib_len) entries_max += lib_len - n;// one entry per position
    }
    LUISA_ASSERT(entries_max > 0, "corpus too small for n-gram range");
    // keep the load factor <= 1/2 so every probe chain reaches an empty slot
    uint32_t cap_log2 = 1;
    while ((uint64_t{1} << cap_log2) < entries_max * 2) ++cap_log2;
    LUISA_ASSERT(cap_log2 < 64u, "table too large");
    const uint64_t cap = uint64_t{1} << cap_log2;
    const uint64_t mask = cap - 1;

    NgramHashIndex index;
    index.cap_log2 = cap_log2;
    index.keys.assign(cap, 0ull);
    index.pos.assign(cap, 0u);

    for (uint32_t n = min_n; n <= max_n; ++n) {
        if (n >= lib_len) break;
        const uint32_t p_end = lib_len - n;// positions must keep p + n < lib_len
        for (uint32_t p = 0; p < p_end; ++p) {
            uint64_t key = detail::fnv1a64_tokens(corpus.data() + p, n);
            if (key == 0ull) key = 1ull;// reserve 0 for empty slots
            uint64_t slot = (key * 0x9E3779B97F4A7C15ull) >> (64 - cap_log2);
            for (;;) {
                const uint64_t k = index.keys[slot];
                if (k == 0ull) {// empty slot: insert (positions grow, so the first wins)
                    index.keys[slot] = key;
                    index.pos[slot] = p;
                    break;
                }
                if (k == key) {
                    // same hash: either the same n-gram (already stored, and
                    // any stored position is earlier than p) or an FNV
                    // collision -- verify the content to decide.
                    const uint32_t q = index.pos[slot];
                    bool same = true;
                    for (uint32_t j = 0; same && j < n; ++j) {
                        same = corpus[q + j] == corpus[p + j];
                    }
                    if (same) break;
                }
                slot = (slot + 1ull) & mask;
            }
        }
    }
    return index;
}

NgramCountIndex build_ngram_count_index(luisa::span<const uint32_t> corpus,
                                        uint32_t min_n, uint32_t max_n) {
    const uint32_t lib_len = static_cast<uint32_t>(corpus.size());
    LUISA_ASSERT(min_n >= 1u && min_n <= max_n, "invalid n-gram range");
    LUISA_ASSERT(lib_len > 0, "corpus must not be empty");
 // per position p with p + n < lib_len: the length-n retrievable prefix
 // window; plus, at every position with p + max_n < lib_len, the
 // length-(max_n+1) continuation window (its last token may be the
 // corpus's final token). Each window is indexed exactly once.
 uint64_t entries_max = 0;
 for (uint32_t n = min_n; n <= max_n; ++n) {
 if (n < lib_len) entries_max += lib_len - n;
 }
 if (max_n < lib_len) entries_max += lib_len - max_n;
    LUISA_ASSERT(entries_max > 0, "corpus too small for n-gram range");
    // keep the load factor <= 1/2 so every probe chain reaches an empty slot
    uint32_t cap_log2 = 1;
    while ((uint64_t{1} << cap_log2) < entries_max * 2) ++cap_log2;
    LUISA_ASSERT(cap_log2 < 64u, "table too large");
    const uint64_t cap = uint64_t{1} << cap_log2;
    const uint64_t mask = cap - 1;

    NgramCountIndex index;
    index.cap_log2 = cap_log2;
    index.keys.assign(cap, 0ull);
    index.pos.assign(cap, 0u);

    // Insert the window of `len` tokens at position p (earliest position wins,
    // FNV collisions disambiguated by content verification -- identical probe
    // sequence to the device kernels).
    auto insert = [&](uint32_t p, uint32_t len) {
        uint64_t key = detail::fnv1a64_tokens(corpus.data() + p, len);
        if (key == 0ull) key = 1ull; // reserve 0 for empty slots
        uint64_t slot = (key * 0x9E3779B97F4A7C15ull) >> (64 - cap_log2);
        for (;;) {
            const uint64_t k = index.keys[slot];
            if (k == 0ull) { // empty slot: insert
                index.keys[slot] = key;
                index.pos[slot] = p;
                break;
            }
            if (k == key) {
                const uint32_t q = index.pos[slot];
                bool same = true;
                for (uint32_t j = 0; same && j < len; ++j) {
                    same = corpus[q + j] == corpus[p + j];
                }
                if (same) break; // already stored (and earlier)
            }
            slot = (slot + 1ull) & mask;
        }
    };

    for (uint32_t n = min_n; n <= max_n; ++n) {
        if (n >= lib_len) break;
        const uint32_t p_end = lib_len - n; // positions keep p + n < lib_len
        for (uint32_t p = 0; p < p_end; ++p) {
            insert(p, n);// retrievable prefix window
        }
    }
    // continuation windows of length max_n + 1: p + max_n < lib_len (the
    // continuation token exists; the window may end exactly at the corpus
    // end, i.e. p + (max_n + 1) == lib_len is allowed)
    if (max_n < lib_len) {
        const uint32_t p_end = lib_len - max_n;
        for (uint32_t p = 0; p < p_end; ++p) {
            insert(p, max_n + 1u);
        }
    }
    return index;
}

}// namespace tokenize

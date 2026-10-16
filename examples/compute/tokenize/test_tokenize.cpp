// Function tests for the n-gram tokenize + retrieve + train example.
//
// This test covers:
//   - the UTF-8/CJK word splitter and the library builder invariants;
//   - the vLLM test_ngram.py vectors against both the faithful KMP port
//     (reference_retrieve_vllm_kmp) and the device kernels;
//   - longest-match / earliest-position tie-breaking of the semantic
//     oracle on a corpus/query pair;
//   - no-match, too-short, empty-query and library-end clamping cases;
//   - seeded fuzz of both DSL kernels against the CPU reference oracle,
//     including batched multi-query retrieves with an empty query row;
//   - the host -> device -> host library buffer round-trip;
//   - n-gram TRAINING: the retrieval-aligned count index layout, device
//     counting vs the host oracle (incl. determinism and the prefix-count
//     identity), trained-index hash retrieval, draft confidence scoring,
//     the parallel_mle continuation variant, and the LM mode (BOS/EOS/UNK
//     preprocessing, Add-k/backoff scoring, perplexity) against the
//     ReferenceNgramModel oracle.

#include "ut/ut.hpp"
#include "test_device.h"

#include "ngram_library.h"
#include "ngram_retriever.h"
#include "ngram_tokenizer.h"
#include "ngram_trainer.h"
#include "reference.h"
#include "reference_model.h"

#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cmath>
#include <random>

using namespace luisa;
using namespace luisa::compute;
using namespace tokenize;
using namespace boost::ut;

namespace {

// ---------- helpers ----------

// std::mt19937::result_type is uint_fast32_t: `unsigned long` (64-bit) on LP64
// Linux but `unsigned int` on MSVC. Mixing raw engine output with uint32_t
// operands breaks std::min/std::max template deduction on GCC/Clang, so every
// fuzz draw goes through this wrapper and stays a plain uint32_t.
class Rng32 {
public:
    explicit Rng32(uint32_t seed) noexcept : m_engine{seed} {}
    // one draw in [0, bound); `bound` must be non-zero
    [[nodiscard]] uint32_t operator()(uint32_t bound) noexcept {
        return static_cast<uint32_t>(m_engine() % bound);
    }

private:
    std::mt19937 m_engine;
};

// Build a library directly from a token-ID corpus (one document),
// bypassing the tokenizer so numeric corpora can be used.
[[nodiscard]] NgramLibrary make_id_library(luisa::span<const uint32_t> corpus) noexcept {
    NgramLibrary lib;
    lib.tokens.assign(corpus.begin(), corpus.end());
    lib.doc_offsets.push_back(0u);
    lib.doc_lengths.push_back(static_cast<uint32_t>(corpus.size()));
    uint32_t vocab = 0;
    for (auto t : corpus) vocab = std::max(vocab, t + 1u);
    lib.vocab_size = vocab;
    lib.finalize();
    return lib;
}

// Same, but with several documents (LM-mode sentences).
[[nodiscard]] NgramLibrary make_multi_doc_id_library(
    luisa::span<const luisa::vector<uint32_t>> docs) noexcept {
    NgramLibrary lib;
    uint32_t off = 0;
    for (auto &d : docs) {
        lib.doc_offsets.push_back(off);
        lib.doc_lengths.push_back(static_cast<uint32_t>(d.size()));
        lib.tokens.insert(lib.tokens.end(), d.begin(), d.end());
        off += static_cast<uint32_t>(d.size());
    }
    for (auto t : lib.tokens) lib.vocab_size = std::max(lib.vocab_size, t + 1u);
    lib.finalize();
    return lib;
}

// Probe a host-built open-addressing table for the n-token window at corpus
// position p, replaying the exact probe sequence of the builders/kernels.
// Returns the slot index or ngram_invalid_id.
[[nodiscard]] uint32_t host_probe_slot(luisa::span<const uint64_t> keys,
                                       luisa::span<const uint32_t> pos,
                                       uint32_t cap_log2,
                                       luisa::span<const uint32_t> corpus,
                                       uint32_t p, uint32_t n) noexcept {
    uint64_t key = tokenize::detail::fnv1a64_tokens(corpus.data() + p, n);
    if (key == 0ull) key = 1ull;
    const uint64_t mask = (uint64_t{1} << cap_log2) - 1ull;
    uint64_t slot = (key * 0x9E3779B97F4A7C15ull) >> (64 - cap_log2);
    for (;;) {
        const uint64_t k = keys[slot];
        if (k == 0ull) return ngram_invalid_id;
        if (k == key) {
            const uint32_t q = pos[slot];
            bool same = true;
            for (uint32_t j = 0; same && j < n; ++j) {
                same = corpus[q + j] == corpus[p + j];
            }
            if (same) return static_cast<uint32_t>(slot);
        }
        slot = (slot + 1ull) & mask;
    }
}

// Decode a byte-key produced by detail::ngram_bytes_key back into tokens.
[[nodiscard]] luisa::vector<uint32_t> decode_ngram_key(const luisa::string &key) noexcept {
    luisa::vector<uint32_t> tokens(key.size() / 4u);
    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t t = 0;
        for (uint32_t b = 0; b < 4; ++b) {
            t |= static_cast<uint32_t>(static_cast<unsigned char>(key[i * 4 + b])) << (b * 8);
        }
        tokens[i] = t;
    }
    return tokens;
}

// Find the first occurrence of `win` in `corpus` (ngram_invalid_id if none).
[[nodiscard]] uint32_t first_occurrence(luisa::span<const uint32_t> corpus,
                                        luisa::span<const uint32_t> win) noexcept {
    const uint32_t n = static_cast<uint32_t>(win.size());
    for (uint32_t p = 0; p + n <= corpus.size(); ++p) {
        bool same = true;
        for (uint32_t j = 0; same && j < n; ++j) same = corpus[p + j] == win[j];
        if (same) return p;
    }
    return ngram_invalid_id;
}

// Run one batch of queries through `variant` and check every draft against
// the reference oracle. Returns the number of failed checks.
// `count_index` is an optional trained NgramCountIndex (required by the
// parallel_mle variant, accepted by the hash variant).
[[nodiscard]] uint32_t check_queries_against_reference(
    Device &device, Stream &stream, NgramLibrary &lib,
    luisa::span<const luisa::vector<uint32_t>> queries,
    uint32_t min_n, uint32_t max_n, uint32_t k,
    NgramKernelVariant variant, uint32_t block_size = 512u,
    const NgramCountIndex *count_index = nullptr) {
    uint32_t max_query_len = max_n;
    for (auto &q : queries) max_query_len = std::max(max_query_len, (uint32_t)q.size());
    NgramRetriever retriever{device, stream, lib, min_n, max_n, k,
                             max_query_len, queries.size(), variant, block_size,
                             count_index};
    luisa::vector<uint32_t> drafts, draft_lens;
    retriever.retrieve(queries, drafts, draft_lens);
    uint32_t failures = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
        auto expected = reference_retrieve(luisa::span{lib.tokens},
                                           luisa::span{queries[i]}, min_n, max_n, k);
        if (draft_lens[i] != expected.size()) {
            ++failures;
            continue;
        }
        for (size_t j = 0; j < expected.size(); ++j) {
            if (drafts[i * k + j] != expected[j]) ++failures;
        }
        // slots past the draft length must stay invalid
        for (size_t j = expected.size(); j < k; ++j) {
            if (drafts[i * k + j] != ngram_invalid_id) ++failures;
        }
    }
    return failures;
}

// Same, but for the parallel_mle variant against the MLE continuation oracle
// (the count index and its host mirror are both required).
[[nodiscard]] uint32_t check_queries_against_mle_reference(
    Device &device, Stream &stream, NgramLibrary &lib,
    const NgramCountIndex &count_index, const NgramCountMap &counts,
    luisa::span<const luisa::vector<uint32_t>> queries,
    uint32_t min_n, uint32_t max_n, uint32_t k, uint32_t block_size = 512u) {
    uint32_t max_query_len = max_n;
    for (auto &q : queries) max_query_len = std::max(max_query_len, (uint32_t)q.size());
    NgramRetriever retriever{device, stream, lib, min_n, max_n, k,
                             max_query_len, queries.size(),
                             NgramKernelVariant::parallel_mle, block_size,
                             &count_index};
    luisa::vector<uint32_t> drafts, draft_lens;
    retriever.retrieve(queries, drafts, draft_lens);
    uint32_t failures = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
        auto expected = reference_retrieve_mle(luisa::span{lib.tokens}, counts,
                                               luisa::span{queries[i]}, min_n, max_n, k);
        if (draft_lens[i] != expected.size()) {
            ++failures;
            continue;
        }
        for (size_t j = 0; j < expected.size(); ++j) {
            if (drafts[i * k + j] != expected[j]) ++failures;
        }
        for (size_t j = expected.size(); j < k; ++j) {
            if (drafts[i * k + j] != ngram_invalid_id) ++failures;
        }
    }
    return failures;
}

// Pack token rows row-major into a flat buffer (zero-padded) + row lengths.
struct PackedRows {
    luisa::vector<uint32_t> flat;
    luisa::vector<uint32_t> lens;
    uint32_t stride = 0;
};
[[nodiscard]] PackedRows pack_rows(luisa::span<const luisa::vector<uint32_t>> rows) noexcept {
    PackedRows out;
    for (auto &r : rows) out.stride = std::max(out.stride, (uint32_t)r.size());
    out.flat.assign(rows.size() * out.stride, 0u);
    out.lens.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        out.lens.push_back((uint32_t)rows[i].size());
        std::copy_n(rows[i].begin(), rows[i].size(), out.flat.begin() + i * out.stride);
    }
    return out;
}

// ---------- host-only tests ----------

void register_host_tests() {
    "tokenizer_normalize_and_cjk"_test = [] {
        expect(NgramTokenizer::normalize("Hello, WORLD") == "hello, world");
        expect(NgramTokenizer::is_cjk(U'中'));
        expect(NgramTokenizer::is_cjk(U'あ'));// hiragana
        expect(!NgramTokenizer::is_cjk(U'a'));
        expect(!NgramTokenizer::is_cjk(U'🙂'));
    };

    "tokenizer_split_ascii_runs"_test = [] {
        auto tokens = NgramTokenizer::split("Hello, World! foo-bar");
        expect(tokens.size() == 4u);
        expect(tokens[0] == "hello");
        expect(tokens[1] == "world");
        expect(tokens[2] == "foo");
        expect(tokens[3] == "bar");
    };

    "tokenizer_split_cjk_per_codepoint"_test = [] {
        auto tokens = NgramTokenizer::split("中文字符 hello");
        expect(tokens.size() == 5u);
        expect(tokens[0] == "中");
        expect(tokens[1] == "文");
        expect(tokens[2] == "字");
        expect(tokens[3] == "符");
        expect(tokens[4] == "hello");
    };

    "library_builder_invariants"_test = [] {
        NgramLibrary lib;
        lib.add_document("the quick brown fox");
        lib.add_document("the quick brown cat");
        lib.finalize();
        expect(lib.finalized());
        expect(lib.num_docs() == 2u);
        expect(lib.doc_offsets.size() == 3u);
        expect(lib.doc_lengths.size() == 2u);
        // offsets[i+1] - offsets[i] == lengths[i]
        for (size_t i = 0; i < lib.num_docs(); ++i) {
            expect(lib.doc_offsets[i + 1] - lib.doc_offsets[i] == lib.doc_lengths[i]);
        }
        // 6 distinct words ("the","quick","brown","fox","cat")
        expect(lib.vocab_size == 5u);
        // "the quick brown" prefix of both docs -> stable ids
        expect(lib.tokens[0] == lib.tokens[4]);
        expect(lib.tokens[1] == lib.tokens[5]);
        expect(lib.tokens[2] == lib.tokens[6]);
        expect(lib.tokens[3] != lib.tokens[7]);
    };

    "reference_kmp_vllm_vectors"_test = [] {
        using V = luisa::vector<uint32_t>;
        // ported vectors from vllm/tests/v1/spec_decode/test_ngram.py
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3, 5, 6}, 2, 2, 2).empty());
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3}, 2, 2, 3) == V({4, 1, 2}));
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3}, 2, 2, 2) == V({4, 1}));
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3}, 1, 1, 3) == V({4, 1, 2}));
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3}, 1, 1, 2) == V({4, 1}));
        expect(reference_retrieve_vllm_kmp(V{1, 3, 6, 2, 3, 4, 1, 2, 3}, 2, 2, 3) == V({4, 1, 2}));
        expect(reference_retrieve_vllm_kmp(V{1, 3, 6, 2, 3, 4, 1, 2, 3}, 1, 1, 2) == V({6, 2}));
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3}, 4, 4, 2).empty());
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 4, 1, 2, 3}, 3, 4, 2) == V({4, 1}));
        expect(reference_retrieve_vllm_kmp(V{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}, 3, 4, 2) == V({1, 2}));
        expect(reference_retrieve_vllm_kmp(V{3, 4, 5, 2, 3, 4, 1, 2, 3, 4}, 2, 4, 2) == V({1, 2}));
        expect(reference_retrieve_vllm_kmp(V{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3}, 3, 3, 2) == V({100, 1}));
        expect(reference_retrieve_vllm_kmp(V{}, 2, 2, 2).empty());
    };

    "reference_semantics_corpus_query"_test = [] {
        using V = luisa::vector<uint32_t>;
        // match must leave >= 1 continuation token: the trailing occurrence
        // itself never counts as a match
        expect(reference_retrieve(V{1, 2, 3, 4, 5}, V{1, 2, 3, 4, 5}, 2, 2, 3).empty());
        // earliest match wins
        expect(reference_retrieve(V{1, 2, 3, 1, 2}, V{1, 2, 3, 1, 2}, 2, 2, 3) == V({3, 1, 2}));
        expect(reference_retrieve(V{1, 2, 3, 1, 2}, V{1, 2, 3, 1, 2}, 2, 2, 2) == V({3, 1}));
        // longest n wins over a shorter earlier match
        expect(reference_retrieve(V{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}, V{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}, 3, 4, 2) == V({1, 2}));
        // earliest position for the winning length
        expect(reference_retrieve(V{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3}, V{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3}, 3, 3, 2) == V({100, 1}));
        // draft clamped to the library end (3 tokens available, k = 5)
        expect(reference_retrieve(V{1, 2, 3, 9, 5}, V{1, 2, 3}, 2, 3, 5) == V({9, 5}));
        // too short / empty / k == 0
        expect(reference_retrieve(V{1, 2, 3, 4, 5}, V{1, 2}, 3, 3, 2).empty());
        expect(reference_retrieve(V{1, 2, 3, 4, 5}, V{}, 1, 2, 2).empty());
        expect(reference_retrieve(V{1, 2, 3, 4, 5}, V{1, 2, 3}, 1, 2, 0).empty());
    };

    "reference_kmp_matches_literal_fuzz"_test = [] {
        Rng32 rng{42};
        for (auto iter = 0; iter < 200; ++iter) {
            const uint32_t n = 1 + rng(48);
            luisa::vector<uint32_t> corpus(n);
            for (auto &t : corpus) t = rng(6);
            // plant a copy of an earlier chunk to create real matches
            if (n > 8 && rng(2) != 0u) {
                uint32_t src = rng(n / 2);
                uint32_t dst = n / 2 + rng(n / 2);
                uint32_t len = std::min(3u + rng(4), n - std::max(src, dst));
                for (uint32_t i = 0; i + 1 < len; ++i) corpus[dst + i] = corpus[src + i];
            }
            const uint32_t min_n = 1 + rng(2);
            const uint32_t max_n = min_n + rng(3);
            const uint32_t k = 1 + rng(5);
            auto via_kmp = reference_retrieve_vllm_kmp(luisa::span{corpus}, min_n, max_n, k);
            auto via_lit = reference_retrieve(luisa::span{corpus}, luisa::span{corpus}, min_n, max_n, k);
            expect(via_kmp == via_lit) << "kmp port matches literal oracle";
        }
    };

    // ---------- training: host-side tests ----------

    "count_index_windows_match_retrieval_rule"_test = [] {
        luisa::vector<uint32_t> corpus = {1, 2, 3, 1, 2, 4, 1, 2, 5, 8, 8, 1, 2, 3, 6, 7};
        const uint32_t min_n = 2, max_n = 3;
        const uint32_t lib_len = static_cast<uint32_t>(corpus.size());
        auto hash_index = build_ngram_hash_index(luisa::span{corpus}, min_n, max_n);
        auto count_index = build_ngram_count_index(luisa::span{corpus}, min_n, max_n);

        // every key/pos pair of the retrieval hash index appears in the count
        // index with the SAME position (the count index's length-n entries
        // reproduce the retrieval layout for the shared range)
        uint32_t shared = 0;
        for (size_t s = 0; s < hash_index.keys.size(); ++s) {
            if (hash_index.keys[s] == 0ull) continue;
            ++shared;
            // recover the window length by probing the count index for all
            // candidate lengths at the stored position
            const uint32_t p = hash_index.pos[s];
            bool found = false;
            for (uint32_t n = min_n; n <= max_n && p + n < lib_len; ++n) {
                uint32_t slot = host_probe_slot(luisa::span{count_index.keys},
                                                luisa::span{count_index.pos},
                                                count_index.cap_log2,
                                                luisa::span{corpus}, p, n);
                if (slot != ngram_invalid_id && count_index.keys[slot] == hash_index.keys[s]) {
                    expect(count_index.pos[slot] == p);
                    found = true;
                }
            }
            expect(found) << "every hash-index entry is present in the count index";
        }
        expect(shared > 0u);

        // the count index contains exactly the retrievable windows (p + n <
        // lib_len, n in [min_n, max_n]) plus the length-(max_n+1)
        // continuation windows (p + max_n < lib_len), and nothing else:
        // occupied slots == number of distinct windows
        NgramCountMap distinct;
        for (uint32_t n = min_n; n <= max_n; ++n) {
            for (uint32_t p = 0; p + n < lib_len; ++p) {
                distinct.emplace(tokenize::detail::ngram_bytes_key(corpus.data() + p, n), 0u);
            }
        }
        for (uint32_t p = 0; p + max_n < lib_len; ++p) {
            distinct.emplace(tokenize::detail::ngram_bytes_key(corpus.data() + p, max_n + 1u), 0u);
        }
        uint32_t occupied = 0;
        for (auto k : count_index.keys) occupied += k != 0ull ? 1u : 0u;
        expect(occupied == distinct.size()) << "count table covers exactly the retrievable windows + continuations";

        // every indexed window is findable by content and stores the earliest
        // position
        for (uint32_t n = min_n; n <= max_n + 1u; ++n) {
            for (uint32_t p = 0; p + n <= lib_len; ++p) {
                // a length-(max_n+1) window is indexed iff p + max_n < lib_len
                const bool should_exist =
                    n <= max_n ? p + n < lib_len : p + max_n < lib_len;
                const uint32_t slot = host_probe_slot(luisa::span{count_index.keys},
                                                      luisa::span{count_index.pos},
                                                      count_index.cap_log2,
                                                      luisa::span{corpus}, p, n);
                if (should_exist) {
                    expect(slot != ngram_invalid_id);
                    expect(count_index.pos[slot] <= p) << "earliest position stored";
                }
            }
        }
    };

    "reference_mle_tie_breaks"_test = [] {
        using V = luisa::vector<uint32_t>;
        // earliest match has the RARER continuation: MLE must move past it.
        // corpus: (1,2,3)->200 once, then (1,2,3)->100 twice
        V corpus{1, 2, 3, 200, 1, 2, 3, 100, 1, 2, 3, 100};
        auto counts = reference_count_ngrams(luisa::span{corpus}, 2, 3);
        V query{1, 2, 3};
        auto mle = reference_retrieve_mle(luisa::span{corpus}, counts,
                                          luisa::span{query}, 2, 3, 3);
        auto plain = reference_retrieve(luisa::span{corpus}, luisa::span{query}, 2, 3, 3);
        expect(plain == V({200, 1, 2})) << "classic rule: earliest position";
        expect(mle == V({100, 1, 2})) << "mle rule: most frequent continuation";

        // equal continuation counts -> earliest position (degenerates to the
        // classic rule)
        V corpus2{1, 2, 3, 100, 1, 2, 3, 200};
        auto counts2 = reference_count_ngrams(luisa::span{corpus2}, 2, 3);
        expect(reference_retrieve_mle(luisa::span{corpus2}, counts2,
                                      luisa::span{query}, 2, 3, 3) == V({100, 1, 2}));

        // longest n still wins overall: a longer match with a rarer
        // continuation beats a shorter match with a frequent one
        V corpus3{9, 9, 7, 1, 2, 3, 50, 9, 9, 7};
        auto counts3 = reference_count_ngrams(luisa::span{corpus3}, 2, 3);
        // query trailing 3-gram (9,9,7) matches at 0 (cont 1) and 7 (none:
        // p+n == lib_len); trailing 2-gram (9,7) matches at 1 (cont 2).
        // named lvalues: std::span rejects spans over rvalue vectors
        V q997{9, 9, 7}, q55{5, 5};
        expect(reference_retrieve_mle(luisa::span{corpus3}, counts3,
                                      luisa::span{q997}, 2, 3, 2) == V({1, 2}));
        // no match / clamped draft
        expect(reference_retrieve_mle(luisa::span{corpus3}, counts3,
                                      luisa::span{q55}, 2, 3, 2).empty());
        V corpus4{1, 2, 3, 9};
        auto counts4 = reference_count_ngrams(luisa::span{corpus4}, 2, 2);
        V q12{1, 2};
        expect(reference_retrieve_mle(luisa::span{corpus4}, counts4,
                                      luisa::span{q12}, 2, 2, 5) == V({3, 9}))
            << "draft clamped to the library end";
    };

    "reference_model_python_example"_test = [] {
        using V = luisa::vector<uint32_t>;
        // the Python BigramModel corpus as padded IDs: 我=0 爱=1 北=2 京=3
        // 学=4 习=5 你=6, <unk>=7 <s>=8 </s>=9, V = 10
        const uint32_t bos = 8, eos = 9;
        V padded{8, 0, 1, 2, 3, 9, 8, 0, 1, 4, 5, 9, 8, 6, 1, 2, 3, 9};
        ReferenceNgramModel model;
        model.train(luisa::span{padded}, 2, bos);
        expect(model.total_tokens() == 18u);
        V w0{0}, w01{0, 1}, w64{6, 4}, weos_bos{9, 8}, wbos{bos};
        expect(model.count(luisa::span{w0}) == 2u);
        expect(model.count(luisa::span{w01}) == 2u);
        expect(model.count(luisa::span{w64}) == 0u) << "unseen bigram";
        // BOS-skip: the cross-sentence window (9, 8) is never counted
        expect(model.count(luisa::span{weos_bos}) == 0u);
        expect(model.count(luisa::span{wbos}) == 3u);

        const float k = 0.1f, Vf = 10.0f;
        V row_wo_ai{0, 1}, row_ni_xx{6, 4}, sent_love_beijing{8, 0, 1, 2, 3, 9};
        // MLE P(爱|我) = 1.0 via backoff
        const float p_mle = std::exp2(model.sentence_log2prob(
            luisa::span{row_wo_ai}, NgramSmoothing::backoff, k, Vf));
        expect(std::abs(p_mle - 1.0f) < 1e-6f) << "MLE P(爱|我) = 1";
        // Add-k P(爱|我) = (2 + 0.1) / (2 + 0.1 * 10) = 0.7
        const float p_addk = std::exp2(model.sentence_log2prob(
            luisa::span{row_wo_ai}, NgramSmoothing::add_k, k, Vf));
        expect(std::abs(p_addk - 0.7f) < 1e-6f) << "Add-k P(爱|我) = 0.7";
        // Add-k P(学习|你) = (0 + 0.1) / (1 + 0.1 * 10) = 0.05 (non-zero)
        const float p_unseen = std::exp2(model.sentence_log2prob(
            luisa::span{row_ni_xx}, NgramSmoothing::add_k, k, Vf));
        expect(std::abs(p_unseen - 0.05f) < 1e-6f) << "smoothed P(学习|你) = 0.05";
        // sentence_prob = P(我|<s>) * P(爱|我) * P(北|爱) * P(京|北) * P(</s>|京)
        //               = 0.525 * 0.7 * 0.525 * 0.7 * 0.7
        const double expected_sp = 0.525 * 0.7 * 0.525 * 0.7 * 0.7;
        const float sp = std::exp2(model.sentence_log2prob(
            luisa::span{sent_love_beijing}, NgramSmoothing::add_k, k, Vf));
        expect(std::abs(sp - expected_sp) < 1e-6) << "sentence_prob equals the manual product";
        // perplexity hand-check on the 3-sentence corpus
        const luisa::vector<luisa::vector<uint32_t>> sents = {
            {8, 0, 1, 2, 3, 9}, {8, 0, 1, 4, 5, 9}, {8, 6, 1, 2, 3, 9}};
        const double ppl = model.perplexity(luisa::span{sents},
                                            NgramSmoothing::add_k, k, Vf);
        expect(std::abs(ppl - 1.833446) < 1e-4) << "perplexity hand-check";
    };
}

// ---------- device tests ----------

void register_device_tests(Device &device) {
    "device_library_roundtrip"_test = [&device] {
        auto stream = device.create_stream();
        luisa::vector<uint32_t> corpus = {5, 3, 9, 1, 2, 3, 9, 1, 2, 7, 5, 3, 9, 1, 2, 3};
        auto lib = make_id_library(luisa::span{corpus});
        NgramRetriever retriever{device, stream, lib, 2, 3, 2, 8, 4,
                                 NgramKernelVariant::naive};
        (void)retriever;
        luisa::vector<uint32_t> tokens_dl(lib.tokens.size());
        luisa::vector<uint32_t> offsets_dl(lib.doc_offsets.size());
        luisa::vector<uint32_t> lengths_dl(lib.doc_lengths.size());
        stream << lib.tokens_buf.view().copy_to(luisa::span{tokens_dl})
               << lib.offsets_buf.view().copy_to(luisa::span{offsets_dl})
               << lib.lengths_buf.view().copy_to(luisa::span{lengths_dl})
               << synchronize();
        expect(tokens_dl == lib.tokens);
        expect(offsets_dl == lib.doc_offsets);
        expect(lengths_dl == lib.doc_lengths);
    };

    "device_vllm_vectors_all_kernels"_test = [&device] {
        auto stream = device.create_stream();
        struct Case {
            luisa::vector<uint32_t> tokens;
            uint32_t min_n, max_n, k;
            luisa::vector<uint32_t> expected;
        };
        luisa::vector<Case> cases = {
            {{1, 2, 3, 4, 1, 2, 3, 5, 6}, 2, 2, 2, {}},
            {{1, 2, 3, 4, 1, 2, 3}, 2, 2, 3, {4, 1, 2}},
            {{1, 2, 3, 4, 1, 2, 3}, 2, 2, 2, {4, 1}},
            {{1, 3, 6, 2, 3, 4, 1, 2, 3}, 2, 2, 3, {4, 1, 2}},
            {{1, 3, 6, 2, 3, 4, 1, 2, 3}, 1, 1, 2, {6, 2}},
            {{1, 2, 3, 4, 1, 2, 3}, 4, 4, 2, {}},
            {{1, 2, 3, 4, 1, 2, 3}, 3, 4, 2, {4, 1}},
            {{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}, 3, 4, 2, {1, 2}},
            {{3, 4, 5, 2, 3, 4, 1, 2, 3, 4}, 2, 4, 2, {1, 2}},
            {{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3}, 3, 3, 2, {100, 1}},
            {{1, 2, 3, 4, 5}, 2, 2, 2, {}},
        };
        for (auto &cs : cases) {
            auto lib = make_id_library(luisa::span{cs.tokens});
            luisa::vector<luisa::vector<uint32_t>> queries;
            queries.emplace_back(cs.tokens.begin(), cs.tokens.end());
            for (auto variant : {NgramKernelVariant::naive, NgramKernelVariant::parallel,
                                NgramKernelVariant::hash}) {
                uint32_t failures = check_queries_against_reference(
                    device, stream, lib, luisa::span{queries},
                    cs.min_n, cs.max_n, cs.k, variant);
                expect(failures == 0u) << "device kernel matches on vLLM vectors";
                // and the draft content is exactly the expected one
                NgramRetriever retriever{device, stream, lib, cs.min_n, cs.max_n, cs.k,
                                         (uint32_t)cs.tokens.size(), 1, variant};
                luisa::vector<uint32_t> drafts, draft_lens;
                retriever.retrieve(luisa::span{queries}, drafts, draft_lens);
                expect(draft_lens[0] == cs.expected.size());
                for (size_t j = 0; j < cs.expected.size(); ++j) {
                    expect(drafts[j] == cs.expected[j]);
                }
            }
        }
    };

    "device_batch_with_empty_and_short_rows"_test = [&device] {
        auto stream = device.create_stream();
        luisa::vector<uint32_t> corpus = {1, 2, 3, 1, 2, 4, 1, 2, 5, 8, 8, 1, 2, 3, 6, 7};
        auto lib = make_id_library(luisa::span{corpus});
        luisa::vector<luisa::vector<uint32_t>> queries = {
            {1, 2},          // too short for min_n = 3 -> empty
            {},              // empty -> empty
            {3, 1, 2},       // trailing 2-gram {1,2} matches at 0 -> draft {3,1,2}
            {1, 2, 3, 6, 7}, // mixed case, oracle-checked below
            {8, 8, 1, 2, 3}, // leading 3-gram of the corpus
        };
        for (auto variant : {NgramKernelVariant::naive, NgramKernelVariant::parallel,
                                NgramKernelVariant::hash}) {
            uint32_t failures = check_queries_against_reference(
                device, stream, lib, luisa::span{queries}, 2, 3, 3, variant);
            expect(failures == 0u) << "batched empty/short rows match oracle";
        }
    };

    // One multi-dispatch command serves several "requests" (unequal row
    // counts); every sub-dispatch finds its row range through kernel_id()
    // and the shared offsets buffer. Exercises the same path as the
    // benchmark's 'multi' strategy.
    "device_multi_dispatch_requests"_test = [&device] {
        auto stream = device.create_stream();
        luisa::vector<uint32_t> corpus = {1, 2, 3, 1, 2, 4, 1, 2, 5, 8, 8, 1, 2, 3, 6, 7};
        auto lib = make_id_library(luisa::span{corpus});
        constexpr uint32_t min_n = 2u, max_n = 3u, k = 3u;
        // three requests with 2/1/3 rows (one row is empty, one too short)
        luisa::vector<luisa::vector<uint32_t>> rows = {
            {1, 2},          // request 0: trailing 2-gram matches at 0
            {8, 8, 1, 2, 3}, // request 1: leading 3-gram of the corpus
            {1, 2, 3, 6, 7}, // request 2
            {5, 5},          // request 2: no match
            {},              // request 2: empty
            {3, 1},          // request 2: too short for min_n = 2... len 2 == min_n ok? {3,1}: trailing 2-gram {3,1} not in corpus -> no match
        };
        const uint32_t counts[3] = {2u, 1u, 3u};
        const uint32_t bases[3] = {0u, 2u, 3u};
        uint32_t max_query_len = max_n;
        for (auto &q : rows) max_query_len = std::max(max_query_len, (uint32_t)q.size());
        const uint32_t total_rows = 6u;

        luisa::vector<uint32_t> lens(total_rows);
        for (auto i = 0u; i < total_rows; ++i) lens[i] = (uint32_t)rows[i].size();
        luisa::vector<uint32_t> flat(total_rows * max_query_len, 0u);
        for (auto i = 0u; i < total_rows; ++i) {
            std::copy_n(rows[i].begin(), rows[i].size(), flat.begin() + i * max_query_len);
        }
        for (auto variant : {NgramKernelVariant::naive, NgramKernelVariant::parallel,
                            NgramKernelVariant::hash}) {
            NgramRetriever retriever{device, stream, lib, min_n, max_n, k,
                                     max_query_len, total_rows, variant};
            retriever.upload_request(stream, luisa::span{flat}, luisa::span{lens}, 0u);
            auto off_buf = retriever.upload_request_offsets(
                stream, luisa::span{bases, 3u});
            const uint32_t block = variant == NgramKernelVariant::parallel ? 512u : 1u;
            const luisa::uint3 sizes[3] = {
                luisa::make_uint3(counts[0] * block, 1u, 1u),
                luisa::make_uint3(counts[1] * block, 1u, 1u),
                luisa::make_uint3(counts[2] * block, 1u, 1u)};
            retriever.dispatch_requests_multi(off_buf, luisa::span{sizes, 3u});
            luisa::vector<uint32_t> drafts, draft_lens;
            retriever.download_request(stream, 0u, total_rows, drafts, draft_lens);
            stream << synchronize();
            for (auto i = 0u; i < total_rows; ++i) {
                auto expected = reference_retrieve(luisa::span{lib.tokens},
                                                   luisa::span{rows[i]}, min_n, max_n, k);
                expect(draft_lens[i] == expected.size());
                for (size_t j = 0; j < expected.size(); ++j) {
                    expect(drafts[i * k + j] == expected[j]);
                }
                for (size_t j = expected.size(); j < k; ++j) {
                    expect(drafts[i * k + j] == ngram_invalid_id);
                }
            }
        }
    };

    "device_fuzz_vs_reference"_test = [&device] {
        auto stream = device.create_stream();
        Rng32 rng{1337};
        for (auto cfg = 0; cfg < 16; ++cfg) {
            const uint32_t min_n = 1 + rng(3);
            const uint32_t max_n = min_n + rng(4);
            const uint32_t k = 1 + rng(5);
            const uint32_t lib_len = 48 + rng(400);
            luisa::vector<uint32_t> corpus(lib_len);
            for (auto &t : corpus) t = rng(7);
            // plant copies of earlier chunks so matches are common
            for (auto p = 0; p < 8; ++p) {
                uint32_t src = rng(lib_len / 2);
                uint32_t dst = lib_len / 2 + rng(lib_len / 2);
                uint32_t len = 4 + rng(8);
                for (uint32_t i = 0; i < len && src + i < lib_len && dst + i < lib_len; ++i) {
                    corpus[dst + i] = corpus[src + i];
                }
            }
            auto lib = make_id_library(luisa::span{corpus});
            luisa::vector<luisa::vector<uint32_t>> queries;
            for (auto q = 0; q < 24; ++q) {
                // mostly corpus suffixes, sometimes unrelated random tokens
                if (rng(4) != 0u) {
                    uint32_t begin = rng(lib_len);
                    uint32_t len = 1 + rng(24);
                    queries.emplace_back(corpus.begin() + begin,
                                         corpus.begin() + std::min(begin + len, lib_len));
                } else {
                    uint32_t len = 1 + rng(8);
                    luisa::vector<uint32_t> rq(len);
                    for (auto &t : rq) t = rng(7);
                    queries.emplace_back(std::move(rq));
                }
            }
            for (auto variant : {NgramKernelVariant::naive, NgramKernelVariant::parallel,
                                NgramKernelVariant::hash}) {
                uint32_t failures = check_queries_against_reference(
                    device, stream, lib, luisa::span{queries}, min_n, max_n, k, variant);
                expect(failures == 0u) << "fuzz: kernel output equals reference";
            }
        }
    };

    // ---------- training: device tests ----------

    "device_unigram_histogram"_test = [&device] {
        auto stream = device.create_stream();
        auto shader = device.compile(make_unigram_histogram_kernel());
        Rng32 rng{777};
        for (auto iter = 0; iter < 8; ++iter) {
            const uint32_t len = 1 + rng(500);
            const uint32_t vocab = 1 + rng(12);// zero-count buckets included
            luisa::vector<uint32_t> corpus(len);
            for (auto &t : corpus) t = rng(vocab);
            luisa::vector<uint32_t> expected(vocab, 0u);
            for (auto t : corpus) ++expected[t];
            auto tokens_buf = device.create_buffer<uint32_t>(len);
            auto counts_buf = device.create_buffer<uint32_t>(vocab);
            const luisa::vector<uint32_t> zeros(vocab, 0u);
            stream << tokens_buf.view().copy_from(luisa::span{corpus})
                   << counts_buf.view().copy_from(luisa::span{zeros})
                   << shader(tokens_buf, len, counts_buf).dispatch(len);
            luisa::vector<uint32_t> got(vocab);
            stream << counts_buf.view().copy_to(luisa::span{got}) << synchronize();
            expect(got == expected) << "histogram kernel matches host counts";
        }
    };

    "device_count_table_matches_oracle"_test = [&device] {
        auto stream = device.create_stream();
        Rng32 rng{2024};
        for (auto cfg = 0; cfg < 6; ++cfg) {
            const uint32_t min_n = 1 + rng(3);
            const uint32_t max_n = min_n + rng(3);
            const uint32_t lib_len = 24 + rng(120);
            luisa::vector<uint32_t> corpus(lib_len);
            for (auto &t : corpus) t = rng(5);
            // plant repeated chunks so counts > 1 are common
            for (auto p = 0; p < 6; ++p) {
                const uint32_t src = rng(lib_len / 2);
                const uint32_t dst = lib_len / 2 + rng(lib_len / 2);
                const uint32_t len = 3 + rng(6);
                for (uint32_t i = 0; i < len && src + i < lib_len && dst + i < lib_len; ++i) {
                    corpus[dst + i] = corpus[src + i];
                }
            }
            auto lib = make_id_library(luisa::span{corpus});
            NgramTrainOptions opt;
            opt.min_n = min_n;
            opt.max_n = max_n;
            NgramTrainer trainer{device, stream, lib, opt};
            trainer.download_counts();
            const auto &index = trainer.index();
            const auto oracle = reference_count_ngrams(luisa::span{corpus}, min_n, max_n);

            // full-map comparison: every oracle window finds its slot with the
            // exact device count and the first-occurrence position
            for (auto &[key, expected_count] : oracle) {
                const auto win = decode_ngram_key(key);
                const uint32_t n = static_cast<uint32_t>(win.size());
                const uint32_t p = first_occurrence(luisa::span{corpus}, luisa::span{win});
                expect(p != ngram_invalid_id);
                if (p == ngram_invalid_id) continue;
                const uint32_t slot = host_probe_slot(luisa::span{index.keys},
                                                      luisa::span{index.pos},
                                                      index.cap_log2,
                                                      luisa::span{corpus}, p, n);
                expect(slot != ngram_invalid_id);
                if (slot == ngram_invalid_id) continue;
                expect(index.counts_host[slot] == expected_count);
                expect(index.pos[slot] == p) << "representative position is the first occurrence";
            }
            // no extra entries: occupied slots == oracle map size
            uint32_t occupied = 0;
            for (auto kv : index.keys) occupied += kv != 0ull ? 1u : 0u;
            expect(static_cast<size_t>(occupied) == oracle.size());

            // prefix-count identity (the Add-k denominator):
            // count(ctx) == sum_w count(ctx, w) for every queryable prefix
            for (uint32_t n = min_n; n <= max_n && n < lib_len; ++n) {
                for (uint32_t p = 0; p + n < lib_len; ++p) {
                    const luisa::span<const uint32_t> ctx{corpus.data() + p, n};
                    if (first_occurrence(luisa::span{corpus}, ctx) != p) continue;// unique prefixes only
                    uint32_t sum_w = 0;
                    for (auto &[key, c] : oracle) {
                        const auto win = decode_ngram_key(key);
                        if (win.size() != n + 1u) continue;
                        bool prefix = true;
                        for (uint32_t j = 0; prefix && j < n; ++j) prefix = win[j] == corpus[p + j];
                        if (prefix) sum_w += c;
                    }
                    // corpus-end edge of the scheme: for |ctx| < max_n a
                    // match whose continuation is the corpus's FINAL token
                    // (occurrence at lib_len - n - 1) is counted in count(ctx)
                    // but its continuation window is not indexed; for
                    // |ctx| == max_n the identity is exact
                    uint32_t end_occurrence = 0u;
                    if (n < max_n) {
                        const uint32_t pe = lib_len - n - 1u;
                        bool same = true;
                        for (uint32_t j = 0; same && j < n; ++j) {
                            same = corpus[pe + j] == corpus[p + j];
                        }
                        end_occurrence = same ? 1u : 0u;
                    }
                    expect(reference_count(oracle, ctx) == sum_w + end_occurrence)
                        << "count(ctx) == sum_w count(ctx, w) (+ corpus-end edge)";
                }
            }

            // determinism: recounting from zero reproduces the identical table
            const luisa::vector<uint32_t> first = index.counts_host;
            trainer.reset_counts();
            trainer.dispatch_count();
            stream << synchronize();
            trainer.download_counts();
            expect(index.counts_host == first) << "counting is deterministic";
        }
    };

    "device_existing_variants_unchanged_with_trained_index"_test = [&device] {
        auto stream = device.create_stream();
        struct Case {
            luisa::vector<uint32_t> tokens;
            uint32_t min_n, max_n, k;
        };
        const luisa::vector<Case> cases = {
            {{1, 2, 3, 4, 1, 2, 3, 5, 6}, 2, 2, 2},
            {{1, 2, 3, 4, 1, 2, 3}, 2, 2, 3},
            {{1, 3, 6, 2, 3, 4, 1, 2, 3}, 2, 2, 3},
            {{1, 3, 6, 2, 3, 4, 1, 2, 3}, 1, 1, 2},
            {{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}, 3, 4, 2},
            {{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3}, 3, 3, 2},
        };
        for (auto &cs : cases) {
            auto lib = make_id_library(luisa::span{cs.tokens});
            NgramTrainOptions opt;
            opt.min_n = cs.min_n;
            opt.max_n = cs.max_n;
            NgramTrainer trainer{device, stream, lib, opt};
            luisa::vector<luisa::vector<uint32_t>> queries;
            queries.emplace_back(cs.tokens.begin(), cs.tokens.end());
            // the hash kernel bound to the TRAINED count index must reproduce
            // the classic oracle exactly (same keys/probe/positions)
            const uint32_t failures = check_queries_against_reference(
                device, stream, lib, luisa::span{queries},
                cs.min_n, cs.max_n, cs.k,
                NgramKernelVariant::hash, 512u, &trainer.index());
            expect(failures == 0u) << "hash variant on the trained index matches the classic oracle";
        }
        // fuzz: trained-index hash retrieval on random planted-repeat corpora
        Rng32 rng{555};
        for (auto cfg = 0; cfg < 8; ++cfg) {
            const uint32_t min_n = 1 + rng(3);
            const uint32_t max_n = min_n + rng(3);
            const uint32_t k = 1 + rng(5);
            const uint32_t lib_len = 32 + rng(160);
            luisa::vector<uint32_t> corpus(lib_len);
            for (auto &t : corpus) t = rng(6);
            for (auto p = 0; p < 6; ++p) {
                const uint32_t src = rng(lib_len / 2);
                const uint32_t dst = lib_len / 2 + rng(lib_len / 2);
                const uint32_t len = 4 + rng(8);
                for (uint32_t i = 0; i < len && src + i < lib_len && dst + i < lib_len; ++i) {
                    corpus[dst + i] = corpus[src + i];
                }
            }
            auto lib = make_id_library(luisa::span{corpus});
            NgramTrainOptions opt;
            opt.min_n = min_n;
            opt.max_n = max_n;
            NgramTrainer trainer{device, stream, lib, opt};
            luisa::vector<luisa::vector<uint32_t>> queries;
            for (auto q = 0; q < 16; ++q) {
                if (rng(4) != 0u) {
                    const uint32_t begin = rng(lib_len);
                    const uint32_t len = 1 + rng(16);
                    queries.emplace_back(corpus.begin() + begin,
                                         corpus.begin() + std::min(begin + len, lib_len));
                } else {
                    const uint32_t len = 1 + rng(8);
                    luisa::vector<uint32_t> rq(len);
                    for (auto &t : rq) t = rng(6);
                    queries.emplace_back(std::move(rq));
                }
            }
            const uint32_t failures = check_queries_against_reference(
                device, stream, lib, luisa::span{queries}, min_n, max_n, k,
                NgramKernelVariant::hash, 512u, &trainer.index());
            expect(failures == 0u) << "fuzz: trained-index hash retrieval matches the classic oracle";
        }
    };

    "device_draft_scoring_matches_oracle"_test = [&device] {
        auto stream = device.create_stream();
        const luisa::vector<uint32_t> corpus = {1, 2, 3, 100, 1, 2, 3, 200,
                                                1, 2, 3, 200, 4, 5, 6};
        auto lib = make_id_library(luisa::span{corpus});
        const uint32_t min_n = 2, max_n = 3, k = 4;
        NgramTrainOptions opt;
        opt.min_n = min_n;
        opt.max_n = max_n;
        opt.add_k = 0.1f;
        NgramTrainer trainer{device, stream, lib, opt};
        const auto counts = reference_count_ngrams(luisa::span{corpus}, min_n, max_n);

        // batch 1: drafts produced by actual retrieval (hash on the trained index)
        const luisa::vector<luisa::vector<uint32_t>> queries = {
            {1, 2, 3},// 3-gram match at 0 -> draft {100, 1, 2, 3}
            {9, 1, 2},// 2-gram match at 0 -> draft {3, 100, 1, 2}
            {7, 7},   // no match -> empty draft
        };
        NgramRetriever retriever{device, stream, lib, min_n, max_n, k, 8,
                                 queries.size(), NgramKernelVariant::hash, 512u,
                                 &trainer.index()};
        luisa::vector<uint32_t> drafts, draft_lens;
        retriever.retrieve(luisa::span{queries}, drafts, draft_lens);
        const auto packed = pack_rows(luisa::span{queries});
        luisa::vector<float> scores;
        trainer.score_drafts(luisa::span{packed.flat}, luisa::span{packed.lens},
                             packed.stride, luisa::span{drafts},
                             luisa::span{draft_lens}, k, scores);
        for (size_t i = 0; i < queries.size(); ++i) {
            const luisa::span<const uint32_t> draft_span{drafts.data() + i * k, draft_lens[i]};
            const float expected = reference_draft_log2prob(
                counts, lib.vocab_size, luisa::span{queries[i]}, draft_span, max_n, opt.add_k);
            expect(std::abs(scores[i] - expected) <= 1e-4f) << "draft log2prob matches the oracle";
        }
        expect(scores[2] == 0.0f) << "empty draft scores exactly 0";

        // batch 2: hand-made drafts exercise the unseen-context floor
        // (queries/drafts need not come from retrieval)
        const luisa::vector<luisa::vector<uint32_t>> mq = {{77, 88}, {1, 2, 3}};
        const luisa::vector<luisa::vector<uint32_t>> md = {{1, 2}, {100, 1, 2, 3}};
        const auto mpacked = pack_rows(luisa::span{mq});
        const uint32_t mk = 4;
        luisa::vector<uint32_t> mdrafts(mq.size() * mk, ngram_invalid_id);
        luisa::vector<uint32_t> mdraft_lens(mq.size());
        for (size_t i = 0; i < mq.size(); ++i) {
            mdraft_lens[i] = static_cast<uint32_t>(md[i].size());
            std::copy_n(md[i].begin(), md[i].size(), mdrafts.begin() + i * mk);
        }
        trainer.score_drafts(luisa::span{mpacked.flat}, luisa::span{mpacked.lens},
                             mpacked.stride, luisa::span{mdrafts},
                             luisa::span{mdraft_lens}, mk, scores);
        for (size_t i = 0; i < mq.size(); ++i) {
            const float expected = reference_draft_log2prob(
                counts, lib.vocab_size, luisa::span{mq[i]}, luisa::span{md[i]}, max_n, opt.add_k);
            expect(std::abs(scores[i] - expected) <= 1e-4f);
        }
        // the {77, 88} row never occurs in the corpus: both positions floor at 1/V
        const float floor_lp = 2.0f * std::log2(1.0f / static_cast<float>(lib.vocab_size));
        expect(std::abs(scores[0] - floor_lp) <= 1e-4f) << "unseen context floors at 1/V";
    };

    "device_parallel_mle_matches_oracle"_test = [&device] {
        auto stream = device.create_stream();
        // (a) hand-crafted: the earliest match has the RARER continuation;
        // parallel_mle must propose the most frequent continuation instead
        {
            const luisa::vector<uint32_t> corpus = {1, 2, 3, 200, 1, 2, 3, 100, 1, 2, 3, 100};
            auto lib = make_id_library(luisa::span{corpus});
            NgramTrainOptions opt;
            opt.min_n = 2;
            opt.max_n = 3;
            NgramTrainer trainer{device, stream, lib, opt};
            const auto counts = reference_count_ngrams(luisa::span{corpus}, 2, 3);
            const luisa::vector<luisa::vector<uint32_t>> queries = {{1, 2, 3}};
            expect(check_queries_against_mle_reference(
                       device, stream, lib, trainer.index(), counts,
                       luisa::span{queries}, 2, 3, 3) == 0u);
            // ... and it really diverges from the classic earliest-position draft
            NgramRetriever classic{device, stream, lib, 2, 3, 3, 8, 1,
                                   NgramKernelVariant::parallel, 512u};
            NgramRetriever mle{device, stream, lib, 2, 3, 3, 8, 1,
                               NgramKernelVariant::parallel_mle, 512u, &trainer.index()};
            luisa::vector<uint32_t> drafts_c, lens_c, drafts_m, lens_m;
            classic.retrieve(luisa::span{queries}, drafts_c, lens_c);
            mle.retrieve(luisa::span{queries}, drafts_m, lens_m);
            expect(lens_c[0] == 3u && lens_m[0] == 3u);
            expect(drafts_c[0] == 200u) << "classic: earliest position";
            expect(drafts_m[0] == 100u) << "mle: most frequent continuation";
        }
        // (b) vLLM vectors through parallel_mle
        {
            struct Case {
                luisa::vector<uint32_t> tokens;
                uint32_t min_n, max_n, k;
            };
            const luisa::vector<Case> cases = {
                {{1, 2, 3, 4, 1, 2, 3, 5, 6}, 2, 2, 2},
                {{1, 2, 3, 4, 1, 2, 3}, 2, 2, 3},
                {{1, 2, 3, 4, 1, 2, 3}, 2, 2, 2},
                {{1, 3, 6, 2, 3, 4, 1, 2, 3}, 2, 2, 3},
                {{1, 3, 6, 2, 3, 4, 1, 2, 3}, 1, 1, 2},
                {{1, 2, 3, 4, 1, 2, 3}, 4, 4, 2},
                {{1, 2, 3, 4, 1, 2, 3}, 3, 4, 2},
                {{2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4}, 3, 4, 2},
                {{3, 4, 5, 2, 3, 4, 1, 2, 3, 4}, 2, 4, 2},
                {{1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3}, 3, 3, 2},
            };
            for (auto &cs : cases) {
                auto lib = make_id_library(luisa::span{cs.tokens});
                NgramTrainOptions opt;
                opt.min_n = cs.min_n;
                opt.max_n = cs.max_n;
                NgramTrainer trainer{device, stream, lib, opt};
                const auto counts = reference_count_ngrams(luisa::span{cs.tokens}, cs.min_n, cs.max_n);
                luisa::vector<luisa::vector<uint32_t>> queries;
                queries.emplace_back(cs.tokens.begin(), cs.tokens.end());
                expect(check_queries_against_mle_reference(
                           device, stream, lib, trainer.index(), counts,
                           luisa::span{queries}, cs.min_n, cs.max_n, cs.k) == 0u)
                    << "parallel_mle matches the MLE oracle on vLLM vectors";
            }
        }
        // (c) planted-repeat fuzz: nontrivial continuation count distributions
        {
            Rng32 rng{31337};
            for (auto cfg = 0; cfg < 8; ++cfg) {
                const uint32_t min_n = 1 + rng(3);
                const uint32_t max_n = min_n + rng(3);
                const uint32_t k = 1 + rng(5);
                const uint32_t lib_len = 48 + rng(200);
                luisa::vector<uint32_t> corpus(lib_len);
                for (auto &t : corpus) t = rng(6);
                for (auto p = 0; p < 10; ++p) {
                    const uint32_t src = rng(lib_len / 2);
                    const uint32_t dst = lib_len / 2 + rng(lib_len / 2);
                    const uint32_t len = 4 + rng(8);
                    for (uint32_t i = 0; i < len && src + i < lib_len && dst + i < lib_len; ++i) {
                        corpus[dst + i] = corpus[src + i];
                    }
                }
                auto lib = make_id_library(luisa::span{corpus});
                NgramTrainOptions opt;
                opt.min_n = min_n;
                opt.max_n = max_n;
                NgramTrainer trainer{device, stream, lib, opt};
                const auto counts = reference_count_ngrams(luisa::span{corpus}, min_n, max_n);
                luisa::vector<luisa::vector<uint32_t>> queries;
                for (auto q = 0; q < 20; ++q) {
                    if (rng(4) != 0u) {
                        const uint32_t begin = rng(lib_len);
                        const uint32_t len = 1 + rng(20);
                        queries.emplace_back(corpus.begin() + begin,
                                             corpus.begin() + std::min(begin + len, lib_len));
                    } else {
                        const uint32_t len = 1 + rng(8);
                        luisa::vector<uint32_t> rq(len);
                        for (auto &t : rq) t = rng(6);
                        queries.emplace_back(std::move(rq));
                    }
                }
                expect(check_queries_against_mle_reference(
                           device, stream, lib, trainer.index(), counts,
                           luisa::span{queries}, min_n, max_n, k) == 0u)
                    << "fuzz: parallel_mle output equals the MLE oracle";
            }
        }
        // (d) degeneration: every n-gram occurs at most once (unique
        // continuations) -> parallel_mle reduces to the classic earliest
        // position, identical to the classic oracle
        {
            luisa::vector<uint32_t> corpus(64);
            for (uint32_t i = 0; i < 64; ++i) corpus[i] = i;
            auto lib = make_id_library(luisa::span{corpus});
            NgramTrainOptions opt;
            opt.min_n = 2;
            opt.max_n = 4;
            NgramTrainer trainer{device, stream, lib, opt};
            luisa::vector<luisa::vector<uint32_t>> queries;
            for (uint32_t begin = 0; begin < 32; begin += 5) {
                queries.emplace_back(corpus.begin() + begin,
                                     corpus.begin() + std::min(begin + 9u, 64u));
            }
            expect(check_queries_against_reference(
                       device, stream, lib, luisa::span{queries}, 2, 4, 5,
                       NgramKernelVariant::parallel_mle, 512u, &trainer.index()) == 0u)
                << "unique continuations degenerate to the classic rule";
        }
    };

    "device_lm_mode_padded_layout_and_counts"_test = [&device] {
        auto stream = device.create_stream();
        // the Python-example sentences as token IDs: 我=0 爱=1 北=2 京=3
        // 学=4 习=5 你=6
        const luisa::vector<luisa::vector<uint32_t>> docs = {
            {0, 1, 2, 3}, {0, 1, 4, 5}, {6, 1, 2, 3}};
        auto lib = make_multi_doc_id_library(luisa::span{docs});
        NgramTrainOptions opt;
        opt.lm_mode = true;
        opt.order = 2;
        opt.add_k = 0.1f;
        NgramTrainer trainer{device, stream, lib, opt};
        expect(trainer.lm_mode());
        expect(trainer.order() == 2u);
        expect(trainer.vocab_effective() == lib.vocab_size + 3u);
        const uint32_t bos = trainer.bos_id();
        const uint32_t eos = trainer.eos_id();
        expect(trainer.unk_id() == lib.vocab_size);
        expect(bos == lib.vocab_size + 1u);
        expect(eos == lib.vocab_size + 2u);
        // padded layout: BOS w1..wL EOS per sentence, concatenated
        const auto padded = trainer.padded_corpus();
        const luisa::vector<uint32_t> expected_padded = {
            bos, 0, 1, 2, 3, eos,
            bos, 0, 1, 4, 5, eos,
            bos, 6, 1, 2, 3, eos};
        expect(padded.size() == expected_padded.size());
        expect(luisa::vector<uint32_t>(padded.begin(), padded.end()) == expected_padded);
        expect(trainer.total_tokens() == expected_padded.size());

        // LM count table vs the ReferenceNgramModel oracle (full map).
        // NOTE: download_counts() only mirrors index(), so the LM counts are
        // downloaded directly from lm_index().counts_buf.
        ReferenceNgramModel oracle;
        oracle.train(padded, 2, bos);
        const auto &lm_index = trainer.lm_index();
        luisa::vector<uint32_t> lm_counts(lm_index.keys.size());
        stream << lm_index.counts_buf.view().copy_to(luisa::span{lm_counts})
               << synchronize();
        for (auto &[key, expected_count] : oracle.counts()) {
            const auto win = decode_ngram_key(key);
            const uint32_t n = static_cast<uint32_t>(win.size());
            const uint32_t p = first_occurrence(padded, luisa::span{win});
            expect(p != ngram_invalid_id);
            if (p == ngram_invalid_id) continue;
            const uint32_t slot = host_probe_slot(luisa::span{lm_index.keys},
                                                  luisa::span{lm_index.pos},
                                                  lm_index.cap_log2,
                                                  padded, p, n);
            expect(slot != ngram_invalid_id);
            if (slot == ngram_invalid_id) continue;
            expect(lm_counts[slot] == expected_count);
        }
        // no extras -> the table holds exactly the oracle windows (which never
        // contain a BOS at offset > 0, i.e. no cross-sentence windows)
        uint32_t occupied = 0;
        for (auto kv : lm_index.keys) occupied += kv != 0ull ? 1u : 0u;
        expect(static_cast<size_t>(occupied) == oracle.counts().size());
    };

    "device_lm_mode_unk_remap"_test = [&device] {
        auto stream = device.create_stream();
        // token 0 occurs 3x, token 1 occurs 2x, token 2 occurs 1x
        const luisa::vector<luisa::vector<uint32_t>> docs = {
            {0, 1, 2}, {0, 1}, {0}};
        auto lib = make_multi_doc_id_library(luisa::span{docs});
        NgramTrainOptions opt;
        opt.lm_mode = true;
        opt.order = 2;
        opt.unk_threshold = 2;// unigram count < 2 -> <unk>
        NgramTrainer trainer{device, stream, lib, opt};
        const uint32_t unk = trainer.unk_id();
        const uint32_t bos = trainer.bos_id();
        const uint32_t eos = trainer.eos_id();
        expect(trainer.lm_remap(0) == 0u);  // 3 >= 2: kept
        expect(trainer.lm_remap(1) == 1u);  // 2 >= 2: kept
        expect(trainer.lm_remap(2) == unk); // 1 < 2: remapped
        expect(trainer.lm_remap(99) == unk) << "out-of-vocabulary maps to <unk>";
        const auto padded = trainer.padded_corpus();
        const luisa::vector<uint32_t> expected_padded = {
            bos, 0, 1, unk, eos,
            bos, 0, 1, eos,
            bos, 0, eos};
        expect(luisa::vector<uint32_t>(padded.begin(), padded.end()) == expected_padded);
        // scoring over the remapped corpus must agree with the oracle
        ReferenceNgramModel oracle;
        oracle.train(padded, 2, bos);
        const float vf = static_cast<float>(trainer.vocab_effective());
        const luisa::vector<luisa::vector<uint32_t>> rows = {
            {bos, 0, 1, unk, eos}, {0, 1}, {1, unk}, {unk, eos}};
        const auto packed = pack_rows(luisa::span{rows});
        luisa::vector<float> scores;
        trainer.score_sentences(luisa::span{packed.flat}, luisa::span{packed.lens}, scores);
        for (size_t i = 0; i < rows.size(); ++i) {
            const float expected = oracle.sentence_log2prob(
                luisa::span{rows[i]}, NgramSmoothing::add_k, opt.add_k, vf);
            expect(std::abs(scores[i] - expected) <= 1e-4f);
        }
    };

    "device_lm_mode_scoring_matches_oracle"_test = [&device] {
        auto stream = device.create_stream();
        const luisa::vector<luisa::vector<uint32_t>> docs = {
            {0, 1, 2, 3}, {0, 1, 4, 5}, {6, 1, 2, 3}};
        auto lib = make_multi_doc_id_library(luisa::span{docs});
        // padded corpus with the default threshold (identity remap)
        const uint32_t bos = lib.vocab_size + 1u;
        const uint32_t eos = lib.vocab_size + 2u;
        const luisa::vector<uint32_t> padded = {
            bos, 0, 1, 2, 3, eos,
            bos, 0, 1, 4, 5, eos,
            bos, 6, 1, 2, 3, eos};
        ReferenceNgramModel oracle;
        oracle.train(luisa::span{padded}, 2, bos);
        const float vf = 10.0f;

        for (auto smoothing : {NgramSmoothing::add_k, NgramSmoothing::backoff}) {
            NgramTrainOptions opt;
            opt.lm_mode = true;
            opt.order = 2;
            opt.add_k = 0.1f;
            opt.smoothing = smoothing;
            NgramTrainer trainer{device, stream, lib, opt};
            // rows: padded sentences + conditional rows + boundary cases
            const luisa::vector<luisa::vector<uint32_t>> rows = {
                {bos, 0, 1, 2, 3, eos},
                {bos, 0, 1, 4, 5, eos},
                {bos, 6, 1, 2, 3, eos},
                {0, 1},     // P(爱|我)
                {6, 4},     // P(学习|你): unseen bigram
                {bos, eos}, // sentence boundary only
                {5},        // single token: score 0
            };
            const auto packed = pack_rows(luisa::span{rows});
            luisa::vector<float> scores;
            trainer.score_sentences(luisa::span{packed.flat}, luisa::span{packed.lens}, scores);
            for (size_t i = 0; i < rows.size(); ++i) {
                const float expected = oracle.sentence_log2prob(
                    luisa::span{rows[i]}, smoothing, opt.add_k, vf);
                expect(std::abs(scores[i] - expected) <= 1e-4f)
                    << "device sentence log2prob matches the oracle";
            }
            expect(scores[6] == 0.0f) << "single-token row scores exactly 0";
            // conditional_prob convenience wrapper: P(爱|我)
            const luisa::vector<uint32_t> prefix = {0u};
            const float p = trainer.conditional_prob(luisa::span{prefix}, 1u);
            const float expected_p = std::exp2(oracle.sentence_log2prob(
                luisa::span{rows[3]}, smoothing, opt.add_k, vf));
            expect(std::abs(p - expected_p) <= 1e-5f);
            // corpus perplexity over the 3 padded sentences
            const luisa::vector<luisa::vector<uint32_t>> sents = {rows[0], rows[1], rows[2]};
            const auto spacked = pack_rows(luisa::span{sents});
            const double ppl = trainer.perplexity(luisa::span{spacked.flat},
                                                  luisa::span{spacked.lens});
            const double ppl_ref = oracle.perplexity(luisa::span{sents}, smoothing,
                                                     opt.add_k, vf);
            expect(std::abs(ppl - ppl_ref) <= 1e-4) << "perplexity matches the oracle";
            if (smoothing == NgramSmoothing::add_k) {
                expect(std::abs(ppl - 1.833446) <= 1e-3) << "Python-example perplexity";
            }
        }
    };

    "device_lm_mode_fuzz_vs_reference"_test = [&device] {
        auto stream = device.create_stream();
        Rng32 rng{999};
        for (auto cfg = 0; cfg < 8; ++cfg) {
            const uint32_t order = 1 + rng(4);// orders 1..4
            const auto smoothing = rng(2) == 0u ? NgramSmoothing::add_k
                                                : NgramSmoothing::backoff;
            const uint32_t unk_threshold = 1 + rng(2);
            const uint32_t num_docs = 2 + rng(4);
            luisa::vector<luisa::vector<uint32_t>> docs(num_docs);
            for (auto &d : docs) {
                d.resize(3 + rng(6));
                for (auto &t : d) t = rng(5);
            }
            auto lib = make_multi_doc_id_library(luisa::span{docs});
            NgramTrainOptions opt;
            opt.lm_mode = true;
            opt.order = order;
            opt.smoothing = smoothing;
            opt.add_k = 0.1f;
            opt.unk_threshold = unk_threshold;
            NgramTrainer trainer{device, stream, lib, opt};
            // oracle over the trainer's OWN padded corpus (unk remap applied)
            ReferenceNgramModel oracle;
            oracle.train(trainer.padded_corpus(), order, trainer.bos_id());
            const float vf = static_cast<float>(trainer.vocab_effective());
            // random scoring rows in the padded token space (len >= 1)
            const uint32_t veff = trainer.vocab_effective();
            luisa::vector<luisa::vector<uint32_t>> rows(12);
            for (auto &r : rows) {
                r.resize(1 + rng(8));
                for (auto &t : r) {
                    // draw raw tokens + BOS/EOS (UNK itself is never an input
                    // token -- it is what rare/OOV tokens map TO; with
                    // threshold 1 its count is 0 and the backoff floor would
                    // be exactly 0)
                    t = rng(veff - 1u);
                    if (t >= trainer.unk_id()) {
                        t += 1u;// skip the unk slot: {bos, eos}
                    } else {
                        t = trainer.lm_remap(t);
                    }
                }
            }
            const auto packed = pack_rows(luisa::span{rows});
            luisa::vector<float> scores;
            trainer.score_sentences(luisa::span{packed.flat}, luisa::span{packed.lens}, scores);
            for (size_t i = 0; i < rows.size(); ++i) {
                const float expected = oracle.sentence_log2prob(
                    luisa::span{rows[i]}, smoothing, opt.add_k, vf);
                // exact match (covers -inf == -inf) or within float tolerance
                expect((scores[i] == expected) |
                       (std::abs(scores[i] - expected) <= 1e-4f))
                    << "fuzz: device sentence log2prob matches the oracle";
            }
            // perplexity over the rows with len >= 2 (sum(m) > 0 guaranteed)
            luisa::vector<luisa::vector<uint32_t>> ppl_rows;
            for (auto &r : rows) {
                if (r.size() >= 2) ppl_rows.push_back(r);
            }
            if (!ppl_rows.empty()) {
                const auto ppacked = pack_rows(luisa::span{ppl_rows});
                const double ppl = trainer.perplexity(luisa::span{ppacked.flat},
                                                      luisa::span{ppacked.lens});
                const double ppl_ref = oracle.perplexity(luisa::span{ppl_rows}, smoothing,
                                                         opt.add_k, vf);
                expect((ppl == ppl_ref) | (std::abs(ppl - ppl_ref) <= 1e-3)) << "fuzz: perplexity matches";
            }
        }
    };
}

}// namespace

int run_tests(int argc, char *argv[]) {
    // Hand Boost.UT a clean argv (executable + backend only): the "--test"
    // dispatch flag would otherwise be parsed as a test-name filter.
    luisa::vector<const char *> ut_argv;
    ut_argv.push_back(argv[0]);
    if (argc > 1) ut_argv.push_back(argv[1]);
    const int ut_argc = static_cast<int>(ut_argv.size());

    auto dc = luisa::test::create_device_from_ut(ut_argc, const_cast<char **>(ut_argv.data()));
    if (!dc) {
        return 1;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(ut_argc, ut_argv.data());

    register_host_tests();
    register_device_tests(dc->device);

    const bool failed = boost::ut::cfg().run({.report_errors = true,
                                              .argc = ut_argc,
                                              .argv = ut_argv.data()});
    return failed ? 1 : 0;
}

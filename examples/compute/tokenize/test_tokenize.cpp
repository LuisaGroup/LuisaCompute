// Function tests for the n-gram tokenize + retrieve example.
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
//   - the host -> device -> host library buffer round-trip.

#include "ut/ut.hpp"
#include "test_device.h"

#include "ngram_library.h"
#include "ngram_retriever.h"
#include "ngram_tokenizer.h"
#include "reference.h"

#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <random>

using namespace luisa;
using namespace luisa::compute;
using namespace tokenize;
using namespace boost::ut;

namespace {

// ---------- helpers ----------

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

// Run one batch of queries through `variant` and check every draft against
// the reference oracle. Returns the number of failed checks.
[[nodiscard]] uint32_t check_queries_against_reference(
    Device &device, Stream &stream, NgramLibrary &lib,
    luisa::span<const luisa::vector<uint32_t>> queries,
    uint32_t min_n, uint32_t max_n, uint32_t k,
    NgramKernelVariant variant, uint32_t block_size = 512u) {
    uint32_t max_query_len = max_n;
    for (auto &q : queries) max_query_len = std::max(max_query_len, (uint32_t)q.size());
    NgramRetriever retriever{device, stream, lib, min_n, max_n, k,
                             max_query_len, queries.size(), variant, block_size};
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
        std::mt19937 rng{42};
        for (auto iter = 0; iter < 200; ++iter) {
            const uint32_t n = 1 + rng() % 48;
            luisa::vector<uint32_t> corpus(n);
            for (auto &t : corpus) t = rng() % 6;
            // plant a copy of an earlier chunk to create real matches
            if (n > 8 && (rng() & 1)) {
                uint32_t src = rng() % (n / 2);
                uint32_t dst = n / 2 + rng() % (n / 2);
                uint32_t len = std::min(3u + rng() % 4u, n - std::max(src, dst));
                for (uint32_t i = 0; i + 1 < len; ++i) corpus[dst + i] = corpus[src + i];
            }
            const uint32_t min_n = 1 + rng() % 2;
            const uint32_t max_n = min_n + rng() % 3;
            const uint32_t k = 1 + rng() % 5;
            auto via_kmp = reference_retrieve_vllm_kmp(luisa::span{corpus}, min_n, max_n, k);
            auto via_lit = reference_retrieve(luisa::span{corpus}, luisa::span{corpus}, min_n, max_n, k);
            expect(via_kmp == via_lit) << "kmp port matches literal oracle";
        }
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
        std::mt19937 rng{1337};
        for (auto cfg = 0; cfg < 16; ++cfg) {
            const uint32_t min_n = 1 + rng() % 3;
            const uint32_t max_n = min_n + rng() % 4;
            const uint32_t k = 1 + rng() % 5;
            const uint32_t lib_len = 48 + rng() % 400;
            luisa::vector<uint32_t> corpus(lib_len);
            for (auto &t : corpus) t = rng() % 7;
            // plant copies of earlier chunks so matches are common
            for (auto p = 0; p < 8; ++p) {
                uint32_t src = rng() % (lib_len / 2);
                uint32_t dst = lib_len / 2 + rng() % (lib_len / 2);
                uint32_t len = 4 + rng() % 8;
                for (uint32_t i = 0; i < len && src + i < lib_len && dst + i < lib_len; ++i) {
                    corpus[dst + i] = corpus[src + i];
                }
            }
            auto lib = make_id_library(luisa::span{corpus});
            luisa::vector<luisa::vector<uint32_t>> queries;
            for (auto q = 0; q < 24; ++q) {
                // mostly corpus suffixes, sometimes unrelated random tokens
                if ((rng() % 4) != 0) {
                    uint32_t begin = rng() % lib_len;
                    uint32_t len = 1 + rng() % 24;
                    queries.emplace_back(corpus.begin() + begin,
                                         corpus.begin() + std::min(begin + len, lib_len));
                } else {
                    uint32_t len = 1 + rng() % 8;
                    luisa::vector<uint32_t> rq(len);
                    for (auto &t : rq) t = rng() % 7;
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

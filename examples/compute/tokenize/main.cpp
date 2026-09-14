// N-gram tokenize + retrieve example.
//
// Ports vLLM's ngram speculative-decoding "prompt lookup" proposer
// (vllm/v1/spec_decode/ngram_proposer.py) to a LuisaCompute DSL kernel.
//
// A corpus of documents is tokenized into a flat token-ID library
// (NgramLibrary) and uploaded to device buffers. Retrieval —
// finding the earliest occurrence of a query's trailing n-gram and
// extracting the k tokens that follow it — runs ENTIRELY in a DSL kernel;
// the host only does I/O, buffer management and dispatch.
//
// Usage:
//   tokenizer <backend>                     run the small demo
//   tokenizer <backend> --test              run the function test suite
//   tokenizer <backend> --benchmark [opts]  run the performance benchmark
//     benchmark options:
//       --kernel naive|parallel|hash  kernel variant (default: parallel)
//       --lib-size N                  corpus size in tokens (default: 4194304)
//       --queries N                   query batch size (default: 1024)
//       --reps N                      timed repetitions (default: 20)
//       --min-n N --max-n N --k N     retrieval parameters (2 / 4 / 5)
//       --block-size N                threads per query block, parallel kernel
//       --vocab N --seed N            corpus generator knobs

#include "ngram_library.h"
#include "ngram_retriever.h"
#include "reference.h"

#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Implemented in test_tokenize.cpp / benchmark_tokenize.cpp.
int run_tests(int argc, char *argv[]);
int run_benchmark(int argc, char *argv[]);

namespace {

using namespace tokenize;

// Small end-to-end demo: build a token library from a few documents,
// retrieve drafts for trailing-token queries with every kernel variant
// and cross-check the results against the CPU reference oracle.
int run_demo(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0') {
        LUISA_ERROR("demo mode requires a backend argument");
    }
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    NgramLibrary library;
    library.add_document("the quick brown fox jumps over the lazy dog");
    library.add_document("the quick brown fox runs and the quick brown fox sleeps");
    library.add_document("gpu compute shaders are fast and gpu compute shaders are fun");
    library.finalize();
    LUISA_INFO("library: {} tokens, {} docs, vocab {}",
               library.size(), library.num_docs(), library.vocab_size);

    // queries: trailing token sequences taken from the corpus
    const auto &tokens = library.tokens;
    luisa::vector<luisa::vector<uint32_t>> queries;
    queries.emplace_back(tokens.end() - 4, tokens.end());  // "fox sleeps" + ...
    queries.emplace_back(tokens.end() - 2, tokens.end());  // short trailing suffix
    queries.emplace_back(tokens.begin() + 3, tokens.begin() + 9);

    constexpr uint32_t min_n = 2, max_n = 3, k = 4;
    uint32_t max_query_len = 0;
    for (auto &q : queries) max_query_len = std::max(max_query_len, (uint32_t)q.size());

    for (auto variant : {NgramKernelVariant::naive, NgramKernelVariant::parallel,
                         NgramKernelVariant::hash}) {
        NgramRetriever retriever{device, stream, library, min_n, max_n, k,
                                 max_query_len, queries.size(), variant};
        luisa::vector<uint32_t> drafts, draft_lens;
        retriever.retrieve(luisa::span{queries}, drafts, draft_lens);
        auto mismatch = 0u;
        const char *variant_name = variant == NgramKernelVariant::naive ? "naive"
                                   : variant == NgramKernelVariant::parallel ? "parallel"
                                                                             : "hash    ";
        for (size_t i = 0; i < queries.size(); ++i) {
            auto expected = reference_retrieve(luisa::span{library.tokens},
                                               luisa::span{queries[i]}, min_n, max_n, k);
            luisa::vector<uint32_t> got(drafts.begin() + i * k,
                                        drafts.begin() + i * k + draft_lens[i]);
            auto ok = got == expected;
            if (!ok) ++mismatch;
            LUISA_INFO("  [{}] query_len={} draft_len={} draft=[{}] {} (expected [{}])",
                       variant_name,
                       queries[i].size(), draft_lens[i],
                       [&] {
                           luisa::string s;
                           for (auto t : got) s.append(luisa::format("{} ", t));
                           return s;
                       }(),
                       ok ? "ok" : "MISMATCH",
                       [&] {
                           luisa::string s;
                           for (auto t : expected) s.append(luisa::format("{} ", t));
                           return s;
                       }());
        }
        if (mismatch != 0u) {
            LUISA_WARNING("demo: {} mismatches for variant", mismatch);
            return 1;
        }
    }
    LUISA_INFO("demo: all variants match the reference oracle");
    return 0;
}

void print_usage(const char *exe) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s <backend>                     demo\n"
                 "  %s <backend> --test              function tests\n"
                 "  %s <backend> --benchmark [opts]  benchmark\n"
                 "    --kernel naive|parallel|hash  kernel variant (default: parallel)\n"
                 "    --lib-size N                  corpus tokens (default: 4194304)\n"
                 "    --queries N                   query batch (default: 1024)\n"
                 "    --reps N                      timed repetitions (default: 20)\n"
                 "    --min-n N --max-n N --k N     retrieval parameters (2 / 4 / 5)\n"
                 "    --block-size N                threads per query, parallel kernel\n"
                 "    --vocab N --seed N            corpus generator knobs\n"
                 "  benchmark multi-request concurrency (stages of N requests):\n"
                 "    --requests N                  request count (0 = off, default)\n"
                 "    --req-queries N               queries per request (default: --queries)\n"
                 "    --multi-mode seq|pipeline|fiber|multi|all  (default: all)\n",
                 exe, exe, exe);
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0' || argv[1][0] == '-') {
        print_usage(argv[0]);
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0) {
            return run_tests(argc, argv);
        }
        if (std::strcmp(argv[i], "--benchmark") == 0) {
            return run_benchmark(argc, argv);
        }
    }
    return run_demo(argc, argv);
}

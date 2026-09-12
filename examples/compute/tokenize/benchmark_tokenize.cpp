// Performance benchmark for the n-gram tokenize + retrieve example.
//
// Generates a synthetic corpus of repeated token patterns (vLLM-like high
// hit rate), samples a batch of corpus-suffix queries, then times the pure
// device dispatch (upload once, then dispatch + synchronize per rep,
// download once). Device output is spot-checked against the CPU reference
// oracle before timing, and single-threaded CPU reference timings are
// reported for scale.
//
// Options: --kernel naive|parallel --lib-size N --queries N --reps N
//          --min-n N --max-n N --k N --block-size N --vocab N --seed N

#include "ngram_library.h"
#include "ngram_retriever.h"
#include "reference.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

using namespace tokenize;

namespace {

struct BenchOptions {
    NgramKernelVariant variant = NgramKernelVariant::parallel;
    uint32_t lib_size = 4194304u;
    uint32_t num_queries = 1024u;
    uint32_t reps = 20u;
    uint32_t min_n = 2u;
    uint32_t max_n = 4u;
    uint32_t k = 5u;
    uint32_t block_size = 512u;
    uint32_t vocab = 4096u;
    uint32_t seed = 12345u;
};

void parse_options(int argc, char *argv[], BenchOptions &opt) {
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0) continue;
        auto need = [&](const char *name) -> uint32_t {
            if (i + 1 >= argc) {
                LUISA_ERROR("missing value for {}", name);
            }
            return static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        };
        if (std::strcmp(argv[i], "--kernel") == 0) {
            if (i + 1 >= argc) LUISA_ERROR("missing value for --kernel");
            const char *v = argv[++i];
            if (std::strcmp(v, "naive") == 0) {
                opt.variant = NgramKernelVariant::naive;
            } else if (std::strcmp(v, "parallel") == 0) {
                opt.variant = NgramKernelVariant::parallel;
            } else if (std::strcmp(v, "hash") == 0) {
                opt.variant = NgramKernelVariant::hash;
            } else {
                LUISA_ERROR("unknown kernel '{}'", v);
            }
        } else if (std::strcmp(argv[i], "--lib-size") == 0) {
            opt.lib_size = need("--lib-size");
        } else if (std::strcmp(argv[i], "--queries") == 0) {
            opt.num_queries = need("--queries");
        } else if (std::strcmp(argv[i], "--reps") == 0) {
            opt.reps = need("--reps");
        } else if (std::strcmp(argv[i], "--min-n") == 0) {
            opt.min_n = need("--min-n");
        } else if (std::strcmp(argv[i], "--max-n") == 0) {
            opt.max_n = need("--max-n");
        } else if (std::strcmp(argv[i], "--k") == 0) {
            opt.k = need("--k");
        } else if (std::strcmp(argv[i], "--block-size") == 0) {
            opt.block_size = need("--block-size");
        } else if (std::strcmp(argv[i], "--vocab") == 0) {
            opt.vocab = need("--vocab");
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            opt.seed = need("--seed");
        } else if (argv[i][0] == '-') {
            LUISA_ERROR("unknown benchmark option '{}'", argv[i]);
        }
        // positional args other than argv[1] (backend) are ignored
    }
}

[[nodiscard]] luisa::vector<uint32_t> generate_corpus(uint32_t lib_size,
                                                      uint32_t vocab,
                                                      uint32_t seed) {
    luisa::vector<uint32_t> corpus(lib_size);
    std::mt19937 rng{seed};
    uint32_t i = 0;
    while (i < lib_size) {
        const uint32_t pattern_len = 8u + rng() % 25u;// 8..32
        const uint32_t step = 1u + rng() % 4u;
        const uint32_t base = rng() % vocab;
        for (uint32_t j = 0; j < pattern_len && i < lib_size; ++j) {
            corpus[i++] = (base + j * step) % vocab;
        }
    }
    return corpus;
}

[[nodiscard]] NgramLibrary make_benchmark_library(luisa::span<const uint32_t> corpus) {
    NgramLibrary lib;
    lib.tokens.assign(corpus.begin(), corpus.end());
    lib.doc_offsets.push_back(0u);
    lib.doc_lengths.push_back(static_cast<uint32_t>(corpus.size()));
    lib.vocab_size = 0;
    for (auto t : corpus) lib.vocab_size = std::max(lib.vocab_size, t + 1u);
    lib.finalize();
    return lib;
}

int run_benchmark_impl(int argc, char *argv[]) {
    BenchOptions opt;
    parse_options(argc, argv, opt);
    LUISA_ASSERT(opt.min_n >= 1u && opt.min_n <= opt.max_n, "invalid min_n/max_n");
    LUISA_ASSERT(opt.k >= 1u, "invalid k");
    LUISA_ASSERT(opt.lib_size > 0u && opt.num_queries > 0u, "sizes must be > 0");

    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    auto corpus = generate_corpus(opt.lib_size, opt.vocab, opt.seed);
    auto lib = make_benchmark_library(luisa::span{corpus});

    // queries: random suffixes from the corpus (length >= max_n)
    std::mt19937 rng{opt.seed ^ 0x5bd1e995u};
    luisa::vector<uint32_t> query_lens(opt.num_queries);
    uint32_t max_query_len = opt.max_n;
    for (auto &len : query_lens) {
        len = opt.max_n + rng() % 8u;
        max_query_len = std::max(max_query_len, len);
    }
    luisa::vector<uint32_t> queries_flat(opt.num_queries * max_query_len, 0u);
    for (uint32_t q = 0; q < opt.num_queries; ++q) {
        uint32_t end = opt.lib_size / 2 + rng() % (opt.lib_size - opt.lib_size / 2);
        uint32_t begin = end - query_lens[q];
        std::copy_n(corpus.begin() + begin, query_lens[q],
                    queries_flat.begin() + q * max_query_len);
    }

    const char *variant_name = "parallel";
    if (opt.variant == NgramKernelVariant::naive) variant_name = "naive";
    if (opt.variant == NgramKernelVariant::hash) variant_name = "hash";
    LUISA_INFO("benchmark: kernel={} block_size={} lib={} tokens queries={} "
               "min_n={} max_n={} k={} reps={}",
               variant_name, opt.block_size, opt.lib_size, opt.num_queries,
               opt.min_n, opt.max_n, opt.k, opt.reps);

    NgramRetriever retriever{device, stream, lib, opt.min_n, opt.max_n, opt.k,
                             max_query_len, opt.num_queries, opt.variant, opt.block_size};

    // ---- correctness spot-check before timing ----
    retriever.upload_queries(luisa::span{queries_flat}, luisa::span{query_lens});
    retriever.dispatch_queries(opt.num_queries);
    retriever.synchronize();
    luisa::vector<uint32_t> drafts, draft_lens;
    retriever.download_results(opt.num_queries, drafts, draft_lens);
    const uint32_t check_n = std::min(opt.num_queries, 64u);
    uint32_t mismatches = 0;
    for (uint32_t q = 0; q < check_n; ++q) {
        auto expected = reference_retrieve(luisa::span{lib.tokens},
                                           luisa::span{queries_flat}.subspan(q * max_query_len, query_lens[q]),
                                           opt.min_n, opt.max_n, opt.k);
        if (draft_lens[q] != expected.size()) {
            ++mismatches;
            continue;
        }
        for (size_t j = 0; j < expected.size(); ++j) {
            if (drafts[q * opt.k + j] != expected[j]) ++mismatches;
        }
    }
    if (mismatches != 0u) {
        LUISA_ERROR("benchmark aborted: {} device/reference mismatches in spot-check", mismatches);
    }
    LUISA_INFO("spot-check: {} queries match the reference oracle", check_n);

    // ---- timed dispatch + synchronize ----
    for (uint32_t w = 0; w < 3; ++w) {// warmup
        retriever.dispatch_queries(opt.num_queries);
        retriever.synchronize();
    }
    luisa::Clock clock;
    double best = 1e30, total = 0.0;
    for (uint32_t r = 0; r < opt.reps; ++r) {
        clock.tic();
        retriever.dispatch_queries(opt.num_queries);
        retriever.synchronize();
        const double ms = clock.toc();
        best = std::min(best, ms);
        total += ms;
    }
    const double avg_ms = total / static_cast<double>(opt.reps);
    const double per_query_us = avg_ms * 1000.0 / static_cast<double>(opt.num_queries);
    const double queries_per_s = static_cast<double>(opt.num_queries) / (avg_ms / 1000.0);
    const double corpus_gbps = static_cast<double>(opt.lib_size) / (avg_ms / 1000.0) / 1e9;

    // ---- CPU reference timing (subset) ----
    const uint32_t cpu_n = std::min(opt.num_queries, 64u);
    luisa::Clock cpu_clock;
    cpu_clock.tic();
    for (uint32_t q = 0; q < cpu_n; ++q) {
        (void)reference_retrieve(luisa::span{lib.tokens},
                                 luisa::span{queries_flat}.subspan(q * max_query_len, query_lens[q]),
                                 opt.min_n, opt.max_n, opt.k);
    }
    const double cpu_ms = cpu_clock.toc();
    const double cpu_per_query_us = cpu_ms * 1000.0 / static_cast<double>(cpu_n);

    std::printf("\n==================== ngram retrieve benchmark ====================\n");
    std::printf(" backend        : %s\n", argv[1]);
    std::printf(" kernel         : %s (block_size=%u)\n", variant_name, opt.block_size);
    std::printf(" corpus         : %u tokens (vocab %u)\n", opt.lib_size, opt.vocab);
    std::printf(" batch          : %u queries, min_n=%u max_n=%u k=%u\n",
                opt.num_queries, opt.min_n, opt.max_n, opt.k);
    std::printf(" ------------------------------------------------------------------\n");
    std::printf(" device dispatch+sync (best) : %9.3f ms\n", best);
    std::printf(" device dispatch+sync (avg)  : %9.3f ms\n", avg_ms);
    std::printf(" per-query latency (avg)     : %9.3f us\n", per_query_us);
    std::printf(" throughput                  : %9.0f queries/s\n", queries_per_s);
    std::printf(" corpus scan rate            : %9.2f Gtokens/s\n", corpus_gbps);
    std::printf(" cpu reference (1 thread)    : %9.3f us/query (%u queries)\n",
                cpu_per_query_us, cpu_n);
    std::printf(" device speedup vs cpu       : %9.1fx\n", cpu_per_query_us / per_query_us);
    std::printf("==================================================================\n\n");
    return 0;
}

}// namespace

int run_benchmark(int argc, char *argv[]) {
    return run_benchmark_impl(argc, argv);
}

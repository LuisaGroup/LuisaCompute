// Performance benchmark for the n-gram tokenize + retrieve example.
//
// Generates a synthetic corpus of repeated token patterns (vLLM-like high
// hit rate), samples a batch of corpus-suffix queries, then times the pure
// device dispatch (upload once, then dispatch + synchronize per rep,
// download once). Device output is spot-checked against the CPU reference
// oracle before timing, and single-threaded CPU reference timings are
// reported for scale.
//
// Multi-request mode (--requests N) simulates N independent retrieve
// requests arriving together and compares four ways to overlap their
// upload/dispatch/download stages:
//   seq      full serialized round trip per request (baseline);
//   pipeline single stream, all stages queued without intermediate syncs
//            (copies of request i+1 overlap compute of request i);
//   fiber    one fiber + one stream per request: host-side staging and
//            submissions run concurrently, device runs the streams in
//            parallel (see luisa/core/fiber.h);
//   multi    ONE multi-dispatch command (Shader::dispatch(span<uint3>))
//            carrying every request's grid; sub-dispatch r reads its row
//            range through kernel_id() (see ngram_kernels.h).
//
// Options: --kernel naive|parallel --lib-size N --queries N --reps N
//          --min-n N --max-n N --k N --block-size N --vocab N --seed N
//          --requests N  --req-queries N  --multi-mode seq|pipeline|fiber|multi|all

#include "ngram_library.h"
#include "ngram_retriever.h"
#include "reference.h"

#include <luisa/core/clock.h>
#include <luisa/core/fiber.h>
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
    // multi-request mode: --requests > 0 switches from the single-request
    // benchmark to the multi-request concurrency comparison.
    uint32_t requests = 0u;       // 0 = legacy single-request benchmark
    uint32_t req_queries = 0u;    // rows per request (0 = use --queries)
    luisa::string multi_mode = "all";// seq | pipeline | fiber | multi | all
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
        } else if (std::strcmp(argv[i], "--requests") == 0) {
            opt.requests = need("--requests");
        } else if (std::strcmp(argv[i], "--req-queries") == 0) {
            opt.req_queries = need("--req-queries");
        } else if (std::strcmp(argv[i], "--multi-mode") == 0) {
            if (i + 1 >= argc) LUISA_ERROR("missing value for --multi-mode");
            opt.multi_mode = argv[++i];
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

// ---------- multi-request concurrency benchmark ----------

// One full query batch: rows of `stride` tokens plus the actual lengths.
struct QueryBatch {
    luisa::vector<uint32_t> flat;
    luisa::vector<uint32_t> lens;
    uint32_t stride = 0;
};

// Sample `count` corpus-suffix queries (same distribution as the
// single-request benchmark).
[[nodiscard]] QueryBatch generate_query_batch(
    const luisa::vector<uint32_t> &corpus, uint32_t count,
    uint32_t max_n, uint32_t seed) {
    std::mt19937 rng{seed};
    QueryBatch qb;
    qb.stride = max_n;
    qb.lens.resize(count);
    for (auto &len : qb.lens) {
        len = max_n + rng() % 8u;
        qb.stride = std::max(qb.stride, len);
    }
    qb.flat.resize(count * qb.stride, 0u);
    const uint32_t lib_len = static_cast<uint32_t>(corpus.size());
    const uint32_t half = lib_len / 2u;
    for (uint32_t q = 0; q < count; ++q) {
        const uint32_t end = half + rng() % (lib_len - half);
        const uint32_t begin = end - qb.lens[q];
        std::copy_n(corpus.begin() + begin, qb.lens[q],
                    qb.flat.begin() + q * qb.stride);
    }
    return qb;
}

// Everything the strategies need. The retriever's query/draft buffers are
// shared; each request owns a disjoint row range [bases[i], bases[i] +
// counts[i]) plus an offsets buffer and (for the fiber strategy) a private
// stream, since a Stream must never be committed from two fibers at once.
struct MultiRequestCtx {
    NgramRetriever &retriever;
    luisa::compute::Stream &stream;// main stream (library upload + multi-dispatch)
    uint32_t min_n, max_n, k;
    uint32_t stride;
    luisa::vector<uint32_t> bases;  // per-request first row
    luisa::vector<uint32_t> counts; // per-request row count
    luisa::vector<uint32_t> flat;   // all rows, row-major
    luisa::vector<uint32_t> lens;   // all rows
    luisa::vector<luisa::compute::Stream> req_streams;
    luisa::vector<luisa::compute::Buffer<uint32_t>> off_bufs;// 1-entry per request
    luisa::compute::Buffer<uint32_t> multi_off_buf;          // all bases
    luisa::vector<luisa::uint3> multi_sizes;                 // grid per request
    luisa::vector<luisa::vector<uint32_t>> d_vec;            // per-request drafts
    luisa::vector<luisa::vector<uint32_t>> l_vec;            // per-request draft lens
};

enum class MultiStrategy : uint32_t { seq, pipeline, fiber, multi };

// Baseline: request i fully completes (upload -> dispatch -> sync ->
// download) before request i+1 starts. No stage ever overlaps.
void rep_seq(MultiRequestCtx &c) {
    const auto n = static_cast<uint32_t>(c.counts.size());
    for (auto i = 0u; i < n; ++i) {
        c.retriever.upload_request(
            c.stream,
            luisa::span{c.flat}.subspan(c.bases[i] * c.stride, c.counts[i] * c.stride),
            luisa::span{c.lens}.subspan(c.bases[i], c.counts[i]), c.bases[i]);
        c.retriever.dispatch_request(c.stream, c.off_bufs[i], c.counts[i]);
        c.stream.synchronize();
        c.retriever.download_request(c.stream, c.bases[i], c.counts[i],
                                     c.d_vec[i], c.l_vec[i]);
    }
    c.stream.synchronize();
}

// One stream, no host sync between stages: every upload is queued first,
// then every dispatch, then a single synchronize, then all downloads. The
// copy work of request i+1 overlaps the kernel of request i on device.
void rep_pipeline(MultiRequestCtx &c) {
    const auto n = static_cast<uint32_t>(c.counts.size());
    for (auto i = 0u; i < n; ++i) {
        c.retriever.upload_request(
            c.stream,
            luisa::span{c.flat}.subspan(c.bases[i] * c.stride, c.counts[i] * c.stride),
            luisa::span{c.lens}.subspan(c.bases[i], c.counts[i]), c.bases[i]);
    }
    for (auto i = 0u; i < n; ++i) {
        c.retriever.dispatch_request(c.stream, c.off_bufs[i], c.counts[i]);
    }
    c.stream.synchronize();
    for (auto i = 0u; i < n; ++i) {
        c.retriever.download_request(c.stream, c.bases[i], c.counts[i],
                                     c.d_vec[i], c.l_vec[i]);
    }
    c.stream.synchronize();
}

// Host + device parallelism: one fiber per request, each with its own
// stream, so the host-side staging (memcpy into upload heaps) and the
// command submission of all requests run concurrently, and the device
// executes the request streams in parallel.
void rep_fiber(MultiRequestCtx &c) {
    const auto n = static_cast<uint32_t>(c.counts.size());
    luisa::vector<luisa::fiber::event> evts;
    evts.reserve(n);
    for (auto i = 0u; i < n; ++i) {
        evts.push_back(luisa::fiber::async([&, i] {
            c.retriever.upload_request(
                c.req_streams[i],
                luisa::span{c.flat}.subspan(c.bases[i] * c.stride, c.counts[i] * c.stride),
                luisa::span{c.lens}.subspan(c.bases[i], c.counts[i]), c.bases[i]);
            c.retriever.dispatch_request(c.req_streams[i], c.off_bufs[i], c.counts[i]);
        }));
    }
    for (auto &e : evts) { e.wait(); }
    for (auto i = 0u; i < n; ++i) { c.req_streams[i].synchronize(); }
    evts.clear();
    for (auto i = 0u; i < n; ++i) {
        evts.push_back(luisa::fiber::async([&, i] {
            c.retriever.download_request(c.req_streams[i], c.bases[i], c.counts[i],
                                         c.d_vec[i], c.l_vec[i]);
        }));
    }
    for (auto &e : evts) { e.wait(); }
    for (auto i = 0u; i < n; ++i) { c.req_streams[i].synchronize(); }
}

// Device-side parallelism: ONE multi-dispatch command carries every
// request's grid; sub-dispatch r sees kernel_id() == r and resolves its row
// range from the shared offsets buffer.
void rep_multi(MultiRequestCtx &c) {
    c.retriever.upload_request(c.stream, luisa::span{c.flat}, luisa::span{c.lens}, 0u);
    c.retriever.dispatch_requests_multi(c.multi_off_buf, c.multi_sizes);
    c.stream.synchronize();
    for (auto i = 0u; i < c.counts.size(); ++i) {
        c.retriever.download_request(c.stream, c.bases[i], c.counts[i],
                                     c.d_vec[i], c.l_vec[i]);
    }
    c.stream.synchronize();
}

using MultiRep = void (*)(MultiRequestCtx &);

// One correctness pass + timed reps of one strategy. Returns avg ms.
[[nodiscard]] double run_multi_strategy(luisa::string_view name, MultiRep rep,
                                        MultiRequestCtx &c, const NgramLibrary &lib,
                                        uint32_t reps) {
    // correctness first: one full pass, then spot-check against the oracle
    rep(c);
    const uint32_t total_rows = static_cast<uint32_t>(c.lens.size());
    const uint32_t check_rows = std::min(total_rows, 64u);
    uint32_t mismatches = 0, checked = 0;
    for (auto i = 0u; i < c.counts.size() && checked < check_rows; ++i) {
        for (auto j = 0u; j < c.counts[i] && checked < check_rows; ++j, ++checked) {
            const uint32_t row = c.bases[i] + j;
            auto expected = reference_retrieve(
                luisa::span{lib.tokens},
                luisa::span{c.flat}.subspan(row * c.stride, c.lens[row]),
                c.min_n, c.max_n, c.k);
            if (c.l_vec[i][j] != expected.size()) {
                ++mismatches;
                continue;
            }
            for (size_t t = 0; t < expected.size(); ++t) {
                if (c.d_vec[i][j * c.k + t] != expected[t]) ++mismatches;
            }
            for (size_t t = expected.size(); t < c.k; ++t) {
                if (c.d_vec[i][j * c.k + t] != ngram_invalid_id) ++mismatches;
            }
        }
    }
    if (mismatches != 0u) {
        LUISA_ERROR("multi-request benchmark aborted: {} mismatches in '{}' mode",
                    mismatches, name);
    }
    // warmup, then time
    rep(c);
    luisa::Clock clock;
    double best = 1e30, total = 0.0;
    for (auto r = 0u; r < reps; ++r) {
        clock.tic();
        rep(c);
        const double ms = clock.toc();
        best = std::min(best, ms);
        total += ms;
    }
    const double avg_ms = total / static_cast<double>(reps);
    const double per_req_us =
        avg_ms * 1000.0 / static_cast<double>(c.counts.size());
    std::printf("  %-9s       %9.3f   %9.3f          %9.3f\n",
                luisa::to_string(name).c_str(), best, avg_ms, per_req_us);
    return avg_ms;
}

int run_multi_request_benchmark(luisa::compute::Device &device,
                                luisa::compute::Stream &stream,
                                NgramLibrary &lib,
                                const luisa::vector<uint32_t> &corpus,
                                const char *backend,
                                const BenchOptions &opt) {
    const uint32_t num_requests = opt.requests;
    const uint32_t rows_per_req =
        opt.req_queries > 0u ? opt.req_queries : opt.num_queries;
    LUISA_ASSERT(num_requests > 0u, "invalid --requests");
    LUISA_ASSERT(rows_per_req >= 2u, "--req-queries must be >= 2");

    // vary the row count per request (-1/0/+1) so the offsets machinery is
    // exercised with unequal request sizes
    luisa::vector<uint32_t> bases, counts;
    uint32_t total_rows = 0;
    for (auto i = 0u; i < num_requests; ++i) {
        counts.push_back(rows_per_req + (i % 3u) - 1u);
        bases.push_back(total_rows);
        total_rows += counts.back();
    }
    auto qb = generate_query_batch(corpus, total_rows, opt.max_n,
                                   opt.seed ^ 0x5bd1e995u);

    // fiber scheduler for the fiber strategy (binds marl to this thread)
    luisa::fiber::scheduler fiber_pool;

    NgramRetriever retriever{device, stream, lib,
                             opt.min_n, opt.max_n, opt.k,
                             qb.stride, total_rows, opt.variant, opt.block_size};
    MultiRequestCtx c{
        retriever, stream,
        opt.min_n, opt.max_n, opt.k, qb.stride,
        std::move(bases), std::move(counts),
        std::move(qb.flat), std::move(qb.lens),
        {}, {}, {}, {}, {}, {}};

    // one-time offsets uploads on the main stream; the host sync makes them
    // (and the library buffers uploaded by the constructor) visible to every
    // per-request stream before any fiber dispatch reads them.
    for (auto i = 0u; i < num_requests; ++i) {
        c.off_bufs.push_back(retriever.upload_request_offsets(stream, c.bases[i]));
        c.multi_sizes.emplace_back(
            opt.variant == NgramKernelVariant::parallel
                ? c.counts[i] * opt.block_size
                : c.counts[i],
            1u, 1u);
        c.req_streams.emplace_back(device.create_stream());
        c.d_vec.emplace_back();
        c.l_vec.emplace_back();
    }
    c.multi_off_buf = retriever.upload_request_offsets(stream, luisa::span{c.bases});
    stream.synchronize();

    const char *variant_name = "parallel";
    if (opt.variant == NgramKernelVariant::naive) variant_name = "naive";
    if (opt.variant == NgramKernelVariant::hash) variant_name = "hash";
    LUISA_INFO("multi-request benchmark: kernel={} requests={} rows/req~{} "
               "total_rows={} lib={} min_n={} max_n={} k={} reps={}",
               variant_name, num_requests, rows_per_req, total_rows,
               opt.lib_size, opt.min_n, opt.max_n, opt.k, opt.reps);

    std::printf("\n============== multi-request retrieve benchmark ==============\n");
    std::printf(" backend        : %s\n", backend);
    std::printf(" kernel         : %s (block_size=%u)\n", variant_name, opt.block_size);
    std::printf(" requests       : %u (rows/req ~%u, total %u)\n",
                num_requests, rows_per_req, total_rows);
    std::printf(" corpus         : %u tokens, min_n=%u max_n=%u k=%u\n",
                opt.lib_size, opt.min_n, opt.max_n, opt.k);
    std::printf(" ------------------------------------------------------------------\n");
    std::printf("  strategy      best (ms)   avg (ms)   req latency avg (us)\n");

    const bool all = opt.multi_mode == "all";
    // 'seq' is always measured first: it is the speedup baseline.
    const double seq_avg = run_multi_strategy("seq", rep_seq, c, lib, opt.reps);
    struct Mode { luisa::string_view flag, name; MultiRep rep; };
    const Mode modes[] = {
        {"pipeline", "pipeline", rep_pipeline},
        {"fiber", "fiber", rep_fiber},
        {"multi", "multi", rep_multi},
    };
    for (auto &m : modes) {
        if (!all && opt.multi_mode != m.flag) continue;
        const double avg = run_multi_strategy(m.name, m.rep, c, lib, opt.reps);
        std::printf("    -> %s speedup vs seq: %.2fx\n",
                    luisa::to_string(m.name).c_str(), seq_avg / avg);
    }
    std::printf("==================================================================\n\n");

    // leave no pending work on the per-request streams
    for (auto &s : c.req_streams) { s.synchronize(); }
    return 0;
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

    // multi-request mode: compare seq/pipeline/fiber/multi stage overlap
    if (opt.requests > 0u) {
        return run_multi_request_benchmark(device, stream, lib, corpus,
                                           argv[1], opt);
    }

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

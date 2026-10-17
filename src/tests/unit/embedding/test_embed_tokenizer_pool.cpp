// test_embed_tokenizer_pool.cpp -- H8 tokenizer-pool scalability gate
// (embedding_model_opt_plan.md §2.3 H8: "Pin a scalability test: batch-64
// tokenize speedup >= 4x on 8 cores", and "assert identical ids vs
// single-threaded order-preserving output").
//
// Host-only test (LUISA_EMBED_HOST_ONLY=1): it creates no LuisaCompute device
// and takes no backend argument. It exercises Tokenizer::encode_parallel --
// the entry point Session::encode_batch uses for the H8 thread-pool wiring.
//
// What it pins:
//  - Correctness (hard): encode_parallel(texts, max_len, W) returns, for every
//    index i, ids BIT-IDENTICAL to a serial `encode(texts[i], max_len)` loop,
//    i.e. output is order-preserving and the parallel split introduces no
//    divergence. Checked for W in {8 (pool), 1 (serial clamp), 0 (auto)} so the
//    pool, the single-thread fallback, and the auto path all agree.
//  - Scalability (hard): on a machine with at least kPoolThreads usable cores,
//    the min-of-N wall time of the pool run is >= 4x faster than the min-of-N
//    serial wall time for a 64-text batch of mixed tokenized length (the texts
//    span the 128/512/2048 padding buckets so the dynamic cursor genuinely load-
//    balances). On machines with fewer cores the speedup requirement is relaxed
//    proportionally (never below 1.0x) so CI without 8 cores still passes, while
//    the correctness gate always holds.
//  - Edge: encode_parallel with an empty batch returns an empty result; with a
//    single text it equals encode of that text.
//
// min-of-N timing follows lc_optimize §10.8 (sub-ms work has large run-to-run
// noise; distrust single runs).
//
// Run: xmake run test_embed_tokenizer_pool
#include "ut/ut.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <thread>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "embedding/embed/tokenizer.h"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

    constexpr uint32_t k_batch = 64u;        // batch-64 (plan §2.3 H8)
    constexpr uint32_t kPoolThreads = 8u;    // "up to 8 threads"
    constexpr uint32_t k_max_len = 512u;     // matches Session default
    constexpr int k_reps = 6;                // min-of-N timing reps (after warmup)

    const std::filesystem::path &resolve_data_dir() {
        // Same three-step resolution as the other host-only embedding tests.
        static const std::filesystem::path dir = [] {
            std::filesystem::path cand{luisa::string{LUISA_EMBED_TEST_DATA_DIR}};
            if(std::filesystem::exists(cand / "corpus.txt")) { return cand; }
            auto p = std::filesystem::path{__FILE__};
            for(int i = 0; i < 3; i++) { p = p.parent_path(); }
            p /= "data";
            p /= "embedding";
            if(std::filesystem::exists(p / "corpus.txt")) { return p; }
            p = std::filesystem::current_path();
            while(!p.empty()) {
                auto q = p / "src" / "tests" / "data" / "embedding";
                if(std::filesystem::exists(q / "corpus.txt")) { return q; }
                auto parent = p.parent_path();
                if(parent == p) { break; }
                p = parent;
            }
            LUISA_ERROR("cannot locate src/tests/data/embedding (corpus.txt) from cwd \"{}\"",
                        std::filesystem::current_path().string());
        }();
        return dir;
    }

    std::filesystem::path tokenizer_json_path() {
        // bge-m3 assets sit at the repo root (walk up: embedding -> data -> tests
        // -> src -> repo root), same as test_embed_tokenizer_mt.cpp; fall back to
        // the copy inside the data dir if the root layout is absent.
        auto root = resolve_data_dir();
        for(int i = 0; i < 4; i++) { root = root.parent_path(); }
        auto p = root / "bge-m3" / "bge-m3" / "tokenizer.json";
        if(std::filesystem::exists(p)) { return p; }
        return resolve_data_dir() / "tokenizer.json";
    }

    // A deterministic, ragged batch of 64 texts of mixed tokenized length. Each
    // text is a base sentence repeated k_i times; k_i cycles over a spread so the
    // batch crosses the 128/512/2048 padding buckets and the pool's dynamic cursor
    // genuinely has to load-balance (a static equal split would idle a thread on a
    // long text). Contents are pure ASCII for stable, hardware-independent timing.
    luisa::vector<luisa::string> make_batch() {
        static constexpr luisa::string_view base =
            "The quick brown fox jumps over the lazy dog while analysts review "
            "quarterly infrastructure spending and maritime logistics across regions. ";
        luisa::vector<luisa::string> texts;
        texts.reserve(k_batch);
        for(uint32_t i = 0u; i < k_batch; i++) {
            // k in {1, 4, 8, 12, 16, 20, 6, 2} pattern -> ~30 .. ~700 tokens each
            const uint32_t reps = 1u + ((i * 7u + 3u) % 20u);
            luisa::string s;
            s.reserve(base.size() * reps + 1u);
            for(uint32_t r = 0u; r < reps; r++) { s += base; }
            texts.emplace_back(std::move(s));
        }
        return texts;
    }

    // Serial reference: one encode per text, in input order (the definition the
    // pool must match bit-for-bit).
    luisa::vector<luisa::vector<uint32_t>> encode_serial(const luisa::embed::Tokenizer &tkn,
                                                         luisa::span<const luisa::string_view> texts) {
        luisa::vector<luisa::vector<uint32_t>> out;
        out.reserve(texts.size());
        for(const auto &t : texts) { out.emplace_back(tkn.encode(t, k_max_len)); }
        return out;
    }

    template<typename F>
    double time_ms(const F &fn, int reps) {
        double best = std::numeric_limits<double>::max();
        for(int r = 0; r < reps; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    }

    // Compare every id vector element-by-element; report the first divergence.
    uint32_t count_id_mismatches(const luisa::vector<luisa::vector<uint32_t>> &got,
                                 const luisa::vector<luisa::vector<uint32_t>> &ref) {
        if(got.size() != ref.size()) {
            LUISA_WARNING("id-vector count mismatch: pool {} vs serial {}", got.size(), ref.size());
            return static_cast<uint32_t>(got.size() + ref.size());
        }
        uint32_t bad = 0u;
        for(size_t i = 0u; i < ref.size(); i++) {
            if(got[i] != ref[i]) {
                if(bad == 0u) {
                    size_t div = 0u;
                    while(div < got[i].size() && div < ref[i].size() && got[i][div] == ref[i][div]) { div++; }
                    LUISA_WARNING("text {} differs: pool {} ids vs serial {} ids; first divergence at {} "
                                  "(pool {}, serial {})",
                                  i, got[i].size(), ref[i].size(), div,
                                  div < got[i].size() ? got[i][div] : 0xFFFFFFFFu,
                                  div < ref[i].size() ? ref[i][div] : 0xFFFFFFFFu);
                }
                bad++;
            }
        }
        return bad;
    }

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    const auto tkn_path = tokenizer_json_path();
    auto tkn = luisa::embed::Tokenizer::load(tkn_path);
    expect(tkn != nullptr) << "tokenizer load must succeed";
    if(tkn == nullptr) { return 1; }

    const auto texts = make_batch();
    LUISA_INFO("test_embed_tokenizer_pool: batch-{} tokenizer pool gate (tokenizer={})",
               k_batch, tkn_path.string());

    // Reference (serial) ids for the correctness gate.
    luisa::vector<luisa::string_view> views;
    views.reserve(texts.size());
    for(const auto &t : texts) { views.emplace_back(luisa::string_view{t.data(), t.size()}); }
    const auto span_views = luisa::span<const luisa::string_view>{views.data(), views.size()};

    const auto ref = encode_serial(*tkn, span_views);
    uint32_t total_tokens = 0u;
    uint32_t min_tok = 0xFFFFFFFFu, max_tok = 0u;
    for(const auto &v : ref) {
        total_tokens += static_cast<uint32_t>(v.size());
        min_tok = std::min(min_tok, static_cast<uint32_t>(v.size()));
        max_tok = std::max(max_tok, static_cast<uint32_t>(v.size()));
    }
    LUISA_INFO("test_embed_tokenizer_pool: batch spans {}..{} tokens/text (mean {}), i.e. ragged buckets",
               min_tok, max_tok, total_tokens / k_batch);
    // The batch must be ragged enough to exercise bucketing and load balancing.
    expect(min_tok < max_tok) << "the batch must contain texts of differing tokenized length";
    expect(max_tok > 128u) << "the batch must reach beyond the 128 bucket to exercise load balancing";

    // ---- correctness: pool == serial, order-preserving ----
    const auto pool8 = tkn->encode_parallel(span_views, k_max_len, kPoolThreads);
    expect(count_id_mismatches(pool8, ref) == 0u)
        << "encode_parallel(8 threads) must be bit-identical to the serial loop, in order";

    const auto pool1 = tkn->encode_parallel(span_views, k_max_len, 1u);
    expect(count_id_mismatches(pool1, ref) == 0u)
        << "encode_parallel(1 thread) must equal the serial loop";

    const auto pool_auto = tkn->encode_parallel(span_views, k_max_len, 0u);
    expect(count_id_mismatches(pool_auto, ref) == 0u)
        << "encode_parallel(auto threads) must equal the serial loop";

    // ---- edge cases ----
    {
        const auto empty = tkn->encode_parallel(luisa::span<const luisa::string_view>{}, k_max_len, 8u);
        expect(empty.empty()) << "encode_parallel on an empty batch returns an empty result";
    }
    {
        luisa::string_view one{"hello world, how are you today?"};
        const auto s = luisa::span<const luisa::string_view>{&one, 1u};
        const auto p = tkn->encode_parallel(s, k_max_len, 8u);
        expect(p.size() == 1u) << "single-text encode_parallel returns one vector";
        expect(p.size() == 1u && p[0] == tkn->encode(one, k_max_len))
            << "single-text encode_parallel must equal encode()";
    }

    // ---- scalability: min-of-N serial vs pool ----
    // Warm up (allocator/branch caches, first-touch of the uninitialised pool path).
    (void)encode_serial(*tkn, span_views);
    (void)tkn->encode_parallel(span_views, k_max_len, kPoolThreads);

    double serial_ms = std::numeric_limits<double>::max();
    double pool_ms = std::numeric_limits<double>::max();
    for(int r = 0; r < k_reps; r++) {
        serial_ms = std::min(serial_ms,
                             time_ms([&] { (void)encode_serial(*tkn, span_views); }, 1));
        pool_ms = std::min(pool_ms,
                           time_ms([&] { (void)tkn->encode_parallel(span_views, k_max_len, kPoolThreads); }, 1));
    }
    const auto speedup = (pool_ms > 0.0) ? (serial_ms / pool_ms) : 0.0;

    uint32_t hw = std::thread::hardware_concurrency();
    if(hw == 0u) { hw = 1u; }
    const uint32_t usable = std::min(hw, kPoolThreads);
    // Require >= 4x when the box offers the full 8-core pool; otherwise scale the
    // bar down (0.5x per usable worker, floor 1.0x) so small-core CI is not a
    // false failure while still proving the pool is never slower than serial.
    const double required = (usable >= kPoolThreads) ? 4.0 : std::max(1.0, 0.5 * static_cast<double>(usable));

    LUISA_INFO("test_embed_tokenizer_pool: serial min={} ms, pool min={} ms (workers={} on hw={}), "
               "speedup={:.2f}x (required >= {:.2f}x)",
               serial_ms, pool_ms, usable, hw, speedup, required);

    expect(speedup >= required)
        << luisa::format("batch-64 tokenizer pool must scale to >= {:.2f}x; measured {:.2f}x "
                         "(serial {:.2f} ms vs pool {:.2f} ms, {} workers)",
                         required, speedup, serial_ms, pool_ms, usable);

    return 0;
}

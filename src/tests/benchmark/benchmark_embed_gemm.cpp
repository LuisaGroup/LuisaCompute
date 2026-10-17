// GEMM-family tile sweep for the bge-m3 embedding engine
// (embedding_model_opt_plan.md §3.1 G1/G2/G3/G4/G8, agent O-A).
//
// What it does
// ------------
// Phase 1 -- correctness gate: for every variant in the engine's GEMM variant
// table (examples/embedding/embed/kernels/kernels_gemm.h, `GemmTileConfig`), run
// a small *ragged* proxy case (M=37, N=69, K=53: no dimension is a multiple of
// any tile in the table, K leaves a tail slab) against an fp64 host reference
// for both epilogues the engine actually uses (Bias, BiasGeluErf), plus an
// optional arbitrary case via --verify.
//
// Phase 2 -- timing: for every passing variant and every real engine shape,
// sample 16 outputs of the timed shape against the fp64 reference (catches
// shape-dependent bugs the proxy cannot reach, e.g. L2/tile-boundary mistakes at
// rows=4096) and time min-of-N dispatches after a warm-up (lc_optimize §10.8).
// A variant that fails any check is reported and NOT timed -- no plausible-but-
// wrong numbers in the table.
//
// The shapes are the four GEMM call sites in embed/executor.cpp (H=1024,
// I=4096):
//   qkv       A[rows,1024] @ B[1024,3072] + bias  (epilogue Bias)
//   attn_out  A[rows,1024] @ B[1024,1024] + bias  (epilogue Bias)
//   ffn_up    A[rows,1024] @ B[1024,4096] + bias  (epilogue BiasGeluErf)
//   ffn_down  A[rows,4096] @ B[4096,1024] + bias  (epilogue Bias)
// rows = padded B*S; the plan's acceptance point is rows=4096 (B=8, S=512).
//
// Usage
// -----
//   benchmark_embed_gemm <backend:dx|vk|fallback> [options]
//     --rows 512,4096        row counts to time (default 512,4096)
//     --iters N              timed repetitions, min-of-N (default 10)
//     --warmup N             untimed warm-up dispatches (default 3)
//     --variants a,b,c       only these variant names (default: whole table)
//     --shapes qkv,ffn_up    only these shapes (default: all four)
//     --no-time              phase 1 only (correctness)
//     --verify M N K [ep]    extra debug case on every selected variant
//                            (ep: 0 none, 1 bias, 2 gelu; default 1); prints
//                            every mismatching (row, col)
//     --csv FILE             append rows (header written once)
//
// Notes
// -----
// * dispatch() counts TOTAL threads, not blocks (lc_optimize trap #1) -- the
//   geometry comes from the engine's own gemm_dispatch_size_config() so the
//   harness can never disagree with the kernel.
// * Debug builds carry a large fixed per-dispatch overhead; compare variants
//   within one run/backend and re-measure in release before freezing the
//   per-backend table (plan §7.2).
#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <utility>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include "embedding/embed/kernels/kernels_gemm.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

//--- engine shapes -------------------------------------------------------------
struct Shape {
    const char *name;
    uint32_t n;
    uint32_t k;
    GemmEpilogue ep;
};

constexpr Shape kShapes[] = {
    {"qkv", 3072u, 1024u, GemmEpilogue::Bias},
    {"attn_out", 1024u, 1024u, GemmEpilogue::Bias},
    {"ffn_up", 4096u, 1024u, GemmEpilogue::BiasGeluErf},
    {"ffn_down", 1024u, 4096u, GemmEpilogue::Bias},
};
constexpr uint32_t kShapeCount = static_cast<uint32_t>(sizeof(kShapes) / sizeof(kShapes[0]));

constexpr bool uses_bias(GemmEpilogue ep) noexcept {
    return ep == GemmEpilogue::Bias || ep == GemmEpilogue::BiasGeluErf || ep == GemmEpilogue::BiasRelu;
}

//--- deterministic inputs ------------------------------------------------------
struct Lcg {
    uint32_t state = 0x12345678u;
    [[nodiscard]] uint32_t next() noexcept {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() % 8388608u) / 8388608.f);
    }
};

[[nodiscard]] luisa::vector<float> make_inputs(Lcg &rng, size_t count, float lo, float hi) {
    luisa::vector<float> v(count);
    for (auto &f : v) {
        f = rng.uniform(lo, hi);
    }
    return v;
}

[[nodiscard]] float gelu_erf_ref(float x) noexcept {
    const double xd = static_cast<double>(x);
    return static_cast<float>(0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440)));
}

// fp64-accumulated reference for one output element (row-major A[M,K] @ B[K,N]).
[[nodiscard]] float ref_element(const luisa::vector<float> &A, const luisa::vector<float> &B,
                                const luisa::vector<float> &aux, GemmEpilogue ep,
                                uint32_t m, uint32_t n, uint32_t k, uint32_t N) noexcept {
    double acc = 0.0;
    for (uint32_t t = 0u; t < k; t++) {
        acc += static_cast<double>(A[static_cast<size_t>(m) * k + t]) *
               static_cast<double>(B[static_cast<size_t>(t) * N + n]);
    }
    float v = static_cast<float>(acc);
    switch (ep) {
        case GemmEpilogue::None: break;
        case GemmEpilogue::Bias: v += aux[n]; break;
        case GemmEpilogue::BiasGeluErf: v = gelu_erf_ref(v + aux[n]); break;
        case GemmEpilogue::BiasRelu: v = std::max(v + aux[n], 0.0f); break;
        case GemmEpilogue::ResidualAdd: v += aux[static_cast<size_t>(m) * N + n]; break;
    }
    return v;
}

//--- options -------------------------------------------------------------------
struct Options {
    luisa::vector<uint32_t> rows{512u, 4096u};
    uint32_t iters{10u};
    uint32_t warmup{3u};
    std::filesystem::path csv{};
    luisa::vector<luisa::string> variants{};
    luisa::vector<luisa::string> shapes{};
    bool time{true};
    bool debug_verify{false};
    uint32_t v_m{0u}, v_n{0u}, v_k{0u};
    GemmEpilogue v_ep{GemmEpilogue::Bias};
};

[[nodiscard]] uint32_t parse_u32_text(const char *first, const char *last, uint32_t &v) noexcept {
    const auto r = std::from_chars(first, last, v);
    return r.ec == std::errc{} && r.ptr == last ? v : 0u;
}

[[nodiscard]] uint32_t parse_u32(std::string_view s, std::string_view who) {
    uint32_t v{};
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    LUISA_ASSERT(r.ec == std::errc{} && r.ptr == s.data() + s.size(), "{} must be an integer, got \"{}\".",
                 who, s);
    return v;
}

// Comma-separated non-negative integer list.
template<typename F>
void split_u32(const char *text, const F &f) {
    const char *p = text;
    while(*p != '\0') {
        const char *q = p;
        while(*q != '\0' && *q != ',') {
            q++;
        }
        if(q > p) {
            f(p, q);
        }
        p = (*q == ',') ? q + 1 : q;
    }
}

// Comma-separated token list.
template<typename F>
void split_names(const char *text, const F &f) {
    split_u32(text, [&](const char *b, const char *e) { f(luisa::string{b, e}); });
}

[[nodiscard]] Options parse_args(int argc, char *argv[]) {
    Options opt;
    for(int i = 2; i < argc; i++) {
        const std::string_view arg{argv[i]};
        const auto need = [&](std::string_view name) -> const char * {
            LUISA_ASSERT(i + 1 < argc, "flag {} requires a value", name);
            return argv[++i];
        };
        if(arg == "--rows") {
            opt.rows.clear();
            split_u32(need("rows"), [&](const char *b, const char *e) {
                uint32_t v{};
                const auto r = std::from_chars(b, e, v);
                LUISA_ASSERT(r.ec == std::errc{} && r.ptr == e, "rows must be integers");
                opt.rows.emplace_back(v);
            });
        } else if(arg == "--iters") {
            opt.iters = parse_u32(need("iters"), "iters");
        } else if(arg == "--warmup") {
            opt.warmup = parse_u32(need("warmup"), "warmup");
        } else if(arg == "--csv") {
            opt.csv = std::string{need("csv")};
        } else if(arg == "--variants") {
            split_names(need("variants"), [&](luisa::string t) { opt.variants.emplace_back(std::move(t)); });
        } else if(arg == "--shapes") {
            split_names(need("shapes"), [&](luisa::string s) { opt.shapes.emplace_back(std::move(s)); });
        } else if(arg == "--no-time") {
            opt.time = false;
        } else if(arg == "--verify") {
            opt.debug_verify = true;
            opt.v_m = parse_u32(need("M"), "M");
            opt.v_n = parse_u32(need("N"), "N");
            opt.v_k = parse_u32(need("K"), "K");
            if(i + 1 < argc && argv[i + 1][0] != '-') {
                opt.v_ep = static_cast<GemmEpilogue>(parse_u32(need("ep"), "ep"));
            }
        } else {
            LUISA_ERROR_WITH_LOCATION("unknown argument: {}", arg);
        }
    }
    LUISA_ASSERT(opt.iters >= 1u, "--iters must be >= 1");
    LUISA_ASSERT(!opt.rows.empty(), "--rows needs at least one value");
    return opt;
}

[[nodiscard]] bool selected(const luisa::vector<luisa::string> &list, const char *name) noexcept {
    if(list.empty()) {
        return true;
    }
    for(const auto &s : list) {
        if(s == name) {
            return true;
        }
    }
    return false;
}

// Compiled type of the engine's GemmKernel (same trick as embed/executor.h's
// ShaderFor<K>, kept local so the harness does not depend on the executor).
using GemmShader = decltype(std::declval<Device &>().compile(std::declval<GemmKernel>(),
                                                             std::declval<ShaderOption &>()));

struct Statistics {
    double min{};
    double median{};
};

[[nodiscard]] Statistics summarize(luisa::vector<double> samples) {
    LUISA_ASSERT(!samples.empty(), "No samples to summarize.");
    luisa::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2u]};
}

//--- buffers + checks ----------------------------------------------------------
struct CaseBuffers {
    uint32_t m, n, k;
    GemmEpilogue ep;
    luisa::vector<float> ha, hb, haux;
    Buffer<float> a, b, c, aux;
};

[[nodiscard]] CaseBuffers make_case(Device &device, uint32_t m, uint32_t n, uint32_t k, GemmEpilogue ep,
                                    float mag) {
    CaseBuffers cb;
    cb.m = m;
    cb.n = n;
    cb.k = k;
    cb.ep = ep;
    Lcg rng{};
    rng.state = 0x9E3779B9u ^ (m << 24u) ^ (n << 12u) ^ (k << 2u) ^ static_cast<uint32_t>(ep);
    cb.ha = make_inputs(rng, static_cast<size_t>(m) * k, -mag, mag);
    cb.hb = make_inputs(rng, static_cast<size_t>(k) * n, -mag, mag);
    const size_t aux_count = uses_bias(ep) ? n : (ep == GemmEpilogue::ResidualAdd ? static_cast<size_t>(m) * n : 1u);
    cb.haux = make_inputs(rng, aux_count, -mag, mag);
    cb.a = device.create_buffer<float>(cb.ha.size());
    cb.b = device.create_buffer<float>(cb.hb.size());
    cb.c = device.create_buffer<float>(static_cast<size_t>(m) * n);
    cb.aux = device.create_buffer<float>(aux_count);
    return cb;
}

[[nodiscard]] uint32_t shared_bytes_of(const GemmTileConfig &cfg) noexcept {
    const auto a = cfg.naive ? cfg.tile_m * cfg.tile_k : cfg.tile_m * (cfg.tile_k + 1u);
    const auto b = cfg.naive ? cfg.tile_k * cfg.tile_n : cfg.tile_k * (cfg.tile_n + 1u);
    return (a + b) * 4u;
}

// Full comparison of every output element (small cases only).
[[nodiscard]] uint32_t verify_full(Stream &stream, const GemmShader &shader, const GemmTileConfig &cfg,
                                   CaseBuffers &cb, uint32_t max_report, float &worst) {
    luisa::vector<float> got(cb.c.size(), 0.f);
    stream << cb.a.copy_from(luisa::span{cb.ha})
           << cb.b.copy_from(luisa::span{cb.hb})
           << cb.aux.copy_from(luisa::span{cb.haux})
           << shader(cb.a.view(), cb.b.view(), cb.c.view(), cb.aux.view(), cb.m, cb.n, cb.k)
                  .dispatch(gemm_dispatch_size_config(cfg, cb.m, cb.n))
           << cb.c.copy_to(luisa::span{got})
           << synchronize();
    uint32_t bad = 0u;
    worst = 0.f;
    for(uint32_t i = 0u; i < cb.m; i++) {
        for(uint32_t j = 0u; j < cb.n; j++) {
            const auto ref = ref_element(cb.ha, cb.hb, cb.haux, cb.ep, i, j, cb.k, cb.n);
            const auto idx = static_cast<size_t>(i) * cb.n + j;
            const float d = std::abs(got[idx] - ref);
            if(d > worst) {
                worst = d;
            }
            if(!(d <= 1e-3f * std::max(1.f, std::abs(ref)))) {
                if(bad < max_report) {
                    LUISA_WARNING("  mismatch (m={}, n={}): got {:.6e} ref {:.6e} diff {:.3e}", i, j,
                                  got[idx], ref, d);
                }
                bad++;
            }
        }
    }
    return bad;
}

}// namespace

int main(int argc, char *argv[]) {
    const auto opt = parse_args(argc, argv);
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();

    const auto backend = argc > 1 ? argv[1] : "?";
    LUISA_INFO("embed gemm sweep: backend {}, engine-resolved config '{}'", backend,
               gemm_resolved_config_source());

    std::ofstream csv_out;
    if(!opt.csv.empty()) {
        const bool need_header = !std::filesystem::exists(opt.csv) || std::filesystem::file_size(opt.csv) == 0u;
        csv_out.open(opt.csv, std::ios::app);
        LUISA_ASSERT(static_cast<bool>(csv_out), "cannot open csv \"{}\"", opt.csv.string());
        if(need_header) {
            csv_out << "backend,variant,tile_m,tile_n,tile_k,reg_m,reg_n,block,naive,swap_grid,"
                       "shape,epilogue,rows,M,N,K,grid_x,grid_y,shared_bytes,proxy_bad,sample_bad,"
                       "worst_proxy_diff,iters,ms_min,ms_median,gflops,status\n";
        }
    }

    const auto vcount = gemm_tile_variant_count();
    luisa::vector<std::optional<GemmShader>> shader_cache(vcount * 5u);// [variant * 5 + epilogue]
    const auto cache_index = [](uint32_t vi, GemmEpilogue ep) noexcept {
        return vi * 5u + static_cast<uint32_t>(ep);
    };
    auto shader_for = [&](uint32_t vi, GemmEpilogue ep) -> GemmShader & {
        auto &slot = shader_cache[cache_index(vi, ep)];
        if(!slot.has_value()) {
            slot.emplace(device.compile(make_gemm_kernel_config(ep, *gemm_tile_variant(vi)),
                                        compute::ShaderOption{.name = gemm_tile_variant(vi)->name}));
        }
        return *slot;
    };

    uint32_t failures = 0u;
    uint32_t timed_rows = 0u;

    //--- phase 0: optional arbitrary debug case -------------------------------
    if(opt.debug_verify) {
        for(auto vi = 0u; vi < vcount; vi++) {
            const auto *cfg = gemm_tile_variant(vi);
            if(!selected(opt.variants, cfg->name)) {
                continue;
            }
            auto cb = make_case(device, opt.v_m, opt.v_n, opt.v_k, opt.v_ep, 1.f);
            float worst = 0.f;
            const auto bad = verify_full(stream, shader_for(vi, opt.v_ep), *cfg, cb, 48u, worst);
            LUISA_INFO("verify '{}' M={} N={} K={} ep={}: bad={} worst={:.3e}", cfg->name, opt.v_m,
                       opt.v_n, opt.v_k, gemm_epilogue_name(opt.v_ep), bad, worst);
            failures += bad;
        }
    }

    //--- phase 1: ragged proxy correctness gate -------------------------------
    constexpr uint32_t kProxyM = 37u, kProxyN = 69u, kProxyK = 53u;
    luisa::vector<char> proxy_ok(vcount, 0);// 1 == passed
    {
        auto cb_bias = make_case(device, kProxyM, kProxyN, kProxyK, GemmEpilogue::Bias, 1.f);
        auto cb_gelu = make_case(device, kProxyM, kProxyN, kProxyK, GemmEpilogue::BiasGeluErf, 1.f);
        for(auto vi = 0u; vi < vcount; vi++) {
            const auto *cfg = gemm_tile_variant(vi);
            if(!selected(opt.variants, cfg->name)) {
                continue;
            }
            float w1 = 0.f, w2 = 0.f;
            // The case data is deterministic, so one buffer set serves every variant.
            const auto bad1 = verify_full(stream, shader_for(vi, GemmEpilogue::Bias), *cfg, cb_bias, 4u, w1);
            const auto bad2 = verify_full(stream, shader_for(vi, GemmEpilogue::BiasGeluErf), *cfg, cb_gelu, 4u, w2);
            const auto bad = bad1 + bad2;
            const auto worst = std::max(w1, w2);
            proxy_ok[vi] = bad == 0u ? 1 : 0;
            failures += bad;
            LUISA_INFO("{}: proxy 37x69x53 bias_bad={} gelu_bad={} worst={:.3e} [{}]", cfg->name, bad1,
                       bad2, worst, bad == 0u ? "PASS" : "FAIL");
            if(csv_out) {
                csv_out << backend << ',' << cfg->name << ',' << cfg->tile_m << ',' << cfg->tile_n << ','
                        << cfg->tile_k << ',' << cfg->reg_m << ',' << cfg->reg_n << ',' << cfg->block << ','
                        << (cfg->naive ? 1 : 0) << ',' << (cfg->swap_grid ? 1 : 0) << ",proxy,"
                        << gemm_epilogue_name(GemmEpilogue::Bias) << ",0," << kProxyM << ',' << kProxyN
                        << ',' << kProxyK << ',' << gemm_dispatch_size_config(*cfg, kProxyM, kProxyN).x << ','
                        << gemm_dispatch_size_config(*cfg, kProxyM, kProxyN).y << ','
                        << shared_bytes_of(*cfg) << ',' << bad << ",0," << worst << ",0,0,0,0,"
                        << (bad == 0u ? "PASS" : "FAIL") << '\n';
            }
        }
    }

    //--- phase 2: timed engine shapes -----------------------------------------
    if(opt.time) {
        for(const auto &shape : kShapes) {
            if(!selected(opt.shapes, shape.name)) {
                continue;
            }
            for(const auto m : opt.rows) {
                auto cb = make_case(device, m, shape.n, shape.k, shape.ep, 1.f);
                LUISA_INFO("--- shape {} M={} N={} K={} (aux {}, C {:.1f} MB) ---", shape.name, m,
                           shape.n, shape.k, uses_bias(shape.ep) ? "bias[N]" : "residual[M,N]",
                           static_cast<double>(m) * shape.n * 4.0 / 1048576.0);
                for(auto vi = 0u; vi < vcount; vi++) {
                    const auto *cfg = gemm_tile_variant(vi);
                    if(!selected(opt.variants, cfg->name)) {
                        continue;
                    }
                    if(proxy_ok[vi] == 0u) {
                        continue;// already reported in phase 1
                    }
                    auto &run_cb = cb;
                    auto &shader = shader_for(vi, shape.ep);
                    const auto disp = gemm_dispatch_size_config(*cfg, m, shape.n);

                    // Sampled ground truth on the timed shape (4 rows x 4 cols).
                    luisa::vector<float> out_row(shape.n, 0.f);
                    uint32_t sample_bad = 0u;
                    {
                        stream << run_cb.a.copy_from(luisa::span{run_cb.ha})
                               << run_cb.b.copy_from(luisa::span{run_cb.hb})
                               << run_cb.aux.copy_from(luisa::span{run_cb.haux})
                               << shader(run_cb.a.view(), run_cb.b.view(), run_cb.c.view(),
                                         run_cb.aux.view(), m, shape.n, shape.k)
                                      .dispatch(disp)
                               << synchronize();
                        const uint32_t rows[] = {0u, m / 3u, m / 2u, m - 1u};
                        const uint32_t cols[] = {0u, shape.n / 4u, shape.n / 2u, shape.n - 1u};
                        for(const auto i : rows) {
                            stream << run_cb.c.view(static_cast<size_t>(i) * shape.n, shape.n)
                                          .copy_to(luisa::span{out_row})
                                   << synchronize();
                            for(const auto j : cols) {
                                const auto ref =
                                    ref_element(run_cb.ha, run_cb.hb, run_cb.haux, shape.ep, i, j, shape.k, shape.n);
                                const float d = std::abs(out_row[j] - ref);
                                if(!(d <= 1e-3f * std::max(1.f, std::abs(ref)))) {
                                    if(sample_bad < 4u) {
                                        LUISA_WARNING("  sample mismatch (m={}, n={}): got {:.6e} ref {:.6e}",
                                                      i, j, out_row[j], ref);
                                    }
                                    sample_bad++;
                                }
                            }
                        }
                    }

                    double ms_min = 0.0, ms_med = 0.0, gflops = 0.0;
                    uint32_t iters_run = 0u;
                    if(sample_bad == 0u) {
                        const auto enqueue = [&] {
                            stream << shader(run_cb.a.view(), run_cb.b.view(), run_cb.c.view(),
                                             run_cb.aux.view(), m, shape.n, shape.k)
                                          .dispatch(disp);
                        };
                        for(auto w = 0u; w < opt.warmup; w++) {
                            enqueue();
                            stream << synchronize();
                        }
                        luisa::vector<double> samples;
                        for(auto it = 0u; it < opt.iters; it++) {
                            Clock clock;
                            enqueue();
                            stream << synchronize();
                            samples.emplace_back(clock.toc());
                            iters_run++;
                        }
                        const auto st = summarize(std::move(samples));
                        ms_min = st.min;
                        ms_med = st.median;
                        gflops = 2.0 * static_cast<double>(m) * shape.n * shape.k / (ms_min * 1.0e6);
                        timed_rows++;
                    } else {
                        failures += sample_bad;
                    }

                    LUISA_INFO("  {:<9} rows={:<5} grid={}x{:<5} shared={:.1f}KiB sample_bad={} "
                               "min={:.3f}ms med={:.3f}ms {:.1f} GFLOPS [{}]",
                               cfg->name, m, disp.x, disp.y, shared_bytes_of(*cfg) / 1024.0, sample_bad,
                               ms_min, ms_med, gflops, sample_bad == 0u ? "PASS" : "FAIL");
                    if(csv_out) {
                        csv_out << backend << ',' << cfg->name << ',' << cfg->tile_m << ',' << cfg->tile_n
                                << ',' << cfg->tile_k << ',' << cfg->reg_m << ',' << cfg->reg_n << ','
                                << cfg->block << ',' << (cfg->naive ? 1 : 0) << ','
                                << (cfg->swap_grid ? 1 : 0) << ',' << shape.name << ','
                                << gemm_epilogue_name(shape.ep) << ',' << m << ',' << m << ',' << shape.n
                                << ',' << shape.k << ',' << disp.x << ',' << disp.y << ','
                                << shared_bytes_of(*cfg) << ",0," << sample_bad << ",0," << iters_run << ','
                                << ms_min << ',' << ms_med << ',' << gflops << ','
                                << (sample_bad == 0u ? "PASS" : "FAIL") << '\n';
                    }
                }
            }
        }
    }

    LUISA_INFO("sweep done: {} variants, {} timed rows, {} mismatches total, csv: {}", vcount, timed_rows,
               failures, opt.csv.empty() ? "(none)" : opt.csv.string());
    expect(failures == 0u) << luisa::format("gemm sweep found {} mismatching outputs", failures);
    return 0;
}

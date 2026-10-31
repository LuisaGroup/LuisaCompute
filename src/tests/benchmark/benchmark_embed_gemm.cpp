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
// The engine's GEMM variant table (gemm_tile_variant) is enumerated in full,
// so the vector register-blocked `rbv_*` family (tensor_optimize.md §4) enters
// this sweep automatically -- no special handling here.
//
// Phase 3 -- coop (cooperative-vector) matvec sweep (tensor_optimize.md §5.3,
// Phase 2 "coop K-slab x N-tile matvec variant"): appended after the GEMM-table
// sweep, mirroring its two-gate-before-timing discipline. It enumerates the coop
// variant table (coop_gemm_variant, examples/embedding/embed/kernels/
// kernels_coop.h -- 4 entries) and times those variants on the SAME four engine
// shapes. The coop host contract (kernels_coop.h header) is stricter than the
// tiled GEMM and the harness honours it:
//   - fp32 weights ONLY. This benchmark already feeds synthetic fp32 host data,
//     so it always allocates plain fp32 A/B/bias and binds them ZERO-COPY as
//     compute::ByteBufferView{buf} (coop_api.md §4). If the engine ever grows a
//     packed (bf16/fp16) weight path, coop variants are simply left on fp32 --
//     packed formats MUST NOT enter coop (plan §1.3/§5.5); a printed note makes
//     that explicit.
//   - K % kslab == 0 and N % ntile == 0 are GUARANTEED BY THE CALLER (ragged
//     K/N is rejected by the kernel with a device_assert -- there is no tail
//     predicate). So the coop Phase-1 gate cannot reuse the ragged 37x69x53
//     proxy the tiled GEMM uses; instead it gates on a small CLEAN shape
//     M=37, N=64, K=128 (K=128 and N=64 are multiples of EVERY kslab/ntile in
//     the coop table: kslab in {64,128}, ntile in {16,32}). NOTE: the plan
//     text wrote "K=64" but the table ships coop_k128_n32_t1 (kslab=128), so
//     K=64 would trip that variant's device_assert and the gate could never
//     pass for it; K=128 = lcm(all kslabs) satisfies the stated intent
//     ("multiples of every kslab/ntile"). M=37 keeps ragged-M coverage (the
//     kernel guards it), verified FULLY against
//     the fp64 reference with the same gate 1e-3*max(1,|ref|) on BOTH epilogues
//     the coop v1 path emits (Bias, BiasGeluErf). Gate (b) is a 16-point sample
//     on each timed shape (the same points as the tiled Phase-2 sample). A
//     failed gate reports + skips timing and counts in failures -- no
//     plausible-but-wrong numbers reach the table.
//   - The four timed shapes have K,N in {1024,3072,4096}, all multiples of
//     64/128, and every coop ntile <= 32, so they satisfy the contract; the
//     harness still asserts this and skips a shape gracefully otherwise.
//   - coop runs on DX / VK ONLY (plan §2/§5: cooperative vector is DX/VK only;
//     CUDA and fallback have NO cooperative codegen path -- the CUDA AST
//     backend and the fallback backend hit LUISA_NOT_IMPLEMENTED, which ABORTS
//     the process rather than throwing). The harness therefore NEVER attempts a
//     coop compile on a non-dx/vk backend: it reads the backend from the device
//     and, for cuda/fallback/others, prints the skip reason and skips the whole
//     coop section before touching the coop factory. The gate is by backend
//     name (the task's scope). On vk coop is attempted directly -- the real
//     per-driver cooperative-vector capability (VK_NV_cooperative_vector +
//     fp32 + replicated composites) is the subject of the dedicated Phase-0
//     probe test (test_embed_coop_probe.cpp), not this benchmark. On dx, WHEN a
//     coop_* variant can be selected at all, the PRIMARY device is created with
//     DirectXDeviceConfigExt::UseExperimental() (SM 6.9) -- the experimental
//     gate is process-global and must be set before ANY device is created, so it
//     is folded into the first dx device (same as test_embed_coop_probe's
//     make_test_device), which is the device the coop kernels then compile
//     against. If ExperimentalFeaturesEnabled() comes back false (OS/runtime
//     rejected the SM 6.9 feature), coop is skipped with the no-go reason so the
//     coop compile -- which would abort via LUISA_ERROR -- is never reached.
//     Without coop in the selection (--variants naming only naive_*/rb_*/rbv_*
//     GEMM variants) the ordinary device is created instead: asking for the
//     SM 6.9 experimental features there can make D3D12CreateDevice fail with
//     0x887e0003 on drivers that do not expose the preview runtime, and the
//     tiled GEMM sweep has no use for them.
//     The Agility-SDK selection is gated by that SAME condition -- gating only
//     UseExperimental() is not enough. The executable exports D3D12SDKVersion
//     (mutable, initialised to 0 == "system D3D12") plus D3D12SDKPath, and
//     main raises the version to 619 immediately before device creation, and
//     only when a coop_* variant is selected. An unconditional 619 tells the
//     D3D12 loader to look for the 1.619 Agility runtime under D3D12SDKPath for
//     EVERY device creation, so a GEMM-only run then fails D3D12CreateDevice
//     with 0x887e0003 even though no experimental feature was requested (this
//     tree ships no D3D12/ runtime next to the binary). The late write is
//     honoured because the harness exe imports neither d3d12.dll nor dxgi.dll:
//     the backend DLL -- the only d3d12.dll importer -- is loaded at the first
//     device creation, i.e. after main has already set the value.
//
// Usage
// -----
//   benchmark_embed_gemm <backend:dx|vk|cuda|fallback> [options]
//     --rows 512,4096        row counts to time (default 512,4096)
//     --iters N              timed repetitions, min-of-N (default 10)
//     --warmup N             untimed warm-up dispatches (default 3)
//     --variants a,b,c       only these variant names (default: whole GEMM +
//                            coop tables). The list matches BOTH the tiled GEMM
//                            variant names (naive_*, rb_*, rbv_* vec4 family --
//                            rbv_* enters automatically) and the coop names
//                            (coop_k64_n16_t1, coop_k64_n32_t1,
//                            coop_k128_n32_t1, coop_k64_n32_t2). Listing no
//                            coop_* name keeps the dx device non-experimental
//                            AND leaves the D3D12SDKVersion export at 0, so a
//                            GEMM-only sweep needs neither the SM 6.9 preview
//                            runtime nor a shipped Agility SDK 1.619 runtime.
//     --shapes qkv,ffn_up    only these shapes (default: all four)
//     --no-time              phase 1 only (correctness)
//     --verify M N K [ep]    extra debug case on every selected tiled variant
//                            (ep: 0 none, 1 bias, 2 gelu; default 1); prints
//                            every mismatching (row, col)
//     --csv FILE             append rows (header written once)
//
// Backends: dx, vk, cuda (legacy + rbv_* vector only; coop skipped with a
// printed reason), and fallback (CPU oracle; coop skipped). CUDA is listed here
// per tensor_optimize.md Phase 2 ("补齐其 CUDA 验证和使用说明") -- the tiled GEMM
// sweep (legacy + vector rbv_*) runs on cuda; the coop section is dx/vk only.
//
// CSV column convention for coop rows (same columns as the tiled sweep):
//   variant = the coop name; tile_n carries cfg.ntile, tile_k carries
//   cfg.kslab, reg_n carries cfg.tokens_per_block, block carries
//   kCoopGemmBlock (256); tile_m, reg_m, naive, swap_grid, shared_bytes are 0
//   (the coop matvec tiles along K and N only -- one accumulator per thread,
//   no M-tiling, no register-blocking, no shared-memory tile; bytes are
//   register coop-vectors, not SMEM). grid_x = coop_gemm_dispatch_size_config
//   total threads, grid_y = 1. The Phase-1 gate rows report shape="coop_proxy"
//   (M=37,N=64,K=128) with proxy_bad set; timed rows carry sample_bad.
//
// Notes
// -----
// * dispatch() counts TOTAL threads, not blocks (lc_optimize trap #1) -- the
//   geometry comes from the engine's own gemm_dispatch_size_config() /
//   coop_gemm_dispatch_size_config() so the harness can never disagree with the
//   kernel.
// * GFLOP/s uses the same 2*m*n*k formula for both the tiled and the coop
//   variants (min-of-N ms).
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
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#ifdef _WIN32
#include <luisa/backends/ext/dx_config_ext.h>
#include <luisa/core/stl/memory.h>
// Required by the D3D12 Agility SDK: these exports must be in the main .exe,
// not in a loaded DLL, so the D3D12 loader can find the SDK runtime. Only the
// coop dx device needs them, to compile/run cooperative-vector (SM 6.9)
// kernels. (Same mechanism as src/tests/unit/ast/test_cooperative_vector.cpp:
// 29-32, except that the version is gated at runtime -- see below.)
//
// D3D12SDKVersion is deliberately NOT const and starts at 0: 0 selects the
// ordinary system D3D12 runtime, which is what every non-coop run (a GEMM-only
// --variants sweep included) must use. Exporting 619 unconditionally makes the
// D3D12 loader look for the 1.619 Agility runtime under D3D12SDKPath for EVERY
// device creation; since this tree ships no D3D12/ runtime next to the binary,
// that turns into D3D12CreateDevice 0x887e0003 even when the experimental
// (SM 6.9) feature was never requested -- i.e. gating UseExperimental() alone
// does not make a GEMM-only run work. create_primary_device() therefore raises
// the value to kCoopAgilitySDKVersion before the first dx device is created,
// and only when coop_requested()/want_coop is true. That write is seen by the
// loader because this exe imports neither d3d12.dll nor dxgi.dll: the backend
// DLL (the only d3d12.dll importer) is loaded at the first device creation,
// long after main has run. D3D12SDKPath stays exported unconditionally -- the
// loader only reads it when the version selects the Agility path.
extern "C" __declspec(dllexport) uint32_t D3D12SDKVersion = 0u;
extern "C" __declspec(dllexport) const char *D3D12SDKPath = ".\\D3D12\\";

// Agility SDK 1.619 == the SM 6.9 / cooperative-vector runtime the coop dx
// kernels need. Assigned to D3D12SDKVersion right before the coop device is
// created (create_primary_device), never at static-init time.
constexpr uint32_t kCoopAgilitySDKVersion = 619u;
#endif

#include "embedding/embed/kernels/kernels_gemm.h"
#include "embedding/embed/kernels/kernels_coop.h"

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
    while (*p != '\0') {
        const char *q = p;
        while (*q != '\0' && *q != ',') {
            q++;
        }
        if (q > p) {
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
    for (int i = 2; i < argc; i++) {
        const std::string_view arg{argv[i]};
        const auto need = [&](std::string_view name) -> const char * {
            LUISA_ASSERT(i + 1 < argc, "flag {} requires a value", name);
            return argv[++i];
        };
        if (arg == "--rows") {
            opt.rows.clear();
            split_u32(need("rows"), [&](const char *b, const char *e) {
                uint32_t v{};
                const auto r = std::from_chars(b, e, v);
                LUISA_ASSERT(r.ec == std::errc{} && r.ptr == e, "rows must be integers");
                opt.rows.emplace_back(v);
            });
        } else if (arg == "--iters") {
            opt.iters = parse_u32(need("iters"), "iters");
        } else if (arg == "--warmup") {
            opt.warmup = parse_u32(need("warmup"), "warmup");
        } else if (arg == "--csv") {
            opt.csv = std::string{need("csv")};
        } else if (arg == "--variants") {
            split_names(need("variants"), [&](luisa::string t) { opt.variants.emplace_back(std::move(t)); });
        } else if (arg == "--shapes") {
            split_names(need("shapes"), [&](luisa::string s) { opt.shapes.emplace_back(std::move(s)); });
        } else if (arg == "--no-time") {
            opt.time = false;
        } else if (arg == "--verify") {
            opt.debug_verify = true;
            opt.v_m = parse_u32(need("M"), "M");
            opt.v_n = parse_u32(need("N"), "N");
            opt.v_k = parse_u32(need("K"), "K");
            if (i + 1 < argc && argv[i + 1][0] != '-') {
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
    if (list.empty()) {
        return true;
    }
    for (const auto &s : list) {
        if (s == name) {
            return true;
        }
    }
    return false;
}

// Whether the --variants filter can select a cooperative-vector (coop_*) kernel.
// Coop is the only section that needs the dx experimental SM 6.9 device, so the
// device is created with DirectXDeviceConfigExt::UseExperimental() iff this is
// true: an empty filter means "whole GEMM + coop tables" (coop selected), while
// an explicit filter selects coop iff at least one name carries the coop table's
// "coop_" prefix. A naive_/rb_/rbv_-only filter therefore keeps the ordinary
// device. The SAME flag also gates the Agility-SDK selection (D3D12SDKVersion is
// raised from 0 to 619 only when this returns true), so this predicate decides
// both dx process-global gates, not just UseExperimental(). (Same matching rule
// as selected(): the coop table names are exactly the coop_* names enumerated in
// kernels_coop.h.)
[[nodiscard]] bool coop_requested(const luisa::vector<luisa::string> &list) noexcept {
    if (list.empty()) {
        return true;
    }
    for (const auto &s : list) {
        if (std::string_view{s}.starts_with("coop_")) {
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
    for (uint32_t i = 0u; i < cb.m; i++) {
        for (uint32_t j = 0u; j < cb.n; j++) {
            const auto ref = ref_element(cb.ha, cb.hb, cb.haux, cb.ep, i, j, cb.k, cb.n);
            const auto idx = static_cast<size_t>(i) * cb.n + j;
            const float d = std::abs(got[idx] - ref);
            if (d > worst) {
                worst = d;
            }
            if (!(d <= 1e-3f * std::max(1.f, std::abs(ref)))) {
                if (bad < max_report) {
                    LUISA_WARNING("  mismatch (m={}, n={}): got {:.6e} ref {:.6e} diff {:.3e}", i, j,
                                  got[idx], ref, d);
                }
                bad++;
            }
        }
    }
    return bad;
}

//=== coop (cooperative-vector matvec) section ===================================
// (tensor_optimize.md §5.3 / Phase 2). Mirrors the tiled Phase 1/2 structure but
// against the coop variant table and the stricter coop host contract: fp32
// weights, K % kslab == 0, N % ntile == 0, DX/VK only.

// Compiled type of the engine CoopGemmKernel (same trick as GemmShader above,
// kept local so the harness does not depend on the executor).
using CoopShader = decltype(std::declval<Device &>().compile(std::declval<CoopGemmKernel>(),
                                                              std::declval<ShaderOption &>()));

#ifdef _WIN32
// DX device config that turns on the experimental shader models (SM 6.9) the
// cooperative-vector kernels require (tensor_optimize.md §2.2). Same pattern as
// src/tests/integration/embedding/test_embed_coop_probe.cpp:168-201.
class DXExperimentalConfigExt final : public DirectXDeviceConfigExt {
public:
    [[nodiscard]] bool UseExperimental() const noexcept override { return true; }
};
#endif

// Create the primary device. On dx, when the filter can select a coop_* variant
// (`want_coop`) the device is created with the experimental-SM-6.9 config ext
// (same as test_embed_coop_probe's make_test_device) so that the coop kernels
// can actually compile on the SAME device the harness uses; when the filter can
// only reach the tiled GEMM variants the ordinary luisa::test::create_device()
// device is used, since requesting the experimental features would fail
// D3D12CreateDevice (0x887e0003) on drivers without the SM 6.9 preview runtime
// for no benefit. D3D12EnableExperimentalFeatures MUST be called before any
// device is created, so folding the ext into the first dx device is the only
// correct place to do it (a throwaway second device would find the feature gate
// already locked by the first). Two process-global gates are gated by
// `want_coop`, not one: UseExperimental() (SM 6.9 feature) AND the D3D12
// Agility-SDK selection -- D3D12SDKVersion is left at 0 (= system D3D12) for a
// non-coop run, otherwise the loader would hunt for the 1.619 runtime under
// D3D12SDKPath and D3D12CreateDevice would fail 0x887e0003 with no experimental
// feature requested. `dx_experimental_enabled` (out) reports whether the coop
// dx device got the SM 6.9 path (false -> coop is skipped, never compiled, to
// avoid the uncatchable LUISA_ERROR abort inside the coop compile).
[[nodiscard]] luisa::test::DeviceContext create_primary_device(int argc, char *argv[],
                                                               luisa::string_view backend,
                                                               bool want_coop,
                                                               bool &dx_experimental_enabled) {
    dx_experimental_enabled = true;// non-dx: irrelevant, set true so coop is not gated on it
#ifdef _WIN32
    if (backend == "dx") {
        // Gate the Agility-SDK selection exactly like the feature above: the
        // export stays 0 (= ordinary system D3D12) unless this run can reach a
        // coop_* variant. Writing it here is early enough -- the exe imports
        // neither d3d12.dll nor dxgi.dll, so the D3D12 loader only comes up
        // with the backend DLL inside create_device(), below.
        if (want_coop) {
            D3D12SDKVersion = kCoopAgilitySDKVersion;
        }
        if (!want_coop) {
            // Tiled GEMM only: neither process-global gate is touched -- no
            // experimental feature is requested and D3D12SDKVersion keeps its
            // initial 0, so the loader uses the ordinary system D3D12 and the
            // device is created exactly like test_embed_kernel_gemm does.
            dx_experimental_enabled = false;
            return luisa::test::create_device(argc, argv);
        }
        const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : luisa::test::safe_argv0();
        Context context{exe != nullptr && exe[0] != '\0' ? exe : "benchmark_embed_gemm"};
        DeviceConfig config;
        auto dx_config = luisa::make_unique<DXExperimentalConfigExt>();
        auto *dx_config_ptr = dx_config.get();
        config.extension = std::move(dx_config);
        auto device = context.create_device("dx", &config);
        luisa::test::log_test_backend("dx", device);
        dx_experimental_enabled = dx_config_ptr->ExperimentalFeaturesEnabled();
        return luisa::test::DeviceContext{std::move(context), std::move(device)};
    }
#else
    static_cast<void>(backend);
    static_cast<void>(want_coop);
#endif
    return luisa::test::create_device(argc, argv);
}

// Coop case buffers: the coop kernel takes opaque ByteBuffers (the coop ops
// address operands by BYTE offset -- kernels_coop.h), so the same fp32 host
// data as the tiled case is bound ZERO-COPY as ByteBufferView{buf}.
struct CoopCaseBuffers {
    uint32_t m, n, k;
    GemmEpilogue ep;
    luisa::vector<float> ha, hb, haux;
    Buffer<float> a, b, c, aux;
};

[[nodiscard]] CoopCaseBuffers make_coop_case(Device &device, uint32_t m, uint32_t n, uint32_t k,
                                             GemmEpilogue ep, float mag) {
    CoopCaseBuffers cb;
    cb.m = m;
    cb.n = n;
    cb.k = k;
    cb.ep = ep;
    Lcg rng{};
    // Distinct seed from make_case() so coop and tiled sweeps validate against
    // the same fp64 reference formula but on independent data.
    rng.state = 0x85EBCA52u ^ (m << 24u) ^ (n << 12u) ^ (k << 2u) ^ static_cast<uint32_t>(ep);
    cb.ha = make_inputs(rng, static_cast<size_t>(m) * k, -mag, mag);
    cb.hb = make_inputs(rng, static_cast<size_t>(k) * n, -mag, mag);
    cb.haux = make_inputs(rng, n, -mag, mag);// bias[N] (coop only emits Bias/BiasGeluErf)
    cb.a = device.create_buffer<float>(cb.ha.size());
    cb.b = device.create_buffer<float>(cb.hb.size());
    cb.c = device.create_buffer<float>(static_cast<size_t>(m) * n);
    cb.aux = device.create_buffer<float>(n);
    return cb;
}

// Shape predicate the coop host contract requires of a timed shape: K a
// multiple of kslab, N a multiple of ntile (ragged K/N is rejected by the
// kernel; ragged M is handled by the per-token guard).
[[nodiscard]] bool coop_shape_ok(const CoopGemmConfig &cfg, uint32_t n, uint32_t k) noexcept {
    return cfg.kslab != 0u && cfg.ntile != 0u && (k % cfg.kslab) == 0u && (n % cfg.ntile) == 0u;
}

// Full comparison of every output element for a coop variant (small cases
// only). The coop kernel reads A as x[M,K], B as weight[K,N] row-major, the
// aux as bias[N]; the reference (ref_element) is the same A@B+bias formula.
[[nodiscard]] uint32_t coop_verify_full(Stream &stream, const CoopShader &shader,
                                        const CoopGemmConfig &cfg, CoopCaseBuffers &cb,
                                        uint32_t max_report, float &worst) {
    luisa::vector<float> got(cb.c.size(), 0.f);
    stream << cb.a.copy_from(luisa::span{cb.ha})
           << cb.b.copy_from(luisa::span{cb.hb})
           << cb.aux.copy_from(luisa::span{cb.haux})
           << shader(ByteBufferView{cb.a}, ByteBufferView{cb.b}, ByteBufferView{cb.aux},
                     ByteBufferView{cb.c}, cb.m, cb.n, cb.k)
                  // Kernel1D: dispatch() takes TOTAL threads (lc_optimize trap #1);
                  // coop_gemm_dispatch_size_config returns uint2 {total,1}.
                  .dispatch(coop_gemm_dispatch_size_config(cfg, cb.m).x)
           << cb.c.copy_to(luisa::span{got})
           << synchronize();
    uint32_t bad = 0u;
    worst = 0.f;
    for (uint32_t i = 0u; i < cb.m; i++) {
        for (uint32_t j = 0u; j < cb.n; j++) {
            const auto ref = ref_element(cb.ha, cb.hb, cb.haux, cb.ep, i, j, cb.k, cb.n);
            const auto idx = static_cast<size_t>(i) * cb.n + j;
            const float d = std::abs(got[idx] - ref);
            if (d > worst) {
                worst = d;
            }
            if (!(d <= 1e-3f * std::max(1.f, std::abs(ref)))) {
                if (bad < max_report) {
                    LUISA_WARNING("  coop mismatch (m={}, n={}): got {:.6e} ref {:.6e} diff {:.3e}",
                                  i, j, got[idx], ref, d);
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
    const auto backend = argc > 1 ? argv[1] : "?";
    // Coop is the only consumer of the dx experimental (SM 6.9) device -- and of
    // the Agility SDK 1.619 selection, which is gated by the same flag -- so the
    // device creation is decided by the --variants filter (see coop_requested).
    const bool want_coop = coop_requested(opt.variants);
    bool dx_experimental_enabled = true;
    auto [context, device] = create_primary_device(argc, argv, luisa::string_view{backend},
                                                   want_coop, dx_experimental_enabled);
    auto stream = device.create_stream();

    LUISA_INFO("embed gemm sweep: backend {}, engine-resolved config '{}'", backend,
               gemm_resolved_config_source());

    std::ofstream csv_out;
    if (!opt.csv.empty()) {
        const bool need_header = !std::filesystem::exists(opt.csv) || std::filesystem::file_size(opt.csv) == 0u;
        csv_out.open(opt.csv, std::ios::app);
        LUISA_ASSERT(static_cast<bool>(csv_out), "cannot open csv \"{}\"", opt.csv.string());
        if (need_header) {
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
        if (!slot.has_value()) {
            slot.emplace(device.compile(make_gemm_kernel_config(ep, *gemm_tile_variant(vi)),
                                        compute::ShaderOption{.name = gemm_tile_variant(vi)->name}));
        }
        return *slot;
    };

    uint32_t failures = 0u;
    uint32_t timed_rows = 0u;

    //--- phase 0: optional arbitrary debug case -------------------------------
    if (opt.debug_verify) {
        for (auto vi = 0u; vi < vcount; vi++) {
            const auto *cfg = gemm_tile_variant(vi);
            if (!selected(opt.variants, cfg->name)) {
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
        for (auto vi = 0u; vi < vcount; vi++) {
            const auto *cfg = gemm_tile_variant(vi);
            if (!selected(opt.variants, cfg->name)) {
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
            if (csv_out) {
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
    if (opt.time) {
        for (const auto &shape : kShapes) {
            if (!selected(opt.shapes, shape.name)) {
                continue;
            }
            for (const auto m : opt.rows) {
                auto cb = make_case(device, m, shape.n, shape.k, shape.ep, 1.f);
                LUISA_INFO("--- shape {} M={} N={} K={} (aux {}, C {:.1f} MB) ---", shape.name, m,
                           shape.n, shape.k, uses_bias(shape.ep) ? "bias[N]" : "residual[M,N]",
                           static_cast<double>(m) * shape.n * 4.0 / 1048576.0);
                for (auto vi = 0u; vi < vcount; vi++) {
                    const auto *cfg = gemm_tile_variant(vi);
                    if (!selected(opt.variants, cfg->name)) {
                        continue;
                    }
                    if (proxy_ok[vi] == 0u) {
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
                        for (const auto i : rows) {
                            stream << run_cb.c.view(static_cast<size_t>(i) * shape.n, shape.n)
                                          .copy_to(luisa::span{out_row})
                                   << synchronize();
                            for (const auto j : cols) {
                                const auto ref =
                                    ref_element(run_cb.ha, run_cb.hb, run_cb.haux, shape.ep, i, j, shape.k, shape.n);
                                const float d = std::abs(out_row[j] - ref);
                                if (!(d <= 1e-3f * std::max(1.f, std::abs(ref)))) {
                                    if (sample_bad < 4u) {
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
                    if (sample_bad == 0u) {
                        const auto enqueue = [&] {
                            stream << shader(run_cb.a.view(), run_cb.b.view(), run_cb.c.view(),
                                             run_cb.aux.view(), m, shape.n, shape.k)
                                          .dispatch(disp);
                        };
                        for (auto w = 0u; w < opt.warmup; w++) {
                            enqueue();
                            stream << synchronize();
                        }
                        luisa::vector<double> samples;
                        for (auto it = 0u; it < opt.iters; it++) {
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
                    if (csv_out) {
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

    //--- phase 3: coop (cooperative-vector) matvec sweep ----------------------
    // (tensor_optimize.md §5.3 / Phase 2). Runs on dx/vk ONLY: cooperative
    // vector has no executable path on cuda/fallback/others, and the coop
    // lowering there ABORTS (LUISA_NOT_IMPLEMENTED / LUISA_ERROR) rather than
    // throwing -- so the whole section is skipped before touching the coop
    // factory on any other backend. The gate is by backend name (the same name
    // the device was created with); on dx the primary device reports whether the
    // SM 6.9 experimental shader models actually got enabled (a DX machine
    // without them would otherwise abort inside the coop compile). The section
    // is also skipped up front when the --variants filter names no coop_*
    // variant: in that case the device was deliberately created WITHOUT the
    // experimental config ext, so a "not enabled" verdict would be meaningless.
    {
        const luisa::string_view backend_name{backend};
        const bool coop_backend = backend_name == "dx" || backend_name == "vk";
        bool coop_go = coop_backend && want_coop;
        if (coop_go && backend_name == "dx") {
            coop_go = dx_experimental_enabled;
        }
        if (!coop_go) {
            luisa::string why;
            if (!coop_backend) {
                why = "cooperative vector is DX/VK only (plan §2/§5); this backend has no coop codegen and the coop compile would abort";
            } else if (!want_coop) {
                why = "--variants selects no coop_* variant (tiled GEMM only), so the experimental dx device is not requested";
            } else {
                why = "DX experimental SM 6.9 features are not enabled on this system (plan §9); coop disabled";
            }
            LUISA_INFO("coop sweep: SKIPPED on backend '{}' -- {}.", backend_name, why);
            if (csv_out) {
                csv_out << backend << ",coop,0,0,0,0,0," << kCoopGemmBlock
                        << ",0,0,(skipped),,0,0,0,0,0,0,0,0,0,0,0,0,0,0,SKIPPED\n";
            }
        } else {
            const auto coop_vcount = coop_gemm_variant_count();
            luisa::vector<std::optional<CoopShader>> coop_cache(coop_vcount * 5u);
            const auto coop_cache_index = [](uint32_t vi, GemmEpilogue ep) noexcept {
                return vi * 5u + static_cast<uint32_t>(ep);
            };
            auto coop_shader_for = [&](uint32_t ci, GemmEpilogue ep) -> CoopShader & {
                auto &slot = coop_cache[coop_cache_index(ci, ep)];
                if (!slot.has_value()) {
                    const auto *ccfg = coop_gemm_variant(ci);
                    slot.emplace(device.compile(
                        make_coop_gemm_kernel(ep, *ccfg, ccfg->name, GemmWeightFormat::Fp32),
                        compute::ShaderOption{.name = ccfg->name}));
                }
                return *slot;
            };
            LUISA_INFO("coop sweep: backend {} (coop GO), {} coop variants.", backend_name, coop_vcount);

            //--- coop phase 1: clean-shape correctness gate (ragged K/N N/A) --
            // M=37 (ragged M is guarded in-kernel), N=64, K=128: K/N are
            // multiples of every coop kslab {64,128} / ntile {16,32}. coop
            // rejects ragged K/N by contract (device_assert), so the tiled
            // 37x69x53 proxy cannot be reused; gate on both emitted epilogues.
            constexpr uint32_t kCoopProxyM = 37u, kCoopProxyN = 64u, kCoopProxyK = 128u;
            luisa::vector<char> coop_proxy_ok(coop_vcount, 0);// 1 == passed
            {
                auto cb_bias = make_coop_case(device, kCoopProxyM, kCoopProxyN, kCoopProxyK,
                                              GemmEpilogue::Bias, 1.f);
                auto cb_gelu = make_coop_case(device, kCoopProxyM, kCoopProxyN, kCoopProxyK,
                                              GemmEpilogue::BiasGeluErf, 1.f);
                for (auto ci = 0u; ci < coop_vcount; ci++) {
                    const auto *ccfg = coop_gemm_variant(ci);
                    if (!selected(opt.variants, ccfg->name)) {
                        continue;
                    }
                    // fp32-only: the coop host contract forbids packed half/bf16
                    // (plan §1.3/§5.5); this harness feeds synthetic fp32 host
                    // data, so it always binds plain fp32 buffers (see the header
                    // note). No packed path is taken here even if the engine adds
                    // one later.
                    float w1 = 0.f, w2 = 0.f;
                    const auto bad1 = coop_verify_full(stream, coop_shader_for(ci, GemmEpilogue::Bias),
                                                       *ccfg, cb_bias, 4u, w1);
                    const auto bad2 = coop_verify_full(stream,
                                                       coop_shader_for(ci, GemmEpilogue::BiasGeluErf),
                                                       *ccfg, cb_gelu, 4u, w2);
                    const auto bad = bad1 + bad2;
                    const auto worst = std::max(w1, w2);
                    coop_proxy_ok[ci] = bad == 0u ? 1 : 0;
                    failures += bad;
                    LUISA_INFO("coop {}: proxy 37x64x128 bias_bad={} gelu_bad={} worst={:.3e} [{}]",
                               ccfg->name, bad1, bad2, worst, bad == 0u ? "PASS" : "FAIL");
                    if (csv_out) {
                        const auto disp = coop_gemm_dispatch_size_config(*ccfg, kCoopProxyM);
                        csv_out << backend << ',' << ccfg->name << ",0," << ccfg->ntile << ','
                                << ccfg->kslab << ",0," << ccfg->tokens_per_block << ','
                                << kCoopGemmBlock << ",0,0,coop_proxy,"
                                << gemm_epilogue_name(GemmEpilogue::Bias) << ",0," << kCoopProxyM
                                << ',' << kCoopProxyN << ',' << kCoopProxyK << ',' << disp.x << ','
                                << disp.y << ",0," << bad << ",0," << worst << ",0,0,0,0,"
                                << (bad == 0u ? "PASS" : "FAIL") << '\n';
                    }
                }
            }

            //--- coop phase 2: timed engine shapes ----------------------------
            if (opt.time) {
                for (const auto &shape : kShapes) {
                    if (!selected(opt.shapes, shape.name)) {
                        continue;
                    }
                    // coop epilogues are Bias / BiasGeluErf only; the four engine
                    // shapes are all Bias/BiasGeluErf, so nothing is dropped here.
                    for (const auto m : opt.rows) {
                        auto cb = make_coop_case(device, m, shape.n, shape.k, shape.ep, 1.f);
                        for (auto ci = 0u; ci < coop_vcount; ci++) {
                            const auto *ccfg = coop_gemm_variant(ci);
                            if (!selected(opt.variants, ccfg->name)) {
                                continue;
                            }
                            if (coop_proxy_ok[ci] == 0u) {
                                continue;// already reported in coop phase 1
                            }
                            // Ragged-K/N guard (coop contract). All four engine
                            // shapes have K,N in {1024,3072,4096} (multiples of
                            // 64/128) and every coop ntile <= 32, so this holds;
                            // skip gracefully otherwise.
                            if (!coop_shape_ok(*ccfg, shape.n, shape.k)) {
                                LUISA_INFO("  coop {:<16} rows={:<5} skipped: shape K={} N={} not a "
                                           "multiple of kslab={} ntile={} (ragged K/N -> tiled GEMM)",
                                           ccfg->name, m, shape.k, shape.n, ccfg->kslab, ccfg->ntile);
                                continue;
                            }
                            auto &shader = coop_shader_for(ci, shape.ep);
                            const auto disp = coop_gemm_dispatch_size_config(*ccfg, m);

                            // Sampled ground truth on the timed shape (same 16
                            // points as the tiled phase-2 sample).
                            luisa::vector<float> out_row(shape.n, 0.f);
                            uint32_t sample_bad = 0u;
                            {
                                stream << cb.a.copy_from(luisa::span{cb.ha})
                                       << cb.b.copy_from(luisa::span{cb.hb})
                                       << cb.aux.copy_from(luisa::span{cb.haux})
                                       << shader(ByteBufferView{cb.a}, ByteBufferView{cb.b},
                                                 ByteBufferView{cb.aux}, ByteBufferView{cb.c}, m,
                                                 shape.n, shape.k)
                                              .dispatch(disp.x)
                                       << synchronize();
                                const uint32_t rows[] = {0u, m / 3u, m / 2u, m - 1u};
                                const uint32_t cols[] = {0u, shape.n / 4u, shape.n / 2u, shape.n - 1u};
                                for (const auto i : rows) {
                                    stream << cb.c.view(static_cast<size_t>(i) * shape.n, shape.n)
                                                  .copy_to(luisa::span{out_row})
                                           << synchronize();
                                    for (const auto j : cols) {
                                        const auto ref = ref_element(cb.ha, cb.hb, cb.haux, shape.ep,
                                                                     i, j, shape.k, shape.n);
                                        const float d = std::abs(out_row[j] - ref);
                                        if (!(d <= 1e-3f * std::max(1.f, std::abs(ref)))) {
                                            if (sample_bad < 4u) {
                                                LUISA_WARNING(
                                                    "  coop sample mismatch (m={}, n={}): got {:.6e} ref {:.6e}",
                                                    i, j, out_row[j], ref);
                                            }
                                            sample_bad++;
                                        }
                                    }
                                }
                            }

                            double ms_min = 0.0, ms_med = 0.0, gflops = 0.0;
                            uint32_t iters_run = 0u;
                            if (sample_bad == 0u) {
                                const auto enqueue = [&] {
                                    stream << shader(ByteBufferView{cb.a}, ByteBufferView{cb.b},
                                                     ByteBufferView{cb.aux}, ByteBufferView{cb.c}, m,
                                                     shape.n, shape.k)
                                                  .dispatch(disp.x);
                                };
                                for (auto w = 0u; w < opt.warmup; w++) {
                                    enqueue();
                                    stream << synchronize();
                                }
                                luisa::vector<double> samples;
                                for (auto it = 0u; it < opt.iters; it++) {
                                    Clock clock;
                                    enqueue();
                                    stream << synchronize();
                                    samples.emplace_back(clock.toc());
                                    iters_run++;
                                }
                                const auto st = summarize(std::move(samples));
                                ms_min = st.min;
                                ms_med = st.median;
                                gflops = 2.0 * static_cast<double>(m) * shape.n * shape.k /
                                         (ms_min * 1.0e6);
                                timed_rows++;
                            } else {
                                failures += sample_bad;
                            }

                            LUISA_INFO("  coop {:<16} rows={:<5} grid={}x{:<5} kslab={} ntile={} tpb={} "
                                       "sample_bad={} min={:.3f}ms med={:.3f}ms {:.1f} GFLOPS [{}]",
                                       ccfg->name, m, disp.x, disp.y, ccfg->kslab, ccfg->ntile,
                                       ccfg->tokens_per_block, sample_bad, ms_min, ms_med, gflops,
                                       sample_bad == 0u ? "PASS" : "FAIL");
                            if (csv_out) {
                                csv_out << backend << ',' << ccfg->name << ",0," << ccfg->ntile << ','
                                        << ccfg->kslab << ",0," << ccfg->tokens_per_block << ','
                                        << kCoopGemmBlock << ",0,0," << shape.name << ','
                                        << gemm_epilogue_name(shape.ep) << ',' << m << ',' << m << ','
                                        << shape.n << ',' << shape.k << ',' << disp.x << ',' << disp.y
                                        << ",0,0," << sample_bad << ",0," << iters_run << ',' << ms_min
                                        << ',' << ms_med << ',' << gflops << ','
                                        << (sample_bad == 0u ? "PASS" : "FAIL") << '\n';
                            }
                        }
                    }
                }
            }
        }
    }

    LUISA_INFO("sweep done: gemm {} variants, {} timed rows, {} mismatches total, csv: {}", vcount, timed_rows,
               failures, opt.csv.empty() ? "(none)" : opt.csv.string());
    expect(failures == 0u) << luisa::format("gemm sweep found {} mismatching outputs", failures);
    return 0;
}

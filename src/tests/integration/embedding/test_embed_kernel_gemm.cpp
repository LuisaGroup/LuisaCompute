// Test for embedding/embed/kernels/kernels_gemm.h -- plan §4 kernel #3 (tiled
// 16x16 GEMM with compile-time epilogues None / Bias / BiasGeluErf / BiasRelu /
// ResidualAdd) and kernel #7 (standalone residual_add).
//
// Coverage: synthetic seeded tensors only (no goldens). Ragged sizes:
// (M, N, K) sampled from {1, 3, 16, 17, 37, 256, 257} in 12 combinations, all
// non-multiples of the 16x16 tile included, so tile-boundary guards, the ragged
// K sub-tile and the tail store are exercised. CPU oracle: naive triple-loop
// reference with fp64 accumulation and the same epilogue math (erf-GELU via
// <cmath> std::erf, exactly the ONNX `Erf` variant).
//
// Gate (plan §8 numerical gate): per case, every output must satisfy
// |diff| <= 1e-3 * max(1, |ref|) (fp32 GEMM tolerance); the per-case
// max-abs-diff is printed with LUISA_INFO on every case and the worst case is
// reported at the end. Failures accumulate; a single expect() at the end.
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_a3 test_embed_kernel_gemm dx
//   xmake run -o build/embed_a3 test_embed_kernel_gemm fallback
//   (vk likewise; CI gate: dx + fallback)
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/kernels/kernels_gemm.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Deterministic seeded LCG (host-side only; no <random> so no RTTI dependency).
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

struct GemmCase {
    uint32_t m, n, k;
};

// 12 sampled combinations; ragged in every dimension (1, 3, 17, 37, 257 are not
// multiples of the 16x16 tile), including one large all-ragged case.
constexpr GemmCase kCases[] = {
    {1u, 1u, 1u},
    {3u, 1u, 17u},
    {1u, 17u, 3u},
    {16u, 16u, 16u},
    {17u, 17u, 17u},
    {37u, 3u, 16u},
    {3u, 37u, 257u},
    {256u, 17u, 3u},
    {16u, 257u, 16u},
    {257u, 256u, 37u},
    {37u, 37u, 37u},
    {257u, 257u, 257u},
};

// ERF-GELU oracle: 0.5 * x * (1 + erf(x / sqrt(2))), std::erf (the DSL kernel
// evaluates the same function with the A&S 7.1.26 approximation, |err| < 1.5e-7).
[[nodiscard]] float gelu_erf_ref(float x) noexcept {
    const double xd = static_cast<double>(x);
    return static_cast<float>(0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440)));
}

// Naive triple-loop fp64 reference with the epilogue of plan §4 #3.
luisa::vector<float> gemm_cpu_reference(const luisa::vector<float> &A,
                                        const luisa::vector<float> &B,
                                        const luisa::vector<float> &aux,
                                        GemmEpilogue ep,
                                        uint32_t m, uint32_t n, uint32_t k) {
    luisa::vector<float> out(static_cast<size_t>(m) * n, 0.f);
    for (uint32_t i = 0u; i < m; i++) {
        for (uint32_t j = 0u; j < n; j++) {
            double acc = 0.0;
            for (uint32_t t = 0u; t < k; t++) {
                acc += static_cast<double>(A[static_cast<size_t>(i) * k + t]) *
                       static_cast<double>(B[static_cast<size_t>(t) * n + j]);
            }
            float v = static_cast<float>(acc);
            switch (ep) {
                case GemmEpilogue::None:
                    break;
                case GemmEpilogue::Bias:
                    v += aux[j];
                    break;
                case GemmEpilogue::BiasGeluErf:
                    v = gelu_erf_ref(v + aux[j]);
                    break;
                case GemmEpilogue::BiasRelu:
                    v = std::max(v + aux[j], 0.0f);
                    break;
                case GemmEpilogue::ResidualAdd:
                    v += aux[static_cast<size_t>(i) * n + j];
                    break;
            }
            out[static_cast<size_t>(i) * n + j] = v;
        }
    }
    return out;
}

struct Parity {
    uint32_t failures = 0u;
    uint32_t cases = 0u;
    float worst_abs_diff = 0.f;
    GemmCase worst{};
    GemmEpilogue worst_ep = GemmEpilogue::None;
};

// One (epilogue, size) case: upload -> dispatch -> download -> compare.
template<typename Shader>
void check_gemm_case(Device &device, Stream &stream, const Shader &shader,
                     GemmEpilogue ep, const GemmCase &c, Parity &parity) {
    Lcg rng{};
    rng.state = 0x9E3779B9u ^ (c.m << 24u) ^ (c.n << 12u) ^ (c.k << 2u) ^
                (static_cast<uint32_t>(ep) << 28u);

    const size_t a_count = static_cast<size_t>(c.m) * c.k;
    const size_t b_count = static_cast<size_t>(c.k) * c.n;
    const size_t c_count = static_cast<size_t>(c.m) * c.n;

    luisa::vector<float> a(a_count);
    for (auto &f : a) {
        f = rng.uniform(-1.f, 1.f);
    }
    luisa::vector<float> b(b_count);
    for (auto &f : b) {
        f = rng.uniform(-1.f, 1.f);
    }
    // aux: bias[N] for the bias flags, residual R[M,N] for ResidualAdd, a dummy
    // 1-element buffer when the epilogue ignores it (plan §4 #3).
    size_t aux_count = 1u;
    if (ep == GemmEpilogue::Bias || ep == GemmEpilogue::BiasGeluErf || ep == GemmEpilogue::BiasRelu) {
        aux_count = c.n;
    } else if (ep == GemmEpilogue::ResidualAdd) {
        aux_count = c_count;
    }
    luisa::vector<float> aux(aux_count);
    for (auto &f : aux) {
        f = rng.uniform(-1.f, 1.f);
    }

    const auto ref = gemm_cpu_reference(a, b, aux, ep, c.m, c.n, c.k);

    auto a_buf = device.create_buffer<float>(a_count);
    auto b_buf = device.create_buffer<float>(b_count);
    auto c_buf = device.create_buffer<float>(c_count);
    auto aux_buf = device.create_buffer<float>(aux_count);

    luisa::vector<float> got(c_count, 0.f);
    stream << a_buf.copy_from(luisa::span{a})
           << b_buf.copy_from(luisa::span{b})
           << aux_buf.copy_from(luisa::span{aux})
           << shader(a_buf.view(), b_buf.view(), c_buf.view(), aux_buf.view(), c.m, c.n, c.k)
                  .dispatch(gemm_dispatch_size(c.m, c.n))// dispatch = TOTAL threads
           << c_buf.copy_to(luisa::span{got})
           << synchronize();

    float max_diff = 0.f;
    uint32_t bad = 0u;
    for (size_t i = 0; i < c_count; i++) {
        const float d = std::abs(got[i] - ref[i]);
        if (d > max_diff) {
            max_diff = d;
        }
        if (!(d <= 1e-3f * std::max(1.f, std::abs(ref[i])))) {
            if (bad < 4u) {
                LUISA_WARNING("gemm[{}] case (M={}, N={}, K={}) idx {}: got {} ref {}",
                              gemm_epilogue_name(ep), c.m, c.n, c.k, i, got[i], ref[i]);
            }
            bad++;
        }
    }
    LUISA_INFO("gemm[{}] M={} N={} K={}: max_abs_diff={:.3e} bad={}/{}",
               gemm_epilogue_name(ep), c.m, c.n, c.k, max_diff, bad, c_count);
    parity.cases++;
    parity.failures += bad;
    if (max_diff > parity.worst_abs_diff) {
        parity.worst_abs_diff = max_diff;
        parity.worst = c;
        parity.worst_ep = ep;
    }
}// namespace

// Whole epilogue: compile once, sweep all size cases.
void check_gemm_epilogue(Device &device, Stream &stream, GemmEpilogue ep, Parity &parity) {
    auto shader = device.compile(make_gemm_kernel(ep));
    for (const auto &c : kCases) {
        check_gemm_case(device, stream, shader, ep, c, parity);
    }
}

// Standalone residual_add (plan §4 #7): elementwise fp32 add must be exact
// against the same fp32 CPU add (tolerance 1e-6 covers only last-bit ordering).
void check_residual_add(Device &device, Stream &stream, Parity &parity) {
    static constexpr uint32_t kCounts[] = {1u, 3u, 255u, 256u, 257u, 4097u, 66000u};
    auto shader = device.compile(make_residual_add_kernel());
    for (const uint32_t count : kCounts) {
        Lcg rng{};
        rng.state = 0x5BD1E995u ^ count;
        luisa::vector<float> x(count);
        luisa::vector<float> y(count);
        luisa::vector<float> ref(count);
        for (uint32_t i = 0u; i < count; i++) {
            x[i] = rng.uniform(-1.f, 1.f);
            y[i] = rng.uniform(-1.f, 1.f);
            ref[i] = x[i] + y[i];
        }
        auto x_buf = device.create_buffer<float>(count);
        auto y_buf = device.create_buffer<float>(count);
        auto o_buf = device.create_buffer<float>(count);
        luisa::vector<float> got(count, 0.f);
        stream << x_buf.copy_from(luisa::span{x})
               << y_buf.copy_from(luisa::span{y})
               << shader(x_buf.view(), y_buf.view(), o_buf.view(), count)
                      .dispatch(count)// TOTAL threads, tail guarded
               << o_buf.copy_to(luisa::span{got})
               << synchronize();
        float max_diff = 0.f;
        uint32_t bad = 0u;
        for (uint32_t i = 0u; i < count; i++) {
            const float d = std::abs(got[i] - ref[i]);
            if (d > max_diff) {
                max_diff = d;
            }
            if (!(d <= 1e-6f)) {
                bad++;
            }
        }
        LUISA_INFO("residual_add count={}: max_abs_diff={:.3e} bad={}/{}", count, max_diff,
                   bad, count);
        parity.cases++;
        parity.failures += bad;
        if (max_diff > parity.worst_abs_diff) {
            parity.worst_abs_diff = max_diff;
            parity.worst = {count, 0u, 0u};
            parity.worst_ep = GemmEpilogue::ResidualAdd;
        }
    }
}

// Vector register-blocked family (tensor_optimize.md §4, rbv_*): the two
// comparator variants MUST be present in the compiled-in variant table (found by
// name). Host-only checks — no device/launch. gemm_tile_index_by_name returns -1
// for an unknown name.
void check_gemm_vec4_table(Parity &parity) {
    const char *const vec_names[] = {"rbv_64x64_b128_r84v4", "rbv_128x64_b256_r84v4"};
    for (const char *name : vec_names) {
        const int32_t idx = gemm_tile_index_by_name(name);
        const bool present = idx >= 0 && gemm_tile_variant(static_cast<uint32_t>(idx)) != nullptr;
        LUISA_INFO("gemm vec4 table presence [{}]: index={} present={}", name, idx,
                   present ? "yes" : "no");
        expect(present) << "gemm variant table must contain the vec4 family entry " << name;
        parity.cases++;
        parity.failures += present ? 0u : 1u;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();

    Parity parity{};
    check_gemm_epilogue(device, stream, GemmEpilogue::None, parity);
    check_gemm_epilogue(device, stream, GemmEpilogue::Bias, parity);
    check_gemm_epilogue(device, stream, GemmEpilogue::BiasGeluErf, parity);
    check_gemm_epilogue(device, stream, GemmEpilogue::BiasRelu, parity);
    check_gemm_epilogue(device, stream, GemmEpilogue::ResidualAdd, parity);
    check_residual_add(device, stream, parity);
    check_gemm_vec4_table(parity);// host-only vec4 family table presence

    LUISA_INFO("gemm/layernorm gate: worst max-abs-diff {:.3e} at gemm[{}] "
               "(M={}, N={}, K={}) over {} cases",
               parity.worst_abs_diff, gemm_epilogue_name(parity.worst_ep), parity.worst.m,
               parity.worst.n, parity.worst.k, parity.cases);
    expect(parity.failures == 0u)
        << luisa::format("gemm kernels produced {} mismatched outputs over {} cases",
                         parity.failures, parity.cases);
    return 0;
}

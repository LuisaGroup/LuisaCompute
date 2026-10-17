// Test for embedding/embed/kernels/kernels_norm.h -- plan §4 kernel #2
// (layernorm: one block per row, two-pass mean/variance, eps runtime arg) and
// kernel #9 (l2_normalize: one block per row, y = x / max(||x||_2, 1e-12)).
//
// Coverage: synthetic seeded tensors only. layernorm row lengths
// H in {1, 2, 3, 255, 256, 257, 300, 1024, 4096} with eps = 1e-5 (bge-m3
// layer_norm_eps) -- 1/2/3 test H < block size, 255/257/300/4096 straddle the
// 256-thread block (H > block size and non-multiples), 1024 is the production
// hidden size. Multiple rows per launch check that the per-block row index is
// independent. l2_normalize runs H in {1, 7, 1024, 4096, 8192} plus a
// near-zero-vector case that pins the 1e-12 clamp (torch F.normalize).
//
// Gate (plan §8 numerical gate): max-abs-diff vs the in-file naive CPU
// reference <= 1e-4 absolute on every element; the per-case max-abs-diff is
// printed with LUISA_INFO and the worst case is reported at the end. Failures
// accumulate; a single expect() at the end.
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_a3 test_embed_kernel_layernorm dx
//   xmake run -o build/embed_a3 test_embed_kernel_layernorm fallback
//   (vk likewise; CI gate: dx + fallback)
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/kernels/kernels_norm.h"

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

// Deterministic seeded LCG (host-side only; no <random>, no RTTI).
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

struct Parity {
    uint32_t failures = 0u;
    uint32_t cases = 0u;
    float worst_abs_diff = 0.f;
    uint32_t worst_h = 0u;
    const char *worst_kernel = "";
};

// layernorm CPU oracle: two-pass mean/variance, biased variance (ONNX
// ReduceMean semantics), y = ((x - mean) * (1 / sqrt(var + eps))) * gamma + beta
// -- the same expression order as the kernel.
luisa::vector<float> layernorm_cpu_reference(const luisa::vector<float> &x,
                                             const luisa::vector<float> &gamma,
                                             const luisa::vector<float> &beta,
                                             uint32_t rows, uint32_t h, float eps) {
    luisa::vector<float> out(static_cast<size_t>(rows) * h, 0.f);
    for (uint32_t r = 0u; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * h;
        float sum = 0.f;
        for (uint32_t i = 0u; i < h; i++) {
            sum += x[base + i];
        }
        const float mean = sum / static_cast<float>(h);
        float vsum = 0.f;
        for (uint32_t i = 0u; i < h; i++) {
            const float d = x[base + i] - mean;
            vsum += d * d;
        }
        const float var = vsum / static_cast<float>(h);
        const float inv = 1.f / std::sqrt(var + eps);
        for (uint32_t i = 0u; i < h; i++) {
            out[base + i] = ((x[base + i] - mean) * inv) * gamma[i] + beta[i];
        }
    }
    return out;
}

// l2_normalize CPU oracle: y = x / max(||x||_2, 1e-12).
luisa::vector<float> l2_cpu_reference(const luisa::vector<float> &x, uint32_t rows,
                                      uint32_t h) {
    luisa::vector<float> out(static_cast<size_t>(rows) * h, 0.f);
    for (uint32_t r = 0u; r < rows; r++) {
        const size_t base = static_cast<size_t>(r) * h;
        float ssq = 0.f;
        for (uint32_t i = 0u; i < h; i++) {
            ssq += x[base + i] * x[base + i];
        }
        float n = std::sqrt(ssq);
        if (n < 1e-12f) {
            n = 1e-12f;
        }
        for (uint32_t i = 0u; i < h; i++) {
            out[base + i] = x[base + i] / n;
        }
    }
    return out;
}

template<typename Shader>
void run_and_compare(Device &device, Stream &stream, const Shader &shader,
                     const luisa::vector<float> &x, const luisa::vector<float> &ref,
                     uint32_t rows, uint32_t h, float tol, const char *what, Parity &parity) {
    const size_t count = static_cast<size_t>(rows) * h;
    auto x_buf = device.create_buffer<float>(count);
    auto y_buf = device.create_buffer<float>(count);
    luisa::vector<float> got(count, 0.f);
    stream << x_buf.copy_from(luisa::span{x})
           << shader(x_buf.view(), y_buf.view(), h).dispatch(rows * kNormBlock)// TOTAL threads
           << y_buf.copy_to(luisa::span{got})
           << synchronize();
    float max_diff = 0.f;
    uint32_t bad = 0u;
    for (size_t i = 0; i < count; i++) {
        const float d = std::abs(got[i] - ref[i]);
        if (d > max_diff) {
            max_diff = d;
        }
        if (!(d <= tol)) {
            if (bad < 4u) {
                LUISA_WARNING("{} rows={} H={} idx {}: got {} ref {}", what, rows, h, i,
                              got[i], ref[i]);
            }
            bad++;
        }
    }
    LUISA_INFO("{} rows={} H={}: max_abs_diff={:.3e} bad={}/{}", what, rows, h, max_diff,
               bad, count);
    parity.cases++;
    parity.failures += bad;
    if (max_diff > parity.worst_abs_diff) {
        parity.worst_abs_diff = max_diff;
        parity.worst_h = h;
        parity.worst_kernel = what;
    }
}

void check_layernorm(Device &device, Stream &stream, Parity &parity) {
    static constexpr uint32_t kH[] = {1u, 2u, 3u, 255u, 256u, 257u, 300u, 1024u, 4096u};
    static constexpr uint32_t kRows[] = {1u, 3u};
    constexpr float kEps = 1e-5f;// bge-m3 layer_norm_eps (config.json)

    auto shader = device.compile(make_layernorm_kernel());
    for (const uint32_t h : kH) {
        for (const uint32_t rows : kRows) {
            Lcg rng{};
            rng.state = 0xC0FFEE00u ^ (h << 8u) ^ rows;
            luisa::vector<float> x(static_cast<size_t>(rows) * h);
            for (auto &f : x) {
                f = rng.uniform(-1.f, 1.f);
            }
            luisa::vector<float> gamma(h);
            for (auto &f : gamma) {
                f = rng.uniform(0.5f, 1.5f);
            }
            luisa::vector<float> beta(h);
            for (auto &f : beta) {
                f = rng.uniform(-0.5f, 0.5f);
            }
            const auto ref = layernorm_cpu_reference(x, gamma, beta, rows, h, kEps);

            auto x_buf = device.create_buffer<float>(x.size());
            auto y_buf = device.create_buffer<float>(x.size());
            auto g_buf = device.create_buffer<float>(gamma.size());
            auto b_buf = device.create_buffer<float>(beta.size());
            luisa::vector<float> got(x.size(), 0.f);
            stream << x_buf.copy_from(luisa::span{x})
                   << g_buf.copy_from(luisa::span{gamma})
                   << b_buf.copy_from(luisa::span{beta})
                   << shader(x_buf.view(), y_buf.view(), g_buf.view(), b_buf.view(), h, kEps)
                          .dispatch(rows * kNormBlock)// TOTAL threads, block per row
                   << y_buf.copy_to(luisa::span{got})
                   << synchronize();

            float max_diff = 0.f;
            uint32_t bad = 0u;
            for (size_t i = 0; i < got.size(); i++) {
                const float d = std::abs(got[i] - ref[i]);
                if (d > max_diff) {
                    max_diff = d;
                }
                if (!(d <= 1e-4f)) {
                    if (bad < 4u) {
                        LUISA_WARNING("layernorm rows={} H={} idx {}: got {} ref {}", rows, h,
                                      i, got[i], ref[i]);
                    }
                    bad++;
                }
            }
            LUISA_INFO("layernorm rows={} H={}: max_abs_diff={:.3e} bad={}/{}", rows, h,
                       max_diff, bad, got.size());
            parity.cases++;
            parity.failures += bad;
            if (max_diff > parity.worst_abs_diff) {
                parity.worst_abs_diff = max_diff;
                parity.worst_h = h;
                parity.worst_kernel = "layernorm";
            }
        }
    }
}

void check_l2_normalize(Device &device, Stream &stream, Parity &parity) {
    static constexpr uint32_t kH[] = {1u, 7u, 1024u, 4096u, 8192u};
    auto shader = device.compile(make_l2_normalize_kernel());

    // Regular rows: random unit-ish vectors.
    for (const uint32_t h : kH) {
        static constexpr uint32_t rows = 2u;
        Lcg rng{};
        rng.state = 0xBEEF0000u ^ h;
        luisa::vector<float> x(static_cast<size_t>(rows) * h);
        for (auto &f : x) {
            f = rng.uniform(-1.f, 1.f);
        }
        run_and_compare(device, stream, shader, x, l2_cpu_reference(x, rows, h), rows, h, 1e-4f,
                        "l2_normalize", parity);
    }

    // Near-zero and zero rows: the norm falls below the 1e-12 clamp, so rows of
    // exact zeros must stay zeros (no NaN) and 1e-9 rows must stay tiny.
    {
        static constexpr uint32_t h = 1024u;
        static constexpr uint32_t rows = 3u;
        luisa::vector<float> x(static_cast<size_t>(rows) * h, 0.f);
        for (uint32_t i = 0u; i < h; i++) {
            x[static_cast<size_t>(h) + i] = 1e-9f;      // row 1: near-zero
            x[static_cast<size_t>(2u) * h + i] = -1e-9f;// row 2: near-zero, other sign
        }
        run_and_compare(device, stream, shader, x, l2_cpu_reference(x, rows, h), rows, h, 1e-4f,
                        "l2_normalize_clamp", parity);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();

    Parity parity{};
    check_layernorm(device, stream, parity);
    check_l2_normalize(device, stream, parity);

    LUISA_INFO("norm gate: worst max-abs-diff {:.3e} at {} H={} over {} cases",
               parity.worst_abs_diff, parity.worst_kernel, parity.worst_h, parity.cases);
    expect(parity.failures == 0u)
        << luisa::format("norm kernels produced {} mismatched outputs over {} cases",
                         parity.failures, parity.cases);
    return 0;
}

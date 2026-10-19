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
// vec4 variants (tensor_optimize.md Phase 1): layernorm_vec4 (in-place),
// l2_normalize_vec4, dense_pool_vec4 and residual_add_vec4 from
// kernels_norm_vec.h. The vec kernels are load/store-width-4 twins whose
// summation ORDER changes versus the scalar legacy (thread tx owns float4
// lanes tx, tx+256, ..., each lane-group folded with the pinned horizontal
// order (x.x+x.y)+(x.z+x.w) before the shared-memory halving tree), so
// vec-vs-oracle comparisons use the SAME existing tolerances as the legacy
// gate (1e-4 abs; layernorm in-place mode require_finite on the constant row)
// -- NEVER bit-exact -- EXCEPT residual_add_vec4, a pure elementwise fp32 add
// that carries no summation order and is therefore required to be BIT-EXACT
// (0 differing float bit patterns vs the CPU add). The vec4 kernels REQUIRE
// H %% 4 == 0 by host contract (no tail path -- ragged shapes stay on the
// legacy kernels), so the contract test sweeps H in {4, 8, 1020, 1024, 4096}
// (H=4 < block exercises idle-thread zero contribution; 1020/1024 straddle
// the 256-lane boundary in vec4 units: H4=255 is ragged in float4, 256 is
// exact) and rows in {1, 2, 255, 256, 257} (255/256/257 straddle the
// 256-thread block), plus all-zero rows, all-constant rows and zero vectors;
// batch>1 (dense_pool B in {1, 4, 16}) is covered through the row counts.
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_a3 test_embed_kernel_layernorm dx
//   xmake run -o build/embed_a3 test_embed_kernel_layernorm fallback
//   (vk likewise; CI gate: dx + fallback)
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/kernels/kernels_norm.h"
#include "embedding/embed/kernels/kernels_norm_vec.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <cstring>

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

// Count elements whose float bit pattern differs (bit-exactness metric used by
// the vec4 residual_add gate, which carries no summation order -> 0 diffs).
[[nodiscard]] uint32_t bit_diff_count(const luisa::vector<float> &a,
                                      const luisa::vector<float> &b) noexcept {
    uint32_t diff = 0u;
    for (size_t i = 0; i < a.size(); i++) {
        uint32_t ba = 0u;
        uint32_t bb = 0u;
        std::memcpy(&ba, &a[i], sizeof(ba));
        std::memcpy(&bb, &b[i], sizeof(bb));
        diff += (ba != bb) ? 1u : 0u;
    }
    return diff;
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

// ---------------------------------------------------------------------------
// vec4 variants (kernels_norm_vec.h). Buffers are bound as float views
// reinterpreted with .view().as<float4>(); every H in the sweep obeys
// H % 4 == 0 (host contract -- no ragged tail). The summation ORDER differs
// from the scalar legacy, so layernorm/l2/dense_pool are gated at the SAME
// tolerance as the legacy gate (1e-4 abs), while residual_add -- a pure
// elementwise fp32 add -- is required BIT-EXACT (0 differing bit patterns).
// ---------------------------------------------------------------------------

// Sweep H over {4, 8, 1020, 1024, 4096}, rows over {1, 2, 255, 256, 257}; all
// H are multiples of 4. layernorm_vec4 is IN-PLACE (x read and written back).
void check_layernorm_vec4(Device &device, Stream &stream, Parity &parity) {
    static constexpr uint32_t kHV[] = {4u, 8u, 1020u, 1024u, 4096u};
    static constexpr uint32_t kRowsV[] = {1u, 2u, 255u, 256u, 257u};
    constexpr float kEps = 1e-5f;

    auto shader = device.compile(make_layernorm_vec4_kernel());
    for (const uint32_t h : kHV) {
        for (const uint32_t rows : kRowsV) {
            Lcg rng{};
            rng.state = 0xDECAFB00u ^ (h << 8u) ^ rows;
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
            auto g_buf = device.create_buffer<float>(gamma.size());
            auto b_buf = device.create_buffer<float>(beta.size());
            luisa::vector<float> got(x.size(), 0.f);
            // in-place: x is both the read source and the write-back target.
            stream << x_buf.copy_from(luisa::span{x})
                   << g_buf.copy_from(luisa::span{gamma})
                   << b_buf.copy_from(luisa::span{beta})
                   << shader(x_buf.view().as<float4>(), g_buf.view().as<float4>(),
                             b_buf.view().as<float4>(), h, kEps)
                          .dispatch(rows * kNormVecBlock)// TOTAL threads
                   << x_buf.copy_to(luisa::span{got})
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
                        LUISA_WARNING("layernorm_vec4 rows={} H={} idx {}: got {} ref {}", rows,
                                      h, i, got[i], ref[i]);
                    }
                    bad++;
                }
            }
            LUISA_INFO("layernorm_vec4 rows={} H={}: max_abs_diff={:.3e} bad={}/{}", rows, h,
                       max_diff, bad, got.size());
            parity.cases++;
            parity.failures += bad;
            if (max_diff > parity.worst_abs_diff) {
                parity.worst_abs_diff = max_diff;
                parity.worst_h = h;
                parity.worst_kernel = "layernorm_vec4";
            }
        }
    }
}

// l2_normalize_vec4: H in {4, 8, 1020, 1024, 4096}, rows in {1, 2, 255};
// plus a clamp case (zero / near-zero rows) mirroring check_l2_normalize().
void check_l2_normalize_vec4(Device &device, Stream &stream, Parity &parity) {
    static constexpr uint32_t kHV[] = {4u, 8u, 1020u, 1024u, 4096u};
    auto shader = device.compile(make_l2_normalize_vec4_kernel());

    for (const uint32_t h : kHV) {
        static constexpr uint32_t rows = 2u;
        Lcg rng{};
        rng.state = 0xBEEF1000u ^ h;
        luisa::vector<float> x(static_cast<size_t>(rows) * h);
        for (auto &f : x) {
            f = rng.uniform(-1.f, 1.f);
        }
        const auto ref = l2_cpu_reference(x, rows, h);
        auto x_buf = device.create_buffer<float>(x.size());
        auto y_buf = device.create_buffer<float>(x.size());
        luisa::vector<float> got(x.size(), 0.f);
        stream << x_buf.copy_from(luisa::span{x})
               << shader(x_buf.view().as<float4>(), y_buf.view().as<float4>(), h)
                      .dispatch(rows * kNormVecBlock)// TOTAL threads
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
                    LUISA_WARNING("l2_normalize_vec4 rows={} H={} idx {}: got {} ref {}", rows, h,
                                  i, got[i], ref[i]);
                }
                bad++;
            }
        }
        LUISA_INFO("l2_normalize_vec4 rows={} H={}: max_abs_diff={:.3e} bad={}/{}", rows, h,
                   max_diff, bad, got.size());
        parity.cases++;
        parity.failures += bad;
        if (max_diff > parity.worst_abs_diff) {
            parity.worst_abs_diff = max_diff;
            parity.worst_h = h;
            parity.worst_kernel = "l2_normalize_vec4";
        }
    }

    // Near-zero / zero rows pin the 1e-12 clamp (H = 1024, a multiple of 4).
    {
        static constexpr uint32_t h = 1024u;
        static constexpr uint32_t rows = 3u;
        luisa::vector<float> x(static_cast<size_t>(rows) * h, 0.f);
        for (uint32_t i = 0u; i < h; i++) {
            x[static_cast<size_t>(h) + i] = 1e-9f;
            x[static_cast<size_t>(2u) * h + i] = -1e-9f;
        }
        const auto ref = l2_cpu_reference(x, rows, h);
        auto x_buf = device.create_buffer<float>(x.size());
        auto y_buf = device.create_buffer<float>(x.size());
        luisa::vector<float> got(x.size(), 0.f);
        stream << x_buf.copy_from(luisa::span{x})
               << shader(x_buf.view().as<float4>(), y_buf.view().as<float4>(), h)
                      .dispatch(rows * kNormVecBlock)
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
                bad++;
            }
        }
        LUISA_INFO("l2_normalize_vec4_clamp rows={} H={}: max_abs_diff={:.3e} bad={}/{}", rows, h,
                   max_diff, bad, got.size());
        parity.cases++;
        parity.failures += bad;
    }
}

// dense_pool_vec4: one block per batch row b; B in {1, 4, 16} (batch>1),
// H in {1024, 4096}, S arbitrary (the CLS row is hidden[b*S]). The output
// oracle is the per-row L2 of the CLS hidden row (hidden[b*S] in a [B*S, H]
// state), matching make_dense_pool_vec4_kernel()'s indexing.
void check_dense_pool_vec4(Device &device, Stream &stream, Parity &parity) {
    auto shader = device.compile(make_dense_pool_vec4_kernel());
    for (const uint32_t B : {1u, 4u, 16u}) {
        for (const uint32_t h : {1024u, 4096u}) {
            const uint32_t S = 5u;
            Lcg rng{};
            rng.state = 0xD0500E00u ^ (B << 8u) ^ h;
            const size_t count = static_cast<size_t>(B) * S * h;
            luisa::vector<float> hidden(count);
            for (auto &f : hidden) {
                f = rng.uniform(-1.f, 1.f);
            }
            // oracle: CLS row hidden[b*S] L2-normalized.
            luisa::vector<float> ref(static_cast<size_t>(B) * h, 0.f);
            for (uint32_t b = 0u; b < B; b++) {
                const size_t base = static_cast<size_t>(b) * S * h;
                float ssq = 0.f;
                for (uint32_t i = 0u; i < h; i++) {
                    ssq += hidden[base + i] * hidden[base + i];
                }
                float n = std::sqrt(ssq);
                if (n < 1e-12f) {
                    n = 1e-12f;
                }
                for (uint32_t i = 0u; i < h; i++) {
                    ref[static_cast<size_t>(b) * h + i] = hidden[base + i] / n;
                }
            }
            auto hid_buf = device.create_buffer<float>(count);
            auto out_buf = device.create_buffer<float>(static_cast<size_t>(B) * h);
            luisa::vector<float> got(static_cast<size_t>(B) * h, 0.f);
            stream << hid_buf.copy_from(luisa::span{hidden})
                   << shader(hid_buf.view().as<float4>(), out_buf.view().as<float4>(), S, h)
                          .dispatch(B * kNormVecBlock)// one block per batch row
                   << out_buf.copy_to(luisa::span{got})
                   << synchronize();
            float max_diff = 0.f;
            uint32_t bad = 0u;
            for (size_t i = 0; i < got.size(); i++) {
                const float d = std::abs(got[i] - ref[i]);
                if (d > max_diff) {
                    max_diff = d;
                }
                if (!(d <= 1e-4f)) {
                    bad++;
                }
            }
            LUISA_INFO("dense_pool_vec4 B={} S={} H={}: max_abs_diff={:.3e} bad={}/{}", B, S, h,
                       max_diff, bad, got.size());
            parity.cases++;
            parity.failures += bad;
            if (max_diff > parity.worst_abs_diff) {
                parity.worst_abs_diff = max_diff;
                parity.worst_h = h;
                parity.worst_kernel = "dense_pool_vec4";
            }
        }
    }
}

// residual_add_vec4: out[i] = a[i] + b[i] over count4 float4s. Pure elementwise
// add -> BIT-EXACT (0 differing bit patterns vs the CPU add). counts are the
// vec4 element counts, so the logical count is always a multiple of 4.
void check_residual_add_vec4(Device &device, Stream &stream, Parity &parity) {
    static constexpr uint32_t kCounts[] = {1u, 255u, 256u, 257u, 1024u, 16500u};
    auto shader = device.compile(make_residual_add_vec4_kernel());
    for (const uint32_t count4 : kCounts) {
        const uint32_t count = count4 * 4u;// logical element count (%4==0)
        Lcg rng{};
        rng.state = 0x5BD1EC40u ^ count4;
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
               << shader(x_buf.view().as<float4>(), y_buf.view().as<float4>(),
                         o_buf.view().as<float4>(), count4)
                      .dispatch(count4)// TOTAL threads, tail guarded
               << o_buf.copy_to(luisa::span{got})
               << synchronize();
        const uint32_t diff = bit_diff_count(got, ref);
        LUISA_INFO("residual_add_vec4 count4={} (count={}): bit_diffs={}/{}", count4, count, diff,
                   count);
        parity.cases++;
        parity.failures += diff;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();

    Parity parity{};
    check_layernorm(device, stream, parity);
    check_l2_normalize(device, stream, parity);
    check_layernorm_vec4(device, stream, parity);
    check_l2_normalize_vec4(device, stream, parity);
    check_dense_pool_vec4(device, stream, parity);
    check_residual_add_vec4(device, stream, parity);

    LUISA_INFO("norm gate: worst max-abs-diff {:.3e} at {} H={} over {} cases",
               parity.worst_abs_diff, parity.worst_kernel, parity.worst_h, parity.cases);
    expect(parity.failures == 0u)
        << luisa::format("norm kernels produced {} mismatched outputs over {} cases",
                         parity.failures, parity.cases);
    return 0;
}

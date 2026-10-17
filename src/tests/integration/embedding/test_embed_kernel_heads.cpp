// Test for embedding/embed/kernels/kernels_heads.h — plan §4 kernels #8
// (slice_cls) and #10 (sparse_head, colbert_proj) of the bge-m3 embedding
// engine.
//
// Coverage (synthetic seeded tensors, ragged sizes so nothing is baked to
// bge-m3's 1024):
// - slice_cls: copy of row 0 of [rows, H] — bit-exact (tolerance formal).
// - sparse_head: out[t] = relu(dot(X[t,:], W_s) + b_s), H in {7, 16, 1024}.
//   Tree-order block reduction on device vs. serial CPU reference ⇒ hybrid
//   tol 1e-5 abs + 1e-5·|ref| (H=1024 fp32 summation-order noise ~1e-5).
// - colbert_proj: Y = X @ W_c + b_c (W_c [H,N] row-major, pre-transposed),
//   H in {7, 16, 1024}, N in {1, 13, 1024}, normalize flag on/off; rel tol
//   1e-3. The dot runs in the same serial h order as the reference.
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_a4 test_embed_kernel_heads dx
//   xmake run -o build/embed_a4 test_embed_kernel_heads fallback
//   (vk likewise; CI gate: dx + fallback)
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/kernels/kernels_heads.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::embed;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Deterministic seeded LCG — identical inputs for every backend; the CPU
// reference consumes it in a fixed order.
struct Lcg {
    uint32_t state = 0x9E3779B9u;
    [[nodiscard]] uint32_t next() noexcept {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() % 8388608u) / 8388608.f);
    }
};

void fill_uniform(Lcg &rng, luisa::vector<float> &v, float lo, float hi) noexcept {
    for (auto &f : v) {
        f = rng.uniform(lo, hi);
    }
}

void test_slice_cls(Device &device, Stream &stream) {
    constexpr uint kH = 64u;
    constexpr uint kRows = 6u;
    Lcg rng{};
    luisa::vector<float> hidden(static_cast<size_t>(kRows) * kH);
    fill_uniform(rng, hidden, -1.f, 1.f);
    auto in_buf = device.create_buffer<float>(hidden.size());
    auto out_buf = device.create_buffer<float>(kH);
    stream << in_buf.copy_from(luisa::span{hidden}) << synchronize();

    auto shader = device.compile(make_slice_cls_kernel());
    luisa::vector<float> out(kH, 0.f);
    stream << shader(in_buf, out_buf, kH)
                  .dispatch(kH)
           << out_buf.copy_to(luisa::span{out})
           << synchronize();

    float max_diff = 0.f;
    for (uint i = 0; i < kH; i++) {
        max_diff = std::max(max_diff, std::abs(out[i] - hidden[i]));
    }
    LUISA_INFO("slice_cls rows={} H={}: max_abs_diff={:.3e}", kRows, kH, max_diff);
    // Pure copy -> expect bit-exactness; the tiny tolerance is formal.
    expect(max_diff < 1e-12f) << "slice_cls must copy row 0 exactly";
}

void test_sparse_head(Device &device, Stream &stream) {
    constexpr uint kRows = 3u;
    const uint hs[] = {7u, 16u, 1024u};
    auto shader = device.compile(make_sparse_head_kernel());
    for (const uint h : hs) {
        Lcg rng{};
        rng.state = 0x5EED1234u ^ (h << 3u);
        luisa::vector<float> x(static_cast<size_t>(kRows) * h);
        fill_uniform(rng, x, -1.f, 1.f);
        luisa::vector<float> w(h);
        fill_uniform(rng, w, -1.f, 1.f);
        const float b = rng.uniform(-1.f, 1.f);

        // CPU reference: out[t] = relu(sum_h X[t,h] * W[h] + b), serial order.
        luisa::vector<float> ref(kRows, 0.f);
        for (uint t = 0; t < kRows; t++) {
            float dot = 0.f;
            for (uint hh = 0; hh < h; hh++) {
                dot += x[static_cast<size_t>(t) * h + hh] * w[hh];
            }
            const float v = dot + b;
            ref[t] = v > 0.f ? v : 0.f;
        }

        auto x_buf = device.create_buffer<float>(x.size());
        auto w_buf = device.create_buffer<float>(w.size());
        auto out_buf = device.create_buffer<float>(kRows);
        stream << x_buf.copy_from(luisa::span{x})
               << w_buf.copy_from(luisa::span{w})
               << synchronize();

        luisa::vector<float> out(kRows, 0.f);
        stream << shader(x_buf, w_buf, b, out_buf, h)
                      .dispatch(kRows * kHeadsBlock)// block per token; total threads
               << out_buf.copy_to(luisa::span{out})
               << synchronize();

        float max_diff = 0.f;
        float worst_ratio = 0.f;// diff / tolerance
        bool ok = true;
        for (uint t = 0; t < kRows; t++) {
            const float d = std::abs(out[t] - ref[t]);
            const float tol = 1e-5f + 1e-5f * std::abs(ref[t]);
            max_diff = std::max(max_diff, d);
            worst_ratio = std::max(worst_ratio, d / tol);
            ok = ok && (d <= tol);
        }
        LUISA_INFO("sparse_head rows={} H={}: max_abs_diff={:.3e} worst_diff/tol={:.3e}",
                   kRows, h, max_diff, worst_ratio);
        expect(ok) << "sparse_head H=" << h
                   << " diff must be <= 1e-5 + 1e-5*|ref| (block-reduction order)";
    }
}

void test_colbert_proj(Device &device, Stream &stream) {
    const uint hs[] = {7u, 16u, 1024u};
    const uint ns[] = {1u, 13u, 1024u};
    auto shader = device.compile(make_colbert_proj_kernel());
    for (const uint rows_u : {2u, 5u}) {
        for (uint hi = 0; hi < 3u; hi++) {
            const uint h = hs[hi];
            const uint n = ns[hi];// ragged pairs (7,1), (16,13), (1024,1024)
            for (const uint norm : {0u, 1u}) {
                Lcg rng{};
                rng.state = 0xC01B3E7u ^ (h << 5u) ^ (n << 1u) ^ norm ^ (rows_u << 12u);
                luisa::vector<float> x(static_cast<size_t>(rows_u) * h);
                fill_uniform(rng, x, -1.f, 1.f);
                luisa::vector<float> wc(static_cast<size_t>(h) * n);
                fill_uniform(rng, wc, -1.f, 1.f);
                luisa::vector<float> bc(n);
                fill_uniform(rng, bc, -1.f, 1.f);

                // CPU reference: y[t,n] = sum_h x[t,h]*W[h,n] + b[n]; then
                // per-row division by max(||y||_2, 1e-12) when normalizing
                // (torch F.normalize semantics, same op order as the kernel).
                luisa::vector<float> ref(static_cast<size_t>(rows_u) * n, 0.f);
                for (uint t = 0; t < rows_u; t++) {
                    float sq = 0.f;
                    for (uint nn = 0; nn < n; nn++) {
                        float acc = 0.f;
                        for (uint hh = 0; hh < h; hh++) {
                            acc += x[static_cast<size_t>(t) * h + hh] *
                                   wc[static_cast<size_t>(hh) * n + nn];
                        }
                        const float y = acc + bc[nn];
                        ref[static_cast<size_t>(t) * n + nn] = y;
                        sq += y * y;
                    }
                    if (norm != 0u) {
                        float d = std::sqrt(sq);
                        d = d < 1e-12f ? 1e-12f : d;
                        for (uint nn = 0; nn < n; nn++) {
                            ref[static_cast<size_t>(t) * n + nn] /= d;
                        }
                    }
                }

                auto x_buf = device.create_buffer<float>(x.size());
                auto wc_buf = device.create_buffer<float>(wc.size());
                auto bc_buf = device.create_buffer<float>(bc.size());
                auto out_buf = device.create_buffer<float>(static_cast<size_t>(rows_u) * n);
                stream << x_buf.copy_from(luisa::span{x})
                       << wc_buf.copy_from(luisa::span{wc})
                       << bc_buf.copy_from(luisa::span{bc})
                       << synchronize();

                luisa::vector<float> out(static_cast<size_t>(rows_u) * n, 0.f);
                stream << shader(x_buf, wc_buf, bc_buf, out_buf, h, n, norm)
                              .dispatch(rows_u * kHeadsBlock)// block per token
                       << out_buf.copy_to(luisa::span{out})
                       << synchronize();

                float max_diff = 0.f;
                float worst_ratio = 0.f;
                bool ok = true;
                for (size_t i = 0; i < out.size(); i++) {
                    const float d = std::abs(out[i] - ref[i]);
                    const float tol = 1e-3f * std::max(1.f, std::abs(ref[i]));
                    max_diff = std::max(max_diff, d);
                    worst_ratio = std::max(worst_ratio, d / tol);
                    ok = ok && (d <= tol);
                }
                LUISA_INFO("colbert_proj rows={} H={} N={} normalize={}: "
                           "max_abs_diff={:.3e} worst_diff/tol={:.3e}",
                           rows_u, h, n, norm, max_diff, worst_ratio);
                expect(ok) << "colbert_proj rows=" << rows_u << " H=" << h
                           << " N=" << n << " norm=" << norm
                           << " rel diff must be <= 1e-3";
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();
    test_slice_cls(device, stream);
    test_sparse_head(device, stream);
    test_colbert_proj(device, stream);
    return 0;
}

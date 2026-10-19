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
#include "embedding/embed/kernels/kernels_heads_vec.h"

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

// Mirror of the vec4 kernels' block reduction (shared-memory halving tree over
// kHeadsVecBlock per-thread partials) — same arithmetic order as
// kernels_heads_vec.cpp so the oracle tracks the device lane grouping: thread tx
// owns strided float4 lanes {tx, tx+256, ...}, each lane-group folded with the
// pinned horizontal order (x.x+x.y)+(x.z+x.w) before the tree.
[[nodiscard]] float vec4_tree_sum(const luisa::vector<float> &partial) noexcept {
    luisa::vector<float> red{partial};
    for (uint32_t stride = kHeadsVecBlock / 2u; stride > 0u; stride >>= 1u) {
        for (uint32_t t = 0u; t < stride; t++) {
            red[t] += red[t + stride];
        }
    }
    return red[0u];
}

// sparse_head_vec4 lane-order oracle for one token row t: thread tx owns the
// strided float4 lanes {tx, tx+256, ...} < H4, accumulates each component into
// its own float4 lane (p4.x/.y/.z/.w), then horizontal-sums ONCE with the
// pinned order (p4.x+p4.y)+(p4.z+p4.w), and the 256 partials go through the
// shared-memory halving tree (vec4_tree_sum). x4 row base = t*H, w base = 0.
[[nodiscard]] float sparse_head_vec4_cpu_row(const luisa::vector<float> &x,
                                             const luisa::vector<float> &w, float b_s,
                                             uint32_t row, uint32_t H) noexcept {
    const uint32_t H4 = H / 4u;
    const size_t base = static_cast<size_t>(row) * H;
    luisa::vector<float> partial(kHeadsVecBlock, 0.f);
    for (uint32_t tx = 0u; tx < kHeadsVecBlock; tx++) {
        float px = 0.f, py = 0.f, pz = 0.f, pw = 0.f;
        for (uint32_t h4 = tx; h4 < H4; h4 += kHeadsVecBlock) {
            const size_t e = base + 4u * h4;
            px += x[e + 0u] * w[4u * h4 + 0u];
            py += x[e + 1u] * w[4u * h4 + 1u];
            pz += x[e + 2u] * w[4u * h4 + 2u];
            pw += x[e + 3u] * w[4u * h4 + 3u];
        }
        partial[tx] = (px + py) + (pz + pw);
    }
    const float dot = vec4_tree_sum(partial);
    const float v = dot + b_s;
    return v > 0.f ? v : 0.f;// exact legacy strict ReLU
}

// NOTE: colbert_proj_vec4 has NO block reduction (each thread serially
// accumulates acc4 over h ascending), so its per-output K order is the plain
// ascending serial dot the existing colbert_cpu_reference already reproduces —
// the same <1e-3 rel tolerance applies. Only the sparse head carries a tree to
// mirror (done above).

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

// vec4 twin of test_sparse_head: H % 4 == 0 shapes only (host contract). The
// oracle mirrors the kernel's vec4 lane order (sparse_head_vec4_cpu_row).
// Same 1e-5 + 1e-5*|ref| hybrid tolerance as the scalar sparse_head gate.
void test_sparse_head_vec4(Device &device, Stream &stream) {
    constexpr uint kRows = 3u;
    const uint hs[] = {4u, 8u, 1020u, 1024u};
    auto shader = device.compile(make_sparse_head_vec4_kernel());
    for (const uint h : hs) {
        Lcg rng{};
        rng.state = 0x5EEDB400u ^ (h << 3u);
        luisa::vector<float> x(static_cast<size_t>(kRows) * h);
        fill_uniform(rng, x, -1.f, 1.f);
        luisa::vector<float> w(h);
        fill_uniform(rng, w, -1.f, 1.f);
        const float b = rng.uniform(-1.f, 1.f);

        luisa::vector<float> ref(kRows, 0.f);
        for (uint t = 0; t < kRows; t++) {
            ref[t] = sparse_head_vec4_cpu_row(x, w, b, t, h);
        }

        auto x_buf = device.create_buffer<float>(x.size());
        auto w_buf = device.create_buffer<float>(w.size());
        auto out_buf = device.create_buffer<float>(kRows);
        stream << x_buf.copy_from(luisa::span{x})
               << w_buf.copy_from(luisa::span{w})
               << synchronize();

        luisa::vector<float> out(kRows, 0.f);
        stream << shader(x_buf.view().as<float4>(), w_buf.view().as<float4>(), b,
                         out_buf.view(), h)
                      .dispatch(kRows * kHeadsVecBlock)// block per token; total threads
               << out_buf.copy_to(luisa::span{out})
               << synchronize();

        float max_diff = 0.f;
        float worst_ratio = 0.f;
        bool ok = true;
        for (uint t = 0; t < kRows; t++) {
            const float d = std::abs(out[t] - ref[t]);
            const float tol = 1e-5f + 1e-5f * std::abs(ref[t]);
            max_diff = std::max(max_diff, d);
            worst_ratio = std::max(worst_ratio, d / tol);
            ok = ok && (d <= tol);
        }
        LUISA_INFO("sparse_head_vec4 rows={} H={}: max_abs_diff={:.3e} worst_diff/tol={:.3e}",
                   kRows, h, max_diff, worst_ratio);
        expect(ok) << "sparse_head_vec4 H=" << h
                   << " diff must be <= 1e-5 + 1e-5*|ref| (vec4 lane-order reduction)";
    }
}

// colbert_proj_vec4 normalize denominator: sum of squares accumulated per-thread
// into a float4 lane over strided owned vec4-columns, horizontal-summed once with
// the pinned order, then folded through the shared-memory halving tree (tot).
[[nodiscard]] float colbert_proj_vec4_norm_denom(const luisa::vector<float> &pre, uint32_t row,
                                                 uint32_t n) noexcept {
    const uint32_t N4 = n / 4u;
    const size_t rbase = static_cast<size_t>(row) * n;
    luisa::vector<float> partial(kHeadsVecBlock, 0.f);
    for (uint32_t tx = 0u; tx < kHeadsVecBlock; tx++) {
        float px = 0.f, py = 0.f, pz = 0.f, pw = 0.f;
        for (uint32_t c = tx; c < N4; c += kHeadsVecBlock) {
            for (uint32_t j = 0u; j < 4u; j++) {
                const float y = pre[rbase + 4u * c + j];
                const float yy = y * y;
                if (j == 0u) {
                    px += yy;
                } else if (j == 1u) {
                    py += yy;
                } else if (j == 2u) {
                    pz += yy;
                } else {
                    pw += yy;
                }
            }
        }
        partial[tx] = (px + py) + (pz + pw);
    }
    const float tot = vec4_tree_sum(partial);
    float d = std::sqrt(tot);
    return d < 1e-12f ? 1e-12f : d;
}

// vec4 twin of test_colbert_proj: H % 4 == 0 AND N % 4 == 0 shapes only. The dot
// matches the serial reference (colbert_proj_vec4 has no K-block reduction); only
// the fused normalize is re-summed in vec4 lane order (colbert_proj_vec4_norm_denom).
void test_colbert_proj_vec4(Device &device, Stream &stream) {
    const uint hs[] = {4u, 16u, 1024u};
    const uint ns[] = {4u, 8u, 1024u};
    auto shader = device.compile(make_colbert_proj_vec4_kernel());
    for (const uint rows_u : {2u, 5u}) {
        for (uint hi = 0; hi < 3u; hi++) {
            const uint h = hs[hi];
            const uint n = ns[hi];
            for (const uint norm : {0u, 1u}) {
                Lcg rng{};
                rng.state = 0xC01B4E7u ^ (h << 5u) ^ (n << 1u) ^ norm ^ (rows_u << 12u);
                luisa::vector<float> x(static_cast<size_t>(rows_u) * h);
                fill_uniform(rng, x, -1.f, 1.f);
                luisa::vector<float> wc(static_cast<size_t>(h) * n);
                fill_uniform(rng, wc, -1.f, 1.f);
                luisa::vector<float> bc(n);
                fill_uniform(rng, bc, -1.f, 1.f);

                // pre-normalize reference (serial ascending K dot, matches kernel).
                luisa::vector<float> ref(static_cast<size_t>(rows_u) * n, 0.f);
                for (uint t = 0; t < rows_u; t++) {
                    for (uint nn = 0; nn < n; nn++) {
                        float acc = 0.f;
                        for (uint hh = 0; hh < h; hh++) {
                            acc += x[static_cast<size_t>(t) * h + hh] *
                                   wc[static_cast<size_t>(hh) * n + nn];
                        }
                        ref[static_cast<size_t>(t) * n + nn] = acc + bc[nn];
                    }
                }
                if (norm != 0u) {
                    for (uint t = 0; t < rows_u; t++) {
                        const float d = colbert_proj_vec4_norm_denom(ref, t, n);
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
                stream << shader(x_buf.view().as<float4>(), wc_buf.view().as<float4>(),
                                 bc_buf.view().as<float4>(), out_buf.view().as<float4>(), h, n,
                                 norm)
                              .dispatch(rows_u * kHeadsVecBlock)// block per token
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
                LUISA_INFO("colbert_proj_vec4 rows={} H={} N={} normalize={}: "
                           "max_abs_diff={:.3e} worst_diff/tol={:.3e}",
                           rows_u, h, n, norm, max_diff, worst_ratio);
                expect(ok) << "colbert_proj_vec4 rows=" << rows_u << " H=" << h
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
    // vec4 variants (H % 4 == 0 / N % 4 == 0 host contract; oracle mirrors the
    // vec4 lane order, same hybrid / rel tolerances as the scalar gates).
    test_sparse_head_vec4(device, stream);
    test_colbert_proj_vec4(device, stream);
    return 0;
}

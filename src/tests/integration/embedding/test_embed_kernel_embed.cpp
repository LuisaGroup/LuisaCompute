// Test for embedding/embed/kernels/kernels_embed.h — plan §4 kernel #1
// (embed_lookup: word-embedding Gather + absolute position Gather + Add +
// embedding LayerNorm, eps=1e-5, gamma/beta affine).
//
// Coverage: synthetic seeded tensors only. H in {8, 64, 1024} (bge-m3 hidden
// size) plus H=2500 (odd size above the 2048-element shared row cache —
// exercises the multi-pass re-gather fallback); S in {1, 5, 128}; batch
// B in {1, 2}; small vocab V=97. CPU reference implements the same two-pass
// mean/var LayerNorm math in the kernel's expression order.
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_a4 test_embed_kernel_embed dx
//   xmake run -o build/embed_a4 test_embed_kernel_embed fallback
//   (vk likewise; CI gate: dx + fallback)
// Gate: max-abs-diff < 1e-4 on every case, printed via LUISA_INFO.
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/kernels/kernels_embed.h"
#include "embedding/embed/kernels/kernels_embed_vec.h"

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

// Deterministic seeded LCG (host-side only; the CPU reference consumes it in a
// fixed order so dx/fallback/vk all compare against the same data).
struct Lcg {
    uint32_t state = 0x12345678u;
    [[nodiscard]] uint32_t next() noexcept {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    [[nodiscard]] float uniform(float lo, float hi) noexcept {
        return lo + (hi - lo) * (static_cast<float>(next() % 8388608u) / 8388608.f);
    }
    [[nodiscard]] uint32_t bounded(uint32_t n) noexcept { return next() % n; }
};

// CPU reference for embed_lookup: per token row, x = word[id] + pos[p],
// two-pass mean/var LayerNorm, y = ((x-mean) * 1/sqrt(var+eps)) * gamma + beta.
// Same expression order as the kernel.
luisa::vector<float> embed_cpu_reference(const luisa::vector<uint32_t> &ids,
                                         const luisa::vector<float> &word,
                                         const luisa::vector<float> &pos,
                                         const luisa::vector<float> &gamma,
                                         const luisa::vector<float> &beta,
                                         uint b, uint s, uint h, float eps) {
    luisa::vector<float> out(static_cast<size_t>(b) * s * h, 0.f);
    const size_t rows = static_cast<size_t>(b) * s;
    for (size_t row = 0; row < rows; row++) {
        const auto p = static_cast<size_t>(row % s);
        const auto id = static_cast<size_t>(ids[row]);
        float sum = 0.f;
        for (uint hh = 0; hh < h; hh++) {
            sum += word[id * h + hh] + pos[p * h + hh];
        }
        const float mean = sum / static_cast<float>(h);
        float vsum = 0.f;
        for (uint hh = 0; hh < h; hh++) {
            const float d = word[id * h + hh] + pos[p * h + hh] - mean;
            vsum += d * d;
        }
        const float var = vsum / static_cast<float>(h);
        const float inv = 1.f / std::sqrt(var + eps);
        for (uint hh = 0; hh < h; hh++) {
            const float x = word[id * h + hh] + pos[p * h + hh];
            out[row * h + hh] = ((x - mean) * inv) * gamma[hh] + beta[hh];
        }
    }
    return out;
}

// Mirror of the vec4 kernel's block reduction: 256 per-thread partials folded
// with the shared-memory halving tree (red[t] += red[t+stride] for the lower
// half, stride 128..1). Same arithmetic order as kernels_embed_vec.cpp so the
// oracle tracks the device lane-order grouping ((x.x+x.y)+(x.z+x.w) horizontal
// sums per lane, strided ownership tx, tx+256, ...).
[[nodiscard]] float vec4_tree_sum(const luisa::vector<float> &partial) noexcept {
    luisa::vector<float> red{partial};
    for (uint32_t stride = kEmbedVecBlock / 2u; stride > 0u; stride >>= 1u) {
        for (uint32_t t = 0u; t < stride; t++) {
            red[t] += red[t + stride];
        }
    }
    return red[0u];
}

// Per-thread strided (vec4) horizontal-sum partial of f over h4 in
// {tx, tx+256, ...} < H4:  partial[tx] = sum  (v.x+v.y)+(v.z+v.w).
template<typename F>
[[nodiscard]] luisa::vector<float> vec4_partials(uint32_t H4, F &&lane_hsum) {
    luisa::vector<float> partial(kEmbedVecBlock, 0.f);
    for (uint32_t tx = 0u; tx < kEmbedVecBlock; tx++) {
        float s = 0.f;
        for (uint32_t h4 = tx; h4 < H4; h4 += kEmbedVecBlock) {
            s += lane_hsum(h4);
        }
        partial[tx] = s;
    }
    return partial;
}

// CPU reference for embed_lookup_vec4: identical math to embed_cpu_reference,
// but the LN mean/var are accumulated in the vec4 kernel's exact lane order
// (per-thread strided ownership + pinned (x.x+x.y)+(x.z+x.w) horizontal order +
// the halving tree). gamma/beta are float4 (H % 4 == 0 per the host contract).
luisa::vector<float> embed_vec4_cpu_reference(const luisa::vector<uint32_t> &ids,
                                              const luisa::vector<float> &word,
                                              const luisa::vector<float> &pos,
                                              const luisa::vector<float> &gamma,
                                              const luisa::vector<float> &beta,
                                              uint b, uint s, uint h, float eps,
                                              uint pos_offset) {
    const uint rows = b * s;
    const uint H4 = h / 4u;
    const float Hf = static_cast<float>(h);
    luisa::vector<float> out(static_cast<size_t>(rows) * h, 0.f);
    for (uint row = 0u; row < rows; row++) {
        const auto p = static_cast<size_t>((row % s) + pos_offset);
        const auto id = static_cast<size_t>(ids[row]);
        // x4 lane base index in `word`/`pos`: word[id*h + 4*hh + j].
        auto xlane = [&](uint32_t hh, uint32_t j) noexcept {
            const size_t w = id * h + 4u * hh + j;
            const size_t q = p * h + 4u * hh + j;
            return word[w] + pos[q];
        };
        // Pass 1: mean via vec4 horizontal order + tree.
        auto sum1 = vec4_partials(H4, [&](uint32_t hh) noexcept {
            return (xlane(hh, 0u) + xlane(hh, 1u)) + (xlane(hh, 2u) + xlane(hh, 3u));
        });
        const float mean = vec4_tree_sum(sum1) / Hf;
        // Pass 2: biased variance.
        auto sum2 = vec4_partials(H4, [&](uint32_t hh) noexcept {
            const float d0 = xlane(hh, 0u) - mean;
            const float d1 = xlane(hh, 1u) - mean;
            const float d2 = xlane(hh, 2u) - mean;
            const float d3 = xlane(hh, 3u) - mean;
            return (d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3);
        });
        const float var = vec4_tree_sum(sum2) / Hf;
        const float inv = 1.f / std::sqrt(var + eps);
        // Pass 3: affine, per-element expression order matches the kernel.
        for (uint hh = 0u; hh < h; hh++) {
            const float x = xlane(hh / 4u, hh % 4u);
            out[static_cast<size_t>(row) * h + hh] =
                ((x - mean) * inv) * gamma[hh] + beta[hh];
        }
    }
    return out;
}

// embed_lookup_vec4: H % 4 == 0 shapes (host contract). eps = 1e-5, pos_offset =
// 0 (matching the legacy test). Same < 1e-4 gate as test_embed_lookup.
void test_embed_lookup_vec4(Device &device, Stream &stream, uint b, uint s, uint h, uint v) {
    const uint rows = b * s;
    Lcg rng{};
    rng.state = 0xF04B3E00u ^ (h << 8u) ^ (s << 4u) ^ b;

    luisa::vector<uint32_t> ids(rows);
    for (auto &i : ids) {
        i = rng.bounded(v);
    }
    luisa::vector<float> word(static_cast<size_t>(v) * h);
    for (auto &f : word) {
        f = rng.uniform(-1.f, 1.f);
    }
    luisa::vector<float> pos(static_cast<size_t>(s) * h);
    for (auto &f : pos) {
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

    const auto ref = embed_vec4_cpu_reference(ids, word, pos, gamma, beta, b, s, h, 1e-5f, 0u);

    auto word_buf = device.create_buffer<float>(word.size());
    auto pos_buf = device.create_buffer<float>(pos.size());
    auto ids_buf = device.create_buffer<uint32_t>(ids.size());
    auto gamma_buf = device.create_buffer<float>(gamma.size());
    auto beta_buf = device.create_buffer<float>(beta.size());
    auto out_buf = device.create_buffer<float>(static_cast<size_t>(rows) * h);

    stream << word_buf.copy_from(luisa::span{word})
           << pos_buf.copy_from(luisa::span{pos})
           << ids_buf.copy_from(luisa::span{ids})
           << gamma_buf.copy_from(luisa::span{gamma})
           << beta_buf.copy_from(luisa::span{beta})
           << synchronize();

    auto shader = device.compile(make_embed_lookup_vec4_kernel());
    luisa::vector<float> out(static_cast<size_t>(rows) * h, 0.f);
    stream << shader(word_buf.view().as<float4>(), ids_buf, pos_buf.view().as<float4>(),
                     gamma_buf.view().as<float4>(), beta_buf.view().as<float4>(),
                     out_buf.view().as<float4>(), s, h, 1e-5f, 0u)
                  .dispatch(rows * kEmbedVecBlock)// block-per-token row; total threads
           << out_buf.copy_to(luisa::span{out})
           << synchronize();

    float max_diff = 0.f;
    size_t worst = 0u;
    bool ok = true;
    for (size_t i = 0; i < out.size(); i++) {
        const float d = std::abs(out[i] - ref[i]);
        if (d > max_diff) {
            max_diff = d;
            worst = i;
        }
        if (!(d < 1e-4f)) {
            ok = false;
        }
    }
    LUISA_INFO("embed_lookup_vec4 B={} S={} H={} V={}: max_abs_diff={:.3e} (idx {})",
               b, s, h, v, max_diff, worst);
    expect(ok) << "embed_lookup_vec4 B=" << b << " S=" << s << " H=" << h
               << " max-abs-diff must be < 1e-4";
}

void test_embed_lookup(Device &device, Stream &stream, uint b, uint s, uint h, uint v) {
    const uint rows = b * s;
    Lcg rng{};
    rng.state = 0xC0FFEEu ^ (h << 8u) ^ (s << 4u) ^ b;

    luisa::vector<uint32_t> ids(rows);
    for (auto &i : ids) {
        i = rng.bounded(v);
    }
    luisa::vector<float> word(static_cast<size_t>(v) * h);
    for (auto &f : word) {
        f = rng.uniform(-1.f, 1.f);
    }
    luisa::vector<float> pos(static_cast<size_t>(s) * h);// positions 0..S-1 only
    for (auto &f : pos) {
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

    const auto ref = embed_cpu_reference(ids, word, pos, gamma, beta, b, s, h, 1e-5f);

    auto word_buf = device.create_buffer<float>(word.size());
    auto pos_buf = device.create_buffer<float>(pos.size());
    auto ids_buf = device.create_buffer<uint32_t>(ids.size());
    auto gamma_buf = device.create_buffer<float>(gamma.size());
    auto beta_buf = device.create_buffer<float>(beta.size());
    auto out_buf = device.create_buffer<float>(static_cast<size_t>(rows) * h);

    stream << word_buf.copy_from(luisa::span{word})
           << pos_buf.copy_from(luisa::span{pos})
           << ids_buf.copy_from(luisa::span{ids})
           << gamma_buf.copy_from(luisa::span{gamma})
           << beta_buf.copy_from(luisa::span{beta})
           << synchronize();

    auto shader = device.compile(make_embed_lookup_kernel());
    luisa::vector<float> out(static_cast<size_t>(rows) * h, 0.f);
    stream << shader(word_buf, pos_buf, ids_buf, gamma_buf, beta_buf, out_buf, s, h, 1e-5f, 0u)
                  .dispatch(rows * kEmbedBlock)// block-per-token row; total threads
           << out_buf.copy_to(luisa::span{out})
           << synchronize();

    float max_diff = 0.f;
    size_t worst = 0u;
    bool ok = true;
    for (size_t i = 0; i < out.size(); i++) {
        const float d = std::abs(out[i] - ref[i]);
        if (d > max_diff) {
            max_diff = d;
            worst = i;
        }
        if (!(d < 1e-4f)) {
            ok = false;
        }
    }
    LUISA_INFO("embed_lookup B={} S={} H={} V={}: max_abs_diff={:.3e} (idx {})",
               b, s, h, v, max_diff, worst);
    expect(ok) << "embed_lookup B=" << b << " S=" << s << " H=" << h
               << " max-abs-diff must be < 1e-4";
}// namespace

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();
    test_embed_lookup(device, stream, 2u, 5u, 8u, 97u);   // tiny ragged row
    test_embed_lookup(device, stream, 1u, 128u, 64u, 97u);// S=128 mid row
    test_embed_lookup(device, stream, 1u, 1u, 1024u, 97u);// single token
    test_embed_lookup(device, stream, 2u, 5u, 1024u, 97u);// bge-m3 hidden size
    test_embed_lookup(device, stream, 1u, 1u, 2500u, 97u);// shared-cache fallback
    // vec4 variants (H % 4 == 0 by the host contract; eps=1e-5, pos_offset=0):
    // 4/8/64 straddle the H4 < block-size boundary (H4=1, 2, 16) so idle threads
    // contribute 0; 1024 is the production hidden size (H4=256 == block, exact);
    // 2052 (H4=513) exceeds the 512 float4 shared-row cache -> re-gather path.
    test_embed_lookup_vec4(device, stream, 2u, 5u, 4u, 97u);
    test_embed_lookup_vec4(device, stream, 1u, 128u, 8u, 97u);
    test_embed_lookup_vec4(device, stream, 1u, 1u, 64u, 97u);
    test_embed_lookup_vec4(device, stream, 2u, 5u, 1024u, 97u);
    test_embed_lookup_vec4(device, stream, 1u, 1u, 2052u, 97u);
    return 0;
}

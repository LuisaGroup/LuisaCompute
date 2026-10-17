// Test for embedding/embed/kernels/kernels_attention.h -- plan §4 kernels #4
// (attention_scores), #5 (softmax_row) and #6 (attention_pv), the naive M3 path
// over an explicit [B*heads, S, S] logits workspace, plus the M4 fused flash
// attention kernel (online softmax, no S*S workspace).
//
// Coverage (plan §4 "Attention & long sequences", §6 milestones M3/M4, §8 gates):
//   * naive trio, S <= 1024: B in {1,3}, S in {1,2,5,64,127,256,1000,1024}
//     (2/5/64/127/1000 are ragged against the 16x16 tile), heads in {1,4,16,32},
//     head_dim in {8,64,128}. Every stage is compared against an in-file naive
//     CPU oracle: logits after #4 (dot*scale + mask bias), probabilities after
//     #5 (in-place softmax) and the [B*S, heads*head_dim] result after #6.
//   * mask semantics: each case runs with no pads, trailing pads (what the
//     tokenizer produces) and -- for S <= 256 -- interleaved pads. The oracle
//     drops padded keys through the same additive bias, so agreeing on the real
//     rows proves padding cannot leak into a real row's output beyond fp32 noise
//     (pad query rows are compared too: both paths compute the same, defined
//     math for them).
//   * flash kernel: every shape with S <= 1024 is additionally compared
//     head-to-head against the naive trio's device output -- the M4 bucket gate
//     "flash == naive" -- and against the CPU oracle. S = 2048 and S = 8192 have
//     no naive comparison because the trio would need a 268 MB / 4.3 GB logits
//     buffer, which is exactly the M4 rationale; those buckets are gated against
//     a row-wise oracle on a deterministic subset of query rows (the row step is
//     printed) so the host reference stays bounded.
//
// Gate: max-abs-diff <= 1e-4 for naive-vs-CPU, <= 2e-4 for flash-vs-naive and
// flash-vs-CPU. Failures accumulate; one expect() at the end. Every case prints
// its worst max-abs-diff via LUISA_INFO and the run ends with a per-family table
// (backend / family / cases / elements / failures / worst diff / worst shape).
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_b2 test_embed_kernel_attention dx
//   xmake run -o build/embed_b2 test_embed_kernel_attention vk
//   xmake run -o build/embed_b2 test_embed_kernel_attention fallback
#include "ut/ut.hpp"
#include "test_device.h"

#include "embedding/embed/kernels/kernels_attention.h"

#include <luisa/core/fiber.h>
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

// Launch helpers defined in kernels_attention.cpp. The frozen kernels_attention.h
// is parent-owned and does not declare them, so they are declared here against
// the single definition in the engine library (the geometry each one produces is
// documented in the kernels_attention.cpp file header).
namespace luisa::embed {
[[nodiscard]] uint2 attention_scores_dispatch(uint32_t batch, uint32_t seq_len,
                                              uint32_t num_heads) noexcept;
[[nodiscard]] uint2 attention_pv_dispatch(uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                                          uint32_t head_dim) noexcept;
[[nodiscard]] uint32_t softmax_row_dispatch(uint32_t batch, uint32_t seq_len,
                                            uint32_t num_heads) noexcept;
[[nodiscard]] uint2 flash_attention_dispatch(uint32_t batch, uint32_t seq_len,
                                             uint32_t num_heads) noexcept;
[[nodiscard]] uint32_t flash_attention_scratch_floats(uint32_t batch, uint32_t seq_len,
                                                      uint32_t num_heads) noexcept;
}// namespace luisa::embed

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

// Runtime shape under test.
struct AttnCase {
    uint32_t b;
    uint32_t s;
    uint32_t heads;
    uint32_t hd;
    [[nodiscard]] constexpr uint32_t row_stride() const noexcept { return 3u * heads * hd; }
    [[nodiscard]] constexpr uint32_t out_stride() const noexcept { return heads * hd; }
    [[nodiscard]] constexpr size_t logits_count() const noexcept {
        return static_cast<size_t>(b) * heads * s * s;
    }
    [[nodiscard]] constexpr size_t out_count() const noexcept {
        return static_cast<size_t>(b) * s * heads * hd;
    }
    // Naive-path budget: the [B*heads,S,S] fp32 workspace must stay <= 64 MiB
    // (plan §4 M3), which also bounds the host logits reference. 8192^2*16*4 B
    // = 4 GB is exactly what the flash kernel exists to avoid (plan §4/§5).
    [[nodiscard]] constexpr bool naive_fits() const noexcept {
        return logits_count() <= 16777216ull && s <= 1024u;
    }
    // Flash-path budget: kernels_attention.cpp sizes its shared tiles for
    // kFlashMaxHeadDim = 64 (bge-m3's head_dim = 1024/16). Wider head_dim is
    // only covered by the naive trio, so the flash comparisons are skipped for
    // those shapes instead of asserting on a documented capability limit.
    [[nodiscard]] constexpr bool flash_fits() const noexcept { return hd <= 64u; }
};

// Pad layout: none / trailing (tokenizer layout) / interleaved.
enum class PadMode : uint32_t {
    none = 0u,
    trailing = 1u,
    interleaved = 2u,
};

constexpr float kScale = 0.125f;// bge-m3: 1 / sqrt(head_dim) = 1/8
constexpr float kNegLarge = -1.0e30f;
constexpr float kTolNaive = 1e-4f;
constexpr float kTolFlash = 2e-4f;
constexpr float kNegInf = -3.40282347e+38f;

luisa::vector<float> make_qkv(const AttnCase &c, uint32_t seed_mix) {
    Lcg rng{};
    rng.state = 0x9E3779B9u ^ (c.b << 24u) ^ (c.s << 12u) ^ (c.heads << 4u) ^
                (c.hd << 28u) ^ seed_mix;
    luisa::vector<float> qkv(static_cast<size_t>(c.b) * c.s * c.row_stride());
    for (auto &f : qkv) {
        f = rng.uniform(-1.f, 1.f);
    }
    return qkv;
}

luisa::vector<float> make_mask(const AttnCase &c, PadMode mode) {
    luisa::vector<float> mask(static_cast<size_t>(c.b) * c.s, 1.f);
    if (mode == PadMode::none || c.s <= 1u) {
        return mask;// position 0 must stay real, so S == 1 is never padded
    }
    const uint32_t real = c.s * 3u / 4u;// trailing layout: last quarter is padding
    for (uint32_t b = 0u; b < c.b; b++) {
        for (uint32_t i = 0u; i < c.s; i++) {
            const bool keep = (mode == PadMode::trailing) ? (i < real) : ((i % 4u) != 3u);
            if (!keep) {
                mask[static_cast<size_t>(b) * c.s + i] = 0.f;
            }
        }
        mask[static_cast<size_t>(b) * c.s] = 1.f;// <s> is always a real token
    }
    return mask;
}

//-----------------------------------------------------------------------------
// CPU oracles: naive triple loops, fp32, std::exp -- the same math order as the
// ONNX graph and the kernels (pattern from examples/ml/attention_cpu_reference).
//-----------------------------------------------------------------------------

// #4: logits[(b*heads+h)*S*S + qi*S + ki] = dot(Q,K)*scale + (1-mask[key])*neg
luisa::vector<float> logits_cpu_reference(const luisa::vector<float> &qkv,
                                          const luisa::vector<float> &mask,
                                          const AttnCase &c) {
    luisa::vector<float> logits(c.logits_count(), 0.f);
    const uint32_t rows = c.b * c.heads * c.s;
    const uint32_t rs = c.row_stride();
    luisa::fiber::parallel(rows, [&](uint32_t row) noexcept {
        const uint32_t bh = row / c.s;
        const uint32_t qi = row % c.s;
        const uint32_t b = bh / c.heads;
        const uint32_t h = bh % c.heads;
        const float *q = qkv.data() + (static_cast<size_t>(b) * c.s + qi) * rs +
                         static_cast<size_t>(h) * c.hd;
        const float *k0 = qkv.data() + static_cast<size_t>(b) * c.s * rs +
                          static_cast<size_t>(c.heads) * c.hd + static_cast<size_t>(h) * c.hd;
        float *dst = logits.data() + static_cast<size_t>(row) * c.s;
        for (uint32_t ki = 0u; ki < c.s; ki++) {
            const float *k = k0 + static_cast<size_t>(ki) * rs;
            float acc = 0.f;
            for (uint32_t d = 0u; d < c.hd; d++) {
                acc += q[d] * k[d];
            }
            dst[ki] = acc * kScale + (1.f - mask[static_cast<size_t>(b) * c.s + ki]) * kNegLarge;
        }
    });
    return logits;
}

// #5: in-place row softmax (subtract the row max, Exp, normalise by ReduceSum).
void softmax_cpu_reference(luisa::vector<float> &logits, size_t rows, uint32_t s) {
    luisa::fiber::parallel(static_cast<uint32_t>(rows), [&](uint32_t row) noexcept {
        float *p = logits.data() + static_cast<size_t>(row) * s;
        float m = kNegInf;
        for (uint32_t i = 0u; i < s; i++) {
            m = std::max(m, p[i]);
        }
        float sum = 0.f;
        for (uint32_t i = 0u; i < s; i++) {
            p[i] = std::exp(p[i] - m);
            sum += p[i];
        }
        const float inv = 1.f / sum;
        for (uint32_t i = 0u; i < s; i++) {
            p[i] *= inv;
        }
    });
}

// #6: out[b*S+qi, h*hd+d] = sum_ki P[bh][qi][ki] * V(b,ki,h,d).
luisa::vector<float> pv_cpu_reference(const luisa::vector<float> &qkv,
                                      const luisa::vector<float> &probs,
                                      const AttnCase &c) {
    luisa::vector<float> out(c.out_count(), 0.f);
    const uint32_t rows = c.b * c.heads * c.s;
    const uint32_t rs = c.row_stride();
    luisa::fiber::parallel(rows, [&](uint32_t row) noexcept {
        const uint32_t bh = row / c.s;
        const uint32_t qi = row % c.s;
        const uint32_t b = bh / c.heads;
        const uint32_t h = bh % c.heads;
        const float *p = probs.data() + static_cast<size_t>(row) * c.s;
        const float *v0 = qkv.data() + static_cast<size_t>(b) * c.s * rs +
                          2u * static_cast<size_t>(c.heads) * c.hd +
                          static_cast<size_t>(h) * c.hd;
        const size_t obase = (static_cast<size_t>(b) * c.s + qi) * c.out_stride() +
                             static_cast<size_t>(h) * c.hd;
        for (uint32_t d = 0u; d < c.hd; d++) {
            float acc = 0.f;
            for (uint32_t ki = 0u; ki < c.s; ki++) {
                acc += p[ki] * v0[static_cast<size_t>(ki) * rs + d];
            }
            out[obase + d] = acc;
        }
    });
    return out;
}

// Row-wise oracle for the flash kernel: O(S) host memory per row, so it also
// covers S = 2048 / 8192 where the full logits tensor cannot be materialised.
// Only flattened rows (bh*S + qi) with row % row_step == 0 are computed.
struct OracleOut {
    luisa::vector<float> out;
    uint32_t row_step = 1u;
    uint32_t sampled_rows = 1u;
};

OracleOut attention_flash_oracle(const AttnCase &c, const luisa::vector<float> &qkv,
                                 const luisa::vector<float> &mask) {
    const uint32_t rows = c.b * c.heads * c.s;
    // Keep the host reference at a couple of hundred million multiply-adds per
    // sweep (a row costs 2*S*hd); the resulting step is printed with the case.
    const double per_row = 2.0 * c.s * c.hd;
    constexpr double kWorkCap = 2.0e8;
    uint32_t step = 1u;
    if (static_cast<double>(rows) * per_row > kWorkCap) {
        const auto used = static_cast<uint32_t>(kWorkCap / per_row);
        step = used > 0u ? (rows + used - 1u) / used : 1u;
    }
    OracleOut o;
    o.row_step = step;
    o.sampled_rows = (rows + step - 1u) / step;
    o.out.assign(c.out_count(), 0.f);
    const uint32_t rs = c.row_stride();
    luisa::fiber::parallel(o.sampled_rows, [&](uint32_t j) noexcept {
        const uint32_t row = j * step;
        const uint32_t bh = row / c.s;
        const uint32_t qi = row % c.s;
        const uint32_t b = bh / c.heads;
        const uint32_t h = bh % c.heads;
        const float *q = qkv.data() + (static_cast<size_t>(b) * c.s + qi) * rs +
                         static_cast<size_t>(h) * c.hd;
        const float *k0 = qkv.data() + static_cast<size_t>(b) * c.s * rs +
                          static_cast<size_t>(c.heads) * c.hd + static_cast<size_t>(h) * c.hd;
        const float *v0 = k0 + static_cast<size_t>(c.heads) * c.hd;
        luisa::vector<float> p(c.s);
        float m = kNegInf;
        for (uint32_t ki = 0u; ki < c.s; ki++) {
            const float *k = k0 + static_cast<size_t>(ki) * rs;
            float acc = 0.f;
            for (uint32_t d = 0u; d < c.hd; d++) {
                acc += q[d] * k[d];
            }
            p[ki] = acc * kScale + (1.f - mask[static_cast<size_t>(b) * c.s + ki]) * kNegLarge;
            m = std::max(m, p[ki]);
        }
        float sum = 0.f;
        for (uint32_t ki = 0u; ki < c.s; ki++) {
            p[ki] = std::exp(p[ki] - m);
            sum += p[ki];
        }
        const float inv = 1.f / sum;
        const size_t obase = (static_cast<size_t>(b) * c.s + qi) * c.out_stride() +
                             static_cast<size_t>(h) * c.hd;
        for (uint32_t d = 0u; d < c.hd; d++) {
            float acc = 0.f;
            for (uint32_t ki = 0u; ki < c.s; ki++) {
                acc += p[ki] * inv * v0[static_cast<size_t>(ki) * rs + d];
            }
            o.out[obase + d] = acc;
        }
    });
    return o;
}

//-----------------------------------------------------------------------------
// Parity accounting
//-----------------------------------------------------------------------------
struct Family {
    const char *name = "";
    uint32_t cases = 0u;
    uint32_t elements = 0u;
    uint32_t failures = 0u;
    float worst = -1.f;// -1 so even an all-zero-diff family reports a shape
    AttnCase worst_case{};
    PadMode worst_mode = PadMode::none;
};

// One slot per comparison family; passed by reference so no name lookup (and no
// dangling reference into a growing vector) is needed.
struct Families {
    Family scores{"naive-vs-cpu/logits"};
    Family probs{"naive-vs-cpu/probs"};
    Family naive_out{"naive-vs-cpu/out"};
    Family flash_cpu{"flash-vs-cpu/out"};
    Family flash_naive{"flash-vs-naive/out"};
};

// Elementwise max-abs-diff against a reference, with a per-element predicate so
// the subsampled oracle can skip the rows it did not compute.
template<typename Valid>
[[nodiscard]] float compare(Family &fam, const luisa::vector<float> &got,
                            const luisa::vector<float> &ref, const AttnCase &c, PadMode mode,
                            float tol, const char *what, Valid valid) {
    float max_diff = 0.f;
    uint32_t bad = 0u;
    uint32_t checked = 0u;
    for (size_t i = 0u; i < got.size(); i++) {
        if (!valid(i)) {
            continue;
        }
        const float d = std::abs(got[i] - ref[i]);
        checked++;
        if (d > max_diff) {
            max_diff = d;
        }
        if (!(d <= tol)) {
            if (bad < 4u) {
                LUISA_WARNING("{} B={} S={} heads={} hd={} pads={} idx {}/{}: got {} ref {}",
                              what, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode), i,
                              got.size(), got[i], ref[i]);
            }
            bad++;
        }
    }
    if (checked == 0u) {
        LUISA_WARNING("{} B={} S={} heads={} hd={}: nothing compared!", what, c.b, c.s, c.heads,
                      c.hd);
        bad++;
    }
    fam.cases++;
    fam.elements += checked;
    fam.failures += bad;
    if (max_diff > fam.worst) {
        fam.worst = max_diff;
        fam.worst_case = c;
        fam.worst_mode = mode;
    }
    return max_diff;
}

[[nodiscard]] constexpr bool every_element(size_t) noexcept { return true; }

// Maps a flat out[b*S+qi, h*hd+d] element index to the oracle's flattened row id
// (bh*S + qi) and tests the subsampling step.
struct RowStepValid {
    uint32_t s;
    uint32_t heads;
    uint32_t hd;
    uint32_t step;
    [[nodiscard]] bool operator()(size_t i) const noexcept {
        const uint32_t out_stride = heads * hd;
        const uint32_t logical_row = static_cast<uint32_t>(i / out_stride);
        const uint32_t b = logical_row / s;
        const uint32_t qi = logical_row % s;
        const uint32_t h = static_cast<uint32_t>(i % out_stride) / hd;
        return (static_cast<size_t>(b * heads + h) * s + qi) % step == 0u;
    }
};

}// namespace

namespace {

// One (shape, pad-mode) case: upload the shared inputs once, then run whichever
// paths the shape allows and compare each against its oracle.
template<typename ShScores, typename ShSoftmax, typename ShPv, typename ShFlash>
void check_attention_case(Device &device, Stream &stream, Families &fam, const ShScores &scores,
                          const ShSoftmax &softmax, const ShPv &pv, const ShFlash &flash,
                          const AttnCase &c, PadMode mode, const char *backend) {
    const auto qkv = make_qkv(c, 0x51u << static_cast<uint32_t>(mode));
    const auto mask = make_mask(c, mode);

    auto qkv_buf = device.create_buffer<float>(qkv.size());
    auto mask_buf = device.create_buffer<float>(mask.size());
    auto out_buf = device.create_buffer<float>(c.out_count());
    auto scratch_buf = device.create_buffer<float>(
        flash_attention_scratch_floats(c.b, c.s, c.heads));
    // Naive-path workspace. Buffer is move-only, so instead of default-creating
    // an empty handle and assigning later, bind a 1-element stand-in whenever the
    // real logits tensor does not fit (it is never touched on those shapes).
    auto logits_buf = c.naive_fits()
                          ? device.create_buffer<float>(static_cast<uint32_t>(c.logits_count()))
                          : device.create_buffer<float>(1u);

    stream << qkv_buf.copy_from(luisa::span{qkv}) << mask_buf.copy_from(luisa::span{mask});

    luisa::vector<float> naive_out;    // device result of the naive trio
    luisa::vector<float> out_ref_full; // CPU result matching it exactly

    if (c.naive_fits()) {
        luisa::vector<float> logits_dev(c.logits_count(), 0.f);

        // #4 attention_scores: dot * scale + mask bias into [B*heads, S, S]
        stream << scores(qkv_buf.view(), mask_buf.view(), logits_buf.view(), c.s, c.heads, c.hd,
                         kScale, kNegLarge)
                      .dispatch(attention_scores_dispatch(c.b, c.s, c.heads))
               << logits_buf.copy_to(luisa::span{logits_dev}) << synchronize();
        const auto logits_ref = logits_cpu_reference(qkv, mask, c);
        float diff = compare(fam.scores, logits_dev, logits_ref, c, mode, kTolNaive, "scores",
                             every_element);
        LUISA_INFO("[{} naive #4] B={} S={} heads={} hd={} pads={}: max_abs_diff={:.3e} "
                   "({} logits)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode), diff,
                   static_cast<uint32_t>(c.logits_count()));

        // #5 softmax_row, in place
        stream << softmax(logits_buf.view(), c.s)
                      .dispatch(softmax_row_dispatch(c.b, c.s, c.heads))
               << logits_buf.copy_to(luisa::span{logits_dev}) << synchronize();
        auto probs_ref = logits_ref;
        softmax_cpu_reference(probs_ref, static_cast<size_t>(c.b) * c.heads * c.s, c.s);
        diff = compare(fam.probs, logits_dev, probs_ref, c, mode, kTolNaive, "softmax",
                       every_element);
        LUISA_INFO("[{} naive #5] B={} S={} heads={} hd={} pads={}: max_abs_diff={:.3e} "
                   "({} probs)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode), diff,
                   static_cast<uint32_t>(c.logits_count()));

        // #6 attention_pv -> [B*S, heads*head_dim]
        naive_out.assign(c.out_count(), 0.f);
        stream << pv(qkv_buf.view(), logits_buf.view(), out_buf.view(), c.s, c.heads, c.hd)
                      .dispatch(attention_pv_dispatch(c.b, c.s, c.heads, c.hd))
               << out_buf.copy_to(luisa::span{naive_out}) << synchronize();
        out_ref_full = pv_cpu_reference(qkv, probs_ref, c);
        diff = compare(fam.naive_out, naive_out, out_ref_full, c, mode, kTolNaive, "pv",
                       every_element);
        LUISA_INFO("[{} naive #6] B={} S={} heads={} hd={} pads={}: max_abs_diff={:.3e} "
                   "({} elems)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode), diff,
                   static_cast<uint32_t>(c.out_count()));
    } else {
        LUISA_INFO("[{} naive] B={} S={} heads={} hd={} pads={}: skipped -- the logits "
                   "workspace would be {:.1f} MB (M4 rationale, plan §4)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode),
                   static_cast<double>(c.logits_count()) * 4.0 / 1048576.0);
    }

    if (!c.flash_fits()) {
        LUISA_INFO("[{} flash] B={} S={} heads={} hd={} pads={}: skipped -- head_dim is wider "
                   "than kFlashMaxHeadDim (64) in kernels_attention.cpp; the naive trio covers "
                   "wide heads",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode));
        return;
    }

    // Flash on the same inputs.
    luisa::vector<float> flash_out(c.out_count(), 0.f);
    stream << flash(qkv_buf.view(), mask_buf.view(), out_buf.view(), scratch_buf.view(), c.b, c.s,
                    c.heads, c.hd, kScale, kNegLarge)
                  .dispatch(flash_attention_dispatch(c.b, c.s, c.heads))
           << out_buf.copy_to(luisa::span{flash_out}) << synchronize();

    if (c.naive_fits()) {
        // The full CPU reference exists, so neither comparison skips anything.
        float diff = compare(fam.flash_cpu, flash_out, out_ref_full, c, mode, kTolFlash,
                             "flash-vs-cpu", every_element);
        LUISA_INFO("[{} flash vs-cpu] B={} S={} heads={} hd={} pads={}: max_abs_diff={:.3e} "
                   "({} elems)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode), diff,
                   static_cast<uint32_t>(c.out_count()));
        diff = compare(fam.flash_naive, flash_out, naive_out, c, mode, kTolFlash,
                       "flash-vs-naive", every_element);
        LUISA_INFO("[{} flash vs-naive] B={} S={} heads={} hd={} pads={}: max_abs_diff={:.3e} "
                   "({} elems)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode), diff,
                   static_cast<uint32_t>(c.out_count()));
    } else {
        // S = 2048 / 8192: only the row-wise oracle can run.
        const auto oracle = attention_flash_oracle(c, qkv, mask);
        const float diff = compare(
            fam.flash_cpu, flash_out, oracle.out, c, mode, kTolFlash, "flash-vs-cpu",
            RowStepValid{c.s, c.heads, c.hd, oracle.row_step});
        LUISA_INFO("[{} flash vs-cpu] B={} S={} heads={} hd={} pads={} row_step={}: "
                   "max_abs_diff={:.3e} ({} of {} rows sampled)",
                   backend, c.b, c.s, c.heads, c.hd, static_cast<uint32_t>(mode),
                   oracle.row_step, diff, oracle.sampled_rows, c.b * c.heads * c.s);
    }
}

constexpr AttnCase kNaiveCases[] = {
    {1u, 1u, 1u, 8u},      {1u, 1u, 16u, 64u},  {1u, 2u, 4u, 8u},
    {3u, 2u, 1u, 64u},     {1u, 5u, 1u, 64u},   {3u, 5u, 4u, 8u},
    {1u, 5u, 16u, 128u},   {1u, 64u, 16u, 64u}, {3u, 64u, 4u, 8u},
    {1u, 64u, 32u, 64u},   {3u, 64u, 1u, 128u}, {1u, 127u, 16u, 64u},
    {3u, 127u, 4u, 64u},   {1u, 127u, 1u, 8u},  {3u, 127u, 32u, 8u},
    {1u, 256u, 16u, 64u},  {3u, 256u, 4u, 8u},  {1u, 256u, 1u, 128u},
    {1u, 1000u, 16u, 64u}, {3u, 1000u, 1u, 8u}, {1u, 1024u, 16u, 64u},
    {1u, 1024u, 4u, 128u}, {3u, 1024u, 1u, 64u},
    // Non-multiple-of-4 head_dims: exercise the ragged tail of the 16-wide
    // head_dim chunk (naive) and of the unroll-by-4 shared tiles (flash).
    {1u, 37u, 4u, 6u},     {3u, 5u, 2u, 7u},   {1u, 127u, 1u, 1u},
};

constexpr AttnCase kFlashCases[] = {
    {1u, 37u, 4u, 64u},   {2u, 37u, 1u, 8u},     {2u, 128u, 16u, 64u},
    {1u, 128u, 32u, 8u},  {1u, 1000u, 4u, 64u},  {2u, 1000u, 16u, 64u},
    {1u, 2048u, 16u, 64u},{2u, 2048u, 1u, 64u},  {1u, 8192u, 16u, 64u},
    {2u, 8192u, 16u, 64u}, {1u, 640u, 16u, 6u},
};

[[nodiscard]] luisa::vector<PadMode> modes_for(const AttnCase &c) noexcept {
    luisa::vector<PadMode> m{PadMode::none, PadMode::trailing};
    if (c.s <= 256u) {
        m.emplace_back(PadMode::interleaved);// interleaved pads on the cheap shapes
    }
    return m;
}

void print_table(const Families &fam, const char *backend) {
    const Family *all[] = {&fam.scores, &fam.probs, &fam.naive_out, &fam.flash_cpu,
                           &fam.flash_naive};
    uint32_t failures = 0u;
    uint32_t cases = 0u;
    for (const Family *f : all) {
        failures += f->failures;
        cases += f->cases;
        LUISA_INFO("[table {}] name={} cases={} elems={} failures={} worst={:.3e} "
                   "at B={} S={} heads={} hd={} pads={}",
                   backend, f->name, f->cases, f->elements, f->failures, f->worst,
                   f->worst_case.b, f->worst_case.s, f->worst_case.heads, f->worst_case.hd,
                   static_cast<uint32_t>(f->worst_mode));
    }
    LUISA_INFO("== backend={}: {} case-runs, {} mismatched elements", backend, cases, failures);
}

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto stream = device.create_stream();
    const char *backend = argc > 1 ? argv[1] : "?";

    // The CPU oracles fan rows out over luisa::fiber::parallel, which needs a
    // bound fiber scheduler -- exactly the examples/ml/attention_cpu_reference
    // pattern (without it marl::Scheduler::get() is null and the oracle crashes).
    luisa::fiber::scheduler ref_scheduler;

    auto scores = device.compile(make_attention_scores_kernel());
    auto softmax = device.compile(make_softmax_row_kernel());
    auto pv = device.compile(make_attention_pv_kernel());
    auto flash = device.compile(make_flash_attention_kernel());

    Families fam;
    LUISA_INFO("== naive attention trio (M3, plan §4 #4-#6) + flash attention (M4), backend={}",
               backend);
    for (const auto &c : kNaiveCases) {
        for (const auto m : modes_for(c)) {
            check_attention_case(device, stream, fam, scores, softmax, pv, flash, c, m, backend);
        }
    }
    for (const auto &c : kFlashCases) {
        for (const auto m : modes_for(c)) {
            check_attention_case(device, stream, fam, scores, softmax, pv, flash, c, m, backend);
        }
    }

    print_table(fam, backend);
    const uint32_t failures = fam.scores.failures + fam.probs.failures + fam.naive_out.failures +
                              fam.flash_cpu.failures + fam.flash_naive.failures;
    expect(failures == 0u) << luisa::format(
        "attention kernels produced {} mismatched elements on {}", failures, backend);
    return 0;
}

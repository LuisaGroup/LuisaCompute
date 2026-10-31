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
// INT8 weight storage (INT8 JIT phase 2, GemmWeightFormat::Int8): the same
// GEMM/epilogue kernels built with a packed-B load path, i.e. B carries the
// positive per-tensor fp32 scale in word 0 followed by four signed int8 codes
// per word (contract in kernels_gemm.h). A small targeted case set covers
// ragged/odd logical element counts (K*N = 1, 18, 35, 247, 1221 -- never a
// multiple of 4, so the last packed word is always partially filled with 1..3
// used bytes and a zero tail), zero tail bytes,
// positive and negative codes including the sign-extension boundaries
// (-127 / -128 / -1 / +1 / +127 / 0), and the None / Bias / BiasGeluErf
// epilogues. The oracle dequantizes the codes host-side (value = code * scale,
// embed/quantization.h) and then runs the same fp64 triple-loop reference, so
// the gate measures the kernel's dequantization + accumulation, not the
// quantizer. Accumulation stays fp32 on the device: INT8 is weight storage, not
// integer accumulation.
//
// Gate (plan §8 numerical gate): per case, every output must satisfy
// |diff| <= 1e-3 * max(1, |ref|) (fp32 GEMM tolerance, used unchanged for the
// INT8 cases); the per-case max-abs-diff and cosine similarity are printed with
// LUISA_INFO on every case and the worst case is reported at the end. Failures
// accumulate; a single expect() at the end.
//
// Phase 1a (packed-half packer + real FP16/BF16 GEMM): two extra blocks.
// 1. make_pack_half_kernel: fp32 -> packed bf16/fp16 words, checked BIT FOR BIT
//    against a host oracle that is a literal mirror of tools/pack_model.cpp
//    f32_to_bf16_rne / f32_to_f16_rne (no tolerance: the packed buffer must be
//    byte-identical to what the packer would have written, so a device-packed
//    weight decodes to the same fp32 values). Covers odd and even element
//    counts, the odd-tail zero high half, signed zero, fp32 normal/subnormal
//    extremes, inf, quiet + signaling NaN (payload quieting), the fp16 finite
//    limit and its overflow, the fp16 subnormal range and its ties, and both
//    formats' rounding ties (round-half-to-even).
// 2. FP16 / BF16 GEMM over a DEVICE-packed B: the pack kernel fills B from the
//    fp32 weight matrix (NaN-poisoned first, so a word it fails to write can
//    never pass), then the half-format GEMM reads it and the existing fp64
//    oracle runs over the HOST-ROUNDED B values, i.e. the gate measures the
//    kernel pair and not the rounding itself. None / Bias / BiasGeluErf.
//
// Run (device test, backend = argv[1]):
//   xmake run -o build/embed_quant_p2 test_embed_kernel_gemm dx
//   xmake run -o build/embed_quant_p2 test_embed_kernel_gemm fallback
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

// Shared output comparison for the fp32 and the INT8 (packed-B) case sweepers:
// same plan §8 gate for both (|diff| <= 1e-3 * max(1, |ref|) -- INT8 does NOT
// get a looser tolerance), and the reported metrics are the max-abs-diff plus
// the cosine similarity over the whole output tensor.
void compare_outputs(const char *tag, const GemmCase &c, GemmEpilogue ep,
                     const luisa::vector<float> &got, const luisa::vector<float> &ref,
                     Parity &parity) {
    float max_diff = 0.f;
    uint32_t bad = 0u;
    double dot = 0.0;
    double norm_got = 0.0;
    double norm_ref = 0.0;
    for (size_t i = 0u; i < ref.size(); i++) {
        const double g = static_cast<double>(got[i]);
        const double r = static_cast<double>(ref[i]);
        dot += g * r;
        norm_got += g * g;
        norm_ref += r * r;
        const float d = std::abs(got[i] - ref[i]);
        if (d > max_diff) {
            max_diff = d;
        }
        if (!(d <= 1e-3f * std::max(1.f, std::abs(ref[i])))) {
            if (bad < 4u) {
                LUISA_WARNING("gemm[{}] case (M={}, N={}, K={}) idx {}: got {} ref {}",
                              tag, c.m, c.n, c.k, i, got[i], ref[i]);
            }
            bad++;
        }
    }
    const double denom = std::sqrt(norm_got) * std::sqrt(norm_ref);
    const double cosine = denom > 0.0 ? dot / denom : 1.0;
    LUISA_INFO("gemm[{}] M={} N={} K={}: max_abs_diff={:.3e} cosine={:.9f} bad={}/{}",
               tag, c.m, c.n, c.k, max_diff, cosine, bad, ref.size());
    parity.cases++;
    parity.failures += bad;
    if (max_diff > parity.worst_abs_diff) {
        parity.worst_abs_diff = max_diff;
        parity.worst = c;
        parity.worst_ep = ep;
    }
}

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

    compare_outputs(gemm_epilogue_name(ep), c, ep, got, ref, parity);
}

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

//----------------------------------------------------------------------------------------------------
// INT8 weight storage (INT8 JIT phase 2, GemmWeightFormat::Int8). The B operand
// is the packed buffer documented in kernels_gemm.h: word 0 = positive
// per-tensor fp32 scale, then four signed int8 codes per word in emission order
// e = k*N + n, little-endian (element 4j+c in bits [8c+7:8c]), unused tail bytes
// zero. Everything else (A, aux, C, accumulators) stays fp32.
//----------------------------------------------------------------------------------------------------

struct Int8Case {
    uint32_t m, n, k;
    float scale;///< per-tensor scale stored in B word 0
};

// K*N is 1, 18, 35, 247, 1221 -- never a multiple of 4, so the last packed word
// is always partially filled (1, 2 or 3 used bytes + zero tail), and K never
// divides the 16-wide slab. The scales are all positive and distinct; 1/127 is
// the value the packer produces for weights in [-1, 1] (quantization.h).
constexpr Int8Case kInt8Cases[] = {
    {1u, 1u, 1u, 0.0078740157f},
    {5u, 3u, 6u, 0.0371f},
    {3u, 5u, 7u, 0.0091f},
    {17u, 13u, 19u, 0.02f},
    {37u, 37u, 33u, 0.0129f},
};

// Deterministic code pattern: the first eight elements pin the interesting bit
// patterns so every case covers sign extension on both sides, the zero code and
// both saturating magnitudes; -128 is outside the packer's [-127, 127] range but
// is a legal signed-int8 storage value, so the decoder must handle it too. The
// rest alternates scattered zeros (zero-code lanes must contribute nothing) and
// a deterministic walk over the full code range.
[[nodiscard]] int8_t int8_test_code(size_t e, Lcg &rng) noexcept {
    switch (e) {
        case 0u: return -127;
        case 1u: return 127;
        case 2u: return -1;
        case 3u: return 1;
        case 4u: return 0;
        case 5u: return -128;
        case 6u: return 2;
        case 7u: return -2;
        default: break;
    }
    if (e % 11u == 0u) { return 0; }
    return static_cast<int8_t>(static_cast<int32_t>(rng.next() % 255u) - 127);
}

// Host mirror of the device decode (kernels_gemm.h contract): 1 + ceil(K*N/4)
// words, scale in word 0, four little-endian int8 codes per following word, zero
// tail bytes.
[[nodiscard]] luisa::vector<float> pack_int8_b(const luisa::vector<int8_t> &codes,
                                               float scale) noexcept {
    const size_t words = 1u + (codes.size() + 3u) / 4u;
    luisa::vector<float> packed(words, 0.f);
    packed[0] = scale;
    for (size_t j = 0u; 1u + j < words; j++) {
        uint32_t w = 0u;
        for (size_t c = 0u; c < 4u; c++) {
            const size_t e = j * 4u + c;
            if (e < codes.size()) {
                w |= static_cast<uint32_t>(static_cast<uint32_t>(codes[e]) & 0xffu) << (8u * c);
            }// else: the unused tail byte stays zero (contract)
        }
        packed[1u + j] = luisa::bit_cast<float>(w);
    }
    return packed;
}

// Host-only layout/decoder check (no device): the packing helper above must
// produce the documented word layout, sign-extend the codes back and leave the
// tail bytes zero. A packing bug here would otherwise show up as a mystery
// kernel mismatch, so it is gated separately.
void check_int8_pack_layout(Parity &parity) {
    // 9 codes -> 1 + 3 words, last word 1 used byte + 3 zero bytes.
    const luisa::vector<int8_t> codes{-127, 127, -1, 1, 0, -128, 2, -2, -37};
    const float scale = 0.0137f;
    const auto packed = pack_int8_b(codes, scale);
    uint32_t bad = 0u;
    if (packed.size() != 4u) { bad++; }
    if (!(packed.size() == 4u && packed[0] == scale)) { bad++; }
    if (packed.size() == 4u) {
        for (size_t j = 0u; j < 3u; j++) {
            const uint32_t w = luisa::bit_cast<uint32_t>(packed[1u + j]);
            for (size_t c = 0u; c < 4u; c++) {
                const size_t e = j * 4u + c;
                const uint32_t byte = (w >> (8u * c)) & 0xffu;
                const uint32_t expect = e < codes.size()
                                            ? (static_cast<uint32_t>(codes[e]) & 0xffu)
                                            : 0u;// zero tail
                if (byte != expect) { bad++; }
                // sign-extend the byte exactly like load_b_element<Int8> and
                // compare with the original code
                const int32_t code = byte >= 0x80u
                                         ? static_cast<int32_t>(byte) - 256
                                         : static_cast<int32_t>(byte);
                const int32_t want = e < codes.size() ? static_cast<int32_t>(codes[e]) : 0;
                if (code != want) { bad++; }
                if (e >= codes.size() && byte != 0u) { bad++; }
            }
        }
    }
    LUISA_INFO("int8 packed-B layout: words={} scale={} mismatches={}", packed.size(), scale, bad);
    parity.cases++;
    parity.failures += bad;
}

// "int8" (plus the unchanged fp32/bf16/fp16 names) -- host-only: the executor and
// the benchmark label shaders from this string.
void check_gemm_weight_format_names(Parity &parity) {
    struct NameCase {
        GemmWeightFormat wf;
        const char *name;
    };
    constexpr NameCase kNames[] = {
        {GemmWeightFormat::Fp32, "fp32"},
        {GemmWeightFormat::Bf16, "bf16"},
        {GemmWeightFormat::Fp16, "fp16"},
        {GemmWeightFormat::Int8, "int8"},
    };
    for (const auto &n : kNames) {
        const bool ok = luisa::string_view{gemm_weight_format_name(n.wf)} == n.name;
        LUISA_INFO("gemm weight format name [{}]: {} -> \"{}\"",
                   static_cast<uint32_t>(n.wf), n.name, gemm_weight_format_name(n.wf));
        expect(ok) << "gemm_weight_format_name must report " << n.name;
        parity.cases++;
        parity.failures += ok ? 0u : 1u;
    }
}

// One (epilogue, size) INT8 case: pack [scale | codes] -> upload -> dispatch ->
// download -> compare against the fp64 oracle over the dequantized B.
template<typename Shader>
void check_gemm_int8_case(Device &device, Stream &stream, const Shader &shader,
                          GemmEpilogue ep, const Int8Case &c, Parity &parity) {
    Lcg rng{};
    rng.state = 0x85EBCA6Bu ^ (c.m << 24u) ^ (c.n << 12u) ^ (c.k << 2u) ^
                (static_cast<uint32_t>(ep) << 28u);

    const size_t a_count = static_cast<size_t>(c.m) * c.k;
    const size_t kn = static_cast<size_t>(c.k) * c.n;
    const size_t c_count = static_cast<size_t>(c.m) * c.n;

    luisa::vector<float> a(a_count);
    for (auto &f : a) {
        f = rng.uniform(-1.f, 1.f);
    }
    luisa::vector<int8_t> codes(kn);
    for (size_t i = 0u; i < kn; i++) {
        codes[i] = int8_test_code(i, rng);
    }
    const auto b_packed = pack_int8_b(codes, c.scale);
    // Oracle operand: the dequantized values exactly as the kernel computes them
    // (value = code * scale, one fp32 multiply -- quantization.h), fed to the
    // fp64 triple-loop reference below.
    luisa::vector<float> b_deq(kn);
    for (size_t i = 0u; i < kn; i++) {
        b_deq[i] = static_cast<float>(codes[i]) * c.scale;
    }
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

    const auto ref = gemm_cpu_reference(a, b_deq, aux, ep, c.m, c.n, c.k);

    auto a_buf = device.create_buffer<float>(a_count);
    auto b_buf = device.create_buffer<float>(b_packed.size());
    auto c_buf = device.create_buffer<float>(c_count);
    auto aux_buf = device.create_buffer<float>(aux_count);

    luisa::vector<float> got(c_count, 0.f);
    stream << a_buf.copy_from(luisa::span{a})
           << b_buf.copy_from(luisa::span{b_packed})
           << aux_buf.copy_from(luisa::span{aux})
           << shader(a_buf.view(), b_buf.view(), c_buf.view(), aux_buf.view(), c.m, c.n, c.k)
                  .dispatch(gemm_dispatch_size(c.m, c.n))// dispatch = TOTAL threads
           << c_buf.copy_to(luisa::span{got})
           << synchronize();

    const auto tag = luisa::format("int8/{}", gemm_epilogue_name(ep));
    compare_outputs(tag.c_str(), {c.m, c.n, c.k}, ep, got, ref, parity);
}

// Whole epilogue: compile the INT8 kernel once, sweep the packed-B cases.
void check_gemm_int8_epilogue(Device &device, Stream &stream, GemmEpilogue ep, Parity &parity) {
    auto shader = device.compile(make_gemm_kernel(ep, GemmWeightFormat::Int8));
    for (const auto &c : kInt8Cases) {
        check_gemm_int8_case(device, stream, shader, ep, c, parity);
    }
}

//----------------------------------------------------------------------------------------------------
// Phase 1a, part 1: fp32 -> packed-half weight packer (make_pack_half_kernel).
// The host oracle below is a literal mirror of tools/pack_model.cpp
// f32_to_bf16_rne / f32_to_f16_rne (the packer that produced every on-disk v2
// blob), so a device-packed word must equal the oracle word BIT FOR BIT --
// there is no tolerance here on purpose: the packed B buffer is not "close" to
// the packer's, it is the same bytes, otherwise a device-packed weight and a
// packer-packed weight would decode to different fp32 values.
//----------------------------------------------------------------------------------------------------

// RNE fp32 -> bf16 (packer f32_to_bf16_rne): inf keeps its high bits, NaN
// additionally gets the quiet payload bit so it stays NaN; otherwise round the
// 16 discarded mantissa bits (which also rounds finite overflow to inf).
[[nodiscard]] uint16_t f32_to_bf16_rne_host(float v) noexcept {
    uint32_t u = luisa::bit_cast<uint32_t>(v);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        if ((u & 0x007fffffu) != 0u) { u |= 0x00400000u; }
        return static_cast<uint16_t>(u >> 16u);
    }
    const uint32_t lsb = (u >> 16u) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16u);
}

// RNE fp32 -> fp16 (packer f32_to_f16_rne): inf/NaN, overflow to inf, normal
// with a mantissa carry, fp16 subnormal with a rounding residue, else +-0.
[[nodiscard]] uint16_t f32_to_f16_rne_host(float v) noexcept {
    const uint32_t u = luisa::bit_cast<uint32_t>(v);
    const uint32_t sign = (u >> 16u) & 0x8000u;
    const uint32_t e = (u >> 23u) & 0xffu;
    const uint32_t m = u & 0x007fffffu;
    if (e == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u |
                                     (m != 0u ? (0x200u | ((m >> 13u) & 0x1ffu)) : 0u));
    }
    if (e >= 143u) { return static_cast<uint16_t>(sign | 0x7c00u); }// |v| >= 2^16
    if (e >= 113u) {// normal: RNE on the 13 discarded mantissa bits
        const uint32_t r = m & 0x1fffu;
        uint32_t h = m >> 13u;
        if (r > 0x1000u || (r == 0x1000u && (h & 1u))) { ++h; }
        if (h == 0x400u) { return static_cast<uint16_t>(sign | ((e - 111u) << 10u)); }
        return static_cast<uint16_t>(sign | ((e - 112u) << 10u) | h);
    }
    if (e >= 103u) {// fp16 subnormal: q = M >> (126 - e), RNE on the residue
        const uint32_t M = m | 0x800000u;
        const uint32_t sh = 126u - e;
        const uint32_t r = M & ((1u << sh) - 1u);
        uint32_t q = M >> sh;
        const uint32_t half = 1u << (sh - 1u);
        if (r > half || (r == half && (q & 1u))) { ++q; }
        return static_cast<uint16_t>(sign | q);
    }
    return static_cast<uint16_t>(sign);// |v| < 2^-25
}

[[nodiscard]] uint16_t to_half_bits_host(float v, GemmWeightFormat wf) noexcept {
    return wf == GemmWeightFormat::Bf16 ? f32_to_bf16_rne_host(v) : f32_to_f16_rne_host(v);
}

// Host pack: word j = element 2j in bits [15:0], element 2j+1 in bits [31:16],
// and an odd element count leaves the last word's high half ZERO.
[[nodiscard]] luisa::vector<float> pack_half_words_host(const luisa::vector<float> &src,
                                                        GemmWeightFormat wf) noexcept {
    luisa::vector<float> words((src.size() + 1u) / 2u, 0.f);
    for (size_t j = 0u; j < words.size(); j++) {
        const uint32_t lo = to_half_bits_host(src[2u * j], wf);
        const uint32_t hi = 2u * j + 1u < src.size() ? to_half_bits_host(src[2u * j + 1u], wf) : 0u;
        words[j] = luisa::bit_cast<float>(lo | (hi << 16u));
    }
    return words;
}

// Half -> fp32 upcast of ONE packed half: exactly the mapping the GEMM's
// load_b_element does (bf16 = bit extension, fp16 = exact integer upcast
// including subnormals), so the fp64 oracle can be fed the host-rounded values.
[[nodiscard]] float half_bits_to_f32(uint16_t h, GemmWeightFormat wf) noexcept {
    if (wf == GemmWeightFormat::Bf16) {
        return luisa::bit_cast<float>(static_cast<uint32_t>(h) << 16u);
    }
    const uint32_t mag = h & 0x7fffu;
    const bool neg = (h >> 15u) != 0u;
    float pos = 0.f;
    if (mag >= 0x7c00u) {
        pos = luisa::bit_cast<float>(0x7f800000u | ((mag & 0x03ffu) << 13u));
    } else if (mag >= 0x0400u) {
        pos = luisa::bit_cast<float>((mag << 13u) + 0x38000000u);// exact rebias
    } else {
        pos = static_cast<float>(mag) * 5.9604644775390625e-8f;// 2^-24
    }
    return neg ? -pos : pos;
}

[[nodiscard]] float packed_word_element(float word, GemmWeightFormat wf, bool high) noexcept {
    const uint32_t bits = luisa::bit_cast<uint32_t>(word);
    return half_bits_to_f32(static_cast<uint16_t>(high ? (bits >> 16u) : (bits & 0xffffu)), wf);
}

// Special / boundary bank. It is deliberately 47 entries (ODD) so the count ==
// bank.size() case also pins the odd tail, and it covers both formats' edges:
// signed zero, fp32 min/max normal and subnormals, inf, quiet and signaling NaNs,
// the fp16 finite limit and its overflow, the fp16 min normal / min subnormal /
// half-min-subnormal ties, exact subnormals, and bf16/fp16 rounding ties
// (mantissa exactly halfway -> round to even).
[[nodiscard]] luisa::vector<float> half_pack_value_bank() noexcept {
    luisa::vector<float> v;
    const auto add_bits = [&v](uint32_t b) { v.push_back(luisa::bit_cast<float>(b)); };
    add_bits(0x00000000u);// +0
    add_bits(0x80000000u);// -0
    v.push_back(1.0f);
    v.push_back(-1.0f);
    v.push_back(1.0009765625f);// 1 + 2^-10
    v.push_back(-1.0009765625f);
    add_bits(0x3f808000u);// bf16 tie, low 16 bits == 0x8000 -> round to even (down)
    add_bits(0x3f818000u);// bf16 tie -> round to even (up)
    add_bits(0x3f801000u);// fp16 tie at 1 + 2^-11 -> round to even (down)
    add_bits(0x3f803000u);// fp16 tie -> round to even (up)
    add_bits(0x3f801001u);// just above the half-ulp -> rounds up
    add_bits(0x3f800fffu);// just below the half-ulp -> rounds down
    add_bits(0x7f7fffffu);// fp32 max finite -> bf16 inf (RNE), fp16 inf
    add_bits(0xff7fffffu);
    add_bits(0x7f7f7fffu);// below the bf16 overflow tie -> stays finite (0x7F7F)
    add_bits(0x00800000u);// fp32 min normal
    add_bits(0x00000001u);// fp32 min subnormal
    add_bits(0x00400000u);// fp32 mid subnormal
    add_bits(0x7f800000u);// +inf
    add_bits(0xff800000u);// -inf
    add_bits(0x7fc00000u);// quiet NaN
    add_bits(0xffc00000u);// -quiet NaN
    add_bits(0x7f800001u);// signaling NaN (payload in the low mantissa bits)
    add_bits(0x7fbfffffu);// NaN whose top mantissa bit is 0 (quieting must set it)
    v.push_back(65504.0f);// fp16 max finite
    v.push_back(-65504.0f);
    v.push_back(65519.0f);// just below the fp16 overflow tie -> 65504
    v.push_back(65520.0f);// the tie itself -> inf (rounds away from zero)
    v.push_back(65536.0f);// 2^16 -> inf
    v.push_back(1e-8f);
    v.push_back(-1e-8f);
    v.push_back(6.103515625e-05f);// 2^-14, fp16 min normal
    v.push_back(6.09755516052246094e-05f);// just below 2^-14 -> max fp16 subnormal
    v.push_back(5.9604644775390625e-08f);// 2^-24, fp16 min subnormal
    v.push_back(2.98023223876953125e-08f);// 2^-25, the subnormal tie -> +-0
    v.push_back(3.5762786865234375e-07f);// 6 * 2^-24, exact subnormal
    v.push_back(4.470348358154296875e-08f);// 1.5 * 2^-25: the packer flushes it to 0
    v.push_back(1.1920928955078125e-07f);// 2 * 2^-24
    v.push_back(3.4028234663852886e38f);// fp32 max finite (same as 0x7F7FFFFF)
    v.push_back(-3.4028234663852886e38f);
    v.push_back(2.0f);
    v.push_back(-2.0f);
    v.push_back(0.1f);
    v.push_back(-0.1f);
    v.push_back(0.3333333432674408f);
    v.push_back(-7.0f);
    v.push_back(1.5f);
    return v;
}

// One (format, count) pack case: GPU pack vs the host oracle, word for word.
void check_pack_half_exact(Device &device, Stream &stream, GemmWeightFormat wf, Parity &parity) {
    auto shader = device.compile(make_pack_half_kernel(wf));
    const auto bank = half_pack_value_bank();
    luisa::vector<uint32_t> counts{1u, 2u, 3u, 7u, 255u, 256u, 257u, 1023u, 1024u, 1025u};
    counts.push_back(static_cast<uint32_t>(bank.size()) - 1u);// even: no tail
    counts.push_back(static_cast<uint32_t>(bank.size()));    // odd: covers the whole bank + tail
    // The destination is pre-filled with a NaN pattern: a word the kernel fails
    // to write stays NaN and cannot accidentally match the oracle.
    const float poison = luisa::bit_cast<float>(0x7fc00000u);
    for (const uint32_t count : counts) {
        Lcg rng{};
        rng.state = 0xC2B2AE35u ^ (count * 2654435761u) ^ (static_cast<uint32_t>(wf) << 16u);
        luisa::vector<float> src(count, 0.f);
        for (uint32_t i = 0u; i < count; i++) {
            if (i < bank.size()) {
                src[i] = bank[i];
            } else if ((i & 3u) == 0u) {
                src[i] = luisa::bit_cast<float>(rng.next() * 0x9E3779B9u);// any bit pattern
            } else {
                src[i] = rng.uniform(-2.f, 2.f);
            }
        }
        const auto expect_words = pack_half_words_host(src, wf);
        auto src_buf = device.create_buffer<float>(count);
        auto dst_buf = device.create_buffer<float>(expect_words.size());
        luisa::vector<float> got(expect_words.size(), poison);
        const luisa::vector<float> poison_words(expect_words.size(), poison);
        stream << src_buf.copy_from(luisa::span{src})
               << dst_buf.copy_from(luisa::span{poison_words})
               << shader(src_buf.view(), dst_buf.view(), count)
                      .dispatch(pack_half_dispatch_size(count))// TOTAL threads
               << dst_buf.copy_to(luisa::span{got})
               << synchronize();
        uint32_t bad = 0u;
        for (size_t j = 0u; j < expect_words.size(); j++) {
            const uint32_t g = luisa::bit_cast<uint32_t>(got[j]);
            const uint32_t e = luisa::bit_cast<uint32_t>(expect_words[j]);
            if (g != e) {
                if (bad < 4u) {
                    LUISA_WARNING("pack[{}] count={} word {}: got 0x{:08x} expect 0x{:08x}",
                                  gemm_weight_format_name(wf), count, j, g, e);
                }
                bad++;
            }
            // odd tail: the last word's high half must be exactly zero
            if (2u * j + 1u == count && (g >> 16u) != 0u) {
                if (bad < 4u) {
                    LUISA_WARNING("pack[{}] count={} word {}: odd tail high half 0x{:04x} != 0",
                                  gemm_weight_format_name(wf), count, j, g >> 16u);
                }
                bad++;
            }
        }
        LUISA_INFO("pack[{}] count={}: words={} mismatches={}",
                   gemm_weight_format_name(wf), count, expect_words.size(), bad);
        parity.cases++;
        parity.failures += bad;
    }
}

//----------------------------------------------------------------------------------------------------
// Phase 1a, part 2: real FP16 / BF16 GEMM coverage. B is packed ON THE DEVICE by
// make_pack_half_kernel from an fp32 weight matrix (that is the integration this
// test exists for), then read back by the matching half-format GEMM. The fp64
// oracle runs over the HOST-ROUNDED B values (host pack -> half -> fp32 upcast),
// so the gate measures the kernel pair, not the rounding itself.
//----------------------------------------------------------------------------------------------------

// Ragged in every dimension; K*N = 1, 40 (even), 35, 247, 9499, 1221 (odd) so
// both the even layout and the odd-tail layout of the packed B are exercised.
constexpr GemmCase kHalfCases[] = {
    {1u, 1u, 1u},
    {3u, 8u, 5u},
    {3u, 5u, 7u},
    {17u, 13u, 19u},
    {5u, 37u, 257u},
    {37u, 37u, 33u},
};

template<typename GemmShader, typename PackShader>
void check_gemm_half_case(Device &device, Stream &stream, const GemmShader &gemm,
                          const PackShader &pack, GemmWeightFormat wf, GemmEpilogue ep,
                          const GemmCase &c, Parity &parity) {
    Lcg rng{};
    rng.state = 0x27D4EB2Fu ^ (c.m << 24u) ^ (c.n << 12u) ^ (c.k << 2u) ^
                (static_cast<uint32_t>(ep) << 28u) ^ (static_cast<uint32_t>(wf) << 20u);

    const size_t a_count = static_cast<size_t>(c.m) * c.k;
    const size_t kn = static_cast<size_t>(c.k) * c.n;
    const size_t c_count = static_cast<size_t>(c.m) * c.n;

    luisa::vector<float> a(a_count);
    for (auto &f : a) {
        f = rng.uniform(-1.f, 1.f);
    }
    luisa::vector<float> b(kn);
    for (auto &f : b) {
        f = rng.uniform(-1.f, 1.f);
    }
    // Host-rounded B: the exact fp32 values the device GEMM will decode from the
    // packed words (so the oracle and the kernel see the same operands).
    const auto b_words = pack_half_words_host(b, wf);
    luisa::vector<float> b_rounded(kn);
    for (size_t i = 0u; i < kn; i++) {
        b_rounded[i] = packed_word_element(b_words[i / 2u], wf, (i & 1u) != 0u);
    }
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

    const auto ref = gemm_cpu_reference(a, b_rounded, aux, ep, c.m, c.n, c.k);

    auto a_buf = device.create_buffer<float>(a_count);
    auto b_f32 = device.create_buffer<float>(kn);// fp32 source of the device pack
    auto b_buf = device.create_buffer<float>(b_words.size());
    auto c_buf = device.create_buffer<float>(c_count);
    auto aux_buf = device.create_buffer<float>(aux_count);

    // NaN-poisoned packed B: a word the packer fails to write stays NaN and the
    // GEMM comparison below catches it (no silent "host words already there").
    const luisa::vector<float> poison_words(b_words.size(), luisa::bit_cast<float>(0x7fc00000u));
    luisa::vector<float> got(c_count, 0.f);
    stream << a_buf.copy_from(luisa::span{a})
           << b_f32.copy_from(luisa::span{b})
           << b_buf.copy_from(luisa::span{poison_words})
           << aux_buf.copy_from(luisa::span{aux})
           << pack(b_f32.view(), b_buf.view(), static_cast<uint32_t>(kn))
                  .dispatch(pack_half_dispatch_size(static_cast<uint32_t>(kn)))
           << gemm(a_buf.view(), b_buf.view(), c_buf.view(), aux_buf.view(), c.m, c.n, c.k)
                  .dispatch(gemm_dispatch_size(c.m, c.n))// dispatch = TOTAL threads
           << c_buf.copy_to(luisa::span{got})
           << synchronize();

    const auto tag = luisa::format("{}/{}", gemm_weight_format_name(wf), gemm_epilogue_name(ep));
    compare_outputs(tag.c_str(), {c.m, c.n, c.k}, ep, got, ref, parity);
}

// Whole format: compile the pack kernel ONCE (it is format-only) and the GEMM
// kernel once per epilogue, then sweep the ragged cases (keeps the JIT count,
// and therefore the runtime, close to the fp32/INT8 blocks above).
void check_gemm_half_format(Device &device, Stream &stream, GemmWeightFormat wf, Parity &parity) {
    auto pack = device.compile(make_pack_half_kernel(wf));
    for (const auto ep : {GemmEpilogue::None, GemmEpilogue::Bias, GemmEpilogue::BiasGeluErf}) {
        auto gemm = device.compile(make_gemm_kernel(ep, wf));
        for (const auto &c : kHalfCases) {
            check_gemm_half_case(device, stream, gemm, pack, wf, ep, c, parity);
        }
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
    // INT8 weight storage (phase 2): packed-B kernels, same fp32 gate.
    check_gemm_weight_format_names(parity);// host-only: format name reporting
    check_int8_pack_layout(parity);        // host-only: packed-B contract layout
    check_gemm_int8_epilogue(device, stream, GemmEpilogue::None, parity);
    check_gemm_int8_epilogue(device, stream, GemmEpilogue::Bias, parity);
    check_gemm_int8_epilogue(device, stream, GemmEpilogue::BiasGeluErf, parity);
    // Phase 1a: fp32 -> packed-half packer (bit-exact host oracle) and the real
    // FP16 / BF16 GEMM path over a DEVICE-packed B.
    check_pack_half_exact(device, stream, GemmWeightFormat::Bf16, parity);
    check_pack_half_exact(device, stream, GemmWeightFormat::Fp16, parity);
    check_gemm_half_format(device, stream, GemmWeightFormat::Bf16, parity);
    check_gemm_half_format(device, stream, GemmWeightFormat::Fp16, parity);

    LUISA_INFO("gemm/layernorm gate: worst max-abs-diff {:.3e} at gemm[{}] "
               "(M={}, N={}, K={}) over {} cases",
               parity.worst_abs_diff, gemm_epilogue_name(parity.worst_ep), parity.worst.m,
               parity.worst.n, parity.worst.k, parity.cases);
    expect(parity.failures == 0u)
        << luisa::format("gemm kernels produced {} mismatched outputs over {} cases",
                         parity.failures, parity.cases);
    return 0;
}

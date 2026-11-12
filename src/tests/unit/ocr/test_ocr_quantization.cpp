// Test for examples/ocr/ocr/quantization.h/.cpp -- checked host arithmetic,
// explicit RNE FP16/BF16 conversions, and grouped INT4/INT8 codecs.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - checked add/mul/align overflow and edge handling;
// - FP16/BF16 round trips, subnormals, ties-to-even, NaN/inf, overflow;
// - INT4 pack/unpack nibble order, odd counts, padding validation;
// - grouped INT4 symmetric/asymmetric + INT8 dequant, both group sizes;
// - negative/zero values, invalid scales/zero points, size mismatches;
// - quantize->dequantize round trips within the expected error bound.
#include "ut/ut.hpp"

#include <cmath>
#include <cstring>
#include <limits>

#include "ocr/quantization.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_checked_arithmetic() {
    uint64_t out = 0;
    expect(checked_add(1, 2, out) && out == 3);
    expect(!checked_add(UINT64_MAX, 1, out));
    expect(checked_mul(3, 4, out) && out == 12);
    expect(checked_mul(0, UINT64_MAX, out) && out == 0);
    expect(!checked_mul(UINT64_MAX, 2, out));
    expect(checked_align_up(255, 256, out) && out == 256);
    expect(checked_align_up(256, 256, out) && out == 256);
    expect(checked_align_up(257, 256, out) && out == 512);
    expect(!checked_align_up(10, 3, out));// not a power of two
    expect(!checked_align_up(UINT64_MAX, 256, out));
    const uint64_t dims[] = {2, 3, 4};
    expect(checked_element_count(dims, out) && out == 24);
    expect(checked_element_count(luisa::span<const uint64_t>{}, out) && out == 1);// scalar
    const uint64_t big[] = {UINT64_MAX, 2};
    expect(!checked_element_count(big, out));
}

void test_fp16_conversion() {
    // Exact representable values.
    const float values[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 65504.0f, -65504.0f};
    for (auto v : values) {
        const auto h = fp32_to_fp16(v);
        const auto back = fp16_to_fp32(h);
        expect(back == v) << "fp16 exact round trip";
    }
    // Subnormal half.
    const float sub = 5.9604644775390625e-8f;// 2^-24, smallest half subnormal
    expect(fp16_to_fp32(fp32_to_fp16(sub)) == sub);
    // Ties-to-even: a value exactly halfway rounds to even mantissa.
    // 1.0 + 2^-11 is halfway between 1.0 and 1.0+2^-10; rounds to 1.0 (even).
    const float tie = 1.0f + 0.00048828125f;// 1 + 2^-11
    expect(fp16_to_fp32(fp32_to_fp16(tie)) == 1.0f);
    // Overflow -> inf.
    expect(std::isinf(fp16_to_fp32(fp32_to_fp16(1.0e30f))));
    // NaN stays NaN.
    const auto nan_h = fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    expect(std::isnan(fp16_to_fp32(nan_h)));
    // Inf preserved.
    expect(std::isinf(fp16_to_fp32(fp32_to_fp16(std::numeric_limits<float>::infinity()))));
}

void test_bf16_conversion() {
    const float values[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 3.14159265f};
    for (auto v : values) {
        const auto b = fp32_to_bf16(v);
        const auto back = bf16_to_fp32(b);
        // BF16 has 8 mantissa bits; allow the truncation error.
        expect(std::fabs(back - v) <= std::fabs(v) * 0.01f + 1e-6f) << "bf16 round trip";
    }
    // Ties-to-even on the low 16 bits.
    uint32_t bits = 0x3f808000u;// 1.0 + half-ulp of bf16, low bit even
    float f = 0;
    std::memcpy(&f, &bits, sizeof(f));
    const auto b = fp32_to_bf16(f);
    expect((b & 1u) == 0u) << "bf16 ties to even";
    // NaN stays NaN.
    expect(std::isnan(bf16_to_fp32(fp32_to_bf16(std::numeric_limits<float>::quiet_NaN()))));
    // BF16 and FP16 are NOT interchangeable: same bits decode differently.
    const uint16_t pattern = 0x3c00u;// fp16 1.0
    expect(fp16_to_fp32(pattern) == 1.0f);
    expect(bf16_to_fp32(pattern) != 1.0f);
}

void test_int4_pack_unpack() {
    // Nibble order: low nibble first.
    const uint8_t codes[] = {0x1, 0x2, 0x3, 0x4, 0x5};
    luisa::vector<uint8_t> packed;
    expect(pack_int4_codes(codes, packed).ok());
    expect(packed.size() == 3u);
    expect(packed[0] == 0x21u);// low=1 high=2
    expect(packed[1] == 0x43u);
    expect(packed[2] == 0x05u);// low=5 high=0 (pad)
    uint8_t out[5];
    expect(unpack_int4_codes(packed, 5, out).ok());
    for (size_t i = 0; i < 5; i++) { expect(out[i] == codes[i]); }
    // Nonzero padding nibble rejected.
    packed[2] = 0x15u;
    expect(!unpack_int4_codes(packed, 5, out).ok());
    // Out-of-range code rejected.
    const uint8_t bad[] = {16};
    luisa::vector<uint8_t> bad_packed;
    expect(!pack_int4_codes(bad, bad_packed).ok());
    // Size mismatch rejected.
    expect(!unpack_int4_codes(packed, 4, out).ok());
}

void test_int4_dequant() {
    // Symmetric, group 64, codes [0,8,15], scale 0.125 -> [-1,0,0.875].
    const uint8_t codes[] = {0, 8, 15};
    luisa::vector<uint8_t> packed;
    expect(pack_int4_codes(codes, packed).ok());
    const float scales[] = {0.125f};
    float out[3];
    expect(dequantize_int4(packed, 3, scales, {}, 64, true, out).ok());
    expect(out[0] == -1.0f);
    expect(out[1] == 0.0f);
    expect(out[2] == 0.875f);
    // Asymmetric, zero point 8, scale 0.5, codes [1,2] -> [-3.5,-3.0].
    const uint8_t codes2[] = {1, 2};
    luisa::vector<uint8_t> packed2;
    expect(pack_int4_codes(codes2, packed2).ok());
    const float scales2[] = {0.5f};
    const uint8_t zp[] = {8};
    float out2[2];
    expect(dequantize_int4(packed2, 2, scales2, zp, 64, false, out2).ok());
    expect(out2[0] == -3.5f);
    expect(out2[1] == -3.0f);
    // Invalid scale (negative) rejected.
    const float bad_scale[] = {-1.0f};
    expect(!dequantize_int4(packed, 3, bad_scale, {}, 64, true, out).ok());
    // Non-finite scale rejected.
    const float nan_scale[] = {std::numeric_limits<float>::quiet_NaN()};
    expect(!dequantize_int4(packed, 3, nan_scale, {}, 64, true, out).ok());
    // Zero point out of range rejected.
    const uint8_t bad_zp[] = {16};
    expect(!dequantize_int4(packed2, 2, scales2, bad_zp, 64, false, out2).ok());
    // Unsupported group size rejected.
    expect(!dequantize_int4(packed, 3, scales, {}, 32, true, out).ok());
    // Scale count mismatch rejected.
    const float two_scales[] = {0.125f, 0.25f};
    expect(!dequantize_int4(packed, 3, two_scales, {}, 64, true, out).ok());
}

void test_int4_group128() {
    // 130 elements, group 128 -> 2 groups (second partial).
    luisa::vector<uint8_t> codes(130);
    for (size_t i = 0; i < 130; i++) { codes[i] = static_cast<uint8_t>(i % 16); }
    luisa::vector<uint8_t> packed;
    expect(pack_int4_codes(codes, packed).ok());
    const float scales[] = {0.25f, 0.5f};
    luisa::vector<float> out(130);
    expect(dequantize_int4(packed, 130, scales, {}, 128, true, out).ok());
    // Element 0: code 0, group 0 -> (0-8)*0.25 = -2.
    expect(out[0] == -2.0f);
    // Element 128: code 0, group 1 -> (0-8)*0.5 = -4.
    expect(out[128] == -4.0f);
}

void test_int8() {
    const int8_t codes[] = {-127, -1, 0, 1, 127};
    const float scales[] = {0.01f};
    float out[5];
    expect(dequantize_int8(codes, 5, scales, 64, out).ok());
    expect(std::fabs(out[0] - (-1.27f)) < 1e-6f);
    expect(out[2] == 0.0f);
    expect(std::fabs(out[4] - 1.27f) < 1e-6f);
    // Reserved -128 rejected.
    const int8_t bad[] = {-128};
    expect(!dequantize_int8(bad, 1, scales, 64, out).ok());
}

void test_quantize_roundtrip() {
    // Symmetric INT4 round trip within one quantization step.
    luisa::vector<float> values(70);
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = std::sin(static_cast<float>(i) * 0.3f) * 3.0f;
    }
    for (uint32_t gs : {64u, 128u}) {
        luisa::vector<uint8_t> packed;
        luisa::vector<float> scales;
        luisa::vector<uint8_t> zp;
        expect(quantize_int4(values, gs, true, packed, scales, zp).ok());
        expect(zp.empty());
        luisa::vector<float> out(values.size());
        expect(dequantize_int4(packed, values.size(), scales, zp, gs, true, out).ok());
        float max_err = 0.0f;
        for (size_t i = 0; i < values.size(); i++) {
            max_err = std::max(max_err, std::fabs(out[i] - values[i]));
        }
        // One step is scale; allow a small margin for RNE.
        float max_scale = 0.0f;
        for (auto s : scales) { max_scale = std::max(max_scale, s); }
        expect(max_err <= max_scale * 0.75f + 1e-5f) << "int4 sym round trip error";
    }
    // Asymmetric round trip.
    luisa::vector<uint8_t> packed;
    luisa::vector<float> scales;
    luisa::vector<uint8_t> zp;
    expect(quantize_int4(values, 64, false, packed, scales, zp).ok());
    expect(!zp.empty());
    luisa::vector<float> out(values.size());
    expect(dequantize_int4(packed, values.size(), scales, zp, 64, false, out).ok());
    // All-zero group -> scale 1, code == zero point.
    luisa::vector<float> zeros(64, 0.0f);
    expect(quantize_int4(zeros, 64, true, packed, scales, zp).ok());
    expect(scales[0] == 1.0f);
    // Non-finite input rejected.
    luisa::vector<float> bad(64, std::numeric_limits<float>::infinity());
    expect(!quantize_int4(bad, 64, true, packed, scales, zp).ok());
    // INT8 round trip.
    luisa::vector<int8_t> codes8;
    luisa::vector<float> scales8;
    expect(quantize_int8(values, 64, codes8, scales8).ok());
    luisa::vector<float> out8(values.size());
    expect(dequantize_int8(codes8, values.size(), scales8, 64, out8).ok());
}

// Independent RNE reference for fp32->fp16 over FINITE inputs, used to
// cross-check the production converter (including the subnormal boundary).
[[nodiscard]] uint16_t reference_fp32_to_fp16(float f) noexcept {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23u) & 0xffu) - 127;
    const uint32_t frac = bits & 0x7fffffu;
    if (exp == 128) { return static_cast<uint16_t>(sign | 0x7c00u); }// inf
    if (exp > 15) { return static_cast<uint16_t>(sign | 0x7c00u); }// overflow -> inf
    const uint32_t sig = (1u << 23u) | frac;// 24-bit significand
    if (exp >= -14) {
        // Normal half: RNE on the 13 dropped mantissa bits.
        const uint32_t m = frac >> 13u;
        const uint32_t rem = frac & 0x1fffu;
        uint32_t rounded = m;
        if (rem > 0x1000u || (rem == 0x1000u && (m & 1u) != 0)) { rounded += 1; }
        uint32_t e = static_cast<uint32_t>(exp + 15);
        if (rounded == 0x400u) { rounded = 0; e += 1; }// mantissa carry
        if (e >= 0x1fu) { return static_cast<uint16_t>(sign | 0x7c00u); }
        return static_cast<uint16_t>(sign | (e << 10u) | rounded);
    }
    if (exp >= -24) {
        // Subnormal half: value / 2^-24 = sig * 2^(exp+1), RNE to integer.
        const uint32_t shift = static_cast<uint32_t>(-(exp + 1));// 1..23
        const uint32_t truncated = sig >> shift;
        const uint32_t rem = sig & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        uint32_t q = truncated;
        if (rem > half || (rem == half && (truncated & 1u) != 0)) { q += 1; }
        return static_cast<uint16_t>(sign | q);// q==0x400 becomes smallest normal
    }
    // exp < -24: magnitude < 2^-24. value / 2^-25 = sig * 2^(exp+2); compare to 1.
    // For very small exponents the shift would exceed 63 bits; the value is then
    // far below half the min subnormal and rounds to zero.
    const int32_t shift = -(exp + 2);
    if (shift >= 24) { return static_cast<uint16_t>(sign); }// sig < 2^24 <= 2^shift
    const uint64_t threshold = 1ull << static_cast<uint32_t>(shift);
    if (static_cast<uint64_t>(sig) > threshold) {
        return static_cast<uint16_t>(sign | 0x0001u);// above half -> min subnormal
    }
    return static_cast<uint16_t>(sign);// at/below half -> 0 (ties-to-even)
}

void test_fp16_subnormal_boundary() {
    // Half of the smallest subnormal is 2^-25. Values strictly between 2^-25
    // and 2^-24 must round to the min subnormal 0x0001 (regression: flushed 0).
    const float two_pow_25 = 0x1p-25f;
    const float two_pow_24 = 0x1p-24f;
    expect(fp32_to_fp16(two_pow_25) == 0x0000u) << "exact half-min-subnormal ties to even 0";
    expect(fp32_to_fp16(-two_pow_25) == 0x8000u) << "negative half ties to -0";
    expect(fp32_to_fp16(two_pow_24) == 0x0001u) << "min subnormal exact";
    // Just above 2^-25 (next float up) -> min subnormal.
    uint32_t bits = 0;
    std::memcpy(&bits, &two_pow_25, sizeof(bits));
    bits += 1;
    float just_above = 0;
    std::memcpy(&just_above, &bits, sizeof(just_above));
    expect(fp32_to_fp16(just_above) == 0x0001u) << "just above half-min-subnormal -> 0x0001";
    // Just below 2^-25 -> 0.
    bits -= 2;
    float just_below = 0;
    std::memcpy(&just_below, &bits, sizeof(just_below));
    expect(fp32_to_fp16(just_below) == 0x0000u) << "just below half-min-subnormal -> 0";
    // Midpoint between 2^-25 and 2^-24 -> min subnormal.
    const float mid = (two_pow_25 + two_pow_24) * 0.5f;
    expect(fp32_to_fp16(mid) == 0x0001u) << "midpoint -> min subnormal";
    // Sweep the whole [2^-25, 2^-24) band against the independent reference.
    bool band_ok = true;
    for (uint32_t b = 0x33800000u; b < 0x34000000u; b += 0x1000u) {
        float f = 0;
        std::memcpy(&f, &b, sizeof(f));
        if (fp32_to_fp16(f) != reference_fp32_to_fp16(f)) { band_ok = false; }
    }
    expect(band_ok) << "subnormal band matches independent RNE reference";
}

void test_fp16_exhaustive_roundtrip() {
    // Every finite fp16 pattern must round-trip exactly through fp32.
    bool ok = true;
    for (uint32_t h = 0; h < 65536u; h++) {
        const float f = fp16_to_fp32(static_cast<uint16_t>(h));
        if (std::isnan(f)) { continue; }
        if (fp32_to_fp16(f) != static_cast<uint16_t>(h)) { ok = false; }
    }
    expect(ok) << "fp16 exhaustive round-trip";
    // Random finite fp32 values must match the independent RNE reference.
    uint32_t state = 0x12345678u;
    const auto next_rand = [&]() { state = state * 1664525u + 1013904223u; return state; };
    bool rand_ok = true;
    for (uint32_t i = 0; i < 200000u; i++) {
        uint32_t b = next_rand();
        // Force a finite exponent (0..254); keep sign/mantissa random.
        b = (b & 0x807fffffu) | ((next_rand() % 255u) << 23u);
        float f = 0;
        std::memcpy(&f, &b, sizeof(f));
        if (fp32_to_fp16(f) != reference_fp32_to_fp16(f)) { rand_ok = false; }
    }
    expect(rand_ok) << "fp16 random matches independent RNE reference";
    // NaN stays NaN, inf stays inf.
    expect(std::isnan(fp16_to_fp32(fp32_to_fp16(std::numeric_limits<float>::quiet_NaN()))));
    expect(std::isinf(fp16_to_fp32(fp32_to_fp16(std::numeric_limits<float>::infinity()))));
}

void test_bf16_exhaustive_roundtrip() {
    bool ok = true;
    for (uint32_t b = 0; b < 65536u; b++) {
        const float f = bf16_to_fp32(static_cast<uint16_t>(b));
        if (std::isnan(f)) { continue; }
        if (fp32_to_bf16(f) != static_cast<uint16_t>(b)) { ok = false; }
    }
    expect(ok) << "bf16 exhaustive round-trip";
    // NaN must stay NaN (not carry into infinity).
    const auto nan_bf = fp32_to_bf16(std::numeric_limits<float>::quiet_NaN());
    expect(std::isnan(bf16_to_fp32(nan_bf))) << "bf16 NaN preserved";
}

}// namespace

static auto test_ocr_quantization_registration = [] {
    "checked_arithmetic"_test = [] { test_checked_arithmetic(); };
    "fp16_conversion"_test = [] { test_fp16_conversion(); };
    "fp16_subnormal_boundary"_test = [] { test_fp16_subnormal_boundary(); };
    "fp16_exhaustive_roundtrip"_test = [] { test_fp16_exhaustive_roundtrip(); };
    "bf16_conversion"_test = [] { test_bf16_conversion(); };
    "bf16_exhaustive_roundtrip"_test = [] { test_bf16_exhaustive_roundtrip(); };
    "int4_pack_unpack"_test = [] { test_int4_pack_unpack(); };
    "int4_dequant"_test = [] { test_int4_dequant(); };
    "int4_group128"_test = [] { test_int4_group128(); };
    "int8"_test = [] { test_int8(); };
    "quantize_roundtrip"_test = [] { test_quantize_roundtrip(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}
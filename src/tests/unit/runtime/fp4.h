#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

// FP4 E2M1 format: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit
// Layout: S EE M
// No Inf/NaN encoding; all 16 bit patterns map to finite values.
// Max finite: E=3, M=1 -> (1 + 1/2) * 2^(3-1) = 1.5 * 4 = 6.0
struct FP4E2M1 {

    static constexpr int kExpBits = 2;
    static constexpr int kMantBits = 1;
    static constexpr int kBias = 1;
    static constexpr int kMaxExp = (1 << kExpBits) - 1;        // 3
    static constexpr uint8_t kMantMask = (1 << kMantBits) - 1; // 0x01
    static constexpr uint8_t kSignMask = 0x08;                 // bit 3 for 4-bit value
    static constexpr float kMaxFinite = 6.0f;

    // Convert FP32 to FP4 E2M1 with round-to-nearest-even
    static uint8_t from_float(float v) {
        if (v == 0.0f) {
            return 0;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        if (v > kMaxFinite) {
            // Clamp to max finite: S 11 1
            return static_cast<uint8_t>((sign ? kSignMask : 0) | (kMaxExp << kMantBits) | kMantMask);
        }

        // Decompose float
        int exp;
        float mant = std::frexp(v, &exp);// v = mant * 2^exp, mant in [0.5, 1.0)
        // Adjust to get mant in [1.0, 2.0)
        mant *= 2.0f;
        exp -= 1;

        int e = exp + kBias;
        if (e <= 0) {
            // Denormal or underflow to zero
            if (e < -kMantBits) {
                return static_cast<uint8_t>(sign ? kSignMask : 0);
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 1 bit
        float mant_scaled = mant * (1 << kMantBits);
        float mant_q = std::floor(mant_scaled);
        float frac = mant_scaled - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & kMantMask;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > kMantMask) {
                m = 0;
                e += 1;
            }
        }

        // Clamp overflow to max finite
        if (e > kMaxExp) {
            e = kMaxExp;
            m = kMantMask;
        }

        if (e == 0 && m == 0) {
            return static_cast<uint8_t>(sign ? kSignMask : 0);
        }

        return static_cast<uint8_t>((sign ? kSignMask : 0) | (e << kMantBits) | m);
    }

    // Convert FP4 E2M1 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & kSignMask) >> 3;
        uint8_t e = (bits >> kMantBits) & ((1 << kExpBits) - 1);
        uint8_t m = bits & kMantMask;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        float value;
        if (e == 0) {
            // Denormal: (m/2) * 2^(1-bias)
            value = static_cast<float>(m) / static_cast<float>(1 << kMantBits) * std::pow(2.0f, 1 - kBias);
        } else {
            // Normal: (1 + m/2) * 2^(e-bias)
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << kMantBits)) * std::pow(2.0f, e - kBias);
        }
        return sign ? -value : value;
    }
};

// CPU helpers for packing two 4-bit nibbles into one byte
inline uint8_t pack_fp4(uint8_t upper, uint8_t lower) noexcept {
    return (upper << 4) | (lower & 0x0f);
}

inline uint8_t unpack_fp4_upper(uint8_t packed) noexcept {
    return packed >> 4;
}

inline uint8_t unpack_fp4_lower(uint8_t packed) noexcept {
    return packed & 0x0f;
}

namespace luisa::compute {

Callable<uint(half)> fp4e2m1_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (v == half(0.0f)) {
            $return(0u);
        };

        UInt sign = 0u;
        $if (v < half(0.0f)) {
            sign = 1u;
            v = -v;
        };

        $if (v > half(FP4E2M1::kMaxFinite)) {
            $return(ite(sign == 1u, 0x0fu, 0x07u));
        };

        // Decompose float: v = mant * 2^exp, mant in [1.0, 2.0)
        Int exp = floor(log2(v)).cast<int>();
        Half mant = v / pow(half(2.0f), exp.cast<half>());
        $if (mant >= half(2.0f)) {
            mant = mant * half(0.5f);
            exp = exp + 1;
        };
        $if (mant < half(1.0f)) {
            mant = mant * half(2.0f);
            exp = exp - 1;
        };

        Int e = exp + FP4E2M1::kBias;
        $if (e <= 0) {
            $if (e < -FP4E2M1::kMantBits) {
                $return(ite(sign == 1u, 0x08u, 0u));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 1 bit
        Half mant_scaled = mant * half(2.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UInt m = mant_q.cast<uint>() & static_cast<uint>(FP4E2M1::kMantMask);

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & (m & 1u) != 0u)) {
            m = m + 1u;
            $if (m > static_cast<uint>(FP4E2M1::kMantMask)) {
                m = 0u;
                e = e + 1;
            };
        };

        // Clamp overflow to max finite
        $if (e > FP4E2M1::kMaxExp) {
            e = FP4E2M1::kMaxExp;
            m = static_cast<uint>(FP4E2M1::kMantMask);
        };

        $if (e == 0 & m == 0u) {
            $return(ite(sign == 1u, 0x08u, 0u));
        };

        UInt bits = (sign << 3u) | (e.cast<uint>() << FP4E2M1::kMantBits) | m;
        return bits;
    }};
    return _c;
}

Callable<half(uint)> fp4e2m1_to_float() {
    static Callable _c{[](UInt bits) noexcept {
        UInt sign = (bits >> 3u) & 1u;
        UInt e = (bits >> FP4E2M1::kMantBits) & static_cast<uint>((1 << FP4E2M1::kExpBits) - 1);
        UInt m = bits & static_cast<uint>(FP4E2M1::kMantMask);

        $if (e == 0u & m == 0u) {
            $return(ite(sign != 0u, -half(0.0f), half(0.0f)));
        };

        Half value;
        $if (e == 0u) {
            // Denormal: (m/2) * 2^(1-bias)
            value = (m.cast<half>() / half(2.0f)) * pow(half(2.0f), half(1.0f) - half(FP4E2M1::kBias));
        }
        $else {
            // Normal: (1 + m/2) * 2^(e-bias)
            value = (half(1.0f) + m.cast<half>() / half(2.0f)) * pow(half(2.0f), e.cast<half>() - half(FP4E2M1::kBias));
        };
        return ite(sign != 0u, -value, value);
    }};
    return _c;
}

// Pack two 4-bit nibbles (in lower 4 bits of each uint) into one byte
Callable<uint(uint, uint)> pack_fp4() {
    static Callable _c{[](UInt upper, UInt lower) noexcept {
        return (upper << 4u) | (lower & 0x0fu);
    }};
    return _c;
}

// Unpack nibble from packed byte at index: idx=0 -> upper, idx=1 -> lower
Callable<uint(uint, uint)> unpack_fp4() {
    static Callable _c{[](UInt packed, UInt idx) noexcept {
        return ite((idx & 1u) == 0u, packed >> 4u, packed & 0x0fu);
    }};
    return _c;
}

}// namespace luisa::compute

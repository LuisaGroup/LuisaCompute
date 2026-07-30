#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

// FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits
// Layout: S EEEE MMM
struct FP8E4M3 {

    static constexpr int kExpBits = 4;
    static constexpr int kMantBits = 3;
    static constexpr int kBias = 7;
    static constexpr int kMaxExp = (1 << kExpBits) - 1;       // 15
    static constexpr int kInfNanExp = kMaxExp;                // 15
    static constexpr uint8_t kMantMask = (1 << kMantBits) - 1;// 0x07
    static constexpr uint8_t kSignMask = 0x80;

    // Convert FP32 to FP8 E4M3 with round-to-nearest-even
    static uint8_t from_float(float v) {
        uint8_t bits{};
        if (std::isnan(v)) {
            bits = 0x7f;// canonical NaN
            return bits;
        }
        if (v == 0.0f) {
            bits = 0;
            return bits;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        // E4M3 max finite value: E=15, M=6 -> (1 + 6/8) * 2^(15-7) = 1.75 * 256 = 448
        constexpr float max_finite = 448.0f;
        if (v > max_finite) {
            // Clamp to max finite
            bits = sign ? 0xfe : 0x7e;// E=15, M=6
            return bits;
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
                bits = sign ? kSignMask : 0;
                return bits;
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 3 bits
        float mant_q = std::floor(mant * (1 << kMantBits));
        float frac = mant * (1 << kMantBits) - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & kMantMask;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > kMantMask) {
                m = 0;
                e += 1;
            }
        }

        // E4M3: E=15, M=7 is NaN, so max finite is E=15, M=6
        if (e >= kInfNanExp) {
            e = kInfNanExp;
            if (m >= kMantMask) {
                m = kMantMask - 1;// clamp to 6
            }
        }

        // Reconstruct
        if (e == 0 && m == 0) {
            bits = sign ? kSignMask : 0;
            return bits;
        }

        bits = (sign ? kSignMask : 0) | static_cast<uint8_t>(e << kMantBits) | m;
        return bits;
    }

    // Convert FP8 E4M3 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & kSignMask) >> 7;
        uint8_t e = (bits >> kMantBits) & ((1 << kExpBits) - 1);
        uint8_t m = bits & kMantMask;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        // E4M3 NaN: E=15, M=7
        if (e == kInfNanExp && m == kMantMask) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float value;
        if (e == 0) {
            // Denormal
            value = static_cast<float>(m) / static_cast<float>(1 << kMantBits) * std::pow(2.0f, 1 - kBias);
        } else {
            // Normal
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << kMantBits)) * std::pow(2.0f, e - kBias);
        }
        return sign ? -value : value;
    }
};

// FP8 E5M2 format: 1 sign bit, 5 exponent bits (bias=15), 2 mantissa bits
// Layout: S EEEEE MM
// E5M2 supports Inf/NaN: E=31, M=0 is Inf; E=31, M!=0 is NaN
struct FP8E5M2 {

    static constexpr int kExpBits = 5;
    static constexpr int kMantBits = 2;
    static constexpr int kBias = 15;
    static constexpr int kMaxExp = (1 << kExpBits) - 1;       // 31
    static constexpr int kInfNanExp = kMaxExp;                // 31
    static constexpr uint8_t kMantMask = (1 << kMantBits) - 1;// 0x03
    static constexpr uint8_t kSignMask = 0x80;

    // Convert FP32 to FP8 E5M2 with round-to-nearest-even
    static uint8_t from_float(float v) {
        uint8_t bits{};
        if (std::isnan(v)) {
            bits = 0x7f;// canonical NaN
            return bits;
        }
        if (std::isinf(v)) {
            bits = (v < 0.0f) ? 0xfc : 0x7c;// E=31, M=0
            return bits;
        }
        if (v == 0.0f) {
            bits = 0;
            return bits;
        }
        uint32_t sign = 0;
        if (v < 0.0f) {
            sign = 1;
            v = -v;
        }

        // E5M2 max finite value: E=30, M=3 -> (1 + 3/4) * 2^(30-15) = 1.75 * 32768 = 57344
        constexpr float max_finite = 57344.0f;
        if (v > max_finite) {
            bits = sign ? 0xfc : 0x7c;// E=31, M=0 -> Inf
            return bits;
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
                bits = sign ? kSignMask : 0;
                return bits;
            }
            // Denormal: shift mantissa right
            int shift = 1 - e;
            mant = mant / static_cast<float>(1 << shift);
            e = 0;
        }

        // Round mantissa to 2 bits
        float mant_q = std::floor(mant * (1 << kMantBits));
        float frac = mant * (1 << kMantBits) - mant_q;
        uint8_t m = static_cast<uint8_t>(mant_q) & kMantMask;

        // Round-to-nearest-even
        if (frac > 0.5f || (frac == 0.5f && (m & 1))) {
            m += 1;
            if (m > kMantMask) {
                m = 0;
                e += 1;
            }
        }

        // E5M2: E=31, M=0 is Inf, any M!=0 is NaN
        if (e >= kInfNanExp) {
            e = kInfNanExp;
            m = 0;// Inf
        }

        // Reconstruct
        if (e == 0 && m == 0) {
            bits = sign ? kSignMask : 0;
            return bits;
        }

        bits = (sign ? kSignMask : 0) | static_cast<uint8_t>(e << kMantBits) | m;
        return bits;
    }

    // Convert FP8 E5M2 to FP32
    static float to_float(uint8_t bits) {
        uint8_t sign = (bits & kSignMask) >> 7;
        uint8_t e = (bits >> kMantBits) & ((1 << kExpBits) - 1);
        uint8_t m = bits & kMantMask;

        if (e == 0 && m == 0) {
            return sign ? -0.0f : 0.0f;
        }

        // E5M2 Inf: E=31, M=0
        if (e == kInfNanExp && m == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        }
        // E5M2 NaN: E=31, M!=0
        if (e == kInfNanExp && m != 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float value;
        if (e == 0) {
            // Denormal
            value = static_cast<float>(m) / static_cast<float>(1 << kMantBits) * std::pow(2.0f, 1 - kBias);
        } else {
            // Normal
            value = (1.0f + static_cast<float>(m) / static_cast<float>(1 << kMantBits)) * std::pow(2.0f, e - kBias);
        }
        return sign ? -value : value;
    }
};

namespace luisa::compute {

Callable<uint(half)> fp8e4m3_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (luisa::compute::dsl::isnan(v)) {
            $return(0x7fu);
        };
        $if (v == half(0.0f)) {
            $return(0u);
        };

        UInt sign = 0u;
        $if (v < half(0.0f)) {
            sign = 1u;
            v = -v;
        };

        $if (v > half(448.0f)) {
            $return(ite(sign == 1u, 0xfeu, 0x7eu));
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

        Int e = exp + FP8E4M3::kBias;
        $if (e <= 0) {
            $if (e < -FP8E4M3::kMantBits) {
                $return(ite(sign == 1u, 0x80u, 0u));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 3 bits
        Half mant_scaled = mant * half(8.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UInt m = mant_q.cast<uint>() & static_cast<uint>(FP8E4M3::kMantMask);

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & (m & 1u) != 0u)) {
            m = m + 1u;
            $if (m > static_cast<uint>(FP8E4M3::kMantMask)) {
                m = 0u;
                e = e + 1;
            };
        };

        // E4M3: E=15, M=7 is NaN, so max finite is E=15, M=6
        $if (e >= FP8E4M3::kInfNanExp) {
            e = FP8E4M3::kInfNanExp;
            $if (m >= static_cast<uint>(FP8E4M3::kMantMask)) {
                m = static_cast<uint>(FP8E4M3::kMantMask - 1);
            };
        };

        $if (e == 0 & m == 0u) {
            $return(ite(sign == 1u, 0x80u, 0u));
        };

        UInt bits = (sign << 7u) | (e.cast<uint>() << FP8E4M3::kMantBits) | m;
        return bits;
    }};
    return _c;
}

Callable<half(uint)> fp8e4m3_to_float() {
    static Callable _c{[](UInt bits) noexcept {
        UInt sign = (bits >> 7u) & 1u;
        UInt e = (bits >> FP8E4M3::kMantBits) & static_cast<uint>((1 << FP8E4M3::kExpBits) - 1);
        UInt m = bits & static_cast<uint>(FP8E4M3::kMantMask);

        $if (e == 0u & m == 0u) {
            $return(ite(sign != 0u, -half(0.0f), half(0.0f)));
        };

        // E4M3 NaN: E=15, M=7
        $if (e == static_cast<uint>(FP8E4M3::kInfNanExp) & m == static_cast<uint>(FP8E4M3::kMantMask)) {
            UInt nan_bits = 0x7fc00000u;
            $return(cast<half>(as<float>(nan_bits)));
        };

        Half value;
        $if (e == 0u) {
            // Denormal
            value = (m.cast<half>() / half(8.0f)) * pow(half(2.0f), half(1.0f) - half(FP8E4M3::kBias));
        }
        $else {
            // Normal
            value = (half(1.0f) + m.cast<half>() / half(8.0f)) * pow(half(2.0f), e.cast<half>() - half(FP8E4M3::kBias));
        };
        return ite(sign != 0u, -value, value);
    }};
    return _c;
}

Callable<uint(half)> fp8e5m2_from_float() {
    static Callable _c{[](Half v) noexcept {
        $if (luisa::compute::dsl::isnan(v)) {
            $return(0x7fu);
        };
        $if (luisa::compute::dsl::isinf(v)) {
            $return(ite(v < half(0.0f), 0xfcu, 0x7cu));
        };
        $if (v == half(0.0f)) {
            $return(0u);
        };

        UInt sign = 0u;
        $if (v < half(0.0f)) {
            sign = 1u;
            v = -v;
        };

        $if (v > half(57344.0f)) {
            $return(ite(sign == 1u, 0xfcu, 0x7cu));
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

        Int e = exp + FP8E5M2::kBias;
        $if (e <= 0) {
            $if (e < -FP8E5M2::kMantBits) {
                $return(ite(sign == 1u, 0x80u, 0u));
            };
            Int shift = 1 - e;
            mant = mant / pow(half(2.0f), shift.cast<half>());
            e = 0;
        };

        // Round mantissa to 2 bits
        Half mant_scaled = mant * half(4.0f);
        Half mant_q = floor(mant_scaled);
        Half frac = mant_scaled - mant_q;
        UInt m = mant_q.cast<uint>() & static_cast<uint>(FP8E5M2::kMantMask);

        // Round-to-nearest-even
        $if (frac > half(0.5f) | (frac == half(0.5f) & (m & 1u) != 0u)) {
            m = m + 1u;
            $if (m > static_cast<uint>(FP8E5M2::kMantMask)) {
                m = 0u;
                e = e + 1;
            };
        };

        // E5M2: E=31, M=0 is Inf, any M!=0 is NaN
        $if (e >= FP8E5M2::kInfNanExp) {
            e = FP8E5M2::kInfNanExp;
            m = 0u;
        };

        $if (e == 0 & m == 0u) {
            $return(ite(sign == 1u, 0x80u, 0u));
        };

        UInt bits = (sign << 7u) | (e.cast<uint>() << FP8E5M2::kMantBits) | m;
        return bits;
    }};
    return _c;
}

Callable<half(uint)> fp8e5m2_to_float() {
    static Callable _c{[](UInt bits) noexcept {
        UInt sign = (bits >> 7u) & 1u;
        UInt e = (bits >> FP8E5M2::kMantBits) & static_cast<uint>((1 << FP8E5M2::kExpBits) - 1);
        UInt m = bits & static_cast<uint>(FP8E5M2::kMantMask);

        $if (e == 0u & m == 0u) {
            $return(ite(sign != 0u, -half(0.0f), half(0.0f)));
        };

        // E5M2 Inf: E=31, M=0
        $if (e == static_cast<uint>(FP8E5M2::kInfNanExp) & m == 0u) {
            UInt inf_bits = ite(sign != 0u, 0xff800000u, 0x7f800000u);
            $return(cast<half>(as<float>(inf_bits)));
        };

        // E5M2 NaN: E=31, M!=0
        $if (e == static_cast<uint>(FP8E5M2::kInfNanExp) & m != 0u) {
            UInt nan_bits = 0x7fc00000u;
            $return(cast<half>(as<float>(nan_bits)));
        };

        Half value;
        $if (e == 0u) {
            // Denormal
            value = (m.cast<half>() / half(4.0f)) * pow(half(2.0f), half(1.0f) - half(FP8E5M2::kBias));
        }
        $else {
            // Normal
            value = (half(1.0f) + m.cast<half>() / half(4.0f)) * pow(half(2.0f), e.cast<half>() - half(FP8E5M2::kBias));
        };
        return ite(sign != 0u, -value, value);
    }};
    return _c;
}

}// namespace luisa::compute

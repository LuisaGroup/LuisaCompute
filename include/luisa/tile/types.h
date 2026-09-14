#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <luisa/core/basic_traits.h>

namespace luisa::compute::tile {

using luisa::half;

// Host storage/conversion type. Tile arithmetic remains staged in TileIR.
// Float32 -> BF16 rounds to nearest, ties to even; overflow becomes infinity.
// NaNs remain NaNs and are quieted; signed zeros and subnormals are preserved.
class BFloat16 final {
private:
    uint16_t _bits{};

public:
    constexpr BFloat16() noexcept = default;
    explicit constexpr BFloat16(float value) noexcept {
        auto bits = std::bit_cast<uint32_t>(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u) {
            _bits = static_cast<uint16_t>((bits >> 16u) | 0x0040u);
        } else {
            _bits = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        }
    }
    [[nodiscard]] static constexpr BFloat16 from_bits(uint16_t bits) noexcept {
        BFloat16 value;
        value._bits = bits;
        return value;
    }
    [[nodiscard]] constexpr uint16_t bits() const noexcept { return _bits; }
    [[nodiscard]] explicit constexpr operator float() const noexcept {
        return std::bit_cast<float>(static_cast<uint32_t>(_bits) << 16u);
    }
    [[nodiscard]] friend constexpr BFloat16 operator-(BFloat16 value) noexcept {
        return from_bits(value._bits ^ 0x8000u);
    }
};
using bfloat16 = BFloat16;
using bf16 = BFloat16;

namespace detail {

// These are explicit interchange formats, not an unspecified "fp8" type.
// E4M3FN: bias 7, finite range +/-448, signed zero, no infinities, 0x7f/ff NaNs.
// E5M2: bias 15, finite range +/-57344, signed zero, IEEE-style inf/NaN.
// FNUZ and scaled/block formats need distinct types, never aliases of these.
template<bool E4M3FN>
class Float8Storage final {
private:
    uint8_t _bits{};

public:
    constexpr Float8Storage() noexcept = default;
    [[nodiscard]] static constexpr Float8Storage from_bits(uint8_t bits) noexcept {
        Float8Storage value;
        value._bits = bits;
        return value;
    }
    [[nodiscard]] constexpr uint8_t bits() const noexcept { return _bits; }
    [[nodiscard]] explicit constexpr operator float() const noexcept {
        constexpr auto mantissa_bits = E4M3FN ? 3u : 2u;
        constexpr auto bias = E4M3FN ? 7 : 15;
        auto sign = static_cast<uint32_t>(_bits & 0x80u) << 24u;
        auto magnitude = static_cast<uint32_t>(_bits & 0x7fu);
        auto exponent = magnitude >> mantissa_bits;
        auto mantissa = magnitude & ((1u << mantissa_bits) - 1u);
        if constexpr (E4M3FN) {
            if (magnitude == 0x7fu) { return std::bit_cast<float>(sign | 0x7fc00000u); }
        } else {
            if (exponent == 31u) {
                return std::bit_cast<float>(sign | 0x7f800000u | (mantissa << 21u));
            }
        }
        if (exponent == 0u) {
            // All FP8 subnormals are exact, normal float32 values.
            auto value = static_cast<float>(mantissa) * (E4M3FN ? 0x1p-9f : 0x1p-16f);
            return std::bit_cast<float>(std::bit_cast<uint32_t>(value) | sign);
        }
        return std::bit_cast<float>(sign | ((exponent + 127u - bias) << 23u) |
                                    (mantissa << (23u - mantissa_bits)));
    }
    [[nodiscard]] friend constexpr Float8Storage operator-(Float8Storage value) noexcept {
        return from_bits(value._bits ^ 0x80u);
    }
    // No implicit host float -> FP8 constructor: rounding and overflow /
    // saturation are conversion policy, not encoded in the storage format.
};

}// namespace detail

using float8_e4m3fn = detail::Float8Storage<true>;
using float8_e5m2 = detail::Float8Storage<false>;

static_assert(sizeof(bfloat16) == 2u && alignof(bfloat16) == 2u && std::is_trivially_copyable_v<bfloat16>);
static_assert(sizeof(float8_e4m3fn) == 1u && alignof(float8_e4m3fn) == 1u && std::is_trivially_copyable_v<float8_e4m3fn>);
static_assert(sizeof(float8_e5m2) == 1u && alignof(float8_e5m2) == 1u && std::is_trivially_copyable_v<float8_e5m2>);

}// namespace luisa::compute::tile

namespace std {

template<>
class numeric_limits<luisa::compute::tile::bfloat16> : public numeric_limits<float> {
    using T = luisa::compute::tile::bfloat16;
public:
    static constexpr int digits = 8, digits10 = 2, max_digits10 = 4;
    static constexpr bool is_iec559 = false;
    [[nodiscard]] static constexpr T min() noexcept { return T::from_bits(0x0080u); }
    [[nodiscard]] static constexpr T max() noexcept { return T::from_bits(0x7f7fu); }
    [[nodiscard]] static constexpr T lowest() noexcept { return T::from_bits(0xff7fu); }
    [[nodiscard]] static constexpr T epsilon() noexcept { return T::from_bits(0x3c00u); }
    [[nodiscard]] static constexpr T round_error() noexcept { return T::from_bits(0x3f00u); }
    [[nodiscard]] static constexpr T infinity() noexcept { return T::from_bits(0x7f80u); }
    [[nodiscard]] static constexpr T quiet_NaN() noexcept { return T::from_bits(0x7fc0u); }
    [[nodiscard]] static constexpr T signaling_NaN() noexcept { return T::from_bits(0x7f81u); }
    [[nodiscard]] static constexpr T denorm_min() noexcept { return T::from_bits(0x0001u); }
};

template<bool E4M3FN>
class numeric_limits<luisa::compute::tile::detail::Float8Storage<E4M3FN>> : public numeric_limits<float> {
    using T = luisa::compute::tile::detail::Float8Storage<E4M3FN>;
public:
    static constexpr int digits = E4M3FN ? 4 : 3, digits10 = 0, max_digits10 = E4M3FN ? 3 : 2;
    static constexpr int min_exponent = E4M3FN ? -5 : -13, max_exponent = E4M3FN ? 9 : 16;
    static constexpr int min_exponent10 = E4M3FN ? -1 : -4, max_exponent10 = E4M3FN ? 2 : 4;
    static constexpr bool is_iec559 = false, has_infinity = !E4M3FN, has_signaling_NaN = !E4M3FN;
    [[nodiscard]] static constexpr T min() noexcept { return T::from_bits(E4M3FN ? 0x08u : 0x04u); }
    [[nodiscard]] static constexpr T max() noexcept { return T::from_bits(E4M3FN ? 0x7eu : 0x7bu); }
    [[nodiscard]] static constexpr T lowest() noexcept { return -max(); }
    [[nodiscard]] static constexpr T epsilon() noexcept { return T::from_bits(E4M3FN ? 0x20u : 0x34u); }
    [[nodiscard]] static constexpr T round_error() noexcept { return T::from_bits(E4M3FN ? 0x30u : 0x38u); }
    [[nodiscard]] static constexpr T infinity() noexcept { return T::from_bits(E4M3FN ? 0x00u : 0x7cu); }
    [[nodiscard]] static constexpr T quiet_NaN() noexcept { return T::from_bits(E4M3FN ? 0x7fu : 0x7eu); }
    [[nodiscard]] static constexpr T signaling_NaN() noexcept { return T::from_bits(E4M3FN ? 0x00u : 0x7du); }
    [[nodiscard]] static constexpr T denorm_min() noexcept { return T::from_bits(0x01u); }
};

}// namespace std

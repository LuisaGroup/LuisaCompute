#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace luisa {

/**
 * @brief bfloat16 software scalar
 *
 * Host-side 16-bit brain floating-point scalar storing the IEEE bfloat16
 * bit pattern (1 sign bit, 8 exponent bits, 7 mantissa bits with an
 * implicit leading bit). The type is a trivially destructible, trivially
 * copyable value type so it can live in buffers, variants, and vectors.
 *
 * Conversions from wider types round to nearest-even; NaNs are preserved
 * as (quieted) NaNs, infinities and signed zeros are preserved, and
 * overflow rounds up to infinity.
 */
struct bfloat16 {

    uint16_t _bits = 0u;

    constexpr bfloat16() noexcept = default;

    /// Narrow a single-precision value with round-to-nearest-even.
    explicit constexpr bfloat16(float value) noexcept {
        auto bits = std::bit_cast<uint32_t>(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u) {// NaN: quieten and keep the payload
            _bits = static_cast<uint16_t>((bits >> 16u) | 0x0040u);
        } else {// round-to-nearest-even on the dropped 16 bits
            _bits = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        }
    }

    /// Narrow a double-precision value (round-to-nearest-even to float first).
    explicit constexpr bfloat16(double value) noexcept
        : bfloat16{static_cast<float>(value)} {}

    /// Narrow an extended-precision value (round-to-nearest-even to float first).
    explicit constexpr bfloat16(long double value) noexcept
        : bfloat16{static_cast<float>(value)} {}

    /// Narrow an integral value (exactly representable up to 2^24).
    template<typename T>
        requires std::is_integral_v<T>
    explicit constexpr bfloat16(T value) noexcept
        : bfloat16{static_cast<float>(value)} {}

    /// Reinterpret raw bfloat16 bits.
    [[nodiscard]] static constexpr bfloat16 from_bits(uint16_t bits) noexcept {
        bfloat16 value;
        value._bits = bits;
        return value;
    }

    /// Raw bfloat16 bits.
    [[nodiscard]] constexpr uint16_t bits() const noexcept { return _bits; }

    /// Widen to single precision (exact).
    [[nodiscard]] constexpr operator float() const noexcept {
        return std::bit_cast<float>(static_cast<uint32_t>(_bits) << 16u);
    }

    /// Widen to double precision (exact).
    [[nodiscard]] constexpr explicit operator double() const noexcept {
        return static_cast<double>(static_cast<float>(*this));
    }

    /// Truncate toward zero (via the exact float value).
    template<typename T>
        requires std::is_integral_v<T>
    [[nodiscard]] constexpr explicit operator T() const noexcept {
        return static_cast<T>(static_cast<float>(*this));
    }

    /// Value assignment with the same explicit-rounding conversion as the constructors.
    constexpr bfloat16 &operator=(float rhs) noexcept {
        return *this = bfloat16{rhs};
    }
    constexpr bfloat16 &operator=(double rhs) noexcept {
        return *this = bfloat16{rhs};
    }
    constexpr bfloat16 &operator=(long double rhs) noexcept {
        return *this = bfloat16{rhs};
    }
    template<typename T>
        requires std::is_integral_v<T>
    constexpr bfloat16 &operator=(T rhs) noexcept {
        return *this = bfloat16{rhs};
    }

    /// Arithmetic assignment. The computation happens in single precision
    /// and the result is rounded to bfloat16 once.
    constexpr bfloat16 &operator+=(float rhs) noexcept {
        return *this = bfloat16{static_cast<float>(*this) + rhs};
    }
    /// Arithmetic assignment.
    constexpr bfloat16 &operator-=(float rhs) noexcept {
        return *this = bfloat16{static_cast<float>(*this) - rhs};
    }
    /// Arithmetic assignment.
    constexpr bfloat16 &operator*=(float rhs) noexcept {
        return *this = bfloat16{static_cast<float>(*this) * rhs};
    }
    /// Arithmetic assignment.
    constexpr bfloat16 &operator/=(float rhs) noexcept {
        return *this = bfloat16{static_cast<float>(*this) / rhs};
    }
};

static_assert(sizeof(bfloat16) == 2u && alignof(bfloat16) == 2u,
              "bfloat16 should be 16-bit sized and aligned.");
static_assert(std::is_trivially_destructible_v<bfloat16> &&
                  std::is_trivially_copyable_v<bfloat16> &&
                  std::is_default_constructible_v<bfloat16>,
              "bfloat16 should be a trivial software scalar.");

namespace literals {

/// bfloat16 literal, e.g. 1.5_bf16. The value is rounded to nearest-even.
[[nodiscard]] constexpr bfloat16 operator""_bf16(long double value) noexcept {
    return bfloat16{static_cast<float>(value)};
}

}// namespace literals

}// namespace luisa

// Free operators live at global scope (like the generic luisa::Vector
// operators in basic_types.h): defining them in namespace luisa would hide
// those operators from unqualified lookup inside namespace luisa.

/// Unary plus.
[[nodiscard]] constexpr luisa::bfloat16 operator+(luisa::bfloat16 x) noexcept { return x; }
/// Unary minus (flips the sign bit, preserving zero/NaN semantics).
[[nodiscard]] constexpr luisa::bfloat16 operator-(luisa::bfloat16 x) noexcept {
    return luisa::bfloat16::from_bits(x.bits() ^ 0x8000u);
}

/// Addition, computed in single precision and rounded once.
[[nodiscard]] constexpr luisa::bfloat16 operator+(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return luisa::bfloat16{static_cast<float>(x) + static_cast<float>(y)};
}
/// Subtraction, computed in single precision and rounded once.
[[nodiscard]] constexpr luisa::bfloat16 operator-(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return luisa::bfloat16{static_cast<float>(x) - static_cast<float>(y)};
}
/// Multiplication, computed in single precision and rounded once.
[[nodiscard]] constexpr luisa::bfloat16 operator*(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return luisa::bfloat16{static_cast<float>(x) * static_cast<float>(y)};
}
/// Division, computed in single precision and rounded once.
[[nodiscard]] constexpr luisa::bfloat16 operator/(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return luisa::bfloat16{static_cast<float>(x) / static_cast<float>(y)};
}

/// Equality comparison.
[[nodiscard]] constexpr bool operator==(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return static_cast<float>(x) == static_cast<float>(y);
}
/// Inequality comparison.
[[nodiscard]] constexpr bool operator!=(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return static_cast<float>(x) != static_cast<float>(y);
}
/// Less-than comparison.
[[nodiscard]] constexpr bool operator<(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return static_cast<float>(x) < static_cast<float>(y);
}
/// Greater-than comparison.
[[nodiscard]] constexpr bool operator>(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return static_cast<float>(x) > static_cast<float>(y);
}
/// Less-or-equal comparison.
[[nodiscard]] constexpr bool operator<=(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return static_cast<float>(x) <= static_cast<float>(y);
}
/// Greater-or-equal comparison.
[[nodiscard]] constexpr bool operator>=(luisa::bfloat16 x, luisa::bfloat16 y) noexcept {
    return static_cast<float>(x) >= static_cast<float>(y);
}

// Regression tests for the integer-power implementation embedded in generated
// CPU device code.

#include "ut/ut.hpp"

#include <limits>
#include <type_traits>

#include <luisa/core/basic_types.h>

#define __device__
#include "../../../rust/luisa_compute_backend_impl/src/cpu/codegen/device_math_powi.h"
#undef __device__

using namespace boost::ut;
using namespace boost::ut::literals;

static_assert(std::is_signed_v<luisa::byte>);
static_assert(std::is_same_v<luisa::half, half_float::half>);

int main() {

    "pow_int_preserves_unsigned_64_bit_parity"_test = [] {
        auto even = luisa::ulong{1} << 32u;
        expect(powi_impl(-1.0f, even) == 1.0f);
        expect(powi_impl(-1.0f, even + 1u) == -1.0f);
        expect(powi_impl(-1.0f, std::numeric_limits<luisa::ulong>::max()) == -1.0f);
    };

    "pow_int_handles_signed_minimum_without_overflow"_test = [] {
        expect(powi_impl(-1.0f, std::numeric_limits<luisa::byte>::min()) == 1.0f);
        expect(powi_impl(-1.0f, std::numeric_limits<short>::min()) == 1.0f);
        expect(powi_impl(-1.0f, std::numeric_limits<int>::min()) == 1.0f);
        expect(powi_impl(-1.0f, std::numeric_limits<luisa::slong>::min()) == 1.0f);
        expect(powi_impl(2.0f, luisa::byte{-1}) == 0.5f);
    };

    "pow_int_broadcasts_scalar_integer_exponent"_test = [] {
        auto base = luisa::float2{2.0f, -2.0f};
        auto exponent = int{-3};
        auto result = luisa::float2{
            powi_impl(base.x, exponent),
            powi_impl(base.y, exponent)};
        expect(result.x == 0.125f);
        expect(result.y == -0.125f);
    };

    "pow_int_uses_integer_vector_lanes"_test = [] {
        auto base = luisa::float2{2.0f, 3.0f};
        auto exponent = luisa::byte2{-2, 3};
        auto narrow = luisa::float2{
            powi_impl(base.x, exponent.x),
            powi_impl(base.y, exponent.y)};
        expect(narrow.x == 0.25f);
        expect(narrow.y == 27.0f);

        auto wide_even = luisa::ulong{1} << 32u;
        auto wide_exponent = luisa::ulong2{wide_even, wide_even + 1u};
        auto wide = luisa::float2{
            powi_impl(-1.0f, wide_exponent.x),
            powi_impl(-1.0f, wide_exponent.y)};
        expect(wide.x == 1.0f);
        expect(wide.y == -1.0f);
    };

    "pow_int_uses_luisa_half"_test = [] {
        auto result = powi_impl(luisa::half{2.0f}, luisa::byte{-1});
        expect(static_cast<float>(result) == 0.5f);
    };
}

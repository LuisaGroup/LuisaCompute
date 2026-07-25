// Regression tests for the integer-power contract shared by XIR constant
// folding and the generated CPU/CUDA device libraries.

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ARM_FP16_FORMAT_IEEE)
// The generated CPU source uses Clang's spelling because the runtime JIT is
// Clang-based. Let the GCC-hosted unit test parse the same representation.
using __fp16 = _Float16;
#endif

#include "../../../rust/luisa_compute_backend_impl/src/cpu/codegen/cpu_prelude.h"
#include "../../../rust/luisa_compute_backend_impl/src/cpu/codegen/device_math.h"

using namespace boost::ut;
using namespace boost::ut::literals;

int main() {

    "pow_int_preserves_unsigned_64_bit_parity"_test = [] {
        auto even = lc_ulong{1} << 32u;
        expect(lc_powi(-1.0f, even) == 1.0f);
        expect(lc_powi(-1.0f, even + 1u) == -1.0f);
        expect(lc_powi(-1.0f, std::numeric_limits<lc_ulong>::max()) == -1.0f);
    };

    "pow_int_handles_signed_minimum_without_overflow"_test = [] {
        expect(lc_powi(-1.0f, std::numeric_limits<lc_byte>::min()) == 1.0f);
        expect(lc_powi(-1.0f, std::numeric_limits<lc_short>::min()) == 1.0f);
        expect(lc_powi(-1.0f, std::numeric_limits<lc_int>::min()) == 1.0f);
        expect(lc_powi(-1.0f, std::numeric_limits<lc_long>::min()) == 1.0f);
        expect(lc_powi(2.0f, lc_byte{-1}) == 0.5f);
    };

    "pow_int_broadcasts_scalar_integer_exponent"_test = [] {
        auto result = lc_powi(lc_make_float2(2.0f, -2.0f), lc_int{-3});
        expect(result.x == 0.125f);
        expect(result.y == -0.125f);
    };

    "pow_int_uses_integer_vector_lanes"_test = [] {
        auto narrow = lc_powi(
            lc_make_float2(2.0f, 3.0f), lc_byte2{-2, 3});
        expect(narrow.x == 0.25f);
        expect(narrow.y == 27.0f);

        auto wide_even = lc_ulong{1} << 32u;
        auto wide = lc_powi(
            lc_make_float2(-1.0f, -1.0f),
            lc_ulong2{wide_even, wide_even + 1u});
        expect(wide.x == 1.0f);
        expect(wide.y == -1.0f);
    };
}

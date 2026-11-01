// Test for bfloat16 host scalar support in <luisa/core/bfloat16.h>.
// This test covers:
// - bit patterns, round-to-nearest-even ties, signed zero, infinity/NaN, range
// - float/double/integral conversions
// - unary/binary/compound arithmetic and comparisons
// - vector aliases, make_* helpers, and formatting

#include "ut/ut.hpp"

#include <luisa/core/basic_traits.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/format.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace luisa;
using namespace luisa::literals;
using namespace boost::ut;
using namespace boost::ut::literals;

// type properties
static_assert(sizeof(bfloat16) == 2u && alignof(bfloat16) == 2u, "bfloat16 should be 16-bit sized and aligned.");
static_assert(std::is_trivially_destructible_v<bfloat16> && std::is_trivially_copyable_v<bfloat16>,
              "bfloat16 should be a trivial software scalar.");
static_assert(sizeof(bfloat162) == 4u && alignof(bfloat162) == 4u, "bfloat162 should be 2x 16-bit.");
static_assert(sizeof(bfloat164) == 8u && alignof(bfloat164) == 8u, "bfloat164 should be 4x 16-bit.");

// luisa type traits treat bfloat16 as a signed floating-point scalar
static_assert(is_floating_point_v<bfloat16>, "bfloat16 should be floating point.");
static_assert(!is_integral_v<bfloat16>, "bfloat16 should not be integral.");
static_assert(is_signed_v<bfloat16>, "bfloat16 should be signed.");
static_assert(!is_unsigned_v<bfloat16>, "bfloat16 should not be unsigned.");
static_assert(is_arithmetic_v<bfloat16>, "bfloat16 should be arithmetic.");
static_assert(is_scalar_v<bfloat16>, "bfloat16 should be scalar.");
static_assert(is_basic_v<bfloat16>, "bfloat16 should be basic.");
static_assert(is_floating_point_vector_v<bfloat162> && is_floating_point_vector_v<bfloat163> && is_floating_point_vector_v<bfloat164>,
              "bfloat16 vectors should be floating point vectors.");
static_assert(std::is_same_v<vector_element_t<bfloat162>, bfloat16>, "bfloat162 element should be bfloat16.");

// bfloat16 arithmetic stays in bfloat16; mixed arithmetic computes in float, like half
static_assert(std::is_same_v<decltype(1._bf16 + 1._bf16), bfloat16>, "bfloat16 + bfloat16 should be bfloat16.");
static_assert(std::is_same_v<decltype(1.0f + 1._bf16), float>, "bfloat16 + float should be float.");
static_assert(std::is_same_v<decltype(-1._bf16), bfloat16>, "-bfloat16 should be bfloat16.");
static_assert(std::is_same_v<decltype(1._bf16 < 2._bf16), bool>, "bfloat16 < bfloat16 should be bool.");

// user-defined literal (parenthesized: a member access cannot bind directly to a literal)
static_assert((1.5_bf16).bits() == 0x3fc0u, "1.5_bf16 should have the IEEE bfloat16 bits of 1.5.");
static_assert(1._bf16 + 1._bf16 == 2._bf16, "bfloat16 literal arithmetic should be constexpr.");
static_assert(bfloat16{0.5f}.bits() == 0x3f00u, "0.5f should convert exactly.");

void test_bfloat16_bit_patterns() {
    expect(bfloat16{0.0f}.bits() == 0x0000u) << "+0.0f bits";
    expect(bfloat16{-0.0f}.bits() == 0x8000u) << "-0.0f bits";
    expect(bfloat16{1.0f}.bits() == 0x3f80u) << "1.0f bits";
    expect(bfloat16{-2.0f}.bits() == 0xc000u) << "-2.0f bits";
    expect(bfloat16{0.5f}.bits() == 0x3f00u) << "0.5f bits";
    expect(bfloat16{256.0f}.bits() == 0x4380u) << "256.0f bits";
    expect(bfloat16{std::numeric_limits<float>::infinity()}.bits() == 0x7f80u) << "+inf bits";
    expect(bfloat16{-std::numeric_limits<float>::infinity()}.bits() == 0xff80u) << "-inf bits";
    // from_bits/bits round trip
    expect(bfloat16::from_bits(0x3f80u).bits() == 0x3f80u) << "from_bits round trip";
    // every non-NaN bit pattern widens exactly and narrows back to the same bits
    bool all_round_trips = true;
    for (uint32_t b = 0u; b <= 0xffffu; ++b) {
        auto x = bfloat16::from_bits(static_cast<uint16_t>(b));
        auto f = static_cast<float>(x);
        if (std::isnan(f)) {
            all_round_trips &= std::isnan(static_cast<float>(bfloat16{f}));
        } else {
            all_round_trips &= (bfloat16{f}.bits() == static_cast<uint16_t>(b));
        }
    }
    expect(all_round_trips) << "all non-NaN bfloat16 bit patterns should round-trip exactly through float";
}

void test_bfloat16_round_to_nearest_even() {
    // halfway between 1.0 (mantissa 0, even) and the next bfloat16 (mantissa 1, odd): ties to even
    auto halfway_up = std::bit_cast<float>(0x3f808000u);// 1 + 2^-8
    expect(bfloat16{halfway_up}.bits() == 0x3f80u) << "tie should round to even";
    expect(bfloat16{std::bit_cast<float>(0x3f807fffu)}.bits() == 0x3f80u) << "just below tie should round down";
    expect(bfloat16{std::bit_cast<float>(0x3f808001u)}.bits() == 0x3f81u) << "just above tie should round up";
    // halfway between mantissa 1 (odd) and mantissa 2 (even): ties to even
    auto halfway_even_up = std::bit_cast<float>(0x3f818000u);// 1 + 3 * 2^-8
    expect(bfloat16{halfway_even_up}.bits() == 0x3f82u) << "tie should round up to even mantissa";
    // ties round the same way for negative values
    expect(bfloat16{std::bit_cast<float>(0xbf808000u)}.bits() == 0xbf80u) << "negative tie should round to even";
    // halfway between the largest finite bfloat16 (mantissa 0x7f, odd) and infinity: ties to even (infinity)
    expect(bfloat16{std::bit_cast<float>(0x7f7f8000u)}.bits() == 0x7f80u) << "overflow tie should round up to infinity";
    expect(bfloat16{std::bit_cast<float>(0x7f7f7fffu)}.bits() == 0x7f7fu) << "just below overflow tie should stay finite";
}

void test_bfloat16_signed_zero() {
    auto pz = bfloat16{0.0f};
    auto nz = bfloat16{-0.0f};
    expect(pz.bits() == 0x0000u) << "+0 bits";
    expect(nz.bits() == 0x8000u) << "-0 bits";
    expect(!std::signbit(static_cast<float>(pz))) << "+0 sign bit clear";
    expect(std::signbit(static_cast<float>(nz))) << "-0 sign bit set";
    expect(pz == nz) << "-0 should equal +0";
    expect((-pz).bits() == 0x8000u) << "-(+0) should be -0";
    expect((-nz).bits() == 0x0000u) << "-(-0) should be +0";
    expect(bfloat16::from_bits(0x8000u).bits() == 0x8000u) << "from_bits should preserve -0";
}

void test_bfloat16_infinity_and_nan() {
    auto inf = bfloat16{std::numeric_limits<float>::infinity()};
    auto nan = bfloat16{std::numeric_limits<float>::quiet_NaN()};
    expect(inf.bits() == 0x7f80u) << "infinity bits";
    expect(std::isinf(static_cast<float>(inf))) << "infinity should widen to inf";
    expect(nan.bits() != 0x7f80u && (nan.bits() & 0x7f80u) == 0x7f80u && (nan.bits() & 0x007fu) != 0u)
        << "NaN should keep all-ones exponent with non-zero mantissa";
    expect(std::isnan(static_cast<float>(nan))) << "NaN should widen to NaN";
    // NaN is preserved (and quieted) when narrowing
    auto signaling = std::bit_cast<float>(0x7f810000u);// signaling NaN
    auto quieted = bfloat16{signaling};
    expect(std::isnan(static_cast<float>(quieted))) << "signaling NaN should stay NaN";
    expect((quieted.bits() & 0x0040u) != 0u) << "signaling NaN should be quieted";
    // NaN propagates through arithmetic
    expect(std::isnan(static_cast<float>(nan + bfloat16{1.0f}))) << "NaN + 1 should be NaN";
    expect(std::isnan(static_cast<float>(bfloat16{1.0f} * nan))) << "1 * NaN should be NaN";
    // NaN compares unequal to everything, including itself
    expect(nan != nan) << "NaN != NaN";
    expect(!(nan == nan)) << "!(NaN == NaN)";
    expect(!(nan < bfloat16{1.0f}) && !(nan > bfloat16{1.0f}) && !(nan <= bfloat16{1.0f}) && !(nan >= bfloat16{1.0f}))
        << "ordered comparisons with NaN should be false";
    // infinity arithmetic
    expect(static_cast<float>(inf + inf) == std::numeric_limits<float>::infinity()) << "inf + inf should be inf";
    expect(static_cast<float>(bfloat16{1.0f} / bfloat16{0.0f}) == std::numeric_limits<float>::infinity())
        << "1 / 0 should be inf";
    expect(std::isnan(static_cast<float>(inf - inf))) << "inf - inf should be NaN";
    expect(std::isnan(static_cast<float>(inf / inf))) << "inf / inf should be NaN";
}

void test_bfloat16_range() {
    // largest finite bfloat16: 0x7f7f << 16 reinterpreted as float
    expect(static_cast<float>(bfloat16::from_bits(0x7f7fu)) == std::bit_cast<float>(0x7f7f0000u)) << "max bfloat16";
    expect(static_cast<float>(bfloat16::from_bits(0xff7fu)) == std::bit_cast<float>(0xff7f0000u)) << "lowest bfloat16";
    // smallest normal and subnormal
    expect(static_cast<float>(bfloat16::from_bits(0x0080u)) == std::bit_cast<float>(0x00800000u)) << "min normal";
    expect(static_cast<float>(bfloat16::from_bits(0x0001u)) == std::bit_cast<float>(0x00010000u)) << "min subnormal";
    // narrowing float/double overflow saturates to infinity
    expect(bfloat16{std::numeric_limits<float>::max()}.bits() == 0x7f80u) << "float max should overflow to inf";
    expect(bfloat16{-std::numeric_limits<float>::max()}.bits() == 0xff80u) << "-float max should overflow to -inf";
    expect(bfloat16{1e300}.bits() == 0x7f80u) << "large double should overflow to inf";
    expect(bfloat16{-1e300}.bits() == 0xff80u) << "large negative double should overflow to -inf";
    expect(bfloat16{1e-300}.bits() == 0x0000u) << "tiny double should underflow to +0";
}

void test_bfloat16_conversions() {
    // exactly representable values convert exactly
    expect(static_cast<float>(bfloat16{0.5f}) == 0.5f) << "0.5f round trip";
    expect(static_cast<float>(bfloat16{-0.5f}) == -0.5f) << "-0.5f round trip";
    expect(static_cast<float>(bfloat16{65536.0f}) == 65536.0f) << "65536.0f round trip";
    // narrowing an inexact value rounds to the nearest bfloat16
    expect(std::fabs(static_cast<float>(bfloat16{0.1f}) - 0.1f) < 0x1p-8f) << "0.1f should round to the nearest bfloat16";
    expect(static_cast<float>(bfloat16{65504.0f}) == 65536.0f) << "65504.0f (half max) is a tie and rounds to even";
    // double
    expect(bfloat16{1.5}.bits() == 0x3fc0u) << "1.5 double bits";
    expect(static_cast<double>(bfloat16{1.5}) == 1.5) << "1.5 double round trip";
    expect(bfloat16{0.1}.bits() == bfloat16{0.1f}.bits()) << "double and float should narrow identically when equal";
    // integrals
    expect(static_cast<float>(bfloat16{42}) == 42.0f) << "int 42";
    expect(static_cast<float>(bfloat16{-7}) == -7.0f) << "int -7";
    expect(static_cast<float>(bfloat16{0}) == 0.0f) << "int 0";
    expect(bfloat16{16777216}.bits() == 0x4b80u) << "2^24 exact";
    // 2^24 + 1 is a tie between two representable values: rounds to even (2^24)
    expect(bfloat16{16777217}.bits() == 0x4b80u) << "2^24 + 1 should round to even";
    // beyond 2^24 the bfloat16 spacing is 2^17: 2^24 + 2 stays at 2^24, 2^24 + 2^17 is exact
    expect(bfloat16{16777218}.bits() == 0x4b80u) << "2^24 + 2 rounds down to 2^24";
    expect(bfloat16{16908288}.bits() == 0x4b81u) << "2^24 + 2^17 exact";
    expect(bfloat16{1099511627776ull}.bits() == 0x5380u) << "2^40 exact";
    // integral conversion truncates toward zero
    expect(static_cast<int>(bfloat16{3.7f}) == 3) << "3.7 truncates to 3";
    expect(static_cast<int>(bfloat16{-3.7f}) == -3) << "-3.7 truncates to -3";
    expect(static_cast<uint32_t>(bfloat16{7.75f}) == 7u) << "7.75 truncates to 7";
    expect(static_cast<int>(bfloat16{8.0f}) == 8) << "8.0 converts to 8";
}

void test_bfloat16_operators() {
    auto a = 1.5_bf16;
    auto b = 2.0_bf16;
    // binary arithmetic
    expect(a + b == 3.5_bf16) << "1.5 + 2.0";
    expect(b - a == 0.5_bf16) << "2.0 - 1.5";
    expect(a * b == 3.0_bf16) << "1.5 * 2.0";
    expect(b / a == bfloat16{4.0f / 3.0f}) << "2.0 / 1.5 rounded once";
    expect(1._bf16 - 2._bf16 == -1._bf16) << "1 - 2";
    // unary arithmetic
    expect(-a == -1.5_bf16) << "-(1.5)";
    expect(+a == a) << "+(1.5)";
    // arithmetic rounds the single-precision result once
    expect(1._bf16 / 3._bf16 == bfloat16{1.0f / 3.0f}) << "1 / 3 rounded once";
    // mixed-type arithmetic promotes to float, like half
    auto mixed = a + 0.5f;
    expect(std::is_same_v<decltype(mixed), float>) << "bfloat16 + float should yield float";
    expect(mixed == 2.0f) << "1.5_bf16 + 0.5f";
    expect(1.0f + a == 2.5f) << "float + bfloat16";
    expect(a * 2 == 3.0f) << "bfloat16 * int";
    // ordinary and compound assignment
    auto x = 1._bf16;
    x = 1.0f;
    expect(x == 1._bf16) << "ordinary float assignment";
    x += 1._bf16;
    expect(x == 2._bf16) << "+= bfloat16";
    x -= 0.5_bf16;
    expect(x == 1.5_bf16) << "-= bfloat16";
    x *= 2._bf16;
    expect(x == 3._bf16) << "*= bfloat16";
    x /= 4._bf16;
    expect(x == 0.75_bf16) << "/= bfloat16";
    // compound assignment with a float operand
    x += 0.25f;
    expect(x == 1.0_bf16) << "+= float";
    x *= 2.0f;
    expect(x == 2._bf16) << "*= float";
    // comparisons
    expect(1._bf16 < 2._bf16) << "<";
    expect(2._bf16 > 1._bf16) << ">";
    expect(1._bf16 <= 1._bf16 && 1._bf16 >= 1._bf16) << "<= and >=";
    expect(1._bf16 != 2._bf16) << "!=";
    expect(-1._bf16 < 0._bf16) << "negative < zero";
    expect(a == 1.5f) << "bfloat16 == float promotes";
    expect(b > 1.5f) << "bfloat16 > float promotes";
}

void test_bfloat16_vectors() {
    // construction and element access
    bfloat162 v{1.0_bf16, 2.0_bf16};
    expect(v.x == 1._bf16 && v.y == 2._bf16) << "bfloat162 elements";
    bfloat163 u{1.0_bf16, 2.0_bf16, 3.0_bf16};
    expect(u.z == 3._bf16) << "bfloat163 elements";
    bfloat164 w{1.0_bf16, 2.0_bf16, 3.0_bf16, 4.0_bf16};
    expect(w.w == 4._bf16) << "bfloat164 elements";
    // subscript
    expect(v[0] == 1._bf16 && v[1] == 2._bf16) << "bfloat162 subscript";
    v[1] = 5._bf16;
    expect(v.y == 5._bf16) << "bfloat162 subscript write";
    // zero/one
    auto z3 = bfloat163::zero();
    expect(z3.x == 0._bf16 && z3.y == 0._bf16 && z3.z == 0._bf16) << "zero()";
    auto o4 = bfloat164::one();
    expect(o4.x == 1._bf16 && o4.y == 1._bf16 && o4.z == 1._bf16 && o4.w == 1._bf16) << "one()";
    // vector arithmetic via the generic Vector operators
    auto sum = v + bfloat162{1._bf16, 1._bf16};
    expect(sum.x == 2._bf16 && sum.y == 6._bf16) << "vector + vector";
    auto scaled = bfloat162{1._bf16, 2._bf16} * 2._bf16;
    expect(scaled.x == 2._bf16 && scaled.y == 4._bf16) << "vector * scalar";
    auto neg = -bfloat163{1._bf16, 2._bf16, 3._bf16};
    expect(neg.x == -1._bf16 && neg.y == -2._bf16 && neg.z == -3._bf16) << "-vector";
    // vector comparisons produce bool vectors (v is (1, 5) after the subscript write)
    auto lt = v < bfloat162{2._bf16, 5._bf16};
    expect(lt.x && !lt.y) << "vector < vector";
    // compound assignment
    auto acc = bfloat162{1._bf16, 1._bf16};
    acc += bfloat162{1._bf16, 2._bf16};
    expect(acc.x == 2._bf16 && acc.y == 3._bf16) << "vector += vector";
    // make_* helpers
    auto m = make_bfloat163(1._bf16, 2._bf16, 3._bf16);
    expect(m.z == 3._bf16) << "make_bfloat163";
    auto m4 = make_bfloat164(bfloat162{1._bf16, 2._bf16}, bfloat162{3._bf16, 4._bf16});
    expect(m4.w == 4._bf16) << "make_bfloat164 from two bfloat162";
    // formatting
    expect(luisa::format("{}", bfloat16{1.5f}) == "1.5") << "bfloat16 formatter";
    expect(luisa::to_string(bfloat162{1._bf16, 2._bf16}) == "(bfloat162(1, 2))") << "bfloat162 formatter";
}

void test_bfloat16_traits() {
    expect(static_cast<bool>(is_floating_point_v<bfloat16>));
    expect(static_cast<bool>(is_signed_v<bfloat16>));
    expect(static_cast<bool>(is_arithmetic_v<bfloat16>));
    expect(static_cast<bool>(is_scalar_v<bfloat16>));
    expect(static_cast<bool>(is_basic_v<bfloat16>));
    expect(static_cast<bool>(is_floating_point_vector_v<bfloat164>));
    expect(static_cast<bool>(!is_integral_v<bfloat16>));
}

static auto test_bfloat16_registration = [] {
    "bfloat16_bit_patterns"_test = [] { test_bfloat16_bit_patterns(); };
    "bfloat16_round_to_nearest_even"_test = [] { test_bfloat16_round_to_nearest_even(); };
    "bfloat16_signed_zero"_test = [] { test_bfloat16_signed_zero(); };
    "bfloat16_infinity_and_nan"_test = [] { test_bfloat16_infinity_and_nan(); };
    "bfloat16_range"_test = [] { test_bfloat16_range(); };
    "bfloat16_conversions"_test = [] { test_bfloat16_conversions(); };
    "bfloat16_operators"_test = [] { test_bfloat16_operators(); };
    "bfloat16_vectors"_test = [] { test_bfloat16_vectors(); };
    "bfloat16_traits"_test = [] { test_bfloat16_traits(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}

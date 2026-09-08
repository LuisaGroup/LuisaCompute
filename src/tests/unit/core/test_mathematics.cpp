#include <luisa/core/mathematics.h>
#include <luisa/core/logging.h>
#include "ut/ut.hpp"
#include <cmath>
#include <limits>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

template<typename T, typename U>
concept HasCeilDiv = requires(T n, U d) { luisa::ceil_div(n, d); };

static_assert(ceil_div(0, 7) == 0);
static_assert(ceil_div(16, 8) == 2);
static_assert(ceil_div(17, 8) == 3);
static_assert(ceil_div(-17, 8) == -2);
static_assert(ceil_div(17, -8) == -2);
static_assert(ceil_div(-17, -8) == 3);
static_assert(ceil_div(std::numeric_limits<uint64_t>::max(), uint64_t{2}) == (uint64_t{1} << 63u));
static_assert(ceil_div(std::numeric_limits<int64_t>::max(), int64_t{2}) == (int64_t{1} << 62u));
static_assert(ceil_div(std::numeric_limits<int64_t>::min(), int64_t{1}) == std::numeric_limits<int64_t>::min());
static_assert(std::is_same_v<decltype(ceil_div(uint64_t{17}, 8u)), uint64_t>);
static_assert(std::is_same_v<decltype(ceil_div(uint8_t{17}, uint8_t{8})), int>);
static_assert(!HasCeilDiv<float, int> && !HasCeilDiv<int, float>);
static_assert(!HasCeilDiv<bool, unsigned> && !HasCeilDiv<unsigned, bool>);

void test_ceil_div() {
    // An independent real-valued oracle covers both signs and all remainders.
    for (int n = -64; n <= 64; n++) {
        for (int d = -16; d <= 16; d++) {
            if (d != 0) {
                expect(eq(ceil_div(n, d), static_cast<int>(std::ceil(static_cast<double>(n) / d))));
            }
        }
    }
    auto maximum = std::numeric_limits<uint64_t>::max();
    expect(eq(ceil_div(maximum, uint64_t{1}), maximum));
    expect(eq(ceil_div(maximum, maximum), uint64_t{1}));
    expect(eq(ceil_div(uint64_t{0}, maximum), uint64_t{0}));
    expect(eq(ceil_div(uint64_t{1}, maximum), uint64_t{1}));
    expect(eq(ceil_div(maximum - 1u, uint64_t{2}), maximum / 2u));
    expect(eq(ceil_div(maximum, uint64_t{3}), maximum / 3u));
    expect(eq(ceil_div(uint64_t{513}, 64u), uint64_t{9}));
    expect(eq(ceil_div(int64_t{513}, 64u), int64_t{9}));
    expect(eq(ceil_div(uint8_t{255}, uint8_t{2}), 128));
    expect(eq(ceil_div(std::numeric_limits<int64_t>::min(), int64_t{-2}), int64_t{1} << 62u));
    expect(eq(ceil_div(std::numeric_limits<int64_t>::min(), int64_t{3}), std::numeric_limits<int64_t>::min() / 3));
}

// Helper for approximate equality check
template<typename T>
bool approx_eq(T a, T b, T epsilon = static_cast<T>(1e-5)) {
    return std::abs(a - b) < epsilon;
}

// Helper for vector approximate equality
template<size_t N>
bool approx_eq_vec(Vector<float, N> a, Vector<float, N> b, float epsilon = 1e-5f) {
    for (size_t i = 0; i < N; ++i) {
        if (!approx_eq(a[i], b[i], epsilon)) return false;
    }
    return true;
}

void test_next_pow2() {
    // Test uint32
    expect(static_cast<bool>(next_pow2(0u) == 0u));
    expect(static_cast<bool>(next_pow2(1u) == 1u));
    expect(static_cast<bool>(next_pow2(2u) == 2u));
    expect(static_cast<bool>(next_pow2(3u) == 4u));
    expect(static_cast<bool>(next_pow2(4u) == 4u));
    expect(static_cast<bool>(next_pow2(5u) == 8u));
    expect(static_cast<bool>(next_pow2(1023u) == 1024u));
    expect(static_cast<bool>(next_pow2(1024u) == 1024u));
    expect(static_cast<bool>(next_pow2(1025u) == 2048u));

    // Test uint64
    expect(static_cast<bool>(next_pow2(0ull) == 0ull));
    expect(static_cast<bool>(next_pow2(1ull) == 1ull));
    expect(static_cast<bool>(next_pow2(3ull) == 4ull));
    expect(static_cast<bool>(next_pow2(1ull << 32) == (1ull << 32)));
    expect(static_cast<bool>(next_pow2((1ull << 32) + 1) == (1ull << 33)));
}

void test_scalar_math() {
    // Test fract
    expect(static_cast<bool>(approx_eq(fract(3.7f), 0.7f)));
    expect(static_cast<bool>(approx_eq(fract(-3.7f), 0.3f)));
    expect(static_cast<bool>(approx_eq(fract(5.0f), 0.0f)));

    // Test radians/degrees conversion
    expect(static_cast<bool>(approx_eq(radians(180.0f), constants::pi)));
    expect(static_cast<bool>(approx_eq(radians(90.0f), constants::pi / 2.0f)));
    expect(static_cast<bool>(approx_eq(degrees(constants::pi), 180.0f)));
    expect(static_cast<bool>(approx_eq(degrees(constants::pi / 2.0f), 90.0f)));

    // Test standard math functions are available
    expect(static_cast<bool>(approx_eq(sin(constants::pi / 2.0f), 1.0f)));
    expect(static_cast<bool>(approx_eq(cos(0.0f), 1.0f)));
    expect(static_cast<bool>(approx_eq(sqrt(4.0f), 2.0f)));
    expect(static_cast<bool>(approx_eq(abs(-5.0f), 5.0f)));
    expect(static_cast<bool>(min(3, 5) == 3));
    expect(static_cast<bool>(max(3, 5) == 5));
}

void test_vector_unary_funcs() {
    // Test vector sin
    float2 v2 = make_float2(0.0f, constants::pi / 2.0f);
    float2 sin2 = sin(v2);
    expect(static_cast<bool>(approx_eq(sin2.x, 0.0f)));
    expect(static_cast<bool>(approx_eq(sin2.y, 1.0f)));

    float3 v3 = make_float3(0.0f, constants::pi / 2.0f, constants::pi);
    float3 cos3 = cos(v3);
    expect(static_cast<bool>(approx_eq(cos3.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(cos3.y, 0.0f)));
    expect(static_cast<bool>(approx_eq(cos3.z, -1.0f)));

    float4 v4 = make_float4(1.0f, 4.0f, 9.0f, 16.0f);
    float4 sqrt4 = sqrt(v4);
    expect(static_cast<bool>(approx_eq(sqrt4.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(sqrt4.y, 2.0f)));
    expect(static_cast<bool>(approx_eq(sqrt4.z, 3.0f)));
    expect(static_cast<bool>(approx_eq(sqrt4.w, 4.0f)));

    // Test vector abs
    float3 v3neg = make_float3(-1.0f, -2.0f, -3.0f);
    float3 abs3 = abs(v3neg);
    expect(static_cast<bool>(approx_eq(abs3.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(abs3.y, 2.0f)));
    expect(static_cast<bool>(approx_eq(abs3.z, 3.0f)));

    // Test fract for vectors
    float3 v3fract = make_float3(1.5f, 2.7f, -3.3f);
    float3 fract3 = fract(v3fract);
    expect(static_cast<bool>(approx_eq(fract3.x, 0.5f)));
    expect(static_cast<bool>(approx_eq(fract3.y, 0.7f)));
    expect(static_cast<bool>(approx_eq(fract3.z, 0.7f)));

    // Test floor/ceil
    float2 v2floor = make_float2(1.9f, -1.1f);
    float2 floor2 = floor(v2floor);
    expect(static_cast<bool>(approx_eq(floor2.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(floor2.y, -2.0f)));

    float2 v2ceil = make_float2(1.1f, -1.9f);
    float2 ceil2 = ceil(v2ceil);
    expect(static_cast<bool>(approx_eq(ceil2.x, 2.0f)));
    expect(static_cast<bool>(approx_eq(ceil2.y, -1.0f)));

    // Test radians/degrees for vectors
    float3 deg3 = make_float3(0.0f, 90.0f, 180.0f);
    float3 rad3 = radians(deg3);
    expect(static_cast<bool>(approx_eq(rad3.x, 0.0f)));
    expect(static_cast<bool>(approx_eq(rad3.y, constants::pi / 2.0f)));
    expect(static_cast<bool>(approx_eq(rad3.z, constants::pi)));
}

void test_vector_binary_funcs() {
    // Test min/max for vectors
    float2 a2 = make_float2(1.0f, 5.0f);
    float2 b2 = make_float2(3.0f, 2.0f);
    float2 min2 = min(a2, b2);
    float2 max2 = max(a2, b2);
    expect(static_cast<bool>(approx_eq(min2.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(min2.y, 2.0f)));
    expect(static_cast<bool>(approx_eq(max2.x, 3.0f)));
    expect(static_cast<bool>(approx_eq(max2.y, 5.0f)));

    // Test scalar-vector min/max
    float3 v3 = make_float3(1.0f, 5.0f, 3.0f);
    float3 min_s3 = min(2.0f, v3);
    float3 max_s3 = max(2.0f, v3);
    float3 min_vs3 = min(v3, 2.0f);
    float3 max_vs3 = max(v3, 2.0f);
    expect(static_cast<bool>(approx_eq(min_s3.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(min_s3.y, 2.0f)));
    expect(static_cast<bool>(approx_eq(min_s3.z, 2.0f)));
    expect(static_cast<bool>(approx_eq(max_s3.x, 2.0f)));
    expect(static_cast<bool>(approx_eq(max_s3.y, 5.0f)));
    expect(static_cast<bool>(approx_eq(max_s3.z, 3.0f)));
    expect(static_cast<bool>(approx_eq(min_vs3.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(max_vs3.x, 2.0f)));

    // Test pow for vectors
    float2 base2 = make_float2(2.0f, 3.0f);
    float2 exp2 = make_float2(3.0f, 2.0f);
    float2 pow2 = pow(base2, exp2);
    expect(static_cast<bool>(approx_eq(pow2.x, 8.0f)));
    expect(static_cast<bool>(approx_eq(pow2.y, 9.0f)));

    // Test atan2 for vectors
    float2 y2 = make_float2(1.0f, -1.0f);
    float2 x2 = make_float2(1.0f, 1.0f);
    float2 atan2_2 = atan2(y2, x2);
    expect(static_cast<bool>(approx_eq(atan2_2.x, constants::pi / 4.0f)));
    expect(static_cast<bool>(approx_eq(atan2_2.y, -constants::pi / 4.0f)));

    // Test fmod for vectors
    float2 fmod_a2 = make_float2(7.0f, 10.0f);
    float2 fmod_b2 = make_float2(3.0f, 4.0f);
    float2 fmod2 = fmod(fmod_a2, fmod_b2);
    expect(static_cast<bool>(approx_eq(fmod2.x, 1.0f)));
    expect(static_cast<bool>(approx_eq(fmod2.y, 2.0f)));
}

void test_isnan_isinf() {
    // Test scalar isnan/isinf
    expect(static_cast<bool>(!std::isnan(1.0f)));
    expect(static_cast<bool>(std::isnan(std::numeric_limits<float>::quiet_NaN())));
    expect(static_cast<bool>(!std::isinf(1.0f)));
    expect(static_cast<bool>(std::isinf(std::numeric_limits<float>::infinity())));
    expect(static_cast<bool>(std::isinf(-std::numeric_limits<float>::infinity())));

    // Test vector isnan/isinf
    float3 nan3 = make_float3(1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f);
    bool3 isnan3 = isnan(nan3);
    expect(static_cast<bool>(!isnan3.x));
    expect(static_cast<bool>(isnan3.y));
    expect(static_cast<bool>(!isnan3.z));

    float4 inf4 = make_float4(1.0f, std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(), 2.0f);
    bool4 isinf4 = isinf(inf4);
    expect(static_cast<bool>(!isinf4.x));
    expect(static_cast<bool>(isinf4.y));
    expect(static_cast<bool>(isinf4.z));
    expect(static_cast<bool>(!isinf4.w));
}

void test_select() {
    // Test scalar select
    expect(static_cast<bool>(select(0.0f, 1.0f, true) == 1.0f));
    expect(static_cast<bool>(select(0.0f, 1.0f, false) == 0.0f));

    // Test vector select
    float2 f2 = make_float2(1.0f, 2.0f);
    float2 t2 = make_float2(10.0f, 20.0f);
    bool2 p2 = make_bool2(true, false);
    float2 r2 = select(f2, t2, p2);
    expect(static_cast<bool>(r2.x == 10.0f));
    expect(static_cast<bool>(r2.y == 2.0f));

    float3 f3 = make_float3(1.0f, 2.0f, 3.0f);
    float3 t3 = make_float3(10.0f, 20.0f, 30.0f);
    bool3 p3 = make_bool3(false, true, false);
    float3 r3 = select(f3, t3, p3);
    expect(static_cast<bool>(r3.x == 1.0f));
    expect(static_cast<bool>(r3.y == 20.0f));
    expect(static_cast<bool>(r3.z == 3.0f));
}

void test_lerp_clamp() {
    // Test scalar lerp
    float lerp_result = lerp(0.0f, 10.0f, 0.5f);
    expect(static_cast<bool>(approx_eq(lerp_result, 5.0f)));
    expect(static_cast<bool>(approx_eq(lerp(0.0f, 10.0f, 0.0f), 0.0f)));
    expect(static_cast<bool>(approx_eq(lerp(0.0f, 10.0f, 1.0f), 10.0f)));

    // Test vector lerp
    float2 a2 = make_float2(0.0f, 10.0f);
    float2 b2 = make_float2(10.0f, 20.0f);
    float2 lerp2 = lerp(a2, b2, 0.5f);
    expect(static_cast<bool>(approx_eq(lerp2.x, 5.0f)));
    expect(static_cast<bool>(approx_eq(lerp2.y, 15.0f)));

    // Test scalar clamp
    expect(static_cast<bool>(clamp(5.0f, 0.0f, 10.0f) == 5.0f));
    expect(static_cast<bool>(clamp(-5.0f, 0.0f, 10.0f) == 0.0f));
    expect(static_cast<bool>(clamp(15.0f, 0.0f, 10.0f) == 10.0f));

    // Test vector clamp
    float3 v3 = make_float3(-1.0f, 5.0f, 15.0f);
    float3 clamp3 = clamp(v3, 0.0f, 10.0f);
    expect(static_cast<bool>(clamp3.x == 0.0f));
    expect(static_cast<bool>(clamp3.y == 5.0f));
    expect(static_cast<bool>(clamp3.z == 10.0f));
}

void test_vector_operations() {
    // Test dot product
    float2 v2a = make_float2(1.0f, 2.0f);
    float2 v2b = make_float2(3.0f, 4.0f);
    expect(static_cast<bool>(approx_eq(dot(v2a, v2b), 11.0f)));

    float3 v3a = make_float3(1.0f, 2.0f, 3.0f);
    float3 v3b = make_float3(4.0f, 5.0f, 6.0f);
    expect(static_cast<bool>(approx_eq(dot(v3a, v3b), 32.0f)));

    float4 v4a = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4 v4b = make_float4(5.0f, 6.0f, 7.0f, 8.0f);
    expect(static_cast<bool>(approx_eq(dot(v4a, v4b), 70.0f)));

    // Test length
    float2 v2len = make_float2(3.0f, 4.0f);
    expect(static_cast<bool>(approx_eq(length(v2len), 5.0f)));

    float3 v3len = make_float3(1.0f, 2.0f, 2.0f);
    expect(static_cast<bool>(approx_eq(length(v3len), 3.0f)));

    // Test normalize
    float2 v2norm = make_float2(3.0f, 4.0f);
    float2 n2 = normalize(v2norm);
    expect(static_cast<bool>(approx_eq(length(n2), 1.0f)));
    expect(static_cast<bool>(approx_eq(n2.x, 0.6f)));
    expect(static_cast<bool>(approx_eq(n2.y, 0.8f)));

    // Test distance
    float2 v2dist1 = make_float2(1.0f, 2.0f);
    float2 v2dist2 = make_float2(4.0f, 6.0f);
    expect(static_cast<bool>(approx_eq(distance(v2dist1, v2dist2), 5.0f)));

    // Test cross product
    float3 v3x = make_float3(1.0f, 0.0f, 0.0f);
    float3 v3y = make_float3(0.0f, 1.0f, 0.0f);
    float3 v3z = cross(v3x, v3y);
    expect(static_cast<bool>(approx_eq(v3z.x, 0.0f)));
    expect(static_cast<bool>(approx_eq(v3z.y, 0.0f)));
    expect(static_cast<bool>(approx_eq(v3z.z, 1.0f)));

    float3 v3a_cross = make_float3(1.0f, 2.0f, 3.0f);
    float3 v3b_cross = make_float3(4.0f, 5.0f, 6.0f);
    float3 cross_result = cross(v3a_cross, v3b_cross);
    // cross((1,2,3), (4,5,6)) = (2*6-3*5, 3*4-1*6, 1*5-2*4) = (-3, 6, -3)
    expect(static_cast<bool>(approx_eq(cross_result.x, -3.0f)));
    expect(static_cast<bool>(approx_eq(cross_result.y, 6.0f)));
    expect(static_cast<bool>(approx_eq(cross_result.z, -3.0f)));
}

void test_matrix_transpose() {
    // Test float2x2 transpose
    float2x2 m2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
    float2x2 t2 = transpose(m2);
    // Original: col0=(1,2), col1=(3,4)
    // Transposed: col0=(1,3), col1=(2,4)
    expect(static_cast<bool>(approx_eq(t2[0][0], 1.0f)));
    expect(static_cast<bool>(approx_eq(t2[0][1], 3.0f)));
    expect(static_cast<bool>(approx_eq(t2[1][0], 2.0f)));
    expect(static_cast<bool>(approx_eq(t2[1][1], 4.0f)));

    // Test float3x3 transpose
    float3x3 m3 = make_float3x3(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f);
    float3x3 t3 = transpose(m3);
    expect(static_cast<bool>(approx_eq(t3[0][0], 1.0f)));
    expect(static_cast<bool>(approx_eq(t3[0][1], 4.0f)));
    expect(static_cast<bool>(approx_eq(t3[0][2], 7.0f)));
    expect(static_cast<bool>(approx_eq(t3[1][0], 2.0f)));
    expect(static_cast<bool>(approx_eq(t3[1][1], 5.0f)));
    expect(static_cast<bool>(approx_eq(t3[1][2], 8.0f)));
    expect(static_cast<bool>(approx_eq(t3[2][0], 3.0f)));
    expect(static_cast<bool>(approx_eq(t3[2][1], 6.0f)));
    expect(static_cast<bool>(approx_eq(t3[2][2], 9.0f)));

    // Test float4x4 transpose
    float4x4 m4 = make_float4x4(
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f);
    float4x4 t4 = transpose(m4);
    expect(static_cast<bool>(approx_eq(t4[0][0], 1.0f)));
    expect(static_cast<bool>(approx_eq(t4[0][3], 13.0f)));
    expect(static_cast<bool>(approx_eq(t4[3][0], 4.0f)));
    expect(static_cast<bool>(approx_eq(t4[3][3], 16.0f)));
}

void test_matrix_inverse() {
    // Test float2x2 inverse
    float2x2 m2 = make_float2x2(4.0f, 2.0f, 3.0f, 1.0f);
    float2x2 inv2 = inverse(m2);
    // det = 4*1 - 2*3 = -2
    // inv = (1/-2) * [[1, -2], [-3, 4]] = [[-0.5, 1], [1.5, -2]]
    expect(static_cast<bool>(approx_eq(inv2[0][0], -0.5f)));
    expect(static_cast<bool>(approx_eq(inv2[0][1], 1.0f)));
    expect(static_cast<bool>(approx_eq(inv2[1][0], 1.5f)));
    expect(static_cast<bool>(approx_eq(inv2[1][1], -2.0f)));

    // Verify: M * M^-1 = I
    float2x2 identity2 = m2 * inv2;
    expect(static_cast<bool>(approx_eq(identity2[0][0], 1.0f, 1e-4f)));
    expect(static_cast<bool>(approx_eq(identity2[0][1], 0.0f, 1e-4f)));
    expect(static_cast<bool>(approx_eq(identity2[1][0], 0.0f, 1e-4f)));
    expect(static_cast<bool>(approx_eq(identity2[1][1], 1.0f, 1e-4f)));

    // Test float3x3 inverse
    float3x3 m3 = make_float3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 4.0f);
    float3x3 inv3 = inverse(m3);
    expect(static_cast<bool>(approx_eq(inv3[0][0], 1.0f)));
    expect(static_cast<bool>(approx_eq(inv3[1][1], 0.5f)));
    expect(static_cast<bool>(approx_eq(inv3[2][2], 0.25f)));

    // Test float4x4 inverse
    float4x4 m4 = make_float4x4(
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 8.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 16.0f);
    float4x4 inv4 = inverse(m4);
    expect(static_cast<bool>(approx_eq(inv4[0][0], 0.5f)));
    expect(static_cast<bool>(approx_eq(inv4[1][1], 0.25f)));
    expect(static_cast<bool>(approx_eq(inv4[2][2], 0.125f)));
    expect(static_cast<bool>(approx_eq(inv4[3][3], 0.0625f)));
}

void test_matrix_determinant() {
    // Test float2x2 determinant
    float2x2 m2 = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
    float det2 = determinant(m2);
    // det = 1*4 - 2*3 = -2
    expect(static_cast<bool>(approx_eq(det2, -2.0f)));

    // Test float3x3 determinant
    float3x3 m3 = make_float3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 3.0f);
    float det3 = determinant(m3);
    expect(static_cast<bool>(approx_eq(det3, 6.0f)));

    // Test float4x4 determinant
    float4x4 m4 = make_float4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f);
    float det4 = determinant(m4);
    expect(static_cast<bool>(approx_eq(det4, 24.0f)));
}

void test_transform_functions() {
    // Test translation matrix
    float4x4 trans = translation(1.0f, 2.0f, 3.0f);
    expect(static_cast<bool>(approx_eq(trans[3][0], 1.0f)));
    expect(static_cast<bool>(approx_eq(trans[3][1], 2.0f)));
    expect(static_cast<bool>(approx_eq(trans[3][2], 3.0f)));
    expect(static_cast<bool>(approx_eq(trans[3][3], 1.0f)));

    float4x4 trans_vec = translation(make_float3(4.0f, 5.0f, 6.0f));
    expect(static_cast<bool>(approx_eq(trans_vec[3][0], 4.0f)));
    expect(static_cast<bool>(approx_eq(trans_vec[3][1], 5.0f)));
    expect(static_cast<bool>(approx_eq(trans_vec[3][2], 6.0f)));

    // Test scaling matrix
    float4x4 scale = scaling(2.0f, 3.0f, 4.0f);
    expect(static_cast<bool>(approx_eq(scale[0][0], 2.0f)));
    expect(static_cast<bool>(approx_eq(scale[1][1], 3.0f)));
    expect(static_cast<bool>(approx_eq(scale[2][2], 4.0f)));
    expect(static_cast<bool>(approx_eq(scale[3][3], 1.0f)));

    float4x4 scale_uni = scaling(5.0f);
    expect(static_cast<bool>(approx_eq(scale_uni[0][0], 5.0f)));
    expect(static_cast<bool>(approx_eq(scale_uni[1][1], 5.0f)));
    expect(static_cast<bool>(approx_eq(scale_uni[2][2], 5.0f)));

    // Test rotation matrix (around z-axis)
    float4x4 rot_z = rotation(make_float3(0.0f, 0.0f, 1.0f), constants::pi / 2.0f);
    float4 test_vec = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    float4 rotated = rot_z * test_vec;
    expect(static_cast<bool>(approx_eq(rotated.x, 0.0f, 1e-4f)));
    expect(static_cast<bool>(approx_eq(rotated.y, 1.0f, 1e-4f)));
    expect(static_cast<bool>(approx_eq(rotated.z, 0.0f, 1e-4f)));
}

void test_sign() {
    // Test float sign
    expect(static_cast<bool>(sign(5.0f) == 1.0f));
    expect(static_cast<bool>(sign(-3.0f) == -1.0f));
    expect(static_cast<bool>(sign(0.0f) == 0.0f));

    // Test float vector sign
    float3 v3 = make_float3(5.0f, -3.0f, 0.0f);
    float3 s3 = sign(v3);
    expect(static_cast<bool>(s3.x == 1.0f));
    expect(static_cast<bool>(s3.y == -1.0f));
    expect(static_cast<bool>(s3.z == 0.0f));

    // Test double sign
    expect(static_cast<bool>(sign(5.0) == 1.0));
    expect(static_cast<bool>(sign(-3.0) == -1.0));

    // Test int sign
    expect(static_cast<bool>(sign(5) == 1));
    expect(static_cast<bool>(sign(-3) == -1));
    expect(static_cast<bool>(sign(0) == 0));

    int2 i2 = make_int2(5, -3);
    int2 si2 = sign(i2);
    expect(static_cast<bool>(si2.x == 1));
    expect(static_cast<bool>(si2.y == -1));
}

void test_fma() {
    // Test scalar fma
    float fma_result = fma(2.0f, 3.0f, 4.0f);
    expect(static_cast<bool>(approx_eq(fma_result, 10.0f)));

    // Test vector fma
    float2 a2 = make_float2(1.0f, 2.0f);
    float2 b2 = make_float2(3.0f, 4.0f);
    float2 c2 = make_float2(5.0f, 6.0f);
    float2 fma2 = fma(a2, b2, c2);
    expect(static_cast<bool>(approx_eq(fma2.x, 8.0f)));
    expect(static_cast<bool>(approx_eq(fma2.y, 14.0f)));

    float3 a3 = make_float3(1.0f, 2.0f, 3.0f);
    float3 b3 = make_float3(4.0f, 5.0f, 6.0f);
    float3 c3 = make_float3(7.0f, 8.0f, 9.0f);
    float3 fma3 = fma(a3, b3, c3);
    expect(static_cast<bool>(approx_eq(fma3.x, 11.0f)));
    expect(static_cast<bool>(approx_eq(fma3.y, 18.0f)));
    expect(static_cast<bool>(approx_eq(fma3.z, 27.0f)));

    // Test mixed scalar-vector fma
    float3 fma_s3 = fma(2.0f, a3, c3);
    expect(static_cast<bool>(approx_eq(fma_s3.x, 9.0f)));
    expect(static_cast<bool>(approx_eq(fma_s3.y, 12.0f)));
    expect(static_cast<bool>(approx_eq(fma_s3.z, 15.0f)));
}

void test_double_precision() {
    // Test double vector operations
    double2 d2a = make_double2(1.0, 2.0);
    double2 d2b = make_double2(3.0, 4.0);
    double dot2 = dot(d2a, d2b);
    expect(static_cast<bool>(approx_eq(dot2, 11.0)));

    double3 d3a = make_double3(1.0, 2.0, 3.0);
    double3 d3b = make_double3(4.0, 5.0, 6.0);
    double dot3 = dot(d3a, d3b);
    expect(static_cast<bool>(approx_eq(dot3, 32.0)));

    // Test double matrix transpose
    double2x2 dm2 = make_double2x2(1.0, 2.0, 3.0, 4.0);
    double2x2 dt2 = transpose(dm2);
    expect(static_cast<bool>(approx_eq(dt2[0][0], 1.0)));
    expect(static_cast<bool>(approx_eq(dt2[0][1], 3.0)));
    expect(static_cast<bool>(approx_eq(dt2[1][0], 2.0)));
    expect(static_cast<bool>(approx_eq(dt2[1][1], 4.0)));

    // Test double matrix inverse
    double2x2 dm2_inv = make_double2x2(4.0, 2.0, 3.0, 1.0);
    double2x2 dinv2 = inverse(dm2_inv);
    expect(static_cast<bool>(approx_eq(dinv2[0][0], -0.5)));
    expect(static_cast<bool>(approx_eq(dinv2[0][1], 1.0)));
    expect(static_cast<bool>(approx_eq(dinv2[1][0], 1.5)));
    expect(static_cast<bool>(approx_eq(dinv2[1][1], -2.0)));

    // Test double determinant
    double ddet2 = determinant(dm2_inv);
    expect(static_cast<bool>(approx_eq(ddet2, -2.0)));
}

void test_nan_propagation() {
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();

    // NaN arithmetic propagation
    expect(static_cast<bool>(std::isnan(nan + 1.0f)));
    expect(static_cast<bool>(std::isnan(nan * 2.0f)));
    expect(static_cast<bool>(std::isnan(nan - nan)));
    expect(static_cast<bool>(std::isnan(0.0f * inf)));

    // NaN comparisons always false
    expect(static_cast<bool>(!(nan == nan)));
    expect(static_cast<bool>(!(nan < 0.0f)));
    expect(static_cast<bool>(!(nan > 0.0f)));
    expect(static_cast<bool>(nan != nan));

    // NaN propagation through math functions
    expect(static_cast<bool>(std::isnan(sin(nan))));
    expect(static_cast<bool>(std::isnan(cos(nan))));
    expect(static_cast<bool>(std::isnan(sqrt(nan))));
    expect(static_cast<bool>(std::isnan(fract(nan))));

    // NaN in vector operations
    float3 nan_vec = make_float3(1.0f, nan, 3.0f);
    float3 result = nan_vec + make_float3(1.0f);
    expect(static_cast<bool>(!std::isnan(result.x)));
    expect(static_cast<bool>(std::isnan(result.y)));
    expect(static_cast<bool>(!std::isnan(result.z)));

    // dot product with NaN
    float dot_nan = dot(nan_vec, make_float3(1.0f, 1.0f, 1.0f));
    expect(static_cast<bool>(std::isnan(dot_nan)));

    // length with NaN
    float len_nan = length(nan_vec);
    expect(static_cast<bool>(std::isnan(len_nan)));
}

void test_inf_arithmetic() {
    float inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();

    // Inf arithmetic
    expect(static_cast<bool>(std::isinf(inf + 1.0f)));
    expect(static_cast<bool>(std::isinf(inf * 2.0f)));
    expect(static_cast<bool>(std::isnan(inf - inf)));
    expect(static_cast<bool>(std::isnan(inf + neg_inf)));
    expect(static_cast<bool>(std::isinf(inf * inf)));
    expect(static_cast<bool>(inf + 1.0f > 0.0f));
    expect(static_cast<bool>(neg_inf < 0.0f));

    // Inf / finite = inf
    expect(static_cast<bool>(std::isinf(inf / 2.0f)));
    // finite / inf = 0
    expect(static_cast<bool>(1.0f / inf == 0.0f));
    // inf / inf = NaN
    expect(static_cast<bool>(std::isnan(inf / inf)));

    // Inf in vector operations
    float3 inf_vec = make_float3(inf, 1.0f, neg_inf);
    bool3 isinf_result = isinf(inf_vec);
    expect(static_cast<bool>(isinf_result.x));
    expect(static_cast<bool>(!isinf_result.y));
    expect(static_cast<bool>(isinf_result.z));

    // clamp with inf
    expect(static_cast<bool>(clamp(inf, 0.0f, 10.0f) == 10.0f));
    expect(static_cast<bool>(clamp(neg_inf, 0.0f, 10.0f) == 0.0f));

    // min/max with inf
    expect(static_cast<bool>(min(inf, 1.0f) == 1.0f));
    expect(static_cast<bool>(max(neg_inf, 1.0f) == 1.0f));
}

void test_boundary_values() {
    float eps = std::numeric_limits<float>::epsilon();
    float denorm_min = std::numeric_limits<float>::denorm_min();
    float max_float = std::numeric_limits<float>::max();
    float min_float = std::numeric_limits<float>::min();

    // Epsilon behavior
    expect(static_cast<bool>(1.0f + eps != 1.0f));
    expect(static_cast<bool>(1.0f + eps / 2.0f == 1.0f));

    // Denormalized numbers
    expect(static_cast<bool>(denorm_min > 0.0f));
    expect(static_cast<bool>(denorm_min < min_float));

    // Max float
    expect(static_cast<bool>(!std::isinf(max_float)));
    expect(static_cast<bool>(std::isinf(max_float * 2.0f)));

    // abs of special values
    expect(static_cast<bool>(abs(-0.0f) == 0.0f));
    expect(static_cast<bool>(std::isinf(abs(-std::numeric_limits<float>::infinity()))));

    // sign of special values
    expect(static_cast<bool>(sign(std::numeric_limits<float>::epsilon()) == 1.0f));
    expect(static_cast<bool>(sign(-std::numeric_limits<float>::epsilon()) == -1.0f));

    // lerp boundary: t=0 and t=1 exact
    expect(static_cast<bool>(lerp(3.0f, 7.0f, 0.0f) == 3.0f));
    expect(static_cast<bool>(lerp(3.0f, 7.0f, 1.0f) == 7.0f));

    // fract of exact integers
    expect(static_cast<bool>(approx_eq(fract(1.0f), 0.0f)));
    expect(static_cast<bool>(approx_eq(fract(100.0f), 0.0f)));
    expect(static_cast<bool>(approx_eq(fract(-1.0f), 0.0f)));

    // next_pow2 edge cases
    expect(static_cast<bool>(next_pow2(0u) == 0u));
    expect(static_cast<bool>(next_pow2(1u) == 1u));
    expect(static_cast<bool>(next_pow2(0x80000000u) == 0x80000000u));

    // normalize zero-length vector (implementation-defined, but should not crash)
    // We just verify it doesn't produce NaN for unit vectors
    float3 unit_x = make_float3(1.0f, 0.0f, 0.0f);
    float3 norm_x = normalize(unit_x);
    expect(static_cast<bool>(approx_eq(norm_x.x, 1.0f) && approx_eq(norm_x.y, 0.0f) && approx_eq(norm_x.z, 0.0f)));

    // cross product anti-commutativity: a×b = -(b×a)
    float3 a = make_float3(1.0f, 2.0f, 3.0f);
    float3 b = make_float3(4.0f, 5.0f, 6.0f);
    float3 axb = cross(a, b);
    float3 bxa = cross(b, a);
    expect(static_cast<bool>(approx_eq(axb.x, -bxa.x) && approx_eq(axb.y, -bxa.y) && approx_eq(axb.z, -bxa.z)));

    // cross product of parallel vectors = zero
    float3 parallel = make_float3(2.0f, 4.0f, 6.0f);
    float3 cross_parallel = cross(a, parallel);
    expect(static_cast<bool>(approx_eq(cross_parallel.x, 0.0f) && approx_eq(cross_parallel.y, 0.0f) && approx_eq(cross_parallel.z, 0.0f)));

    // determinant of identity = 1
    expect(static_cast<bool>(approx_eq(determinant(float2x2{}), 1.0f)));
    expect(static_cast<bool>(approx_eq(determinant(float3x3{}), 1.0f)));
    expect(static_cast<bool>(approx_eq(determinant(float4x4{}), 1.0f)));

    // inverse of identity = identity
    float3x3 inv_id = inverse(float3x3{});
    expect(static_cast<bool>(approx_eq(inv_id[0][0], 1.0f) && approx_eq(inv_id[1][1], 1.0f) && approx_eq(inv_id[2][2], 1.0f)));
    expect(static_cast<bool>(approx_eq(inv_id[0][1], 0.0f) && approx_eq(inv_id[1][0], 0.0f)));

    // transpose of identity = identity
    float3x3 trans_id = transpose(float3x3{});
    expect(static_cast<bool>(approx_eq(trans_id[0][0], 1.0f) && approx_eq(trans_id[1][1], 1.0f) && approx_eq(trans_id[2][2], 1.0f)));

    // double-transpose = original
    float3x3 m3 = make_float3x3(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f);
    float3x3 tt = transpose(transpose(m3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            expect(static_cast<bool>(approx_eq(tt[i][j], m3[i][j])));
}

static auto test_mathematics_registration = [] {
    "test_ceil_div"_test = test_ceil_div;
    "test_next_pow2"_test = [] {
        LUISA_INFO("Testing next_pow2...");
        test_next_pow2();
    };
    "test_scalar_math"_test = [] {
        LUISA_INFO("Testing scalar math functions...");
        test_scalar_math();
    };
    "test_vector_unary_funcs"_test = [] {
        LUISA_INFO("Testing vector unary functions...");
        test_vector_unary_funcs();
    };
    "test_vector_binary_funcs"_test = [] {
        LUISA_INFO("Testing vector binary functions...");
        test_vector_binary_funcs();
    };
    "test_isnan_isinf"_test = [] {
        LUISA_INFO("Testing isnan/isinf...");
        test_isnan_isinf();
    };
    "test_select"_test = [] {
        LUISA_INFO("Testing select...");
        test_select();
    };
    "test_lerp_clamp"_test = [] {
        LUISA_INFO("Testing lerp/clamp...");
        test_lerp_clamp();
    };
    "test_vector_operations"_test = [] {
        LUISA_INFO("Testing vector operations...");
        test_vector_operations();
    };
    "test_matrix_transpose"_test = [] {
        LUISA_INFO("Testing matrix transpose...");
        test_matrix_transpose();
    };
    "test_matrix_inverse"_test = [] {
        LUISA_INFO("Testing matrix inverse...");
        test_matrix_inverse();
    };
    "test_matrix_determinant"_test = [] {
        LUISA_INFO("Testing matrix determinant...");
        test_matrix_determinant();
    };
    "test_transform_functions"_test = [] {
        LUISA_INFO("Testing transform functions...");
        test_transform_functions();
    };
    "test_sign"_test = [] {
        LUISA_INFO("Testing sign...");
        test_sign();
    };
    "test_fma"_test = [] {
        LUISA_INFO("Testing fma...");
        test_fma();
    };
    "test_double_precision"_test = [] {
        LUISA_INFO("Testing double precision...");
        test_double_precision();
        LUISA_INFO("All mathematics tests passed!");
    };
    "test_nan_propagation"_test = [] {
        LUISA_INFO("Testing NaN propagation...");
        test_nan_propagation();
    };
    "test_inf_arithmetic"_test = [] {
        LUISA_INFO("Testing inf arithmetic...");
        test_inf_arithmetic();
    };
    "test_boundary_values"_test = [] {
        LUISA_INFO("Testing boundary values...");
        test_boundary_values();
    };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}

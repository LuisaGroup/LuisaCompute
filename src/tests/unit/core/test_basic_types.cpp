// Test for basic_types.h
//
// This test verifies vector and matrix types, their constructors,
// operators, and utility functions defined in luisa/core/basic_types.h
//

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include "ut/ut.hpp"
#include <limits>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

// Helper to check float equality with epsilon
bool float_eq(float a, float b, float eps = std::numeric_limits<float>::epsilon() * 100) {
    return abs(a - b) <= eps;
}

// Helper to check double equality with epsilon
bool double_eq(double a, double b, double eps = std::numeric_limits<double>::epsilon() * 100) {
    return abs(a - b) <= eps;
}

void test_vector_construction() {
    // Test single value construction
    float2 f2(1.0f);
    expect(static_cast<bool>(f2.x == 1.0f && f2.y == 1.0f));

    float3 f3(2.0f);
    expect(static_cast<bool>(f3.x == 2.0f && f3.y == 2.0f && f3.z == 2.0f));

    float4 f4(3.0f);
    expect(static_cast<bool>(f4.x == 3.0f && f4.y == 3.0f && f4.z == 3.0f && f4.w == 3.0f));

    // Test component-wise construction
    int2 i2(1, 2);
    expect(static_cast<bool>(i2.x == 1 && i2.y == 2));

    int3 i3(1, 2, 3);
    expect(static_cast<bool>(i3.x == 1 && i3.y == 2 && i3.z == 3));

    int4 i4(1, 2, 3, 4);
    expect(static_cast<bool>(i4.x == 1 && i4.y == 2 && i4.z == 3 && i4.w == 4));

    // Test zero() and one()
    auto z2 = float2::zero();
    expect(static_cast<bool>(z2.x == 0.0f && z2.y == 0.0f));

    auto o2 = float2::one();
    expect(static_cast<bool>(o2.x == 1.0f && o2.y == 1.0f));

    auto z4 = int4::zero();
    expect(static_cast<bool>(z4.x == 0 && z4.y == 0 && z4.z == 0 && z4.w == 0));

    auto o4 = uint4::one();
    expect(static_cast<bool>(o4.x == 1 && o4.y == 1 && o4.z == 1 && o4.w == 1));

    LUISA_INFO("test_vector_construction passed");
}

void test_vector_element_access() {
    float3 v(1.0f, 2.0f, 3.0f);

    // Test operator[]
    expect(static_cast<bool>(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f));

    // Test modification through operator[]
    v[1] = 5.0f;
    expect(static_cast<bool>(v.y == 5.0f));

    // Test const access
    const float4 cv(1.0f, 2.0f, 3.0f, 4.0f);
    expect(static_cast<bool>(cv[0] == 1.0f && cv[3] == 4.0f));

    LUISA_INFO("test_vector_element_access passed");
}

void test_vector_unary_operators() {
    // Test unary minus
    float2 f2(1.0f, -2.0f);
    float2 nf2 = -f2;
    expect(static_cast<bool>(nf2.x == -1.0f && nf2.y == 2.0f));

    float3 f3(1.0f, 2.0f, -3.0f);
    float3 nf3 = -f3;
    expect(static_cast<bool>(nf3.x == -1.0f && nf3.y == -2.0f && nf3.z == 3.0f));

    // Test unary plus
    float2 pf2 = +f2;
    expect(static_cast<bool>(pf2.x == 1.0f && pf2.y == -2.0f));

    // Test unary not (!)
    bool2 b2(true, false);
    bool2 nb2 = !b2;
    expect(static_cast<bool>(nb2.x == false && nb2.y == true));

    bool3 b3(true, false, true);
    bool3 nb3 = !b3;
    expect(static_cast<bool>(nb3.x == false && nb3.y == true && nb3.z == false));

    // Test unary xor (~) for integral types
    int2 i2(0x0F, 0xF0);
    int2 xi2 = ~i2;
    expect(static_cast<bool>(xi2.x == ~0x0F && xi2.y == ~0xF0));

    LUISA_INFO("test_vector_unary_operators passed");
}

void test_vector_binary_operators() {
    // Test arithmetic operators
    float2 a(1.0f, 2.0f);
    float2 b(3.0f, 4.0f);

    float2 sum = a + b;
    expect(static_cast<bool>(float_eq(sum.x, 4.0f) && float_eq(sum.y, 6.0f)));

    float2 diff = b - a;
    expect(static_cast<bool>(float_eq(diff.x, 2.0f) && float_eq(diff.y, 2.0f)));

    float2 prod = a * b;
    expect(static_cast<bool>(float_eq(prod.x, 3.0f) && float_eq(prod.y, 8.0f)));

    float2 quot = b / a;
    expect(static_cast<bool>(float_eq(quot.x, 3.0f) && float_eq(quot.y, 2.0f)));

    // Test scalar operators
    float2 scaled = a * 2.0f;
    expect(static_cast<bool>(float_eq(scaled.x, 2.0f) && float_eq(scaled.y, 4.0f)));

    float2 scaled2 = 3.0f * a;
    expect(static_cast<bool>(float_eq(scaled2.x, 3.0f) && float_eq(scaled2.y, 6.0f)));

    // Test integral operators
    int3 i1(10, 20, 30);
    int3 i2(3, 5, 7);

    int3 mod = i1 % i2;
    expect(static_cast<bool>(mod.x == 1 && mod.y == 0 && mod.z == 2));

    int3 shifted = i1 << 1;
    expect(static_cast<bool>(shifted.x == 20 && shifted.y == 40 && shifted.z == 60));

    int3 shifted_r = i1 >> 1;
    expect(static_cast<bool>(shifted_r.x == 5 && shifted_r.y == 10 && shifted_r.z == 15));

    // Test bitwise operators
    uint2 u1(0x0F, 0xF0);
    uint2 u2(0xFF, 0x00);

    uint2 uor = u1 | u2;
    expect(static_cast<bool>(uor.x == 0xFF && uor.y == 0xF0));

    uint2 uand = u1 & u2;
    expect(static_cast<bool>(uand.x == 0x0F && uand.y == 0x00));

    uint2 uxor = u1 ^ u2;
    expect(static_cast<bool>(uxor.x == 0xF0 && uxor.y == 0xF0));

    LUISA_INFO("test_vector_binary_operators passed");
}

void test_vector_logic_operators() {
    // Test comparison operators
    float2 a(1.0f, 2.0f);
    float2 b(1.0f, 3.0f);

    bool2 eq = (a == b);
    expect(static_cast<bool>(eq.x == true && eq.y == false));

    bool2 neq = (a != b);
    expect(static_cast<bool>(neq.x == false && neq.y == true));

    bool2 lt = (a < b);
    expect(static_cast<bool>(lt.x == false && lt.y == true));

    bool2 gt = (b > a);
    expect(static_cast<bool>(gt.x == false && gt.y == true));

    bool2 le = (a <= b);
    expect(static_cast<bool>(le.x == true && le.y == true));

    bool2 ge = (b >= a);
    expect(static_cast<bool>(ge.x == true && ge.y == true));

    // Test boolean logic operators
    bool2 b1(true, false);
    bool2 b2(true, true);

    bool2 bor = b1 || b2;
    expect(static_cast<bool>(bor.x == true && bor.y == true));

    bool2 band = b1 && b2;
    expect(static_cast<bool>(band.x == true && band.y == false));

    LUISA_INFO("test_vector_logic_operators passed");
}

void test_vector_assignment_operators() {
    float2 a(1.0f, 2.0f);
    float2 b(3.0f, 4.0f);

    a += b;
    expect(static_cast<bool>(float_eq(a.x, 4.0f) && float_eq(a.y, 6.0f)));

    a -= b;
    expect(static_cast<bool>(float_eq(a.x, 1.0f) && float_eq(a.y, 2.0f)));

    a *= b;
    expect(static_cast<bool>(float_eq(a.x, 3.0f) && float_eq(a.y, 8.0f)));

    a /= b;
    expect(static_cast<bool>(float_eq(a.x, 1.0f) && float_eq(a.y, 2.0f)));

    // Test integral assignment operators
    int2 i(10, 20);
    int2 j(3, 5);

    i %= j;
    expect(static_cast<bool>(i.x == 1 && i.y == 0));

    i = int2(4, 8);
    i <<= 1;
    expect(static_cast<bool>(i.x == 8 && i.y == 16));

    i >>= 2;
    expect(static_cast<bool>(i.x == 2 && i.y == 4));

    uint2 u(0x0F, 0xF0);
    u |= uint2(0xF0, 0x0F);
    expect(static_cast<bool>(u.x == 0xFF && u.y == 0xFF));

    u = uint2(0xFF, 0xFF);
    u &= uint2(0x0F, 0xF0);
    expect(static_cast<bool>(u.x == 0x0F && u.y == 0xF0));

    u = uint2(0xFF, 0x00);
    u ^= uint2(0x0F, 0xF0);
    expect(static_cast<bool>(u.x == 0xF0 && u.y == 0xF0));

    LUISA_INFO("test_vector_assignment_operators passed");
}

void test_bool_vector_functions() {
    bool2 b2_true(true, true);
    bool2 b2_mixed(true, false);
    bool2 b2_false(false, false);

    expect(static_cast<bool>(any(b2_true) == true));
    expect(static_cast<bool>(any(b2_mixed) == true));
    expect(static_cast<bool>(any(b2_false) == false));

    expect(static_cast<bool>(all(b2_true) == true));
    expect(static_cast<bool>(all(b2_mixed) == false));
    expect(static_cast<bool>(all(b2_false) == false));

    expect(static_cast<bool>(luisa::none(b2_true) == false));
    expect(static_cast<bool>(luisa::none(b2_mixed) == false));
    expect(static_cast<bool>(luisa::none(b2_false) == true));

    bool3 b3_true(true, true, true);
    bool3 b3_mixed(true, false, true);
    bool3 b3_false(false, false, false);

    expect(static_cast<bool>(any(b3_true) == true));
    expect(static_cast<bool>(any(b3_mixed) == true));
    expect(static_cast<bool>(any(b3_false) == false));

    expect(static_cast<bool>(all(b3_true) == true));
    expect(static_cast<bool>(all(b3_mixed) == false));
    expect(static_cast<bool>(all(b3_false) == false));

    bool4 b4_true(true, true, true, true);
    bool4 b4_mixed(true, false, true, false);
    bool4 b4_false(false, false, false, false);

    expect(static_cast<bool>(any(b4_true) == true));
    expect(static_cast<bool>(any(b4_mixed) == true));
    expect(static_cast<bool>(any(b4_false) == false));

    expect(static_cast<bool>(all(b4_true) == true));
    expect(static_cast<bool>(all(b4_mixed) == false));
    expect(static_cast<bool>(all(b4_false) == false));

    LUISA_INFO("test_bool_vector_functions passed");
}

void test_matrix_construction() {
    // Test default construction (identity matrix)
    float2x2 m2;
    expect(static_cast<bool>(m2[0][0] == 1.0f && m2[0][1] == 0.0f));
    expect(static_cast<bool>(m2[1][0] == 0.0f && m2[1][1] == 1.0f));

    float3x3 m3;
    expect(static_cast<bool>(m3[0][0] == 1.0f && m3[1][1] == 1.0f && m3[2][2] == 1.0f));
    expect(static_cast<bool>(m3[0][1] == 0.0f && m3[1][0] == 0.0f));

    float4x4 m4;
    expect(static_cast<bool>(m4[0][0] == 1.0f && m4[1][1] == 1.0f && m4[2][2] == 1.0f && m4[3][3] == 1.0f));

    // Test column vector construction
    float2x2 m2c(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    expect(static_cast<bool>(m2c[0][0] == 1.0f && m2c[0][1] == 2.0f));
    expect(static_cast<bool>(m2c[1][0] == 3.0f && m2c[1][1] == 4.0f));

    // Test eye() and fill()
    float2x2 m2eye = float2x2::eye(2.0f);
    expect(static_cast<bool>(m2eye[0][0] == 2.0f && m2eye[1][1] == 2.0f));
    expect(static_cast<bool>(m2eye[0][1] == 0.0f && m2eye[1][0] == 0.0f));

    float2x2 m2fill = float2x2::fill(3.0f);
    expect(static_cast<bool>(m2fill[0][0] == 3.0f && m2fill[0][1] == 3.0f));
    expect(static_cast<bool>(m2fill[1][0] == 3.0f && m2fill[1][1] == 3.0f));

    LUISA_INFO("test_matrix_construction passed");
}

void test_matrix_element_access() {
    float3x3 m(float3(1.0f, 2.0f, 3.0f), float3(4.0f, 5.0f, 6.0f), float3(7.0f, 8.0f, 9.0f));

    // Test column access
    expect(static_cast<bool>(m[0][0] == 1.0f && m[0][1] == 2.0f && m[0][2] == 3.0f));
    expect(static_cast<bool>(m[1][0] == 4.0f && m[1][1] == 5.0f && m[1][2] == 6.0f));
    expect(static_cast<bool>(m[2][0] == 7.0f && m[2][1] == 8.0f && m[2][2] == 9.0f));

    // Test column modification
    m[1] = float3(10.0f, 11.0f, 12.0f);
    expect(static_cast<bool>(m[1][0] == 10.0f && m[1][1] == 11.0f && m[1][2] == 12.0f));

    LUISA_INFO("test_matrix_element_access passed");
}

void test_matrix_scalar_operators() {
    float2x2 m(float2(1.0f, 2.0f), float2(3.0f, 4.0f));

    // Test scalar multiplication
    float2x2 scaled = m * 2.0f;
    expect(static_cast<bool>(float_eq(scaled[0][0], 2.0f) && float_eq(scaled[0][1], 4.0f)));
    expect(static_cast<bool>(float_eq(scaled[1][0], 6.0f) && float_eq(scaled[1][1], 8.0f)));

    float2x2 scaled2 = 3.0f * m;
    expect(static_cast<bool>(float_eq(scaled2[0][0], 3.0f) && float_eq(scaled2[0][1], 6.0f)));
    expect(static_cast<bool>(float_eq(scaled2[1][0], 9.0f) && float_eq(scaled2[1][1], 12.0f)));

    // Test scalar division
    float2x2 divided = m / 2.0f;
    expect(static_cast<bool>(float_eq(divided[0][0], 0.5f) && float_eq(divided[0][1], 1.0f)));
    expect(static_cast<bool>(float_eq(divided[1][0], 1.5f) && float_eq(divided[1][1], 2.0f)));

    LUISA_INFO("test_matrix_scalar_operators passed");
}

void test_matrix_vector_multiplication() {
    // Test float2x2 * float2
    float2x2 m2(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    float2 v2(1.0f, 1.0f);
    float2 r2 = m2 * v2;
    // m2 * v2 = v2.x * m2[0] + v2.y * m2[1] = 1 * (1,2) + 1 * (3,4) = (4,6)
    expect(static_cast<bool>(float_eq(r2.x, 4.0f) && float_eq(r2.y, 6.0f)));

    // Test float3x3 * float3
    float3x3 m3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 2.0f, 0.0f), float3(0.0f, 0.0f, 3.0f));
    float3 v3(1.0f, 1.0f, 1.0f);
    float3 r3 = m3 * v3;
    expect(static_cast<bool>(float_eq(r3.x, 1.0f) && float_eq(r3.y, 2.0f) && float_eq(r3.z, 3.0f)));

    // Test float4x4 * float4
    float4x4 m4(float4(1.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 2.0f, 0.0f, 0.0f),
                float4(0.0f, 0.0f, 3.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 4.0f));
    float4 v4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 r4 = m4 * v4;
    expect(static_cast<bool>(float_eq(r4.x, 1.0f) && float_eq(r4.y, 2.0f) && float_eq(r4.z, 3.0f) && float_eq(r4.w, 4.0f)));

    LUISA_INFO("test_matrix_vector_multiplication passed");
}

void test_matrix_matrix_multiplication() {
    // Test float2x2 * float2x2
    float2x2 a(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    float2x2 b(float2(5.0f, 6.0f), float2(7.0f, 8.0f));
    float2x2 c = a * b;
    // a * b = [a * b[0], a * b[1]]
    // a * b[0] = 5*(1,2) + 6*(3,4) = (5+18, 10+24) = (23, 34)
    // a * b[1] = 7*(1,2) + 8*(3,4) = (7+24, 14+32) = (31, 46)
    expect(static_cast<bool>(float_eq(c[0][0], 23.0f) && float_eq(c[0][1], 34.0f)));
    expect(static_cast<bool>(float_eq(c[1][0], 31.0f) && float_eq(c[1][1], 46.0f)));

    // Test identity multiplication
    float2x2 i;
    float2x2 r = a * i;
    expect(static_cast<bool>(float_eq(r[0][0], a[0][0]) && float_eq(r[0][1], a[0][1])));
    expect(static_cast<bool>(float_eq(r[1][0], a[1][0]) && float_eq(r[1][1], a[1][1])));

    LUISA_INFO("test_matrix_matrix_multiplication passed");
}

void test_matrix_addition_subtraction() {
    float2x2 a(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    float2x2 b(float2(5.0f, 6.0f), float2(7.0f, 8.0f));

    float2x2 sum = a + b;
    expect(static_cast<bool>(float_eq(sum[0][0], 6.0f) && float_eq(sum[0][1], 8.0f)));
    expect(static_cast<bool>(float_eq(sum[1][0], 10.0f) && float_eq(sum[1][1], 12.0f)));

    float2x2 diff = b - a;
    expect(static_cast<bool>(float_eq(diff[0][0], 4.0f) && float_eq(diff[0][1], 4.0f)));
    expect(static_cast<bool>(float_eq(diff[1][0], 4.0f) && float_eq(diff[1][1], 4.0f)));

    LUISA_INFO("test_matrix_addition_subtraction passed");
}

void test_make_functions() {
    // Test make_float2 variations
    float2 f2a = make_float2(1.0f);
    expect(static_cast<bool>(f2a.x == 1.0f && f2a.y == 1.0f));

    float2 f2b = make_float2(1.0f, 2.0f);
    expect(static_cast<bool>(f2b.x == 1.0f && f2b.y == 2.0f));

    float2 f2c = make_float2(float3(1.0f, 2.0f, 3.0f));
    expect(static_cast<bool>(f2c.x == 1.0f && f2c.y == 2.0f));

    float2 f2d = make_float2(float4(1.0f, 2.0f, 3.0f, 4.0f));
    expect(static_cast<bool>(f2d.x == 1.0f && f2d.y == 2.0f));

    // Test make_float3 variations
    float3 f3a = make_float3(1.0f);
    expect(static_cast<bool>(f3a.x == 1.0f && f3a.y == 1.0f && f3a.z == 1.0f));

    float3 f3b = make_float3(1.0f, 2.0f, 3.0f);
    expect(static_cast<bool>(f3b.x == 1.0f && f3b.y == 2.0f && f3b.z == 3.0f));

    float3 f3c = make_float3(float2(1.0f, 2.0f), 3.0f);
    expect(static_cast<bool>(f3c.x == 1.0f && f3c.y == 2.0f && f3c.z == 3.0f));

    float3 f3d = make_float3(1.0f, float2(2.0f, 3.0f));
    expect(static_cast<bool>(f3d.x == 1.0f && f3d.y == 2.0f && f3d.z == 3.0f));

    float3 f3e = make_float3(float4(1.0f, 2.0f, 3.0f, 4.0f));
    expect(static_cast<bool>(f3e.x == 1.0f && f3e.y == 2.0f && f3e.z == 3.0f));

    // Test make_float4 variations
    float4 f4a = make_float4(1.0f);
    expect(static_cast<bool>(f4a.x == 1.0f && f4a.y == 1.0f && f4a.z == 1.0f && f4a.w == 1.0f));

    float4 f4b = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    expect(static_cast<bool>(f4b.x == 1.0f && f4b.y == 2.0f && f4b.z == 3.0f && f4b.w == 4.0f));

    float4 f4c = make_float4(float2(1.0f, 2.0f), 3.0f, 4.0f);
    expect(static_cast<bool>(f4c.x == 1.0f && f4c.y == 2.0f && f4c.z == 3.0f && f4c.w == 4.0f));

    float4 f4d = make_float4(float3(1.0f, 2.0f, 3.0f), 4.0f);
    expect(static_cast<bool>(f4d.x == 1.0f && f4d.y == 2.0f && f4d.z == 3.0f && f4d.w == 4.0f));

    float4 f4e = make_float4(1.0f, float3(2.0f, 3.0f, 4.0f));
    expect(static_cast<bool>(f4e.x == 1.0f && f4e.y == 2.0f && f4e.z == 3.0f && f4e.w == 4.0f));

    float4 f4f = make_float4(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    expect(static_cast<bool>(f4f.x == 1.0f && f4f.y == 2.0f && f4f.z == 3.0f && f4f.w == 4.0f));

    // Test make_float2x2
    float2x2 m2a = make_float2x2(2.0f);
    expect(static_cast<bool>(m2a[0][0] == 2.0f && m2a[1][1] == 2.0f));
    expect(static_cast<bool>(m2a[0][1] == 0.0f && m2a[1][0] == 0.0f));

    float2x2 m2b = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
    expect(static_cast<bool>(m2b[0][0] == 1.0f && m2b[0][1] == 2.0f));
    expect(static_cast<bool>(m2b[1][0] == 3.0f && m2b[1][1] == 4.0f));

    float2x2 m2c = make_float2x2(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
    expect(static_cast<bool>(m2c[0][0] == 1.0f && m2c[1][0] == 3.0f));

    // Test make_float3x3
    float3x3 m3a = make_float3x3(2.0f);
    expect(static_cast<bool>(m3a[0][0] == 2.0f && m3a[1][1] == 2.0f && m3a[2][2] == 2.0f));

    float3x3 m3b = make_float3x3(
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f);
    expect(static_cast<bool>(m3b[0][0] == 1.0f && m3b[0][1] == 2.0f && m3b[0][2] == 3.0f));
    expect(static_cast<bool>(m3b[1][0] == 4.0f && m3b[1][1] == 5.0f && m3b[1][2] == 6.0f));
    expect(static_cast<bool>(m3b[2][0] == 7.0f && m3b[2][1] == 8.0f && m3b[2][2] == 9.0f));

    // Test make_float4x4
    float4x4 m4a = make_float4x4(2.0f);
    expect(static_cast<bool>(m4a[0][0] == 2.0f && m4a[1][1] == 2.0f && m4a[2][2] == 2.0f && m4a[3][3] == 2.0f));

    float4x4 m4b = make_float4x4(
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f);
    expect(static_cast<bool>(m4b[0][0] == 1.0f && m4b[0][3] == 4.0f));
    expect(static_cast<bool>(m4b[3][0] == 13.0f && m4b[3][3] == 16.0f));

    LUISA_INFO("test_make_functions passed");
}

void test_type_aliases() {
    // Test that all vector type aliases exist and work
    bool2 b2;
    bool3 b3;
    bool4 b4;
    short2 s2;
    short3 s3;
    short4 s4;
    ushort2 us2;
    ushort3 us3;
    ushort4 us4;
    byte2 by2;
    byte3 by3;
    byte4 by4;
    ubyte2 ub2;
    ubyte3 ub3;
    ubyte4 ub4;
    int2 i2;
    int3 i3;
    int4 i4;
    uint2 u2;
    uint3 u3;
    uint4 u4;
    slong2 l2;
    slong3 l3;
    slong4 l4;
    ulong2 ul2;
    ulong3 ul3;
    ulong4 ul4;
    half2 h2;
    half3 h3;
    half4 h4;
    float2 f2;
    float3 f3;
    float4 f4;
    double2 d2;
    double3 d3;
    double4 d4;

    // Test that matrix type aliases exist and work
    float2x2 f22;
    float3x3 f33;
    float4x4 f44;
    double2x2 d22;
    double3x3 d33;
    double4x4 d44;
    half2x2 h22;
    half3x3 h33;
    half4x4 h44;

    LUISA_INFO("test_type_aliases passed");
}

void test_swizzle_operations() {
    // float2 swizzles
    float2 f2(1.0f, 2.0f);
    float2 f2_xx = f2.xx();
    expect(static_cast<bool>(f2_xx.x == 1.0f && f2_xx.y == 1.0f));
    float2 f2_yx = f2.yx();
    expect(static_cast<bool>(f2_yx.x == 2.0f && f2_yx.y == 1.0f));
    float2 f2_yy = f2.yy();
    expect(static_cast<bool>(f2_yy.x == 2.0f && f2_yy.y == 2.0f));
    // float2 -> float3 swizzles
    float3 f2_xxy = f2.xxy();
    expect(static_cast<bool>(f2_xxy.x == 1.0f && f2_xxy.y == 1.0f && f2_xxy.z == 2.0f));
    float3 f2_yyx = f2.yyx();
    expect(static_cast<bool>(f2_yyx.x == 2.0f && f2_yyx.y == 2.0f && f2_yyx.z == 1.0f));
    // float2 -> float4 swizzles
    float4 f2_xyxy = f2.xyxy();
    expect(static_cast<bool>(f2_xyxy.x == 1.0f && f2_xyxy.y == 2.0f && f2_xyxy.z == 1.0f && f2_xyxy.w == 2.0f));
    float4 f2_yyyy = f2.yyyy();
    expect(static_cast<bool>(f2_yyyy.x == 2.0f && f2_yyyy.y == 2.0f && f2_yyyy.z == 2.0f && f2_yyyy.w == 2.0f));

    // float3 swizzles
    float3 f3(1.0f, 2.0f, 3.0f);
    float2 f3_xz = f3.xz();
    expect(static_cast<bool>(f3_xz.x == 1.0f && f3_xz.y == 3.0f));
    float2 f3_zy = f3.zy();
    expect(static_cast<bool>(f3_zy.x == 3.0f && f3_zy.y == 2.0f));
    float3 f3_zyx = f3.zyx();
    expect(static_cast<bool>(f3_zyx.x == 3.0f && f3_zyx.y == 2.0f && f3_zyx.z == 1.0f));
    float3 f3_xyz = f3.xyz();
    expect(static_cast<bool>(f3_xyz.x == 1.0f && f3_xyz.y == 2.0f && f3_xyz.z == 3.0f));
    float4 f3_xyzx = f3.xyzx();
    expect(static_cast<bool>(f3_xyzx.x == 1.0f && f3_xyzx.y == 2.0f && f3_xyzx.z == 3.0f && f3_xyzx.w == 1.0f));
    float4 f3_zzzy = f3.zzzy();
    expect(static_cast<bool>(f3_zzzy.x == 3.0f && f3_zzzy.y == 3.0f && f3_zzzy.z == 3.0f && f3_zzzy.w == 2.0f));

    // float4 swizzles
    float4 f4(1.0f, 2.0f, 3.0f, 4.0f);
    float2 f4_xw = f4.xw();
    expect(static_cast<bool>(f4_xw.x == 1.0f && f4_xw.y == 4.0f));
    float2 f4_wz = f4.wz();
    expect(static_cast<bool>(f4_wz.x == 4.0f && f4_wz.y == 3.0f));
    float3 f4_xyz = f4.xyz();
    expect(static_cast<bool>(f4_xyz.x == 1.0f && f4_xyz.y == 2.0f && f4_xyz.z == 3.0f));
    float3 f4_wzy = f4.wzy();
    expect(static_cast<bool>(f4_wzy.x == 4.0f && f4_wzy.y == 3.0f && f4_wzy.z == 2.0f));
    float4 f4_wzyx = f4.wzyx();
    expect(static_cast<bool>(f4_wzyx.x == 4.0f && f4_wzyx.y == 3.0f && f4_wzyx.z == 2.0f && f4_wzyx.w == 1.0f));
    float4 f4_xxxx = f4.xxxx();
    expect(static_cast<bool>(f4_xxxx.x == 1.0f && f4_xxxx.y == 1.0f && f4_xxxx.z == 1.0f && f4_xxxx.w == 1.0f));
    float4 f4_xywz = f4.xywz();
    expect(static_cast<bool>(f4_xywz.x == 1.0f && f4_xywz.y == 2.0f && f4_xywz.z == 4.0f && f4_xywz.w == 3.0f));

    // int vector swizzles
    int3 i3(10, 20, 30);
    int2 i3_zx = i3.zx();
    expect(static_cast<bool>(i3_zx.x == 30 && i3_zx.y == 10));
    int4 i3_yzxz = i3.yzxz();
    expect(static_cast<bool>(i3_yzxz.x == 20 && i3_yzxz.y == 30 && i3_yzxz.z == 10 && i3_yzxz.w == 30));

    // bool vector swizzles
    bool3 b3(true, false, true);
    bool2 b3_xz = b3.xz();
    expect(static_cast<bool>(b3_xz.x == true && b3_xz.y == true));
    bool2 b3_yz = b3.yz();
    expect(static_cast<bool>(b3_yz.x == false && b3_yz.y == true));

    LUISA_INFO("test_swizzle_operations passed");
}

void test_cross_type_construction() {
    // float -> int construction via make_
    float3 f3(1.5f, 2.7f, 3.9f);
    int3 i3 = make_int3(f3);
    expect(static_cast<bool>(i3.x == 1 && i3.y == 2 && i3.z == 3));

    // int -> float construction via make_
    int2 i2(3, 4);
    float2 f2 = make_float2(i2);
    expect(static_cast<bool>(f2.x == 3.0f && f2.y == 4.0f));

    // uint -> int construction
    uint3 u3(10u, 20u, 30u);
    int3 i3_from_u = make_int3(u3);
    expect(static_cast<bool>(i3_from_u.x == 10 && i3_from_u.y == 20 && i3_from_u.z == 30));

    // int -> uint construction
    int4 i4(1, 2, 3, 4);
    uint4 u4 = make_uint4(i4);
    expect(static_cast<bool>(u4.x == 1u && u4.y == 2u && u4.z == 3u && u4.w == 4u));

    // float -> double construction
    float2 f2d(1.5f, 2.5f);
    double2 d2 = make_double2(f2d);
    expect(static_cast<bool>(d2.x == static_cast<double>(1.5f) && d2.y == static_cast<double>(2.5f)));

    // double -> float construction
    double3 d3(1.0, 2.0, 3.0);
    float3 f3_from_d = make_float3(d3);
    expect(static_cast<bool>(f3_from_d.x == 1.0f && f3_from_d.y == 2.0f && f3_from_d.z == 3.0f));

    // Construct higher-dimension from lower + scalars (already tested, add more edge cases)
    float4 f4_from_f3 = make_float4(make_float3(0.0f), 1.0f);
    expect(static_cast<bool>(f4_from_f3.x == 0.0f && f4_from_f3.y == 0.0f && f4_from_f3.z == 0.0f && f4_from_f3.w == 1.0f));

    float4 f4_from_scalar_f3 = make_float4(1.0f, make_float3(0.0f));
    expect(static_cast<bool>(f4_from_scalar_f3.x == 1.0f && f4_from_scalar_f3.y == 0.0f && f4_from_scalar_f3.z == 0.0f && f4_from_scalar_f3.w == 0.0f));

    // Edge case: max/min values
    int2 max_int = make_int2(std::numeric_limits<int>::max());
    expect(static_cast<bool>(max_int.x == std::numeric_limits<int>::max() && max_int.y == std::numeric_limits<int>::max()));

    int2 min_int = make_int2(std::numeric_limits<int>::min());
    expect(static_cast<bool>(min_int.x == std::numeric_limits<int>::min() && min_int.y == std::numeric_limits<int>::min()));

    uint2 max_uint = make_uint2(std::numeric_limits<uint>::max());
    expect(static_cast<bool>(max_uint.x == std::numeric_limits<uint>::max()));

    // Edge case: zero and negative zero
    float2 neg_zero = make_float2(-0.0f);
    float2 pos_zero = make_float2(0.0f);
    // -0.0f and 0.0f compare equal
    expect(static_cast<bool>(neg_zero.x == pos_zero.x && neg_zero.y == pos_zero.y));

    LUISA_INFO("test_cross_type_construction passed");
}

void test_double_matrix_operations() {
    // Test double matrix scalar multiplication
    double2x2 m(double2(1.0, 2.0), double2(3.0, 4.0));
    double2x2 scaled = m * 2.0;
    expect(static_cast<bool>(double_eq(scaled[0][0], 2.0) && double_eq(scaled[0][1], 4.0)));
    expect(static_cast<bool>(double_eq(scaled[1][0], 6.0) && double_eq(scaled[1][1], 8.0)));

    double2x2 scaled2 = 3.0 * m;
    expect(static_cast<bool>(double_eq(scaled2[0][0], 3.0)));

    // Test double matrix-vector multiplication
    double2 v(1.0, 1.0);
    double2 r = m * v;
    expect(static_cast<bool>(double_eq(r.x, 4.0) && double_eq(r.y, 6.0)));

    // Test double matrix-matrix multiplication
    double2x2 a(double2(1.0, 2.0), double2(3.0, 4.0));
    double2x2 b(double2(5.0, 6.0), double2(7.0, 8.0));
    double2x2 c = a * b;
    expect(static_cast<bool>(double_eq(c[0][0], 23.0) && double_eq(c[0][1], 34.0)));
    expect(static_cast<bool>(double_eq(c[1][0], 31.0) && double_eq(c[1][1], 46.0)));

    // Test double matrix addition
    double2x2 sum = a + b;
    expect(static_cast<bool>(double_eq(sum[0][0], 6.0)));

    // Test double matrix subtraction
    double2x2 diff = b - a;
    expect(static_cast<bool>(double_eq(diff[0][0], 4.0)));

    LUISA_INFO("test_double_matrix_operations passed");
}

static auto test_basic_types_registration = [] {
    "test_vector_construction"_test = [] { test_vector_construction(); };
    "test_vector_element_access"_test = [] { test_vector_element_access(); };
    "test_vector_unary_operators"_test = [] { test_vector_unary_operators(); };
    "test_vector_binary_operators"_test = [] { test_vector_binary_operators(); };
    "test_vector_logic_operators"_test = [] { test_vector_logic_operators(); };
    "test_vector_assignment_operators"_test = [] { test_vector_assignment_operators(); };
    "test_bool_vector_functions"_test = [] { test_bool_vector_functions(); };
    "test_matrix_construction"_test = [] { test_matrix_construction(); };
    "test_matrix_element_access"_test = [] { test_matrix_element_access(); };
    "test_matrix_scalar_operators"_test = [] { test_matrix_scalar_operators(); };
    "test_matrix_vector_multiplication"_test = [] { test_matrix_vector_multiplication(); };
    "test_matrix_matrix_multiplication"_test = [] { test_matrix_matrix_multiplication(); };
    "test_matrix_addition_subtraction"_test = [] { test_matrix_addition_subtraction(); };
    "test_make_functions"_test = [] { test_make_functions(); };
    "test_type_aliases"_test = [] { test_type_aliases(); };
    "test_swizzle_operations"_test = [] { test_swizzle_operations(); };
    "test_cross_type_construction"_test = [] { test_cross_type_construction(); };
    "test_double_matrix_operations"_test = [] { test_double_matrix_operations(); };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}

#include <luisa/core/basic_traits.h>
#include <luisa/core/basic_types.h>
#include "ut/ut.hpp"
#include <type_traits>
#include <cstdint>

using namespace boost::ut;
using namespace boost::ut::literals;

// Test always_false and always_true
static_assert(!luisa::always_false_v<>, "always_false_v<> should be false");
static_assert(!luisa::always_false_v<int>, "always_false_v<int> should be false");
static_assert(!luisa::always_false_v<int, float>, "always_false_v<int, float> should be false");
static_assert(luisa::always_true_v<>, "always_true_v<> should be true");
static_assert(luisa::always_true_v<int>, "always_true_v<int> should be true");
static_assert(luisa::always_true_v<int, float>, "always_true_v<int, float> should be true");

// Test to_underlying
enum class TestEnum : int { A = 1,
                            B = 2 };
static_assert(luisa::to_underlying(TestEnum::A) == 1, "to_underlying should return underlying value");
static_assert(luisa::to_underlying(TestEnum::B) == 2, "to_underlying should return underlying value");

// Test half type properties
static_assert(sizeof(luisa::half) == 2u, "half should be 2 bytes");
static_assert(alignof(luisa::half) == 2u, "half should be 2-byte aligned");
// NOTE: half_float::half is a class type and cannot satisfy std::is_arithmetic.
// static_assert(std::is_arithmetic_v<luisa::half>, "half should be arithmetic");

// Test type aliases
static_assert(sizeof(luisa::byte) == 1, "byte should be 1 byte");
static_assert(sizeof(luisa::ubyte) == 1, "ubyte should be 1 byte");
static_assert(sizeof(luisa::ushort) == 2, "ushort should be 2 bytes");
static_assert(sizeof(luisa::uint) == 4, "uint should be 4 bytes");
static_assert(sizeof(luisa::ulong) == 8, "ulong should be 8 bytes");
static_assert(sizeof(luisa::slong) == 8, "slong should be 8 bytes");

// Test is_integral
static_assert(luisa::is_integral_v<int>, "int should be integral");
static_assert(luisa::is_integral_v<luisa::uint>, "uint should be integral");
static_assert(luisa::is_integral_v<long>, "long should be integral");
static_assert(luisa::is_integral_v<unsigned long>, "unsigned long should be integral");
static_assert(luisa::is_integral_v<long long>, "long long should be integral");
static_assert(luisa::is_integral_v<unsigned long long>, "unsigned long long should be integral");
static_assert(luisa::is_integral_v<short>, "short should be integral");
static_assert(luisa::is_integral_v<luisa::ushort>, "ushort should be integral");
static_assert(luisa::is_integral_v<char>, "char should be integral");
static_assert(luisa::is_integral_v<luisa::uchar>, "uchar should be integral");
static_assert(luisa::is_integral_v<luisa::byte>, "byte should be integral");
static_assert(luisa::is_integral_v<luisa::ubyte>, "ubyte should be integral");
static_assert(!luisa::is_integral_v<float>, "float should not be integral");
static_assert(!luisa::is_integral_v<double>, "double should not be integral");
static_assert(!luisa::is_integral_v<luisa::half>, "half should not be integral");
static_assert(!luisa::is_integral_v<bool>, "bool should not be integral");

// Test is_boolean
static_assert(luisa::is_boolean_v<bool>, "bool should be boolean");
static_assert(!luisa::is_boolean_v<int>, "int should not be boolean");
static_assert(!luisa::is_boolean_v<float>, "float should not be boolean");

// Test is_floating_point
static_assert(luisa::is_floating_point_v<luisa::half>, "half should be floating point");
static_assert(luisa::is_floating_point_v<float>, "float should be floating point");
static_assert(luisa::is_floating_point_v<double>, "double should be floating point");
static_assert(!luisa::is_floating_point_v<int>, "int should not be floating point");
static_assert(!luisa::is_floating_point_v<bool>, "bool should not be floating point");

// Test is_signed
static_assert(luisa::is_signed_v<luisa::half>, "half should be signed");
static_assert(luisa::is_signed_v<float>, "float should be signed");
static_assert(luisa::is_signed_v<double>, "double should be signed");
static_assert(luisa::is_signed_v<char>, "char should be signed");
static_assert(luisa::is_signed_v<luisa::byte>, "byte should be signed");
static_assert(luisa::is_signed_v<short>, "short should be signed");
static_assert(luisa::is_signed_v<int>, "int should be signed");
static_assert(luisa::is_signed_v<luisa::slong>, "slong should be signed");

// Test is_unsigned
static_assert(luisa::is_unsigned_v<luisa::uchar>, "uchar should be unsigned");
static_assert(luisa::is_unsigned_v<luisa::ubyte>, "ubyte should be unsigned");
static_assert(luisa::is_unsigned_v<luisa::ushort>, "ushort should be unsigned");
static_assert(luisa::is_unsigned_v<luisa::uint>, "uint should be unsigned");
static_assert(luisa::is_unsigned_v<luisa::ulong>, "ulong should be unsigned");
static_assert(!luisa::is_unsigned_v<int>, "int should not be unsigned");
static_assert(!luisa::is_unsigned_v<float>, "float should not be unsigned");

// Test is_signed_integral and is_unsigned_integral
static_assert(luisa::is_signed_integral_v<int>, "int should be signed integral");
static_assert(luisa::is_signed_integral_v<short>, "short should be signed integral");
static_assert(!luisa::is_signed_integral_v<luisa::uint>, "uint should not be signed integral");
static_assert(!luisa::is_signed_integral_v<float>, "float should not be signed integral");

static_assert(luisa::is_unsigned_integral_v<luisa::uint>, "uint should be unsigned integral");
static_assert(luisa::is_unsigned_integral_v<luisa::ulong>, "ulong should be unsigned integral");
static_assert(!luisa::is_unsigned_integral_v<int>, "int should not be unsigned integral");
static_assert(!luisa::is_unsigned_integral_v<float>, "float should not be unsigned integral");

// Test is_scalar
static_assert(luisa::is_scalar_v<int>, "int should be scalar");
static_assert(luisa::is_scalar_v<float>, "float should be scalar");
static_assert(luisa::is_scalar_v<luisa::half>, "half should be scalar");
static_assert(luisa::is_scalar_v<bool>, "bool should be scalar");
static_assert(!luisa::is_scalar_v<luisa::Vector<float, 2>>, "Vector should not be scalar");

// Test vector traits with actual Vector types
using float2 = luisa::Vector<float, 2>;
using float3 = luisa::Vector<float, 3>;
using float4 = luisa::Vector<float, 4>;
using int2 = luisa::Vector<int, 2>;
using bool3 = luisa::Vector<bool, 3>;
using half4 = luisa::Vector<luisa::half, 4>;

// Test is_vector
static_assert(luisa::is_vector_v<float2>, "float2 should be a vector");
static_assert(luisa::is_vector_v<float3>, "float3 should be a vector");
static_assert(luisa::is_vector_v<float4>, "float4 should be a vector");
static_assert(luisa::is_vector_v<int2>, "int2 should be a vector");
static_assert(!luisa::is_vector_v<float>, "float should not be a vector");
static_assert(!luisa::is_vector_v<int>, "int should not be a vector");

// Test is_vector2, is_vector3, is_vector4
static_assert(luisa::is_vector2_v<float2>, "float2 should be a vector2");
static_assert(luisa::is_vector2_v<int2>, "int2 should be a vector2");
static_assert(!luisa::is_vector2_v<float3>, "float3 should not be a vector2");

static_assert(luisa::is_vector3_v<float3>, "float3 should be a vector3");
static_assert(luisa::is_vector3_v<bool3>, "bool3 should be a vector3");
static_assert(!luisa::is_vector3_v<float2>, "float2 should not be a vector3");

static_assert(luisa::is_vector4_v<float4>, "float4 should be a vector4");
static_assert(luisa::is_vector4_v<half4>, "half4 should be a vector4");
static_assert(!luisa::is_vector4_v<float3>, "float3 should not be a vector4");

// Test is_vector with specific dimension
static_assert(luisa::is_vector_v<float2, 2>, "float2 should be a vector of dimension 2");
static_assert(luisa::is_vector_v<float3, 3>, "float3 should be a vector of dimension 3");
static_assert(luisa::is_vector_v<float4, 4>, "float4 should be a vector of dimension 4");
static_assert(!luisa::is_vector_v<float2, 3>, "float2 should not be a vector of dimension 3");

// Test vector_element_t
static_assert(std::is_same_v<luisa::vector_element_t<float2>, float>, "float2 element should be float");
static_assert(std::is_same_v<luisa::vector_element_t<float3>, float>, "float3 element should be float");
static_assert(std::is_same_v<luisa::vector_element_t<int2>, int>, "int2 element should be int");
static_assert(std::is_same_v<luisa::vector_element_t<bool3>, bool>, "bool3 element should be bool");
static_assert(std::is_same_v<luisa::vector_element_t<half4>, luisa::half>, "half4 element should be half");
static_assert(std::is_same_v<luisa::vector_element_t<float>, float>, "vector_element_t<float> should be float");

// Test vector_dimension_v
static_assert(luisa::vector_dimension_v<float2> == 2, "float2 dimension should be 2");
static_assert(luisa::vector_dimension_v<float3> == 3, "float3 dimension should be 3");
static_assert(luisa::vector_dimension_v<float4> == 4, "float4 dimension should be 4");
static_assert(luisa::vector_dimension_v<float> == 1, "vector_dimension_v<float> should be 1");

// Test is_boolean_vector
static_assert(luisa::is_boolean_vector_v<luisa::Vector<bool, 2>>, "Vector<bool, 2> should be boolean vector");
static_assert(luisa::is_boolean_vector_v<bool3>, "bool3 should be boolean vector");
static_assert(!luisa::is_boolean_vector_v<float2>, "float2 should not be boolean vector");
static_assert(!luisa::is_boolean_vector_v<bool>, "bool should not be boolean vector");

// Test is_floating_point_vector
static_assert(luisa::is_floating_point_vector_v<float2>, "float2 should be floating point vector");
static_assert(luisa::is_floating_point_vector_v<float3>, "float3 should be floating point vector");
static_assert(luisa::is_floating_point_vector_v<half4>, "half4 should be floating point vector");
static_assert(!luisa::is_floating_point_vector_v<int2>, "int2 should not be floating point vector");
static_assert(!luisa::is_floating_point_vector_v<float>, "float should not be floating point vector");

// Test is_integral_vector
static_assert(luisa::is_integral_vector_v<int2>, "int2 should be integral vector");
static_assert(luisa::is_integral_vector_v<luisa::Vector<luisa::uint, 4>>, "Vector<uint, 4> should be integral vector");
static_assert(!luisa::is_integral_vector_v<float2>, "float2 should not be integral vector");
static_assert(!luisa::is_integral_vector_v<int>, "int should not be integral vector");

// Test is_signed_integral_vector
static_assert(luisa::is_signed_integral_vector_v<int2>, "int2 should be signed integral vector");
static_assert(!luisa::is_signed_integral_vector_v<luisa::Vector<luisa::uint, 2>>, "Vector<uint, 2> should not be signed integral vector");
static_assert(!luisa::is_signed_integral_vector_v<float2>, "float2 should not be signed integral vector");

// Test is_unsigned_integral_vector
static_assert(luisa::is_unsigned_integral_vector_v<luisa::Vector<luisa::uint, 2>>, "Vector<uint, 2> should be unsigned integral vector");
static_assert(luisa::is_unsigned_integral_vector_v<luisa::Vector<luisa::ulong, 4>>, "Vector<ulong, 4> should be unsigned integral vector");
static_assert(!luisa::is_unsigned_integral_vector_v<int2>, "int2 should not be unsigned integral vector");
static_assert(!luisa::is_unsigned_integral_vector_v<float2>, "float2 should not be unsigned integral vector");

// Test is_vector_same_dimension
static_assert(luisa::is_vector_same_dimension_v<float2, luisa::Vector<int, 2>>, "float2 and Vector<int, 2> should have same dimension");
static_assert(luisa::is_vector_same_dimension_v<float3, bool3>, "float3 and bool3 should have same dimension");
static_assert(!luisa::is_vector_same_dimension_v<float2, float3>, "float2 and float3 should not have same dimension");
static_assert(luisa::is_vector_same_dimension_v<float2, float2, float2>, "Three float2 should have same dimension");
static_assert(!luisa::is_vector_same_dimension_v<float2, float3, float2>, "float2, float3, float2 should not have same dimension");

// Test is_boolean_or_vector
static_assert(luisa::is_boolean_or_vector_v<bool>, "bool should be boolean or vector");
static_assert(luisa::is_boolean_or_vector_v<luisa::Vector<bool, 2>>, "Vector<bool, 2> should be boolean or vector");
static_assert(!luisa::is_boolean_or_vector_v<float>, "float should not be boolean or vector");
static_assert(!luisa::is_boolean_or_vector_v<float2>, "float2 should not be boolean or vector");

// Test is_floating_point_or_vector
static_assert(luisa::is_floating_point_or_vector_v<float>, "float should be floating point or vector");
static_assert(luisa::is_floating_point_or_vector_v<float2>, "float2 should be floating point or vector");
static_assert(luisa::is_floating_point_or_vector_v<half4>, "half4 should be floating point or vector");
static_assert(!luisa::is_floating_point_or_vector_v<int>, "int should not be floating point or vector");
static_assert(!luisa::is_floating_point_or_vector_v<int2>, "int2 should not be floating point or vector");

// Test is_integral_or_vector
static_assert(luisa::is_integral_or_vector_v<int>, "int should be integral or vector");
static_assert(luisa::is_integral_or_vector_v<int2>, "int2 should be integral or vector");
static_assert(luisa::is_integral_or_vector_v<luisa::Vector<luisa::uint, 4>>, "Vector<uint, 4> should be integral or vector");
static_assert(!luisa::is_integral_or_vector_v<float>, "float should not be integral or vector");
static_assert(!luisa::is_integral_or_vector_v<float2>, "float2 should not be integral or vector");

// Test is_signed_integral_or_vector
static_assert(luisa::is_signed_integral_or_vector_v<int>, "int should be signed integral or vector");
static_assert(luisa::is_signed_integral_or_vector_v<int2>, "int2 should be signed integral or vector");
static_assert(!luisa::is_signed_integral_or_vector_v<luisa::uint>, "uint should not be signed integral or vector");
static_assert(!luisa::is_signed_integral_or_vector_v<luisa::Vector<luisa::uint, 2>>, "Vector<uint, 2> should not be signed integral or vector");

// Test is_unsigned_integral_or_vector
static_assert(luisa::is_unsigned_integral_or_vector_v<luisa::uint>, "uint should be unsigned integral or vector");
static_assert(luisa::is_unsigned_integral_or_vector_v<luisa::Vector<luisa::uint, 2>>, "Vector<uint, 2> should be unsigned integral or vector");
static_assert(!luisa::is_unsigned_integral_or_vector_v<int>, "int should not be unsigned integral or vector");
static_assert(!luisa::is_unsigned_integral_or_vector_v<int2>, "int2 should not be unsigned integral or vector");

// Test matrix traits with actual Matrix types
using float2x2 = luisa::Matrix<float, 2>;
using float3x3 = luisa::Matrix<float, 3>;
using float4x4 = luisa::Matrix<float, 4>;

// Test is_matrix
static_assert(luisa::is_matrix_v<float2x2>, "float2x2 should be a matrix");
static_assert(luisa::is_matrix_v<float3x3>, "float3x3 should be a matrix");
static_assert(luisa::is_matrix_v<float4x4>, "float4x4 should be a matrix");
static_assert(!luisa::is_matrix_v<float>, "float should not be a matrix");
static_assert(!luisa::is_matrix_v<float2>, "float2 should not be a matrix");

// Test is_matrix2, is_matrix3, is_matrix4
static_assert(luisa::is_matrix2_v<float2x2>, "float2x2 should be a matrix2");
static_assert(!luisa::is_matrix2_v<float3x3>, "float3x3 should not be a matrix2");

static_assert(luisa::is_matrix3_v<float3x3>, "float3x3 should be a matrix3");
static_assert(!luisa::is_matrix3_v<float2x2>, "float2x2 should not be a matrix3");

static_assert(luisa::is_matrix4_v<float4x4>, "float4x4 should be a matrix4");
static_assert(!luisa::is_matrix4_v<float3x3>, "float3x3 should not be a matrix4");

// Test is_matrix with specific dimension
static_assert(luisa::is_matrix_v<float2x2, 2>, "float2x2 should be a matrix of dimension 2");
static_assert(luisa::is_matrix_v<float3x3, 3>, "float3x3 should be a matrix of dimension 3");
static_assert(luisa::is_matrix_v<float4x4, 4>, "float4x4 should be a matrix of dimension 4");
static_assert(!luisa::is_matrix_v<float2x2, 3>, "float2x2 should not be a matrix of dimension 3");

// Test matrix_element_t
static_assert(std::is_same_v<luisa::matrix_element_t<float2x2>, float>, "float2x2 element should be float");
static_assert(std::is_same_v<luisa::matrix_element_t<float3x3>, float>, "float3x3 element should be float");
static_assert(std::is_same_v<luisa::matrix_element_t<float>, float>, "matrix_element_t<float> should be float");

// Test matrix_dimension_v
static_assert(luisa::matrix_dimension_v<float2x2> == 2, "float2x2 dimension should be 2");
static_assert(luisa::matrix_dimension_v<float3x3> == 3, "float3x3 dimension should be 3");
static_assert(luisa::matrix_dimension_v<float4x4> == 4, "float4x4 dimension should be 4");
static_assert(luisa::matrix_dimension_v<float> == 1, "matrix_dimension_v<float> should be 1");

// Test is_basic
static_assert(luisa::is_basic_v<int>, "int should be basic");
static_assert(luisa::is_basic_v<float>, "float should be basic");
static_assert(luisa::is_basic_v<bool>, "bool should be basic");
static_assert(luisa::is_basic_v<luisa::half>, "half should be basic");
static_assert(luisa::is_basic_v<float2>, "float2 should be basic");
static_assert(luisa::is_basic_v<float3>, "float3 should be basic");
static_assert(luisa::is_basic_v<float4>, "float4 should be basic");
static_assert(luisa::is_basic_v<float2x2>, "float2x2 should be basic");
static_assert(luisa::is_basic_v<float3x3>, "float3x3 should be basic");
static_assert(luisa::is_basic_v<float4x4>, "float4x4 should be basic");
static_assert(!luisa::is_basic_v<int *>, "int* should not be basic");
static_assert(!luisa::is_basic_v<void>, "void should not be basic");

// Runtime test function
void test_basic_traits() {
    // Test to_underlying at runtime
    expect(static_cast<bool>(luisa::to_underlying(TestEnum::A) == 1));
    expect(static_cast<bool>(luisa::to_underlying(TestEnum::B) == 2));

    // Test type traits at runtime
    expect(static_cast<bool>(luisa::is_integral_v<int> == true));
    expect(static_cast<bool>(luisa::is_integral_v<float> == false));
    expect(static_cast<bool>(luisa::is_floating_point_v<float> == true));
    expect(static_cast<bool>(luisa::is_floating_point_v<int> == false));
    expect(static_cast<bool>(luisa::is_boolean_v<bool> == true));
    expect(static_cast<bool>(luisa::is_boolean_v<int> == false));
    expect(static_cast<bool>(luisa::is_scalar_v<int> == true));
    expect(static_cast<bool>(luisa::is_scalar_v<float2> == false));
    expect(static_cast<bool>(luisa::is_vector_v<float2> == true));
    expect(static_cast<bool>(luisa::is_vector_v<float> == false));
    expect(static_cast<bool>(luisa::is_matrix_v<float2x2> == true));
    expect(static_cast<bool>(luisa::is_matrix_v<float> == false));
    expect(static_cast<bool>(luisa::is_basic_v<int> == true));
    expect(static_cast<bool>(luisa::is_basic_v<float2> == true));
    expect(static_cast<bool>(luisa::is_basic_v<float2x2> == true));
}

static auto test_basic_traits_registration = [] {
    "test_basic_traits"_test = [] { test_basic_traits(); };
    return 0;
}();
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

}

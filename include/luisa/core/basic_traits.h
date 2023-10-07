#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

#define HALF_NO_THROW
#define HALF_ARITHMETIC_TYPE float
#include <half.hpp>

namespace luisa {

template<typename... T>
struct always_false : std::false_type {};

template<typename... T>
constexpr auto always_false_v = always_false<T...>::value;

template<typename... T>
struct always_true : std::true_type {};

template<typename... T>
constexpr auto always_true_v = always_true<T...>::value;

template<typename T>
    requires std::is_enum_v<T>
[[nodiscard]] constexpr auto to_underlying(T e) noexcept {
    return static_cast<std::underlying_type_t<T>>(e);
}

using half = half_float::half;
using namespace half_float::literal;

using byte = int8_t;
using ubyte = uint8_t;

static_assert(sizeof(half) == 2u && alignof(half) == 2u,
              "half should be 16-bit sized and aligned.");

static_assert(std::is_same_v<decltype(1._h + 1._h), half>,
              "half should support arithmetic operations.");

static_assert(std::is_same_v<decltype(sin(1._h)), half>,
              "half should support std::sin.");

static_assert(std::is_arithmetic_v<half>,
              "half should be arithmetic.");

using uchar = uint8_t;
using ushort = uint16_t;
using uint = uint32_t;
using ulong = uint64_t;
using slong = int64_t;// long has different size on different platforms

template<typename T>
using is_integral = std::disjunction<
    std::is_same<std::remove_cvref_t<T>, int>,
    std::is_same<std::remove_cvref_t<T>, uint>,
    std::is_same<std::remove_cvref_t<T>, slong>,
    std::is_same<std::remove_cvref_t<T>, ulong>,
    std::is_same<std::remove_cvref_t<T>, short>,
    std::is_same<std::remove_cvref_t<T>, ushort>>;

template<typename T>
constexpr auto is_integral_v = is_integral<T>::value;

template<typename T>
using is_boolean = std::is_same<std::remove_cvref_t<T>, bool>;

template<typename T>
constexpr auto is_boolean_v = is_boolean<T>::value;

template<typename T>
using is_floating_point = std::disjunction<
    std::is_same<std::remove_cvref_t<T>, half>,
    std::is_same<std::remove_cvref_t<T>, float>,
    std::is_same<std::remove_cvref_t<T>, double>>;

template<typename T>
constexpr auto is_floating_point_v = is_floating_point<T>::value;

template<typename T>
using is_signed = std::disjunction<
    is_floating_point<T>,
    std::is_same<std::remove_cvref_t<T>, short>,
    std::is_same<std::remove_cvref_t<T>, int>,
    std::is_same<std::remove_cvref_t<T>, slong>>;

template<typename T>
constexpr auto is_signed_v = is_signed<T>::value;

template<typename T>
using is_unsigned = std::disjunction<
    std::is_same<std::remove_cvref_t<T>, ushort>,
    std::is_same<std::remove_cvref_t<T>, uint>,
    std::is_same<std::remove_cvref_t<T>, ulong>>;

template<typename T>
constexpr auto is_unsigned_v = is_unsigned<T>::value;

template<typename T>
using is_signed_integral = std::conjunction<
    is_signed<T>,
    is_integral<T>>;

template<typename T>
using is_unsigned_integral = std::conjunction<
    is_unsigned<T>,
    is_integral<T>>;

template<typename T>
constexpr auto is_signed_integral_v = is_signed_integral<T>::value;

template<typename T>
constexpr auto is_unsigned_integral_v = is_unsigned_integral<T>::value;

template<typename T>
using is_scalar = std::disjunction<
    is_integral<T>,
    is_boolean<T>,
    is_floating_point<T>>;

template<typename T>
constexpr auto is_scalar_v = is_scalar<T>::value;

template<typename T, size_t N>
struct Vector;

template<size_t N>
struct Matrix;

namespace detail {

template<typename T, size_t N = 0u>
struct is_vector_impl : std::false_type {};

template<typename T, size_t N>
struct is_vector_impl<Vector<T, N>, N> : std::true_type {};

template<typename T, size_t N>
struct is_vector_impl<Vector<T, N>, 0u> : std::true_type {};

template<typename T, size_t N = 0u>
struct is_matrix_impl : std::false_type {};

template<size_t N>
struct is_matrix_impl<Matrix<N>, N> : std::true_type {};

template<size_t N>
struct is_matrix_impl<Matrix<N>, 0u> : std::true_type {};

template<typename T>
struct vector_element_impl {
    using type = T;
};

template<typename T, size_t N>
struct vector_element_impl<Vector<T, N>> {
    using type = T;
};

template<typename T>
struct vector_dimension_impl {
    static constexpr auto value = static_cast<size_t>(1u);
};

template<typename T, size_t N>
struct vector_dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
struct matrix_dimension_impl {
    static constexpr auto value = static_cast<size_t>(1u);
};

template<size_t N>
struct matrix_dimension_impl<Matrix<N>> {
    static constexpr auto value = N;
};

template<typename U, typename V>
struct is_vector_same_dimension_impl : std::false_type {};

template<typename U, typename V, size_t N>
struct is_vector_same_dimension_impl<Vector<U, N>, Vector<V, N>> : std::true_type {};

template<typename... T>
struct is_vector_all_same_dimension_impl : std::true_type {};

template<typename First, typename... Other>
struct is_vector_all_same_dimension_impl<First, Other...> : std::conjunction<is_vector_same_dimension_impl<First, Other>...> {};

}// namespace detail

template<typename... T>
using is_vector_same_dimension = detail::is_vector_all_same_dimension_impl<std::remove_cvref_t<T>...>;

template<typename... T>
constexpr auto is_vector_same_dimension_v = is_vector_same_dimension<T...>::value;

template<typename T>
using vector_dimension = detail::vector_dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto vector_dimension_v = vector_dimension<T>::value;

template<typename T>
using matrix_dimension = detail::matrix_dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto matrix_dimension_v = matrix_dimension<T>::value;

template<typename T>
using vector_element = detail::vector_element_impl<std::remove_cvref_t<T>>;

template<typename T>
using vector_element_t = typename vector_element<T>::type;

template<typename T, size_t N = 0u>
using is_vector = detail::is_vector_impl<std::remove_cvref_t<T>, N>;

template<typename T>
using is_vector2 = is_vector<T, 2u>;

template<typename T>
using is_vector3 = is_vector<T, 3u>;

template<typename T>
using is_vector4 = is_vector<T, 4u>;

template<typename T>
using is_boolean_vector = std::conjunction<is_vector<T>, is_boolean<vector_element_t<T>>>;

template<typename T>
constexpr auto is_boolean_vector_v = is_boolean_vector<T>::value;

template<typename T>
using is_floating_point_vector = std::conjunction<is_vector<T>, is_floating_point<vector_element_t<T>>>;

template<typename T>
constexpr auto is_floating_point_vector_v = is_floating_point_vector<T>::value;

template<typename T>
using is_integral_vector = std::conjunction<is_vector<T>, is_integral<vector_element_t<T>>>;

template<typename T>
constexpr auto is_integral_vector_v = is_integral_vector<T>::value;

template<typename T>
using is_signed_integral_vector = std::conjunction<is_vector<T>, is_signed_integral<vector_element_t<T>>>;

template<typename T>
constexpr auto is_signed_integral_vector_v = is_signed_integral_vector<T>::value;

template<typename T>
using is_unsigned_integral_vector = std::conjunction<is_vector<T>, is_unsigned_integral<vector_element_t<T>>>;

template<typename T>
constexpr auto is_unsigned_integral_vector_v = is_unsigned_integral_vector<T>::value;

template<typename T>
using is_boolean_or_vector = std::disjunction<is_boolean<T>, is_boolean_vector<T>>;

template<typename T>
constexpr auto is_boolean_or_vector_v = is_boolean_or_vector<T>::value;

template<typename T>
using is_floating_point_or_vector = std::disjunction<is_floating_point<T>, is_floating_point_vector<T>>;

template<typename T>
constexpr auto is_floating_point_or_vector_v = is_floating_point_or_vector<T>::value;

template<typename T>
using is_integral_or_vector = std::disjunction<is_integral<T>, is_integral_vector<T>>;

template<typename T>
constexpr auto is_integral_or_vector_v = is_integral_or_vector<T>::value;

template<typename T>
using is_signed_integral_or_vector = std::disjunction<is_signed_integral<T>, is_signed_integral_vector<T>>;

template<typename T>
constexpr auto is_signed_integral_or_vector_v = is_signed_integral_or_vector<T>::value;

template<typename T>
using is_unsigned_integral_or_vector = std::disjunction<is_unsigned_integral<T>, is_unsigned_integral_vector<T>>;

template<typename T>
constexpr auto is_unsigned_integral_or_vector_v = is_unsigned_integral_or_vector<T>::value;

template<typename T, size_t N = 0u>
constexpr auto is_vector_v = is_vector<T, N>::value;

template<typename T>
constexpr auto is_vector2_v = is_vector2<T>::value;

template<typename T>
constexpr auto is_vector3_v = is_vector3<T>::value;

template<typename T>
constexpr auto is_vector4_v = is_vector4<T>::value;

template<typename T, size_t N = 0u>
using is_matrix = detail::is_matrix_impl<std::remove_cvref_t<T>, N>;

template<typename T>
using is_matrix2 = is_matrix<T, 2u>;

template<typename T>
using is_matrix3 = is_matrix<T, 3u>;

template<typename T>
using is_matrix4 = is_matrix<T, 4u>;

template<typename T, size_t N = 0u>
constexpr auto is_matrix_v = is_matrix<T, N>::value;

template<typename T>
constexpr auto is_matrix2_v = is_matrix2<T>::value;

template<typename T>
constexpr auto is_matrix3_v = is_matrix3<T>::value;

template<typename T>
constexpr auto is_matrix4_v = is_matrix4<T>::value;

template<typename T>
using is_basic = std::disjunction<is_scalar<T>, is_vector<T>, is_matrix<T>>;

template<typename T>
constexpr auto is_basic_v = is_basic<T>::value;

}// namespace luisa

//
// Created by Mike Smith on 2021/8/9.
//

#pragma once

#include <type_traits>
#include <concepts>
#include <core/basic_types.h>
#include <core/concepts.h>

namespace luisa::compute {

template<typename T>
struct Expr;

template<typename T>
struct Ref;

template<typename T>
struct Var;

namespace detail {

template<typename T>
struct expr_value_impl {
    using type = T;
};

template<typename T>
struct expr_value_impl<Expr<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Ref<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Var<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using expr_value = detail::expr_value_impl<std::remove_cvref_t<T>>;

template<typename T>
using expr_value_t = typename expr_value<T>::type;

template<typename T>
using vector_expr_element = vector_element<expr_value_t<T>>;

template<typename T>
using vector_expr_element_t = typename vector_expr_element<T>::type;

template<typename T>
using vector_expr_dimension = vector_dimension<expr_value_t<T>>;

template<typename T>
constexpr auto vector_expr_dimension_v = vector_expr_dimension<T>::value;

template<typename... T>
using is_vector_expr_same_dimension = is_vector_same_dimension<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_vector_expr_same_dimension_v = is_vector_expr_same_dimension<T...>::value;

template<typename... T>
using is_vector_expr_same_element = concepts::is_same<vector_expr_element_t<T>...>;

template<typename... T>
constexpr auto is_vector_expr_same_element_v = is_vector_expr_same_element<T...>::value;

namespace detail {
template<typename T>
struct is_dsl_impl : std::false_type {};

template<typename T>
struct is_dsl_impl<Expr<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Ref<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Var<T>> : std::true_type {};

}// namespace detail

template<typename T>
using is_dsl = typename detail::is_dsl_impl<std::remove_cvref_t<T>>::type;

template<typename T>
constexpr auto is_dsl_v = is_dsl<T>::value;

template<typename... T>
using any_dsl = std::disjunction<is_dsl<T>...>;

template<typename... T>
constexpr auto any_dsl_v = any_dsl<T...>::value;

template<typename... T>
using is_same_expr = concepts::is_same<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_same_expr_v = is_same_expr<T...>::value;

template<typename T>
using is_integral_expr = is_integral<expr_value_t<T>>;

template<typename T>
constexpr auto is_integral_expr_v = is_integral_expr<T>::value;

template<typename T>
using is_floating_point_expr = is_floating_point<expr_value_t<T>>;

template<typename T>
constexpr auto is_floating_point_expr_v = is_floating_point_expr<T>::value;

template<typename T>
using is_scalar_expr = is_scalar<expr_value_t<T>>;

template<typename T>
constexpr auto is_scalar_expr_v = is_scalar_expr<T>::value;

template<typename T>
using is_vector_expr = is_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector_expr_v = is_vector_expr<T>::value;

template<typename T>
using is_vector2_expr = is_vector2<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector2_expr_v = is_vector2_expr<T>::value;

template<typename T>
using is_vector3_expr = is_vector3<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector3_expr_v = is_vector3_expr<T>::value;

template<typename T>
using is_vector4_expr = is_vector4<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector4_expr_v = is_vector4_expr<T>::value;

template<typename T>
using is_bool_vector_expr = is_bool_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_bool_vector_expr_v = is_bool_vector_expr<T>::value;

template<typename T>
using is_float_vector_expr = is_float_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_float_vector_expr_v = is_float_vector_expr<T>::value;

template<typename T>
using is_int_vector_expr = is_int_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_int_vector_expr_v = is_int_vector_expr<T>::value;

template<typename T>
using is_uint_vector_expr = is_uint_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_uint_vector_expr_v = is_uint_vector_expr<T>::value;

template<typename T>
using is_matrix_expr = is_matrix<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix_expr_v = is_matrix_expr<T>::value;

template<typename T>
using is_matrix2_expr = is_matrix2<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix2_expr_v = is_matrix2_expr<T>::value;

template<typename T>
using is_matrix3_expr = is_matrix3<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix3_expr_v = is_matrix3_expr<T>::value;

template<typename T>
using is_matrix4_expr = is_matrix4<expr_value_t<T>>;

template<typename T>
constexpr auto is_matrix4_expr_v = is_matrix4_expr<T>::value;

template<typename T>
using is_float_or_vector_expr = std::disjunction<
    is_floating_point_expr<T>,
    is_float_vector_expr<T>>;

template<typename T>
constexpr auto is_float_or_vector_expr_v = is_float_or_vector_expr<T>::value;

template<typename T>
using is_int_or_vector_expr = std::disjunction<
    std::is_same<expr_value_t<T>, int>,
    is_int_vector_expr<T>>;

template<typename T>
constexpr auto is_int_or_vector_expr_v = is_int_or_vector_expr<T>::value;

template<typename T>
using is_bool_or_vector_expr = std::disjunction<
    std::is_same<expr_value_t<T>, bool>,
    is_bool_vector_expr<T>>;

template<typename T>
constexpr auto is_bool_or_vector_expr_v = is_bool_or_vector_expr<T>::value;

template<typename T>
using is_uint_or_vector_expr = std::disjunction<
    std::is_same<expr_value_t<T>, uint>,
    is_uint_vector_expr<T>>;

template<typename T>
constexpr auto is_uint_or_vector_expr_v = is_uint_or_vector_expr<T>::value;

template<typename T>
using is_struct_expr = is_struct<expr_value_t<T>>;

template<typename T>
constexpr auto is_struct_expr_v = is_struct_expr<T>::value;

template<typename T>
using is_buffer_expr = is_buffer_or_view<expr_value_t<T>>;

template<typename T>
constexpr auto is_buffer_expr_v = is_buffer_expr<T>::value;

template<typename T>
using buffer_expr_element = buffer_element<expr_value_t<T>>;

template<typename T>
using buffer_expr_element_t = typename buffer_expr_element<T>::type;

}// namespace luisa::compute

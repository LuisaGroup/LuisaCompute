#pragma once
#include "./../attributes.hpp"
#include "builtins.hpp"

namespace luisa::shader {

namespace detail {
template<class T>
trait is_vec : public false_type { using scalar_type = T; };
template<class T, uint64 N>
trait is_vec<vec<T, N>> : public true_type { using scalar_type = T; };

template<typename T>
trait is_matrix { static constexpr bool value = false; };
template<uint64 N>
trait is_matrix<matrix<N>> { static constexpr bool value = true; };
}// namespace detail

template<typename T>
using scalar_type = typename detail::is_vec<decay_t<T>>::scalar_type;

template<typename T>
static constexpr bool is_matrix_v = detail::is_matrix<decay_t<T>>::value;

template<typename T>
static constexpr bool is_vec_v = detail::is_vec<decay_t<T>>::value;

template<typename T>
static constexpr bool is_float_family_v = is_same_v<scalar_type<T>, float> | is_same_v<scalar_type<T>, double> | is_same_v<scalar_type<T>, half>;

template<typename T>
static constexpr bool is_sint_family_v = is_same_v<scalar_type<T>, int16> | is_same_v<scalar_type<T>, int32> | is_same_v<scalar_type<T>, int64>;

template<typename T>
static constexpr bool is_uint_family_v = is_same_v<scalar_type<T>, uint16> | is_same_v<scalar_type<T>, uint32> | is_same_v<scalar_type<T>, uint64>;

template<typename T>
static constexpr bool is_bool_family_v = is_same_v<scalar_type<T>, bool>;

template<typename T>
static constexpr bool is_arithmetic_v = is_float_family_v<T> || is_bool_family_v<T> || is_sint_family_v<T> || is_uint_family_v<T>;
template<typename T>
static constexpr bool is_arithmetic_scalar_v = is_arithmetic_v<T> && !is_vec_v<T>;

template<typename T>
concept vec_family = is_vec_v<T>;

template<typename T>
concept non_vec_family = !is_vec_v<T>;

template<typename T>
concept float_family = is_float_family_v<T>;

template<typename T>
concept bool_family = is_bool_family_v<T>;

template<typename T>
concept int_family = is_sint_family_v<T>;

template<typename T>
concept uint_family = is_uint_family_v<T>;

template<typename T>
concept arithmetic = is_arithmetic_v<T>;
template<typename T>
concept arithmetic_scalar = is_arithmetic_scalar_v<T>;

}// namespace luisa::shader
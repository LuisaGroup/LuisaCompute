#pragma once
#include "./../attributes.hpp"
#include "builtins.hpp"

namespace luisa::shader {

namespace detail {
template<class T>
trait vec_or_matrix : public false_type 
{ 
    using scalar_type = T; 
    static constexpr bool is_vec = false;
    static constexpr bool is_matrix = false;
};

template<class T, uint64 N>
trait vec_or_matrix<vec<T, N>> : public true_type 
{ 
    using scalar_type = T; 
    static constexpr bool is_vec = true;
    static constexpr bool is_matrix = false;    
};

template<uint64 N>
trait vec_or_matrix<matrix<N>> : public true_type {
    using scalar_type = float;
    static constexpr bool is_vec = false;
    static constexpr bool is_matrix = true;
};
}// namespace detail

template<typename T>
using scalar_type = typename detail::vec_or_matrix<decay_t<T>>::scalar_type;

template<typename T>
static constexpr bool is_matrix_v = detail::vec_or_matrix<decay_t<T>>::is_matrix;

template<typename T>
static constexpr bool is_vec_v = detail::vec_or_matrix<decay_t<T>>::is_vec;

template<typename T>
static constexpr bool is_vec_or_matrix_v = detail::vec_or_matrix<decay_t<T>>::value;

template<typename T>
static constexpr bool is_float_family_v = is_same_v<scalar_type<T>, float> | is_same_v<scalar_type<T>, double> | is_same_v<scalar_type<T>, half>;

template<typename T>
static constexpr bool is_sint_family_v = is_same_v<scalar_type<T>, int16> | is_same_v<scalar_type<T>, int32> | is_same_v<scalar_type<T>, int64>;

template<typename T>
static constexpr bool is_uint_family_v = is_same_v<scalar_type<T>, uint16> | is_same_v<scalar_type<T>, uint32> | is_same_v<scalar_type<T>, uint64>;

template<typename T>
static constexpr bool is_int_family_v = is_sint_family_v<T> || is_uint_family_v<T>;

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
concept sint_family = is_sint_family_v<T>;

template<typename T>
concept uint_family = is_uint_family_v<T>;

template<typename T>
concept int_family = is_int_family_v<T>;

template<typename T>
concept arithmetic = is_arithmetic_v<T>;

template<typename T>
concept arithmetic_scalar = is_arithmetic_scalar_v<T>;

template<typename T>
concept primitive = is_arithmetic_v<T> || is_vec_or_matrix_v<T>;

}// namespace luisa::shader
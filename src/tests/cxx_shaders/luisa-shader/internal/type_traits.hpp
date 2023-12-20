#pragma once
#include "attributes.hpp"

namespace luisa::shader {
using int16 = short;
using uint16 = unsigned short;
using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint64 = unsigned long long;

template<typename T, typename U>
static constexpr bool is_same_v = __is_same_as(T, U);

template<typename T>
trait remove_cvref { using type = T; };
template<typename T>
trait remove_cvref<T &> { using type = T; };
template<typename T>
trait remove_cvref<T const> { using type = T; };
template<typename T>
trait remove_cvref<T volatile> { using type = T; };
template<typename T>
trait remove_cvref<T &&> { using type = T; };

/*
template<typename T>
[[ignore]] constexpr T&& forward(typename remove_cvref<T>::type& Arg) {
    return static_cast<T&&>(Arg);
}

template<typename T>
[[ignore]] constexpr T&& forward(typename remove_cvref<T>::type&& Arg) {
    return static_cast<T&&>(Arg);
}
*/

struct [[builtin("half")]] half {
    [[ignore]] half() = default;
    [[ignore]] half(float);
    [[ignore]] half(uint32);
    [[ignore]] half(int32);
private:
    short v;
};

template<typename T, uint64 N>
struct vec;

template<uint64 N>
struct matrix;
template<typename T>
trait is_float_family { static constexpr bool value = is_same_v<T, float> | is_same_v<T, double> | is_same_v<T, half>; };
template<typename T>
trait is_sint_family { static constexpr bool value = is_same_v<T, int16> | is_same_v<T, int32> | is_same_v<T, int64>; };
template<typename T>
trait is_uint_family { static constexpr bool value = is_same_v<T, uint16> | is_same_v<T, uint32> | is_same_v<T, uint64>;};
template<typename T>
trait is_bool_family { static constexpr bool value = is_same_v<T, bool>; };

template<typename T, size_t N>
trait is_float_family<vec<T, N>> { static constexpr bool value = is_same_v<T, float> | is_same_v<T, double> | is_same_v<T, half>; };
template<typename T, size_t N>
trait is_sint_family<vec<T, N>> { static constexpr bool value = is_same_v<T, int16> | is_same_v<T, int32> | is_same_v<T, int64>; };
template<typename T, size_t N>
trait is_uint_family<vec<T, N>> { static constexpr bool value = is_same_v<T, uint16> | is_same_v<T, uint32> | is_same_v<T, uint64>;};
template<typename T, size_t N>
trait is_bool_family<vec<T, N>> { static constexpr bool value = is_same_v<T, bool>; };

template <typename T>
trait is_arithmetic{ static constexpr bool value = is_float_family<T>::value || is_bool_family<T>::value || is_sint_family<T>::value || is_uint_family<T>::value; };
template <typename T>
trait is_arithmetic_scalar { static constexpr bool value = false;};
template <>
trait is_arithmetic_scalar<float>{ static constexpr bool value = true;};
template <>
trait is_arithmetic_scalar<uint32>{ static constexpr bool value = true;};
template <>
trait is_arithmetic_scalar<int32>{ static constexpr bool value = true;};
template <>
trait is_arithmetic_scalar<uint16>{ static constexpr bool value = true;};
template <>
trait is_arithmetic_scalar<int16>{ static constexpr bool value = true;};
template <>
trait is_arithmetic_scalar<uint64>{ static constexpr bool value = true;};
template <>
trait is_arithmetic_scalar<int64>{ static constexpr bool value = true;};

template<typename T>
concept floatN = is_float_family<typename remove_cvref<T>::type>::value;

template<typename T>
concept boolN = is_bool_family<typename remove_cvref<T>::type>::value;

template<typename T>
concept intN = is_sint_family<typename remove_cvref<T>::type>::value;

template<typename T>
concept uintN = is_uint_family<typename remove_cvref<T>::type>::value;

template<typename T>
concept arithmetic = is_arithmetic<T>::value;
template<typename T>
concept arithmetic_scalar = is_arithmetic_scalar<T>::value;

template<typename T>
trait vec_dim { static constexpr uint64 value = 1; };

template<typename T, uint64 N>
trait vec_dim<vec<T, N>> { static constexpr uint64 value = N; };
template<typename T>
[[ignore]] constexpr uint64 vec_dim_v = vec_dim<typename remove_cvref<T>::type>::value;

template<uint64 dim, typename T, typename... Ts>
[[ignore]] consteval uint64 sum_dim() {
    constexpr auto new_dim = dim + vec_dim_v<T>;
    if constexpr (sizeof...(Ts) == 0) {
        return new_dim;
    } else {
        return sum_dim<new_dim, Ts...>();
    }
}

template <typename...T>
static constexpr auto sum_dim_v = sum_dim<0ull, T...>();

template <typename T>
trait is_basic_type{ static constexpr bool value = false; };
template <typename T, uint64 N>
trait is_basic_type<vec<T, N>>{ static constexpr bool value = true; };
template <uint64 N>
trait is_basic_type<matrix<N>>{ static constexpr bool value = true; };
template <>
trait is_basic_type<half>{ static constexpr bool value = true; };
template <>
trait is_basic_type<float>{ static constexpr bool value = true; };
template <>
trait is_basic_type<double>{ static constexpr bool value = true; };
template <>
trait is_basic_type<int32>{ static constexpr bool value = true; };
template <>
trait is_basic_type<uint32>{ static constexpr bool value = true; };
template <>
trait is_basic_type<int64>{ static constexpr bool value = true; };
template <>
trait is_basic_type<uint64>{ static constexpr bool value = true; };
template <>
trait is_basic_type<int16>{ static constexpr bool value = true; };
template <>
trait is_basic_type<uint16>{ static constexpr bool value = true; };
template<typename T>
concept basic_type = is_basic_type<typename remove_cvref<T>::type>::value;

template <typename T>
trait is_matrix{ static constexpr bool value = false;};
template <uint64 N>
trait is_matrix<matrix<N>>{ static constexpr bool value = true;};

template<typename T>
trait element { using type = T;};
template<typename T, uint64 N>
trait element<vec<T, N>> { using type = T;};
template<uint64 N>
trait element<matrix<N>> { using type = float;};

namespace detail
{
    template<uint64 N, typename T, typename...Ts>
    trait element_of {
        using type = typename element_of<N - 1, Ts...>::type; 
        using type2 = typename element<T>::type; 
        static_assert(__is_same_as(type, type2), "!!!");
    };

    template<typename T, typename...Ts>
    trait element_of<1, T, Ts...> { 
        using type = typename element<T>::type; 
    };
}

template<typename...Ts>
trait element_of { 
    using type = typename detail::element_of<sizeof...(Ts), Ts...>::type; 
};

}
#pragma once
#include "attributes.hpp"

namespace luisa::shader {
using int16 = short;
using uint16 = unsigned short;
using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint64 = unsigned long long;

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
trait is_floatN { static constexpr bool value = false; };
template<typename T>
trait is_intN { static constexpr bool value = false; };
template<typename T>
trait is_uintN { static constexpr bool value = false; };
template<typename T>
trait is_boolN { static constexpr bool value = false; };
template<> trait is_floatN<float> { static constexpr bool value = true; };
template<> trait is_floatN<half> { static constexpr bool value = true; };
template<> trait is_floatN<double> { static constexpr bool value = true; };
template<> trait is_intN<int16> { static constexpr bool value = true; };
template<> trait is_intN<int32> { static constexpr bool value = true; };
template<> trait is_intN<int64> { static constexpr bool value = true; };
template<> trait is_uintN<uint16> { static constexpr bool value = true; };
template<> trait is_uintN<uint32> { static constexpr bool value = true; };
template<> trait is_uintN<uint64> { static constexpr bool value = true; };
template<> trait is_boolN<bool> { static constexpr bool value = true; };
template<uint64 N> trait is_boolN<vec<bool, N>> { static constexpr bool value = true; };

template<uint64 N> trait is_floatN<vec<half, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_floatN<vec<float, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_floatN<vec<double, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_intN<vec<int16, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_intN<vec<int32, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_intN<vec<int64, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_uintN<vec<uint16, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_uintN<vec<uint32, N>> { static constexpr bool value = true; };
template<uint64 N> trait is_uintN<vec<uint64, N>> { static constexpr bool value = true; };

template <typename T>
trait is_arithmetic{ static constexpr bool value = is_floatN<T>::value || is_boolN<T>::value || is_intN<T>::value || is_uintN<T>::value; };
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
concept floatN = is_floatN<typename remove_cvref<T>::type>::value;

template<typename T>
concept boolN = is_boolN<typename remove_cvref<T>::type>::value;

template<typename T>
concept intN = is_intN<typename remove_cvref<T>::type>::value;

template<typename T>
concept uintN = is_uintN<typename remove_cvref<T>::type>::value;

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
        using type = element_of<N - 1, Ts...>::type; 
        using type2 = element<T>::type; 
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
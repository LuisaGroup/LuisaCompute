#pragma once
#include "attributes.hpp"
#include "type_traits/builtins.hpp"
#include "type_traits/concepts.hpp"

namespace luisa::shader {

namespace detail
{
    template<typename T>
    trait vec_dim { static constexpr uint64 value = 1; };

    template<typename T, uint64 N>
    trait vec_dim<vec<T, N>> { static constexpr uint64 value = N; };
}

template<typename T>
[[ignore]] constexpr uint64 vec_dim_v = detail::vec_dim<decay_t<T>>::value;

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
concept basic_type = is_basic_type<decay_t<T>>::value;

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
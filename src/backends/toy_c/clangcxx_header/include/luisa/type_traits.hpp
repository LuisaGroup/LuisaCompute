#pragma once
#include "attributes.hpp"
#include "type_traits/builtins.hpp"
#include "type_traits/concepts.hpp"

namespace luisa::shader {

namespace detail {
template<typename T>
trait vec_dim { static constexpr uint64 value = 1; };

template<typename T, uint64 N>
trait vec_dim<vec<T, N>> { static constexpr uint64 value = N; };

}// namespace detail

template<typename T, typename OtherType>
trait copy_dim { using type = T; };

template<typename T, typename ElemType, uint64 N>
trait copy_dim<T, vec<ElemType, N>> { using type = vec<T, N>; };

template<typename T>
[[ignore]] constexpr uint64 vec_dim_v = detail::vec_dim<decay_t<T>>::value;

template<typename T, typename U>
[[ignore]] constexpr bool same_dim_v = (vec_dim_v<T> == vec_dim_v<U>);

template<uint64 dim, typename T, typename... Ts>
[[ignore]] consteval uint64 sum_dim() {
	constexpr auto new_dim = dim + vec_dim_v<T>;
	if constexpr (sizeof...(Ts) == 0) {
		return new_dim;
	} else {
		return sum_dim<new_dim, Ts...>();
	}
}

template<typename... T>
static constexpr auto sum_dim_v = sum_dim<0ull, T...>();

namespace detail {
template<uint64 N, typename T, typename... Ts>
trait element_of {
	using type = typename element_of<N - 1, Ts...>::type;
	static_assert(__is_same_as(type, scalar_type<T>), "!!!");
};

template<typename T, typename... Ts>
trait element_of<1, T, Ts...> {
	using type = scalar_type<T>;
};
}// namespace detail

template<typename... Ts>
trait element_of {
	using type = typename detail::element_of<sizeof...(Ts), Ts...>::type;
};

}// namespace luisa::shader
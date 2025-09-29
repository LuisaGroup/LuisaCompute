#pragma once

#include "std/ex/utility.hpp"

namespace stdex {

template<class... Ts>
trait_struct type_list {
public:
	template<class T>
	static constexpr bool contains = (std::is_same_v<T, Ts> || ...);

	template<template<class> class T>
	static constexpr bool all = (T<Ts>::value && ...);

	template<template<class> class T>
	static constexpr bool any = (T<Ts>::value || ...);

	static constexpr size_t size = sizeof...(Ts);

	template<template<class> class W>
	using wrap = type_list<W<Ts>...>;

	template<template<class> class M>
	using map = type_list<typename M<Ts>::type...>;

	template<class T>
	using push_back = type_list<Ts..., T>;

	template<class T>
	using push_front = type_list<T, Ts...>;

	template<template<class...> class U>
	using to = U<Ts...>;

	template<template<class...> class U, class... Us>
	using apply_type = U<Us..., Ts...>;

	template<size_t N>
	using get = get_type_t<N + 1, void, Ts...>;

	template<class T>
	static constexpr size_t index = index_of<T, Ts...>::value;

	template<class T>
	trait_struct instance {
		using type = T;
		static constexpr size_t index = type_list::index<T>;
	};

	template<class Fn>
	static constexpr void for_each(Fn && fn) {
		unroll_type<instance<Ts>...>(static_cast<Fn&&>(fn));
	}

	template<class Ret, class Fn, class... Args>
	static constexpr decltype(auto) visit(size_t index, Fn&& fn, Args&&... args) {
		return (visit_index<Ret, size>(index, [&]<size_t I>() -> decltype(auto) {
			return (static_cast<Fn&&>(fn).template operator()<instance<get<I>>>(static_cast<decltype(args)&&>(args)...));
		}));
	}

	template<class Fn, class... Args>
	static constexpr decltype(auto) visit(size_t index, Fn&& fn, Args&&... args) {
		using Ret = decltype((std::declval<Fn>().template operator()<instance<get<0>>>(std::declval<Args>()...)));
		return (visit_index<Ret, size>(index, [&]<size_t I>() -> decltype(auto) {
			return (static_cast<Fn&&>(fn).template operator()<instance<get<I>>>(static_cast<decltype(args)&&>(args)...));
		}));
	}
};

}// namespace stdex
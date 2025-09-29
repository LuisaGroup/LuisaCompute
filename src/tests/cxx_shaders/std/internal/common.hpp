#pragma once

#include <luisa/attributes.hpp>

namespace std {

template<class T, T v>
trait_struct integral_constant {
	static constexpr T value = v;

	using value_type = T;
	using type = integral_constant;

	constexpr operator value_type() const noexcept {
		return value;
	}

	[[nodiscard]] constexpr value_type operator()() const noexcept {
		return value;
	}
};

template<bool v>
using bool_constant = integral_constant<bool, v>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template<class T, class U>
constexpr bool is_same_v = __is_same(T, U);

template<class T, class U>
trait_struct is_same : bool_constant<__is_same(T, U)>{};

template<class T>
trait_struct remove_cv {// remove top-level const and volatile qualifiers
	using type = T;

	template<template<class> class F>
	using apply = F<T>;// apply cv-qualifiers from the class template argument to F<T>
};

template<class T>
trait_struct remove_cv<const T> {
	using type = T;

	template<template<class> class F>
	using apply = const F<T>;
};

template<class T>
trait_struct remove_cv<volatile T> {
	using type = T;

	template<template<class> class F>
	using apply = volatile F<T>;
};

template<class T>
trait_struct remove_cv<const volatile T> {
	using type = T;

	template<template<class> class F>
	using apply = const volatile F<T>;
};

template<class T>
using remove_cv_t = typename remove_cv<T>::type;

template<class T>
trait_struct remove_reference {
	using type = __remove_reference_t(T);
};

template<class T>
using remove_reference_t = __remove_reference_t(T);

template<class T, class... Ts>
constexpr bool is_any_of_v = (is_same_v<T, Ts> || ...);

template<class T>
constexpr bool is_integral_v = is_any_of_v<remove_cv_t<T>, bool, char, signed char, unsigned char,
										   short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long>;

template<class T, size_t N>
struct [[builtin("array")]] array;

}// namespace std
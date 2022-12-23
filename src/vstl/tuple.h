#pragma once
#include <type_traits>
namespace vstd {
template<typename... Args>
struct tuple;
template<>
struct tuple<> {};

template<typename T>
struct tuple<T> {
	using value_type = T;
	T value;
	template<typename Arg>
		requires(std::is_constructible_v<T, Arg&&>)
	tuple(Arg&& a)
		: value(std::forward<Arg>(a)) {
	}
	tuple(){}
	template<size_t i>
		requires(i == 0)
	auto& get() & {
		return value;
	}
	template<size_t i>
		requires(i == 0)
	auto const& get() const& {
		return value;
	}
	template<size_t i>
		requires(i == 0)
	auto&& get() && {
		return std::move(value);
	}
};
template<typename A, typename B, typename... Values>
struct tuple<A, B, Values...> {
	A value;
	tuple<B, Values...> args;
	using value_type = A;
	template<typename Arg0, typename Arg1, typename... Args>
		requires(std::is_constructible_v<A, Arg0&&>&& std::is_constructible_v<tuple<B, Values...>, B&&, Args&&...>)
	tuple(Arg0&& a, Arg1&& b, Args&&... c)
		: value(std::forward<Arg0&&>(a)), args(std::forward<Arg1>(b), std::forward<Args>(c)...) {}
	template<size_t i>
		requires(i < (sizeof...(Values) + 2))
	auto& get() & {
		if constexpr (i == 0) {
			return value;
		} else {
			return args.template get<i - 1>();
		}
	}
	tuple(){}
	template<size_t i>
		requires(i < (sizeof...(Values) + 2))
	auto const& get() const& {
		if constexpr (i == 0) {
			return value;
		} else {
			return args.template get<i - 1>();
		}
	}
	template<size_t i>
		requires(i < (sizeof...(Values) + 2))
	auto&& get() && {
		if constexpr (i == 0) {
			return std::move(value);
		} else {
			return std::move(args).template get<i - 1>();
		}
	}
};

}// namespace vstd
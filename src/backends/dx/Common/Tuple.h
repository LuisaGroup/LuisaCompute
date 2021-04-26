#pragma once
#include <type_traits>
#include <stdint.h>
template<typename... Args>
class TupleElement;

template<>
class TupleElement<> {
public:
	using Type = TupleElement<>;
};

template<>
class TupleElement<void> {
public:
	using Type = TupleElement<>;
};

template<typename T>
class TupleElement<T> {
	T obj;

public:
	using Type = TupleElement<T>;

	constexpr TupleElement(T const& t)
		: obj(t) {}
	constexpr TupleElement(T&& t)
		: obj(std::move(t)) {
	}

	constexpr TupleElement(Type const& t)
		: obj(t.obj) {}

	constexpr TupleElement(Type&& t)
		: obj(std::move(t.obj)) {}

	template<size_t index>
	constexpr T& Get() noexcept {
		static_assert(index == 0, "Tuple Index Out of Range!");
		return obj;
	}
	template <size_t index>
	constexpr T const& Get() const noexcept {
		static_assert(index == 0, "Tuple Index Out of Range!");
		return obj;
	}
};

template<typename A, typename B, typename... Args>
class TupleElement<A, B, Args...> {
	A obj;
	TupleElement<B, Args...> other;

public:
	using Type = TupleElement<A, B, Args...>;

	constexpr TupleElement(A const& a, B const& b, Args const&... args)
		: obj(a),
		  other(b, args...) {}
	constexpr TupleElement(A&& a, B&& b, Args&&... args)
		: obj(std::move(a)),
		  other(std::move(b), std::move(args)...) {}

	constexpr TupleElement(Type const& t)
		: obj(t.obj), other(t.other) {}

	constexpr TupleElement(Type&& t)
		: obj(std::move(t.obj)), other(std::move(t.other)) {}


	template<size_t index>
	constexpr decltype(auto) Get() noexcept {
		if constexpr (index == 0) {
			return static_cast<A&>(obj);
		} else {
			return other.Get<index - 1>();
		}
	}

	template<size_t index>
	constexpr decltype(auto) Get() const noexcept {
		if constexpr (index == 0) {
			return static_cast<A const&>(obj);
		} else {
			return other.Get<index - 1>();
		}
	}
};

template<typename... Args>
using Tuple = typename TupleElement<std::remove_cvref_t<Args>...>::Type;
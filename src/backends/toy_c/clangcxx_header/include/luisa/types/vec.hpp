#pragma once
#include "array.hpp"
#include "../type_traits.hpp"

namespace luisa::shader {

template<typename T, uint64 N>
struct [[builtin("vec")]] vec;

template<typename T>
struct alignas(8) [[builtin("vec")]] vec<T, 2> {
	using ThisType = vec<T, 2>;
	static constexpr uint32 dim = 2;
	vec() noexcept = default;

	template<typename... Args>
		requires(sum_dim<0ull, Args...>() == 2)
	explicit constexpr vec(Args&&... args)
		: _v()// active union member
	{
		set<0>(args...);
	}

	template<concepts::arithmetic_scalar U>
	explicit constexpr vec(U v)
		: _v(v, v) {}

	template<concepts::arithmetic_scalar U>
	[[nodiscard, noignore]] operator vec<U, 2>() {
		return vec<U, 2>(static_cast<U>(x), static_cast<U>(y));
	}

#include "ops/vec_ops.inl"

	// DONT EDIT THIS FIELD LAYOUT
	union {
		Array<T, 2> _v;
#include "ops/swizzle2.inl"
	};
};

template<typename T>
struct alignas(16) [[builtin("vec")]] vec<T, 3> {
	using ThisType = vec<T, 3>;
	static constexpr uint32 dim = 3;
	vec() noexcept = default;

	template<typename... Args>
		requires(sum_dim<0ull, Args...>() == 3)
	explicit constexpr vec(Args&&... args)
		: _v()// active union member
	{
		set<0>(args...);
	}

	template<concepts::arithmetic_scalar U>
	explicit constexpr vec(U v)
		: _v(v, v, v) {}

	template<concepts::arithmetic_scalar U>
	[[nodiscard, noignore]] operator vec<U, 3>() {
		return vec<U, 3>(static_cast<U>(x), static_cast<U>(y), static_cast<U>(z));
	}

#include "ops/vec_ops.inl"

	// DONT EDIT THIS FIELD LAYOUT
	union {
		Array<T, 3> _v;
#include "ops/swizzle3.inl"
	};
};

template<typename T>
struct alignas(16) [[builtin("vec")]] vec<T, 4> {
	using ThisType = vec<T, 4>;
	static constexpr uint32 dim = 4;
	vec() noexcept = default;

	template<typename... Args>
		requires(sum_dim<0ull, Args...>() == 4)
	explicit constexpr vec(Args&&... args)
		: _v()// active union member
	{
		set<0>(args...);
	}

	template<concepts::arithmetic_scalar U>
	explicit constexpr vec(U v)
		: _v(v, v, v, v) {}

	template<concepts::arithmetic_scalar U>
	[[nodiscard, noignore]] operator vec<U, 4>() {
		return vec<U, 4>(static_cast<U>(x), static_cast<U>(y), static_cast<U>(z), static_cast<U>(w));
	}

#include "ops/vec_ops.inl"

	// DONT EDIT THIS FIELD LAYOUT
	union {
		Array<T, 4> _v;
#include "ops/swizzle4.inl"
	};
};

template<typename T, uint64 N>
[[binop("ADD")]] vec<T, N> operator+(T, vec<T, N>);

template<typename T, uint64 N>
[[binop("MUL")]] vec<T, N> operator*(T, vec<T, N>);

template<typename... T>
auto make_vector(const T&... ts) {
	return vec<typename element_of<T...>::type, sum_dim_v<T...>>(ts...);
}

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
// using double2 = vec<double, 2>;
// using double3 = vec<double, 3>;
// using double4 = vec<double, 4>;
using int2 = vec<int32, 2>;
using int3 = vec<int32, 3>;
using int4 = vec<int32, 4>;
using uint2 = vec<uint32, 2>;
using uint3 = vec<uint32, 3>;
using uint4 = vec<uint32, 4>;
using bool2 = vec<bool, 2>;
using bool3 = vec<bool, 3>;
using bool4 = vec<bool, 4>;

};// namespace luisa::shader
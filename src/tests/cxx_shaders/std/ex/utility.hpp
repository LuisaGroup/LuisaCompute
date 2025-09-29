#pragma once

#ifdef __SHADER_LANG__
#include <std/cstddef>
#include <std/type_traits>
#include <std/utility>
#else
#include <cstddef>
#include <type_traits>
#include <utility>
#define trait_struct struct
#endif

#ifdef _MSC_VER

#ifndef O_EBO
#define O_EBO __declspec(empty_bases)
#endif

#ifndef O_FORCEINLINE
#define O_FORCEINLINE __forceinline
#endif

#ifndef O_NOINLINE
#define O_NOINLINE __declspec(noinline)
#endif

#ifndef __SHADER_LANG__
#ifndef O_RETURN_ADDRESS
#define O_RETURN_ADDRESS _ReturnAddress()
#endif
#endif

#ifndef O_CURRENT_LINE
#define O_CURRENT_LINE __builtin_LINE()
#endif

#ifndef O_CURRENT_COLUMN
#define O_CURRENT_COLUMN __builtin_COLUMN()
#endif

#ifndef O_CURRENT_FILE
#define O_CURRENT_FILE __builtin_FILE()
#endif

#ifndef O_CURRENT_FUNCTION
#define O_CURRENT_FUNCTION __builtin_FUNCTION()
#endif

#else

#ifndef O_EBO
#define O_EBO
#endif

#ifndef O_FORCEINLINE
#define O_FORCEINLINE inline __attribute__((always_inline))
#endif

#ifndef O_NOINLINE
#define O_NOINLINE __attribute__((noinline))
#endif

#ifndef __SHADER_LANG__
#ifndef O_RETURN_ADDRESS
#define O_RETURN_ADDRESS __builtin_return_address(0)
#endif
#endif

#ifndef O_CURRENT_LINE
#define O_CURRENT_LINE __builtin_LINE()
#endif

#ifndef O_CURRENT_COLUMN
#define O_CURRENT_COLUMN 0
#endif

#ifndef O_CURRENT_FILE
#define O_CURRENT_FILE __builtin_FILE()
#endif

#ifndef O_CURRENT_FUNCTION
#define O_CURRENT_FUNCTION __builtin_FUNCTION()
#endif

#endif

#ifndef O_UNREACHABLE
#ifdef _MSVC_STL_VERSION
#define O_UNREACHABLE _STL_UNREACHABLE
#else
#define O_UNREACHABLE __builtin_unreachable()
#endif
#endif

#ifndef __SHADER_LANG__
#ifndef O_MACHINE_PAUSE
#if (defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)) && !defined(_CHPE_ONLY_) && !defined(_M_ARM64EC)
#define O_MACHINE_PAUSE _mm_pause()
#elif defined(__ARM_ARCH_7A__) || defined(__aarch64__)
#define O_MACHINE_PAUSE __asm__ __volatile__("isb sy" ::: "memory")
#endif
#endif
#endif

#ifndef O_UNIQUE_TYPE
#ifdef __INTELLISENSE__
#define O_UNIQUE_TYPE decltype(nullptr)
#else
#define O_UNIQUE_TYPE decltype([] {})
#endif
#endif

namespace stdex {

template<size_t N, class T, class... Ts>
trait_struct get_type {
	using type = typename get_type<N - 1, Ts...>::type;
};

template<class T, class... Ts>
trait_struct get_type<0, T, Ts...> {
	using type = T;
};

template<size_t N, class... Ts>
using get_type_t = typename get_type<N, Ts...>::type;

template<class T, class... Ts>
trait_struct index_of;

template<class T, class... Ts>
trait_struct index_of<T, T, Ts...> : std::integral_constant<size_t, 0>{};

template<class T, class U, class... Ts>
trait_struct index_of<T, U, Ts...> : std::integral_constant<size_t, 1 + index_of<T, Ts...>::value>{};

template<size_t N, int Strategy>
trait_struct visit_strategy;

template<size_t N>
trait_struct visit_strategy<N, 0>{
	template<class Ret, class Fn, class... Args>
	static constexpr Ret invoke(size_t idx, Fn&& fn, Args&&... args){
		return static_cast<Fn&&>(fn).template operator()<0>(static_cast<Args&&>(args)...);
}// namespace stdex
}
;
#ifndef __SHADER_LANG__
inline void force_case() {}
#else
[[callop("FORCE_CASE")]] extern void force_case();// make next switch force-case (no if else)
#endif
#ifndef O_VISIT_CASE
#define O_VISIT_CASE(n)                                                                          \
	case (n):                                                                                    \
		if constexpr ((n) < N) {                                                                 \
			return static_cast<Fn&&>(fn).template operator()<(n)>(static_cast<Args&&>(args)...); \
		}                                                                                        \
		O_UNREACHABLE;                                                                           \
		[[fallthrough]]

#define O_VISIT_STAMP(stamper, n)           \
	static_assert(N > (n) / 4 && N <= (n)); \
	force_case();                           \
	switch (idx) {                          \
		stamper(0, O_VISIT_CASE);           \
		default:                            \
			O_UNREACHABLE;                  \
	}

#define O_STAMP4(n, x) \
	x(n);              \
	x(n + 1);          \
	x(n + 2);          \
	x(n + 3)
#define O_STAMP16(n, x) \
	O_STAMP4(n, x);     \
	O_STAMP4(n + 4, x); \
	O_STAMP4(n + 8, x); \
	O_STAMP4(n + 12, x)
#define O_STAMP64(n, x)   \
	O_STAMP16(n, x);      \
	O_STAMP16(n + 16, x); \
	O_STAMP16(n + 32, x); \
	O_STAMP16(n + 48, x)
#define O_STAMP256(n, x)   \
	O_STAMP64(n, x);       \
	O_STAMP64(n + 64, x);  \
	O_STAMP64(n + 128, x); \
	O_STAMP64(n + 192, x)

#define O_STAMP(n, x) x(O_STAMP##n, n)

template<size_t N>
trait_struct visit_strategy<N, 1>{
	template<class Ret, class Fn, class... Args>
	static constexpr Ret invoke(size_t idx, Fn&& fn, Args&&... args){
		O_STAMP(4, O_VISIT_STAMP);
#ifdef __SHADER_LANG__
if constexpr (!std::is_void_v<Ret>) {
	return Ret();// make compiler happy
}
#endif
}
}
;

template<size_t N>
trait_struct visit_strategy<N, 2>{
	template<class Ret, class Fn, class... Args>
	static constexpr Ret invoke(size_t idx, Fn&& fn, Args&&... args){
		O_STAMP(16, O_VISIT_STAMP);
#ifdef __SHADER_LANG__
if constexpr (!std::is_void_v<Ret>) {
	return Ret();// make compiler happy
}
#endif
}
}
;

template<size_t N>
trait_struct visit_strategy<N, 3>{
	template<class Ret, class Fn, class... Args>
	static constexpr Ret invoke(size_t idx, Fn&& fn, Args&&... args){
		O_STAMP(64, O_VISIT_STAMP);
#ifdef __SHADER_LANG__
if constexpr (!std::is_void_v<Ret>) {
	return Ret();// make compiler happy
}
#endif
}
}
;

template<size_t N>
trait_struct visit_strategy<N, 4>{
	template<class Ret, class Fn, class... Args>
	static constexpr Ret invoke(size_t idx, Fn&& fn, Args&&... args){
		O_STAMP(256, O_VISIT_STAMP);
#ifdef __SHADER_LANG__
if constexpr (!std::is_void_v<Ret>) {
	return Ret();// make compiler happy
}
#endif
}
}
;

#undef O_VISIT_CASE
#undef O_VISIT_STAMP
#undef O_STAMP
#undef O_STAMP256
#undef O_STAMP64
#undef O_STAMP16
#undef O_STAMP4
#endif

template<size_t N>
trait_struct priority_t : priority_t<N - 1>{};

template<>
trait_struct priority_t<0>{};

template<size_t N>
constexpr priority_t<N> priority{};

template<class... Ts>
trait_struct overloaded : Ts... {
	using is_transparent = void;
	using Ts::operator()...;
};
template<class... Ts>
trait_struct opaque_overloaded : Ts... {
	using Ts::operator()...;
};

template<class... Ts, class Fn>
constexpr void unroll_type(Fn&& fn) {
	[&]<size_t... I>(std::index_sequence<I...>) {
		(void(static_cast<Fn&&>(fn).template operator()<Ts>(I)), ...);
	}(std::index_sequence_for<Ts...>());
}

template<size_t N, class Fn>
constexpr void unroll(Fn&& fn) {
	[&]<size_t... I>(std::index_sequence<I...>) {
		(void(static_cast<Fn&&>(fn)(I)), ...);
	}(std::make_index_sequence<N>());
}

template<size_t N, class Fn, class... Args>
	requires(N <= 256)
constexpr decltype(auto) visit_index(size_t index, Fn&& fn, Args&&... args) {
	constexpr int s = N == 1 ? 0 : N <= 4 ? 1 :
							   N <= 16	  ? 2 :
							   N <= 64	  ? 3 :
											4;
	using Ret = decltype((std::declval<Fn>().template operator()<0>(std::declval<Args>()...)));
	return (visit_strategy<N, s>::template invoke<Ret>(index, static_cast<Fn&&>(fn), static_cast<Args&&>(args)...));
}

template<class Ret, size_t N, class Fn, class... Args>
	requires(N <= 256)
constexpr Ret visit_index(size_t index, Fn&& fn, Args&&... args) {
	constexpr int s = N == 1 ? 0 : N <= 4 ? 1 :
							   N <= 16	  ? 2 :
							   N <= 64	  ? 3 :
											4;
	return visit_strategy<N, s>::template invoke<Ret>(index, static_cast<Fn&&>(fn), static_cast<Args&&>(args)...);
}

}// namespace stdex
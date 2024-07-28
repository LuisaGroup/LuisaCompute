#pragma once
#include "./../attributes.hpp"
#include "builtins.hpp"

namespace luisa::shader {

trait ArrayFlags {
	static constexpr uint32 None = 0;
	static constexpr uint32 Shared = 1;
};

trait CacheFlags {
	static constexpr uint32 None = 0;
	static constexpr uint32 Coherent = 1;
};

template<typename T, uint64 N>
struct vec;

template<uint64 N>
struct matrix;

template<typename Type, uint32 size, uint32 Flags>
struct Array;

struct Ray;
struct Accel;
struct CommittedHit;
struct TriangleHit;
struct ProceduralHit;

namespace detail {
template<class T>
trait vec_or_matrix : public false_type {
	using scalar_type = T;
	static constexpr bool is_vec = false;
	static constexpr bool is_matrix = false;
};

template<class T, uint64 N>
trait vec_or_matrix<vec<T, N>> : public true_type {
	using scalar_type = T;
	static constexpr bool is_vec = true;
	static constexpr bool is_matrix = false;
};

template<uint64 N>
trait vec_or_matrix<matrix<N>> : public true_type {
	using scalar_type = float;
	static constexpr bool is_vec = false;
	static constexpr bool is_matrix = true;
};
template<typename T>
trait is_char {
	static constexpr bool value = false;
};
template<size_t n>
trait is_char<const char (&)[n]> {
	static constexpr bool value = true;
};
}// namespace detail

template<typename T>
using scalar_type = typename detail::vec_or_matrix<decay_t<T>>::scalar_type;

template<typename T>
static constexpr bool is_scalar_v = !detail::vec_or_matrix<decay_t<T>>::is_vec;

template<typename T>
static constexpr bool is_vec_v = detail::vec_or_matrix<decay_t<T>>::is_vec;

template<typename T>
static constexpr bool is_matrix_v = detail::vec_or_matrix<decay_t<T>>::is_matrix;

template<typename T>
static constexpr bool is_vec_or_matrix_v = detail::vec_or_matrix<decay_t<T>>::value;

template<typename T>
inline constexpr bool is_shared_array_v = false;

template<typename U, uint32 N, uint32 F>
inline constexpr bool is_shared_array_v<Array<U, N, F>> = F & ArrayFlags::Shared;

template<typename T>
inline constexpr bool is_array_v = false;

template<typename U, uint32 N, uint32 F>
inline constexpr bool is_array_v<Array<U, N, F>> = true;

template<typename T>
static constexpr bool is_float_family_v = is_same_v<scalar_type<T>, float> | is_same_v<scalar_type<T>, double>;

template<typename T>
static constexpr bool is_sint_family_v = is_same_v<scalar_type<T>, int16> | is_same_v<scalar_type<T>, int32> | is_same_v<scalar_type<T>, int64>;

template<typename T>
static constexpr bool is_uint_family_v = is_same_v<scalar_type<T>, uint16> | is_same_v<scalar_type<T>, uint32> | is_same_v<scalar_type<T>, uint64>;

template<typename T>
static constexpr bool is_int_family_v = is_sint_family_v<T> || is_uint_family_v<T>;

template<typename T>
static constexpr bool is_bool_family_v = is_same_v<scalar_type<T>, bool>;

template<typename T>
static constexpr bool is_arithmetic_v = is_float_family_v<T> || is_bool_family_v<T> || is_int_family_v<T>;

template<typename T>
static constexpr bool is_signed_arithmetic_v = is_float_family_v<T> || is_sint_family_v<T>;

template<typename T>
static constexpr bool is_arithmetic_scalar_v = is_arithmetic_v<T> && !is_vec_v<T>;

namespace concepts {

template<typename T>
concept vec = is_vec_v<T>;

template<typename T>
concept non_vec = !is_vec_v<T>;

template<typename T>
concept matrix = is_matrix_v<T>;

template<typename T>
concept array = is_array_v<T>;

template<typename T>
concept float_family = is_float_family_v<T>;

template<typename T>
concept float_vec_family = is_float_family_v<T> && is_vec_v<T>;

template<typename T>
concept bool_family = is_bool_family_v<T>;

template<typename T>
concept bool_vec_family = is_bool_family_v<T> && is_vec_v<T>;

template<typename T>
concept sint_family = is_sint_family_v<T>;

template<typename T>
concept sint_vec_family = is_sint_family_v<T> && is_vec_v<T>;

template<typename T>
concept uint_family = is_uint_family_v<T>;

template<typename T>
concept uint_vec_family = is_uint_family_v<T> && is_vec_v<T>;

template<typename T>
concept int_family = is_int_family_v<T>;

template<typename T>
concept int_vec_family = is_int_family_v<T> && is_vec_v<T>;

template<typename T>
concept arithmetic = is_arithmetic_v<T>;

template<typename T>
concept arithmetic_vec = is_arithmetic_v<T> && is_vec_v<T>;

template<typename T>
concept signed_arithmetic = is_signed_arithmetic_v<T>;

template<typename T>
concept arithmetic_scalar = is_arithmetic_scalar_v<T>;

template<typename T>
concept primitive = is_arithmetic_v<T> || is_vec_or_matrix_v<T>;

template<typename T>
concept string_literal = detail::is_char<T>::value;

template<typename T>
concept can_be_volatile_v = is_same_v<T, int32> || is_same_v<T, int64> || is_same_v<T, uint32> || is_same_v<T, uint64>;
}// namespace concepts
template<typename Func, typename T>
trait invocable;

template<typename Func, typename Ret, typename... Args>
trait invocable<Func, Ret(Args...)> {
	static constexpr bool value = is_same_v<Func, Ret(Args...)>;
};
template<typename Func, typename T>
trait functor_invocable;

template<typename Func, typename Ret, typename... Args>
trait functor_invocable<Func, Ret(Args...)> {
	static constexpr bool value = is_same_v<Func, Ret(uint64, Args...)>;
};
template<typename Func, typename T>
static constexpr bool invocable_v = invocable<Func, T>::value;
template<typename Func, typename T>
static constexpr bool functor_invocable_v = functor_invocable<Func, T>::value;
}// namespace luisa::shader
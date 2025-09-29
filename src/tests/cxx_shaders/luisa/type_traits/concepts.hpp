#pragma once
#include "./../attributes.hpp"
#include "builtins.hpp"

#include <std/type_traits>

namespace luisa::shader {

trait_struct ArrayFlags {
	static constexpr uint32 None = 0;
	static constexpr uint32 Shared = 1;
};

trait_struct CacheFlags {
	static constexpr uint32 None = 0;
	static constexpr uint32 Coherent = 1;
	static constexpr uint32 ReadOnly = 2;
};

template<typename T, uint64 N>
struct vec;

template<uint64 N>
struct matrix;

template<typename Type, uint32 cache_flags>
struct Buffer;

struct Ray;
struct Accel;
struct CommittedHit;
struct TriangleHit;
struct ProceduralHit;
struct IndirectBuffer;

namespace detail {
template<class T>
trait_struct vec_or_matrix : public std::false_type {
	using scalar_type = T;
	static constexpr bool is_vec = false;
	static constexpr bool is_matrix = false;
};

template<class T, uint64 N>
trait_struct vec_or_matrix<vec<T, N>> : public std::true_type {
	using scalar_type = T;
	static constexpr bool is_vec = true;
	static constexpr bool is_matrix = false;
};

template<uint64 N>
trait_struct vec_or_matrix<matrix<N>> : public std::true_type {
	using scalar_type = float;
	static constexpr bool is_vec = false;
	static constexpr bool is_matrix = true;
};
#ifdef DEBUG
template<typename T>
trait_struct is_char {
	static constexpr bool value = false;
};
template<size_t n>
trait_struct is_char<const char (&)[n]> {
	static constexpr bool value = true;
};
#endif
}// namespace detail

template<typename T>
using scalar_type = typename detail::vec_or_matrix<std::decay_t<T>>::scalar_type;

template<typename T, template<typename, uint32> typename Template>
inline constexpr bool is_specialization_resource_v = false;// true if && only if T is a specialization of Template

template<template<typename, uint32> typename Template, typename Arg>
inline constexpr bool is_specialization_resource_v<Template<Arg, 0u>, Template> = true;

template<typename T>
static constexpr bool is_scalar_v = !detail::vec_or_matrix<std::decay_t<T>>::is_vec;

template<typename T>
static constexpr bool is_vec_v = detail::vec_or_matrix<std::decay_t<T>>::is_vec;

template<typename T>
static constexpr bool is_matrix_v = detail::vec_or_matrix<std::decay_t<T>>::is_matrix;

template<typename T>
static constexpr bool is_vec_or_matrix_v = detail::vec_or_matrix<std::decay_t<T>>::value;

template<typename T>
inline constexpr bool is_buffer_v = is_specialization_resource_v<T, Buffer>;

template<typename T>
static constexpr bool is_float_family_v = std::is_same_v<scalar_type<T>, float> | std::is_same_v<scalar_type<T>, double> | std::is_same_v<scalar_type<T>, half>;

template<typename T>
static constexpr bool is_sint_family_v = std::is_same_v<scalar_type<T>, int16> | std::is_same_v<scalar_type<T>, int32> | std::is_same_v<scalar_type<T>, int64>;

template<typename T>
static constexpr bool is_uint_family_v = std::is_same_v<scalar_type<T>, uint16> | std::is_same_v<scalar_type<T>, uint32> | std::is_same_v<scalar_type<T>, uint64>;

template<typename T>
static constexpr bool is_int_family_v = is_sint_family_v<T> || is_uint_family_v<T>;

template<typename T>
static constexpr bool is_bool_family_v = std::is_same_v<scalar_type<T>, bool>;

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
concept buffer = is_buffer_v<T>;

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
#ifdef DEBUG
template<typename T>
concept string_literal = detail::is_char<T>::value;
#endif
}// namespace concepts

template<concepts::arithmetic_scalar T, uint32 cache_flag>
struct Image;

template<concepts::arithmetic_scalar T, uint32 cache_flag>
struct Volume;

template<typename T>
inline constexpr bool is_image_v = is_specialization_resource_v<T, Image>;

template<typename T>
inline constexpr bool is_volume_v = is_specialization_resource_v<T, Volume>;

template<typename T>
inline constexpr bool is_texture_v = is_image_v<T> || is_volume_v<T>;

namespace concepts {

template<typename T>
concept image = is_image_v<T>;

template<typename T>
concept volume = is_volume_v<T>;

template<typename T>
concept texture = is_texture_v<T>;

}// namespace concepts

}// namespace luisa::shader
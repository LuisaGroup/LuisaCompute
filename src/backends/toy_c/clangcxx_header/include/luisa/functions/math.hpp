#pragma once
#include "./../attributes.hpp"
#include "./../type_traits.hpp"
#include "./../types/vec.hpp"
#include "./../types/matrix.hpp"

namespace luisa::shader {

template<concepts::arithmetic T, concepts::arithmetic U>
	requires(sizeof(T) == sizeof(U))
[[expr("bit_cast")]] extern T bit_cast(U v);

template<concepts::bool_family T>
[[callop("ALL")]] extern bool all(const T& x);

template<concepts::bool_family T>
[[callop("ANY")]] extern bool any(const T& x);

template<concepts::primitive T, concepts::bool_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("SELECT")]] extern T select(const T& false_v, const T& true_v, const B& bool_v);

template<concepts::primitive T, concepts::bool_family B>
	requires(vec_dim_v<T> == vec_dim_v<B> || vec_dim_v<B> == 1)
extern T ite(B bool_v, T true_v, T false_v) {
	return select(false_v, true_v, bool_v);
}

template<concepts::arithmetic T, concepts::arithmetic A, concepts::arithmetic B>
	requires((same_dim_v<T, B> || is_scalar_v<B>) && (same_dim_v<T, A> || is_scalar_v<A>))
[[callop("CLAMP")]] extern T clamp(const T& v, const A& min_v, const B& max_v);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("LERP")]] extern T lerp(const T& left_v, const T& right_v, const B& step);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("SMOOTHSTEP")]] extern T smoothstep(const T& left_v, const B& right_v, const B& step);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("STEP")]] extern T step(const T& left_v, const B& right_v);

template<concepts::float_family T>
[[callop("SATURATE")]] extern T saturate(const T& v);

template<concepts::signed_arithmetic T>
[[callop("ABS")]] extern T abs(const T& v);

template<concepts::arithmetic T, concepts::arithmetic U>
[[callop("MIN")]] extern T min(const T& a, U b)
	requires(vec_dim_v<T> == vec_dim_v<U>);

template<concepts::arithmetic T, concepts::arithmetic U>
[[callop("MAX")]] extern T max(const T& v, U b)
	requires(vec_dim_v<T> == vec_dim_v<U>);

template<concepts::uint_family T>
[[callop("CLZ")]] extern T clz(const T& v);

template<concepts::uint_family T>
[[callop("CTZ")]] extern T ctz(const T& v);

template<concepts::uint_family T>
[[callop("POPCOUNT")]] extern T popcount(const T& v);

template<concepts::uint_family T>
[[callop("REVERSE")]] extern T reverse(const T& v);

template<concepts::float_vec_family T>
[[callop("ISINF")]] extern vec<bool, vec_dim_v<T>> is_inf(const T& v);

template<concepts::float_vec_family T>
[[callop("ISNAN")]] extern vec<bool, vec_dim_v<T>> is_nan(const T& v);

[[callop("ISINF")]] extern bool is_inf(float v);

[[callop("ISNAN")]] extern bool is_nan(float v);

template<concepts::float_family T>
[[callop("ACOS")]] extern T acos(const T& v);

template<concepts::float_family T>
[[callop("ACOSH")]] extern T acosh(const T& v);

template<concepts::float_family T>
[[callop("ASIN")]] extern T asin(const T& v);

template<concepts::float_family T>
[[callop("ASINH")]] extern T asinh(const T& v);

template<concepts::float_family T>
[[callop("ATAN")]] extern T atan(const T& v);

template<concepts::float_family T, concepts::float_family U>
[[callop("ATAN2")]] extern T atan2(const T& a, const U& b)
	requires(vec_dim_v<T> == vec_dim_v<U>);

template<concepts::float_family T>
[[callop("ATANH")]] extern T atanh(const T& v);

template<concepts::float_family T>
[[callop("COS")]] extern T cos(const T& v);

template<concepts::float_family T>
[[callop("COSH")]] extern T cosh(const T& v);

template<concepts::float_family T>
[[callop("SIN")]] extern T sin(const T& v);

template<concepts::float_family T>
[[callop("SINH")]] extern T sinh(const T& v);

template<concepts::float_family T>
[[callop("TAN")]] extern T tan(const T& v);

template<concepts::float_family T>
[[callop("TANH")]] extern T tanh(const T& v);

template<concepts::float_family T>
[[callop("EXP")]] extern T exp(const T& v);

template<concepts::float_family T>
[[callop("EXP2")]] extern T exp2(const T& v);

template<concepts::float_family T>
[[callop("EXP10")]] extern T exp10(const T& v);

template<concepts::float_family T>
[[callop("LOG")]] extern T log(const T& v);

template<concepts::float_family T>
[[callop("LOG2")]] extern T log2(const T& v);

template<concepts::float_family T>
[[callop("LOG10")]] extern T log10(const T& v);

template<concepts::float_family T, concepts::float_family B>
	requires(same_dim_v<T, B> || is_scalar_v<B>)
[[callop("POW")]] extern T pow(const T& base, const B& rate);

template<concepts::float_family T>
[[callop("SQRT")]] extern T sqrt(const T& v);

template<concepts::float_family T>
[[callop("RSQRT")]] extern T rsqrt(const T& v);

template<concepts::float_family T>
[[callop("CEIL")]] extern T ceil(const T& v);

template<concepts::float_family T>
[[callop("FLOOR")]] extern T floor(const T& v);

template<concepts::float_family T>
[[callop("FRACT")]] extern T fract(const T& v);

template<concepts::float_family T>
[[callop("TRUNC")]] extern T trunc(const T& v);

template<concepts::float_family T>
[[callop("ROUND")]] extern T round(const T& v);

template<concepts::float_family T>
[[callop("FMA")]] extern T fma(const T& a, const T& b, const T& c);

template<concepts::float_family T>
[[callop("COPYSIGN")]] extern T copysign(const T& a, T b);

template<concepts::float_vec_family T>
[[callop("CROSS")]] extern T cross(const T& a, const T& b);

[[callop("FACEFORWARD")]] extern float3 faceforward(const float3& a, const float3& b, const float3& c);

[[callop("REFLECT")]] extern float3 reflect(const float3& i, const float3& n);
template<concepts::float_vec_family T>
[[callop("DOT")]] extern scalar_type<T> dot(const T& a, const T& b);

template<concepts::float_vec_family T>
[[callop("LENGTH")]] extern scalar_type<T> length(const T& v);

template<concepts::float_vec_family T>
scalar_type<T> distance(const T& a, const T& b) { return length(a - b); }

template<concepts::float_vec_family T>
[[callop("LENGTH_SQUARED")]] extern scalar_type<T> length_squared(const T& v);

template<concepts::float_vec_family T>
[[callop("NORMALIZE")]] extern T normalize(const T& v);
template<concepts::matrix T>
[[callop("DETERMINANT")]] extern T determinant(const T& v);

template<concepts::matrix T>
[[callop("TRANSPOSE")]] extern T transpose(const T& v);

template<concepts::matrix T>
[[callop("INVERSE")]] extern T inverse(const T& v);

template<typename T>
[[callop("ADDRESS_OF")]] uint64 address_of(T& ref);

template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_FETCH_ADD")]] T atomic_add(T& ref, T rhs);
template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_FETCH_SUB")]] T atomic_sub(T& ref, T rhs);
template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_FETCH_AND")]] T atomic_and(T& ref, T rhs);
template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_FETCH_OR")]] T atomic_or(T& ref, T rhs);
template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_FETCH_XOR")]] T atomic_xor(T& ref, T rhs);
template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_EXCHANGE")]] T atomic_exchange(T& ref, T desire);
template<concepts::can_be_volatile_v T>
[[callop("ATOMIC_COMPARE_EXCHANGE")]] T atomic_compare_exchange(T& ref, T expected, T desire);


float3 refract(float3 I, float3 N, float eta) {
	float3 result = float3(0.0);
	float k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));
	if (k > 0.0)
		result = eta * I - (eta * dot(N, I) + sqrt(k)) * N;
	return result;
}

template<concepts::float_family T>
auto sign(T v) {
	using Type = copy_dim<int, T>::type;
	using MaskType = copy_dim<uint, T>::type;
	auto low = select(Type(1), Type(0), (bit_cast<MaskType>(v) & MaskType(0x7fffffffu)) == MaskType(0));
	auto high = select(Type(2), Type(0), (bit_cast<MaskType>(v) & MaskType(0x80000000u)) == MaskType(0));
	return low - high;
}

template<concepts::float_family T>
T rcp(T t) {
	return T(1.f) / t;
}
}// namespace luisa::shader
#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

[[expr("dispatch_id")]] extern uint3 dispatch_id();
template <boolN T> 
[[callop("ALL")]] extern bool all(T x);
template <boolN T> 
[[callop("ANY")]] extern bool any(T x);
template <basic_type T, boolN B>
requires(vec_dim<T>::value == vec_dim<B>::value)
[[callop("SELECT")]] extern T select(T false_v, T true_v, B bool_v);
template<arithmetic T>
[[callop("CLAMP")]] extern T clamp(T v, T min_v, T max_v);
template<floatN T>
[[callop("LERP")]] extern T lerp(T left_v, T right_v, T step);
template<floatN T>
[[callop("SMOOTHSTEP")]] extern T smoothstep(T left_v, T right_v, T step);
template<floatN T>
[[callop("SATURATE")]] extern T saturate(T v);
template<typename T>
requires(is_floatN<T>::value || is_intN<T>::value)
[[callop("ABS")]] extern T abs(T v);
template<arithmetic T>
[[callop("MIN")]] extern T min(T v);
template<arithmetic T>
[[callop("MAX")]] extern T max(T v);
template <uintN T>
[[callop("CLZ")]] extern T clz(T v);
template <uintN T>
[[callop("CTZ")]] extern T ctz(T v);
template <uintN T>
[[callop("POPCOUNT")]] extern T popcount(T v);
template <uintN T>
[[callop("REVERSE")]] extern T reverse(T v);
template <floatN T>
[[callop("ISINF")]] extern vec<bool, vec_dim<T>::value> is_inf(T v);
template <floatN T>
[[callop("ISNAN")]] extern vec<bool, vec_dim<T>::value> is_nan(T v);
template <floatN T>
[[callop("ACOS")]] extern T acos(T v);
template <floatN T>
[[callop("ACOSH")]] extern T acosh(T v);
template <floatN T>
[[callop("ASIN")]] extern T  asin(T v);
template <floatN T>
[[callop("ASINH")]] extern T asinh(T v);
template <floatN T>
[[callop("ATAN")]] extern T  atan(T v);
template <floatN T>
[[callop("ATAN2")]] extern T atan2(T v);
template <floatN T>
[[callop("ATANH")]] extern T atanh(T v);
template <floatN T>
[[callop("COS")]] extern T cos(T v);
template <floatN T>
[[callop("COSH")]] extern T cosh(T v);
template <floatN T>
[[callop("SIN")]] extern T sin(T v);
template <floatN T>
[[callop("SINH")]] extern T sinh(T v);
template <floatN T>
[[callop("TAN")]] extern T tan(T v);
template <floatN T>
[[callop("TANH")]] extern T tanh(T v);
template <floatN T>
[[callop("EXP")]] extern T exp(T v);
template <floatN T>
[[callop("EXP2")]] extern T exp2(T v);
template <floatN T>
[[callop("EXP10")]] extern T exp10(T v);
template <floatN T>
[[callop("LOG")]] extern T log(T v);
template <floatN T>
[[callop("LOG2")]] extern T log2(T v);
template <floatN T>
[[callop("LOG10")]] extern T log10(T v);
template <floatN T>
[[callop("POW")]] extern T pow(T v);
template <floatN T>
[[callop("SQRT")]] extern T sqrt(T v);
template <floatN T>
[[callop("RSQRT")]] extern T rsqrt(T v);
template <floatN T>
[[callop("CEIL")]] extern T ceil(T v);
template <floatN T>
[[callop("FLOOR")]] extern T floor(T v);
template <floatN T>
[[callop("FRACT")]] extern T fract(T v);
template <floatN T>
[[callop("TRUNC")]] extern T trunc(T v);
template <floatN T>
[[callop("ROUND")]] extern T round(T v);
template <floatN T>
[[callop("FMA")]] extern T fma(T a, T b, T c);
template <floatN T>
[[callop("COPYSIGN")]] extern T fma(T a, T b);
template <floatN T>
requires(vec_dim<T>::value > 1)
[[callop("CROSS")]] extern T cross(T a, T b);
[[callop("FACEFORWARD")]] extern float3 faceforward(float3 a, float3 b, float3 c);
[[callop("FACEFORWARD")]] extern half3 faceforward(half3 a, half3 b, half3 c);
[[callop("REFLECT")]] extern float3 reflect(float3 i, float3 n);
[[callop("REFLECT")]] extern half3 reflect(half3 i, half3 n);
template <floatN T>
requires(vec_dim<T>::value > 1)
[[callop("DOT")]] extern typename element<T>::type dot(T a, T b);
template <floatN T>
requires(vec_dim<T>::value > 1)
[[callop("LENGTH")]] extern typename element<T>::type length(T v);
template <floatN T>
requires(vec_dim<T>::value > 1)
[[callop("LENGTH_SQUARED")]] extern typename element<T>::type length_squared(T v);
template <floatN T>
requires(vec_dim<T>::value > 1)
[[callop("NORMALIZE")]] extern typename element<T>::type normalize(T v);
template <arithmetic T>
requires(vec_dim<T>::value > 1)
[[callop("REDUCE_SUM")]] extern typename element<T>::type reduce_sum(T v);
template <arithmetic T>
requires(vec_dim<T>::value > 1)
[[callop("REDUCE_PRODUCT")]] extern typename element<T>::type reduce_product(T v);
template <arithmetic T>
requires(vec_dim<T>::value > 1)
[[callop("REDUCE_MIN")]] extern typename element<T>::type reduce_min(T v);
template <arithmetic T>
requires(vec_dim<T>::value > 1)
[[callop("REDUCE_MAX")]] extern typename element<T>::type reduce_max(T v);
template <typename T>
requires(is_matrix<T>::value)
[[callop("DETERMINANT")]] extern typename T determinant(T v);
template <typename T>
requires(is_matrix<T>::value)
[[callop("TRANSPOSE")]] extern typename T transpose(T v);
template <typename T>
requires(is_matrix<T>::value)
[[callop("INVERSE")]] extern typename T inverse(T v);
[[callop("SYNCHRONIZE_BLOCK")]] void sync_block();
}// namespace luisa::shader
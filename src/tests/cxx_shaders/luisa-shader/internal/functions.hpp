#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"
#include "vec.hpp"
#include "matrix.hpp"

namespace luisa::shader {

[[expr("dispatch_id")]] extern uint3 dispatch_id();

[[expr("block_id")]] extern uint3 block_id();

[[expr("thread_id")]] extern uint3 thread_id();

[[expr("dispatch_size")]] extern uint3 dispatch_size();

[[expr("kernel_id")]] extern uint32 kernel_id();

[[expr("warp_lane_count")]] extern uint32 warp_lane_count();
[[expr("warp_lane_count")]] extern uint32 wave_lane_count();

[[expr("warp_lane_id")]] extern uint32 warp_lane_id();
[[expr("warp_lane_id")]] extern uint32 wave_lane_id();

template<arithmetic T, arithmetic U>
[[expr("bit_cast")]] extern T bit_cast(U v);

template<bool_family T>
[[callop("ALL")]] extern bool all(T x);

template<bool_family T>
[[callop("ANY")]] extern bool any(T x);

template<primitive T, bool_family B>
    requires(vec_dim_v<T> == vec_dim_v<B> || vec_dim_v<B> == 1)
[[callop("SELECT")]] extern T select(T false_v, T true_v, B bool_v);

template<arithmetic T>
[[callop("CLAMP")]] extern T clamp(T v, T min_v, T max_v);

template<float_family T, float_family B>
    requires(vec_dim_v<T> == vec_dim_v<B> || vec_dim_v<B> == 1)
[[callop("LERP")]] extern T lerp(T left_v, T right_v, B step);

template<float_family T>
[[callop("SMOOTHSTEP")]] extern T smoothstep(T left_v, T right_v, T step);

template<float_family T>
[[callop("SATURATE")]] extern T saturate(T v);

template<typename T>
    requires(is_float_family_v<T> || is_int_family_v<T>)
[[callop("ABS")]] extern T abs(T v);

template<arithmetic T>
[[callop("MIN")]] extern T min(T a, T b);

template<arithmetic T>
[[callop("MAX")]] extern T max(T v, T b);

template<uint_family T>
[[callop("CLZ")]] extern T clz(T v);

template<uint_family T>
[[callop("CTZ")]] extern T ctz(T v);

template<uint_family T>
[[callop("POPCOUNT")]] extern T popcount(T v);

template<uint_family T>
[[callop("REVERSE")]] extern T reverse(T v);

template<float_family T>
[[callop("ISINF")]] extern vec<bool, vec_dim_v<T>> is_inf(T v);

template<float_family T>
[[callop("ISNAN")]] extern vec<bool, vec_dim_v<T>> is_nan(T v);

template<float_family T>
[[callop("ACOS")]] extern T acos(T v);

template<float_family T>
[[callop("ACOSH")]] extern T acosh(T v);

template<float_family T>
[[callop("ASIN")]] extern T asin(T v);

template<float_family T>
[[callop("ASINH")]] extern T asinh(T v);

template<float_family T>
[[callop("ATAN")]] extern T atan(T v);

template<float_family T>
[[callop("ATAN2")]] extern T atan2(T v);

template<float_family T>
[[callop("ATANH")]] extern T atanh(T v);
template<float_family T>
[[callop("COS")]] extern T cos(T v);
template<float_family T>
[[callop("COSH")]] extern T cosh(T v);
template<float_family T>
[[callop("SIN")]] extern T sin(T v);
template<float_family T>
[[callop("SINH")]] extern T sinh(T v);
template<float_family T>
[[callop("TAN")]] extern T tan(T v);

template<float_family T>
[[callop("TANH")]] extern T tanh(T v);

template<float_family T>
[[callop("EXP")]] extern T exp(T v);

template<float_family T>
[[callop("EXP2")]] extern T exp2(T v);

template<float_family T>
[[callop("EXP10")]] extern T exp10(T v);

template<float_family T>
[[callop("LOG")]] extern T log(T v);

template<float_family T>
[[callop("LOG2")]] extern T log2(T v);

template<float_family T>
[[callop("LOG10")]] extern T log10(T v);

template<float_family T>
[[callop("POW")]] extern T pow(T v);

template<float_family T>
[[callop("SQRT")]] extern T sqrt(T v);

template<float_family T>
[[callop("RSQRT")]] extern T rsqrt(T v);

template<float_family T>
[[callop("CEIL")]] extern T ceil(T v);

template<float_family T>
[[callop("FLOOR")]] extern T floor(T v);

template<float_family T>
[[callop("FRACT")]] extern T fract(T v);

template<float_family T>
[[callop("TRUNC")]] extern T trunc(T v);

template<float_family T>
[[callop("ROUND")]] extern T round(T v);

template<float_family T>
[[callop("FMA")]] extern T fma(T a, T b, T c);

template<float_family T>
[[callop("COPYSIGN")]] extern T fma(T a, T b);

template<float_family T>
    requires(vec_dim_v<T> > 1)
[[callop("CROSS")]] extern T cross(T a, T b);

[[callop("FACEFORWARD")]] extern float3 faceforward(float3 a, float3 b, float3 c);

[[callop("FACEFORWARD")]] extern half3 faceforward(half3 a, half3 b, half3 c);

[[callop("REFLECT")]] extern float3 reflect(float3 i, float3 n);

[[callop("REFLECT")]] extern half3 reflect(half3 i, half3 n);

template<float_family T>
    requires(vec_dim_v<T> > 1)
[[callop("DOT")]] extern scalar_type<T> dot(T a, T b);

template<float_family T>
    requires(vec_dim_v<T> > 1)
[[callop("LENGTH")]] extern scalar_type<T> length(T v);

template<float_family T>
    requires(vec_dim_v<T> > 1)
scalar_type<T> distance(T a, T b){
    return length(a - b);
}

template<float_family T>
    requires(vec_dim_v<T> > 1)
[[callop("LENGTH_SQUARED")]] extern scalar_type<T> length_squared(T v);

template<float_family T>
    requires(vec_dim_v<T> > 1)
[[callop("NORMALIZE")]] extern T normalize(T v);

template<arithmetic T>
    requires(vec_dim_v<T> > 1)
[[callop("REDUCE_SUM")]] extern scalar_type<T> reduce_sum(T v);

template<arithmetic T>
    requires(vec_dim_v<T> > 1)
[[callop("REDUCE_PRODUCT")]] extern scalar_type<T> reduce_product(T v);

template<arithmetic T>
    requires(vec_dim_v<T> > 1)
[[callop("REDUCE_MIN")]] extern scalar_type<T> reduce_min(T v);

template<arithmetic T>
    requires(vec_dim_v<T> > 1)
[[callop("REDUCE_MAX")]] extern scalar_type<T> reduce_max(T v);

template<typename T>
    requires(is_matrix_v<T>)
[[callop("DETERMINANT")]] extern T determinant(T v);

template<typename T>
    requires(is_matrix_v<T>)
[[callop("TRANSPOSE")]] extern T transpose(T v);

template<typename T>
    requires(is_matrix_v<T>)
[[callop("INVERSE")]] extern T inverse(T v);

[[callop("SYNCHRONIZE_BLOCK")]] void sync_block();

// raster
[[callop("RASTER_DISCARD")]] void discard();

template<float_family T>
[[callop("DDX")]] T ddx();

template<float_family T>
[[callop("DDY")]] T ddy();

// warp
[[callop("WARP_IS_FIRST_ACTIVE_LANE")]] bool warp_is_first_active_lane();
[[callop("WARP_IS_FIRST_ACTIVE_LANE")]] bool wave_is_first_lane();

template<arithmetic T>
[[callop("WARP_ACTIVE_ALL_EQUAL")]] vec<bool, vec_dim_v<T>> warp_active_all_equal();
template<arithmetic T>
[[callop("WARP_ACTIVE_ALL_EQUAL")]] vec<bool, vec_dim_v<T>> wave_active_all_equal();

template<typename T>
    requires(is_int_family_v<T> && is_uint_family_v<T>)
[[callop("WARP_ACTIVE_BIT_AND")]] extern T warp_active_bit_and(T v);
template<typename T>
    requires(is_int_family_v<T> && is_uint_family_v<T>)
[[callop("WARP_ACTIVE_BIT_AND")]] extern T wave_active_bit_and(T v);

template<typename T>
    requires(is_int_family_v<T> && is_uint_family_v<T>)
[[callop("WARP_ACTIVE_BIT_OR")]] extern T warp_active_bit_or(T v);
template<typename T>
    requires(is_int_family_v<T> && is_uint_family_v<T>)
[[callop("WARP_ACTIVE_BIT_OR")]] extern T wave_active_bit_or(T v);

template<typename T>
    requires(is_int_family_v<T> && is_uint_family_v<T>)
[[callop("WARP_ACTIVE_BIT_XOR")]] extern T warp_active_bit_xor(T v);
template<typename T>
    requires(is_int_family_v<T> && is_uint_family_v<T>)
[[callop("WARP_ACTIVE_BIT_XOR")]] extern T wave_active_bit_xor(T v);

[[callop("WARP_ACTIVE_COUNT_BITS")]] extern uint32 warp_active_count_bits(bool val);
[[callop("WARP_ACTIVE_COUNT_BITS")]] extern uint32 wave_active_count_bits(bool val);

template<arithmetic T>
[[callop("WARP_ACTIVE_MAX")]] extern T warp_active_max(T v);
template<arithmetic T>
[[callop("WARP_ACTIVE_MAX")]] extern T wave_active_max(T v);

template<arithmetic T>
[[callop("WARP_ACTIVE_MIN")]] extern T warp_active_min(T v);
template<arithmetic T>
[[callop("WARP_ACTIVE_MIN")]] extern T wave_active_min(T v);

template<arithmetic T>
[[callop("WARP_ACTIVE_PRODUCT")]] extern T warp_active_product(T v);
template<arithmetic T>
[[callop("WARP_ACTIVE_PRODUCT")]] extern T wave_active_product(T v);

template<arithmetic T>
[[callop("WARP_ACTIVE_SUM")]] extern T warp_active_sum(T v);
template<arithmetic T>
[[callop("WARP_ACTIVE_SUM")]] extern T wave_active_sum(T v);

[[callop("WARP_ACTIVE_ALL")]] extern bool warp_active_all(bool val);
[[callop("WARP_ACTIVE_ALL")]] extern bool wave_active_all_true(bool val);

[[callop("WARP_ACTIVE_ANY")]] extern bool warp_active_any(bool val);
[[callop("WARP_ACTIVE_ANY")]] extern bool wave_active_any_true(bool val);

[[callop("WARP_ACTIVE_BIT_MASK")]] extern uint4 warp_active_bit_mask(bool val);
[[callop("WARP_ACTIVE_BIT_MASK")]] extern uint4 wave_active_ballot(bool val);

[[callop("WARP_PREFIX_COUNT_BITS")]] extern uint32 warp_prefix_count_bits(bool val);
[[callop("WARP_PREFIX_COUNT_BITS")]] extern uint32 wave_prefix_count_bits(bool val);

template<arithmetic T>
[[callop("WARP_PREFIX_PRODUCT")]] extern T warp_prefix_product(T v);
template<arithmetic T>
[[callop("WARP_PREFIX_PRODUCT")]] extern T wave_prefix_product(T v);

template<arithmetic T>
[[callop("WARP_PREFIX_SUM")]] extern T warp_prefix_sum(T v);
template<arithmetic T>
[[callop("WARP_PREFIX_SUM")]] extern T wave_prefix_sum(T v);

template<primitive T>
[[callop("WARP_READ_LANE")]] extern T warp_read_lane(uint32 lane_index);
template<primitive T>
[[callop("WARP_READ_LANE")]] extern T wave_read_lane_at(uint32 lane_index);

template<primitive T>
[[callop("WARP_READ_FIRST_ACTIVE_LANE")]] extern T warp_read_first_active_lane(uint32 lane_index);

template<primitive T>
[[callop("WARP_READ_FIRST_ACTIVE_LANE")]] extern T wave_read_lane_first(uint32 lane_index);

// cuda
[[callop("SHADER_EXECUTION_REORDER")]] extern void shader_execution_reorder();

// [[callop("WARP_FIRST_ACTIVE_LANE")]] bool warp_first_active_lane();

}// namespace luisa::shader
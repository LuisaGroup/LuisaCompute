#pragma once
#include "./../attributes.hpp"
#include "./../type_traits.hpp"

namespace luisa::shader {

[[expr("warp_lane_count")]] extern uint32 warp_lane_count();
[[expr("warp_lane_count")]] extern uint32 wave_lane_count();

[[expr("warp_lane_id")]] extern uint32 warp_lane_id();
[[expr("warp_lane_id")]] extern uint32 wave_lane_id();

[[callop("WARP_IS_FIRST_ACTIVE_LANE")]] extern bool warp_is_first_active_lane();
[[callop("WARP_IS_FIRST_ACTIVE_LANE")]] extern bool wave_is_first_lane();

template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_ALL_EQUAL")]] extern vec<bool, vec_dim_v<T>> warp_active_all_equal();
template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_ALL_EQUAL")]] extern vec<bool, vec_dim_v<T>> wave_active_all_equal();

template<concepts::int_family T>
[[callop("WARP_ACTIVE_BIT_AND")]] extern T warp_active_bit_and(T v);
template<concepts::int_family T>
[[callop("WARP_ACTIVE_BIT_AND")]] extern T wave_active_bit_and(T v);

template<concepts::int_family T>
[[callop("WARP_ACTIVE_BIT_OR")]] extern T warp_active_bit_or(T v);
template<concepts::int_family T>
[[callop("WARP_ACTIVE_BIT_OR")]] extern T wave_active_bit_or(T v);

template<concepts::int_family T>
[[callop("WARP_ACTIVE_BIT_XOR")]] extern T warp_active_bit_xor(T v);
template<concepts::int_family T>
[[callop("WARP_ACTIVE_BIT_XOR")]] extern T wave_active_bit_xor(T v);

[[callop("WARP_ACTIVE_COUNT_BITS")]] extern uint32 warp_active_count_bits(bool val);
[[callop("WARP_ACTIVE_COUNT_BITS")]] extern uint32 wave_active_count_bits(bool val);

template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_MAX")]] extern T warp_active_max(T v);
template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_MAX")]] extern T wave_active_max(T v);

template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_MIN")]] extern T warp_active_min(T v);
template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_MIN")]] extern T wave_active_min(T v);

template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_PRODUCT")]] extern T warp_active_product(T v);
template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_PRODUCT")]] extern T wave_active_product(T v);

template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_SUM")]] extern T warp_active_sum(T v);
template<concepts::arithmetic T>
[[callop("WARP_ACTIVE_SUM")]] extern T wave_active_sum(T v);

[[callop("WARP_ACTIVE_ALL")]] extern bool warp_active_all(bool val);
[[callop("WARP_ACTIVE_ALL")]] extern bool wave_active_all_true(bool val);

[[callop("WARP_ACTIVE_ANY")]] extern bool warp_active_any(bool val);
[[callop("WARP_ACTIVE_ANY")]] extern bool wave_active_any_true(bool val);

[[callop("WARP_ACTIVE_BIT_MASK")]] extern uint4 warp_active_bit_mask(bool val);
[[callop("WARP_ACTIVE_BIT_MASK")]] extern uint4 wave_active_ballot(bool val);

[[callop("WARP_PREFIX_COUNT_BITS")]] extern uint32 warp_prefix_count_bits(bool val);
[[callop("WARP_PREFIX_COUNT_BITS")]] extern uint32 wave_prefix_count_bits(bool val);

template<concepts::arithmetic T>
[[callop("WARP_PREFIX_PRODUCT")]] extern T warp_prefix_product(T v);
template<concepts::arithmetic T>
[[callop("WARP_PREFIX_PRODUCT")]] extern T wave_prefix_product(T v);

template<concepts::arithmetic T>
[[callop("WARP_PREFIX_SUM")]] extern T warp_prefix_sum(T v);
template<concepts::arithmetic T>
[[callop("WARP_PREFIX_SUM")]] extern T wave_prefix_sum(T v);

template<concepts::primitive T>
[[callop("WARP_READ_LANE")]] extern T warp_read_lane(uint32 lane_index);
template<concepts::primitive T>
[[callop("WARP_READ_LANE")]] extern T wave_read_lane_at(uint32 lane_index);

template<concepts::primitive T>
[[callop("WARP_READ_FIRST_ACTIVE_LANE")]] extern T warp_read_first_active_lane(uint32 lane_index);

template<concepts::primitive T>
[[callop("WARP_READ_FIRST_ACTIVE_LANE")]] extern T wave_read_lane_first(uint32 lane_index);

// cuda
[[callop("SHADER_EXECUTION_REORDER")]] extern void shader_execution_reorder();
// [[callop("WARP_FIRST_ACTIVE_LANE")]] bool warp_first_active_lane();

}
// Function: benchmark_gemm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
kernel void benchmark_gemm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  constexpr auto mpp_descriptor = mpp::tensor_ops::matmul2d_descriptor(16, 16, mpp::tensor_ops::dynamic_length_v<int>, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  using mpp_operation = mpp::tensor_ops::matmul2d<mpp_descriptor, execution_simdgroups<1>>;
  using mpp_tensor = mpp_operation::cooperative_tensor_destination_t<mpp_operation::cooperative_tensor_left_input_t<float, float, float>, mpp_operation::cooperative_tensor_right_input_t<float, float, float>, float>;
  mpp_tensor tile_i_10_mpp_c_fragment;
  long cse_v1 = ((((long)blockIdx) / (long)13) * (long)32);
  long cse_v2 = ((((long)threadIdx) >> (long)7) * (long)16);
  long cse_v3 = ((((long)blockIdx) % (long)13) * (long)64);
  long cse_v4 = max(((((long)blockIdx) / (long)13) * (long)32), (long)225);
  long cse_v5 = (((((long)blockIdx) / (long)13) * (long)32) + ((((long)threadIdx) >> (long)7) * (long)16));
  long cse_v6 = max(((((long)blockIdx) % (long)13) * (long)64), (long)705);
  long cse_v7 = (((((long)threadIdx) & (long)127) >> (long)5) * (long)16);
  long cse_v8 = (((((long)blockIdx) % (long)13) * (long)64) + (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  long cse_v9 = (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16));
  long cse_v10 = (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  long condval;
  if (((long)-257 < (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))) {
    condval = (((((long)blockIdx) / (long)13) * (long)32) + ((((long)threadIdx) >> (long)7) * (long)16));
  } else {
    condval = (long)0;
  }
  long condval_1;
  if (((long)-257 < (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))) {
    condval_1 = (long)0;
  } else {
    condval_1 = (long)0;
  }
  long condval_2;
  if (((long)-769 < (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16)))) {
    condval_2 = (long)0;
  } else {
    condval_2 = (long)0;
  }
  long condval_3;
  if (((long)-769 < (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16)))) {
    condval_3 = (((((long)blockIdx) % (long)13) * (long)64) + (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  } else {
    condval_3 = (long)0;
  }
  long condval_4;
  if (((long)-257 < (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))) {
    condval_4 = (((((long)blockIdx) / (long)13) * (long)32) + ((((long)threadIdx) >> (long)7) * (long)16));
  } else {
    condval_4 = (long)0;
  }
  long condval_5;
  if (((long)-257 < (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))) {
    condval_5 = (long)0;
  } else {
    condval_5 = (long)0;
  }
  long condval_6;
  if (((long)-769 < (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16)))) {
    condval_6 = (long)0;
  } else {
    condval_6 = (long)0;
  }
  long condval_7;
  if (((long)-769 < (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16)))) {
    condval_7 = (((((long)blockIdx) % (long)13) * (long)64) + (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  } else {
    condval_7 = (long)0;
  }
  long condval_8;
  if (((long)-257 < (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))) {
    condval_8 = (((((long)blockIdx) / (long)13) * (long)32) + ((((long)threadIdx) >> (long)7) * (long)16));
  } else {
    condval_8 = (long)0;
  }
  long condval_9;
  if (((long)-257 < (((long)0 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))) {
    condval_9 = (long)0;
  } else {
    condval_9 = (long)0;
  }
  long condval_10;
  if (((long)-769 < (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16)))) {
    condval_10 = (long)0;
  } else {
    condval_10 = (long)0;
  }
  long condval_11;
  if (((long)-769 < (((long)0 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16)))) {
    condval_11 = (((((long)blockIdx) % (long)13) * (long)64) + (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  } else {
    condval_11 = (long)0;
  }
  { int mpp_actual_m = int(max((long)0, min((long)16, (((long)257 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16))))), mpp_actual_n = int(max((long)0, min((long)16, (((long)769 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))), mpp_actual_k = int(113); if (mpp_actual_m == 0 || mpp_actual_n == 0) { auto mpp_left_pointer = (&(arg0_ptr[((condval_4 * (long)113) + condval_5)])); auto mpp_right_pointer = (&(arg1_ptr[((condval_6 * (long)769) + condval_7)])); uint mpp_empty_lane = simd_prefix_exclusive_sum(1u); for (uint mpp_empty_chunk = 0; mpp_empty_chunk < 1; ++mpp_empty_chunk) { uint mpp_empty_outer = mpp_empty_chunk * 32u + mpp_empty_lane; uint mpp_owned_classification = 0u; if (mpp_empty_outer < uint(mpp_actual_m == 0 ? mpp_actual_n : mpp_actual_m)) { uint mpp_invalid_product = 0u, mpp_zero_sign = 0x80000000u; for (int mpp_empty_k = 0; mpp_empty_k < mpp_actual_k; ++mpp_empty_k) { uint mpp_operand_bits; if (mpp_actual_m == 0) mpp_operand_bits = as_type<uint>(mpp_right_pointer[ulong(mpp_empty_k) * 769 + mpp_empty_outer]); else mpp_operand_bits = as_type<uint>(mpp_left_pointer[ulong(mpp_empty_outer) * 113 + mpp_empty_k]); mpp_invalid_product |= (mpp_operand_bits & 0x7fffffffu) >= 0x7f800000u; mpp_zero_sign &= mpp_operand_bits; } mpp_owned_classification = mpp_invalid_product | mpp_zero_sign; }
#pragma unroll
for (uint mpp_empty_element = 0; mpp_empty_element < tile_i_10_mpp_c_fragment.get_capacity(); ++mpp_empty_element) { bool mpp_valid_output = tile_i_10_mpp_c_fragment.is_valid_element(mpp_empty_element); uint mpp_classification_source = 0u; if (mpp_valid_output) { auto mpp_empty_xy = tile_i_10_mpp_c_fragment.get_multidimensional_index(mpp_empty_element); mpp_classification_source = mpp_empty_xy[mpp_actual_m == 0 ? 0 : 1]; } uint mpp_selected_classification = simd_shuffle(mpp_owned_classification, mpp_classification_source & 31u); if (mpp_valid_output && mpp_classification_source / 32u == mpp_empty_chunk) { uint mpp_empty_bits = 0u; if ((mpp_selected_classification & 1u) || (mpp_empty_bits & 0x7fffffffu) > 0x7f800000u) mpp_empty_bits = 0x7fc00000u; else if ((mpp_empty_bits & 0x7fffffffu) == 0u) mpp_empty_bits &= mpp_selected_classification; tile_i_10_mpp_c_fragment[mpp_empty_element] = as_type<float>(mpp_empty_bits); } } } } else if (mpp_actual_m == 16 && mpp_actual_n == 16) { auto mpp_left = tensor<device float, extents<int, dynamic_extent, 16>, tensor_inline>((&(arg0_ptr[((condval_8 * (long)113) + condval_9)])), extents<int, dynamic_extent, 16>{int(113)}, array<int, 2>{1, 113}); auto mpp_right = tensor<device float, extents<int, 16, dynamic_extent>, tensor_inline>((&(arg1_ptr[((condval_10 * (long)769) + condval_11)])), extents<int, 16, dynamic_extent>{int(113)}, array<int, 2>{1, 769}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); } else { auto mpp_left = tensor<device float, extents<int, dynamic_extent, dynamic_extent>, tensor_inline>((&(arg0_ptr[((condval * (long)113) + condval_1)])), extents<int, dynamic_extent, dynamic_extent>{int(113), int(max((long)0, min((long)16, (((long)257 - max(((((long)blockIdx) / (long)13) * (long)32), (long)225)) - ((((long)threadIdx) >> (long)7) * (long)16)))))}, array<int, 2>{1, 113}); auto mpp_right = tensor<device float, extents<int, dynamic_extent, dynamic_extent>, tensor_inline>((&(arg1_ptr[((condval_2 * (long)769) + condval_3)])), extents<int, dynamic_extent, dynamic_extent>{int(max((long)0, min((long)16, (((long)769 - max(((((long)blockIdx) % (long)13) * (long)64), (long)705)) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))), int(113)}, array<int, 2>{1, 769}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); } };
  metal::threadgroup_barrier(metal::mem_flags(2));
  long cse_v11 = (min((long)32, ((long)257 - ((((long)blockIdx) / (long)13) * (long)32))) - ((((long)threadIdx) >> (long)7) * (long)16));
  long cse_v12 = (min((long)64, ((long)769 - ((((long)blockIdx) % (long)13) * (long)64))) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  long condval_12;
  if ((((long)0 < (min((long)32, ((long)257 - ((((long)blockIdx) / (long)13) * (long)32))) - ((((long)threadIdx) >> (long)7) * (long)16))) && ((long)0 < (min((long)64, ((long)769 - ((((long)blockIdx) % (long)13) * (long)64))) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))) {
    condval_12 = (((((long)blockIdx) / (long)13) * (long)32) + ((((long)threadIdx) >> (long)7) * (long)16));
  } else {
    condval_12 = (long)0;
  }
  long condval_13;
  if ((((long)0 < (min((long)32, ((long)257 - ((((long)blockIdx) / (long)13) * (long)32))) - ((((long)threadIdx) >> (long)7) * (long)16))) && ((long)0 < (min((long)64, ((long)769 - ((((long)blockIdx) % (long)13) * (long)64))) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))) {
    condval_13 = (((((long)blockIdx) % (long)13) * (long)64) + (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  } else {
    condval_13 = (long)0;
  }
  long condval_14;
  if ((((long)0 < (min((long)32, ((long)257 - ((((long)blockIdx) / (long)13) * (long)32))) - ((((long)threadIdx) >> (long)7) * (long)16))) && ((long)0 < (min((long)64, ((long)769 - ((((long)blockIdx) % (long)13) * (long)64))) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))) {
    condval_14 = (((((long)blockIdx) / (long)13) * (long)32) + ((((long)threadIdx) >> (long)7) * (long)16));
  } else {
    condval_14 = (long)0;
  }
  long condval_15;
  if ((((long)0 < (min((long)32, ((long)257 - ((((long)blockIdx) / (long)13) * (long)32))) - ((((long)threadIdx) >> (long)7) * (long)16))) && ((long)0 < (min((long)64, ((long)769 - ((((long)blockIdx) % (long)13) * (long)64))) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))) {
    condval_15 = (((((long)blockIdx) % (long)13) * (long)64) + (((((long)threadIdx) & (long)127) >> (long)5) * (long)16));
  } else {
    condval_15 = (long)0;
  }
  { const int mpp_store_rows = int(max((long)0, min((long)16, (min((long)32, ((long)257 - ((((long)blockIdx) / (long)13) * (long)32))) - ((((long)threadIdx) >> (long)7) * (long)16))))); const int mpp_store_columns = int(max((long)0, min((long)16, (min((long)64, ((long)769 - ((((long)blockIdx) % (long)13) * (long)64))) - (((((long)threadIdx) & (long)127) >> (long)5) * (long)16))))); if (mpp_store_rows == 16 && mpp_store_columns == 16) { tile_i_10_mpp_c_fragment.store(tensor<device float, extents<int, 16, 16>, tensor_inline>((&(arg2_ptr[((condval_14 * (long)769) + condval_15)])), extents<int, 16, 16>{}, array<int, 2>{1, 769})); } else { auto mpp_memory = (&(arg2_ptr[((condval_12 * (long)769) + condval_13)]));
#pragma unroll
for (uint mpp_element = 0; mpp_element < tile_i_10_mpp_c_fragment.get_capacity(); ++mpp_element) { if (tile_i_10_mpp_c_fragment.is_valid_element(mpp_element)) { auto mpp_coordinate = tile_i_10_mpp_c_fragment.get_multidimensional_index(mpp_element); if (mpp_coordinate[1] < mpp_store_rows && mpp_coordinate[0] < mpp_store_columns) mpp_memory[mpp_coordinate[1] * 769 + mpp_coordinate[0]] = tile_i_10_mpp_c_fragment[mpp_element]; } } } };
  metal::threadgroup_barrier(metal::mem_flags(2));
}



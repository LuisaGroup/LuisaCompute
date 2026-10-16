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
  threadgroup float tile_storage_0_shared[4096];
  for (int tile_i_1_chunk = 0; tile_i_1_chunk < 32; ++tile_i_1_chunk) {
    tile_storage_0_shared[((((long)tile_i_1_chunk) * (long)128) + ((long)threadIdx))] = 0.000000e+00f;
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  constexpr auto mpp_descriptor = mpp::tensor_ops::matmul2d_descriptor(32, 32, mpp::tensor_ops::dynamic_length_v<int>, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  using mpp_operation = mpp::tensor_ops::matmul2d<mpp_descriptor, execution_simdgroups<1>>;
  using mpp_tensor = mpp_operation::cooperative_tensor_destination_t<mpp_operation::cooperative_tensor_left_input_t<float, float, float>, mpp_operation::cooperative_tensor_right_input_t<float, float, float>, float>;
  mpp_tensor tile_i_10_mpp_c_fragment;
  long cse_v1 = (((long)threadIdx) >> (long)5);
  long cse_v5 = ((((long)threadIdx) >> (long)5) * (long)1024);
  tile_i_10_mpp_c_fragment.load(tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>((&(tile_storage_0_shared[((((long)threadIdx) >> (long)5) * (long)1024)])), extents<int, 32, 32>{}, array<int, 2>{1, 32}));
  long cse_v2 = (((long)blockIdx) / (long)129);
  long cse_v6 = ((((long)blockIdx) / (long)129) * (long)128);
  long cse_v8 = ((((long)blockIdx) % (long)129) * (long)32);
  for (long pipeline_6 = (long)0; pipeline_6 < (long)256; ++pipeline_6) {
    long cse_v7 = ((((long)threadIdx) >> (long)5) * (long)32);
    long cse_v9 = max(((((long)blockIdx) / (long)129) * (long)128), (long)3969);
    long cse_v10 = (((long)0 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32));
    long condval;
    if (((long)-4097 < (((long)0 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32)))) {
      condval = (((((long)blockIdx) / (long)129) * (long)128) + ((((long)threadIdx) >> (long)5) * (long)32));
    } else {
      condval = (long)0;
    }
    long condval_1;
    if (((long)-4097 < (((long)0 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32)))) {
      condval_1 = (pipeline_6 * (long)16);
    } else {
      condval_1 = (long)0;
    }
    long condval_2;
    if (((long)-4097 < (((long)0 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32)))) {
      condval_2 = (((((long)blockIdx) / (long)129) * (long)128) + ((((long)threadIdx) >> (long)5) * (long)32));
    } else {
      condval_2 = (long)0;
    }
    long condval_3;
    if (((long)-4097 < (((long)0 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32)))) {
      condval_3 = (pipeline_6 * (long)16);
    } else {
      condval_3 = (long)0;
    }
    { int mpp_actual_m = int(max((long)0, min((long)32, (((long)4097 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32))))), mpp_actual_n = int(((long)4097 - max(((((long)blockIdx) % (long)129) * (long)32), (long)4065))), mpp_actual_k = int(16); if (mpp_actual_m == 0 || mpp_actual_n == 0) { auto mpp_left_pointer = (&(arg0_ptr[((condval_2 * (long)4096) + condval_3)])); auto mpp_right_pointer = (&(arg1_ptr[((pipeline_6 * (long)65552) + ((((long)blockIdx) % (long)129) * (long)32))]));
#pragma unroll
for (uint mpp_empty_element = 0; mpp_empty_element < tile_i_10_mpp_c_fragment.get_capacity(); ++mpp_empty_element) { if (tile_i_10_mpp_c_fragment.is_valid_element(mpp_empty_element)) { auto mpp_empty_xy = tile_i_10_mpp_c_fragment.get_multidimensional_index(mpp_empty_element); uint mpp_empty_bits = as_type<uint>(tile_i_10_mpp_c_fragment[mpp_empty_element]); uint mpp_zero_sign = 0x80000000u; bool mpp_invalid_product = (mpp_empty_bits & 0x7fffffffu) > 0x7f800000u; for (int mpp_empty_k = 0; mpp_empty_k < mpp_actual_k; ++mpp_empty_k) { uint mpp_left_bits = 0u, mpp_right_bits = 0u; if (mpp_empty_xy[1] < mpp_actual_m) mpp_left_bits = as_type<uint>(mpp_left_pointer[ulong(mpp_empty_xy[1]) * 4096 + mpp_empty_k]); if (mpp_empty_xy[0] < mpp_actual_n) mpp_right_bits = as_type<uint>(mpp_right_pointer[ulong(mpp_empty_k) * 4097 + mpp_empty_xy[0]]); mpp_invalid_product |= (mpp_left_bits & 0x7fffffffu) >= 0x7f800000u || (mpp_right_bits & 0x7fffffffu) >= 0x7f800000u; mpp_zero_sign &= mpp_left_bits ^ mpp_right_bits; } if (mpp_invalid_product) mpp_empty_bits = 0x7fc00000u; else if ((mpp_empty_bits & 0x7fffffffu) == 0u) mpp_empty_bits &= mpp_zero_sign; tile_i_10_mpp_c_fragment[mpp_empty_element] = as_type<float>(mpp_empty_bits); } } } else { auto mpp_left = tensor<device float, extents<int, dynamic_extent, dynamic_extent>, tensor_inline>((&(arg0_ptr[((condval * (long)4096) + condval_1)])), extents<int, dynamic_extent, dynamic_extent>{int(16), int(max((long)0, min((long)32, (((long)4097 - max(((((long)blockIdx) / (long)129) * (long)128), (long)3969)) - ((((long)threadIdx) >> (long)5) * (long)32)))))}, array<int, 2>{1, 4096}); auto mpp_right = tensor<device float, extents<int, dynamic_extent, dynamic_extent>, tensor_inline>((&(arg1_ptr[((pipeline_6 * (long)65552) + ((((long)blockIdx) % (long)129) * (long)32))])), extents<int, dynamic_extent, dynamic_extent>{int(((long)4097 - max(((((long)blockIdx) % (long)129) * (long)32), (long)4065))), int(16)}, array<int, 2>{1, 4097}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); } };
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  tile_i_10_mpp_c_fragment.store(tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>((&(tile_storage_0_shared[((((long)threadIdx) >> (long)5) * (long)1024)])), extents<int, 32, 32>{}, array<int, 2>{1, 32}));
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_15_chunk = 0; tile_i_15_chunk < 32; ++tile_i_15_chunk) {
    long cse_v3 = ((long)tile_i_15_chunk);
    long cse_v4 = (((long)threadIdx) & (long)31);
    if ((((((((long)blockIdx) / (long)129) * (long)128) + (((long)tile_i_15_chunk) * (long)4)) + (((long)threadIdx) >> (long)5)) < (long)4097) && ((((((long)blockIdx) % (long)129) * (long)32) + (((long)threadIdx) & (long)31)) < (long)4097)) {
      arg2_ptr[((((((((long)blockIdx) / (long)129) * (long)524416) + (((long)tile_i_15_chunk) * (long)16388)) + ((((long)threadIdx) >> (long)5) * (long)4097)) + ((((long)blockIdx) % (long)129) * (long)32)) + (((long)threadIdx) & (long)31))] = tile_storage_0_shared[((((long)tile_i_15_chunk) * (long)128) + ((long)threadIdx))];
    }
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
}



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
  constexpr auto mpp_descriptor = mpp::tensor_ops::matmul2d_descriptor(16, 16, mpp::tensor_ops::dynamic_length_v<int>, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  using mpp_operation = mpp::tensor_ops::matmul2d<mpp_descriptor, execution_simdgroups<1>>;
  using mpp_tensor = mpp_operation::cooperative_tensor_destination_t<mpp_operation::cooperative_tensor_left_input_t<float, float, float>, mpp_operation::cooperative_tensor_right_input_t<float, float, float>, float>;
  mpp_tensor tile_i_10_mpp_c_fragment;
  {
#pragma unroll
for (uint mpp_element = 0; mpp_element < tile_i_10_mpp_c_fragment.get_capacity(); ++mpp_element) { if (tile_i_10_mpp_c_fragment.is_valid_element(mpp_element)) tile_i_10_mpp_c_fragment[mpp_element] = 0.000000e+00f; } };
  int cse_v1 = (((int)blockIdx) / 344);
  int cse_v4 = (((int)threadIdx) >> 6);
  int cse_v5 = ((((int)blockIdx) % 344) * 32);
  int cse_v6 = (((((int)threadIdx) & 63) >> 5) * 16);
  for (int pipeline_6 = 0; pipeline_6 < 128; ++pipeline_6) {
    threadgroup float tile_storage_3_shared[1024];
    threadgroup float tile_storage_6_shared[1024];
    int cse_v2 = (((int)threadIdx) >> 5);
    int cse_v3 = (((int)threadIdx) & 31);
    for (int tile_i_4_chunk = 0; tile_i_4_chunk < 8; ++tile_i_4_chunk) {
      tile_storage_3_shared[((tile_i_4_chunk * 128) + ((int)threadIdx))] = arg0_ptr[((((((((int)blockIdx) / 344) * 131072) + (tile_i_4_chunk * 16384)) + ((((int)threadIdx) >> 5) * 4096)) + (pipeline_6 * 32)) + (((int)threadIdx) & 31))];
    }
    for (int tile_i_7_chunk = 0; tile_i_7_chunk < 8; ++tile_i_7_chunk) {
      tile_storage_6_shared[((tile_i_7_chunk * 128) + ((int)threadIdx))] = arg1_ptr[(((((pipeline_6 * 352256) + (tile_i_7_chunk * 44032)) + ((((int)threadIdx) >> 5) * 11008)) + ((((int)blockIdx) % 344) * 32)) + (((int)threadIdx) & 31))];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    { auto mpp_left = tensor<threadgroup float, extents<int, 32, 16>, tensor_inline>((&(tile_storage_3_shared[((((int)threadIdx) >> 6) * 512)])), extents<int, 32, 16>{}, array<int, 2>{1, 32}); auto mpp_right = tensor<threadgroup float, extents<int, 16, 32>, tensor_inline>((&(tile_storage_6_shared[(((((int)threadIdx) & 63) >> 5) * 16)])), extents<int, 16, 32>{}, array<int, 2>{1, 32}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); };
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  tile_i_10_mpp_c_fragment.store(tensor<device float, extents<int, 16, 16>, tensor_inline>((&(arg2_ptr[(((((((int)blockIdx) / 344) * 352256) + ((((int)threadIdx) >> 6) * 176128)) + ((((int)blockIdx) % 344) * 32)) + (((((int)threadIdx) & 63) >> 5) * 16))])), extents<int, 16, 16>{}, array<int, 2>{1, 11008}));
  metal::threadgroup_barrier(metal::mem_flags(2));
}



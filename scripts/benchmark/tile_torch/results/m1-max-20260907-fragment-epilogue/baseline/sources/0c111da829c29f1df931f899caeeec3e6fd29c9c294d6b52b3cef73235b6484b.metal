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
    tile_storage_0_shared[((tile_i_1_chunk * 128) + ((int)threadIdx))] = 0.000000e+00f;
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  constexpr auto mpp_descriptor = mpp::tensor_ops::matmul2d_descriptor(32, 32, mpp::tensor_ops::dynamic_length_v<int>, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  using mpp_operation = mpp::tensor_ops::matmul2d<mpp_descriptor, execution_simdgroups<1>>;
  using mpp_tensor = mpp_operation::cooperative_tensor_destination_t<mpp_operation::cooperative_tensor_left_input_t<float, float, float>, mpp_operation::cooperative_tensor_right_input_t<float, float, float>, float>;
  mpp_tensor tile_i_10_mpp_c_fragment;
  int cse_v1 = (((int)threadIdx) >> 6);
  int cse_v2 = (((int)threadIdx) & 63);
  int cse_v7 = (((((int)threadIdx) & 63) >> 5) * 32);
  int cse_v8 = (((((int)threadIdx) >> 6) * 2048) + (((((int)threadIdx) & 63) >> 5) * 32));
  tile_i_10_mpp_c_fragment.load(tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>((&(tile_storage_0_shared[(((((int)threadIdx) >> 6) * 2048) + (((((int)threadIdx) & 63) >> 5) * 32))])), extents<int, 32, 32>{}, array<int, 2>{1, 64}));
  int cse_v3 = ((((int)blockIdx) >> 6) * 262144);
  int cse_v4 = ((((int)blockIdx) & 63) * 64);
  { auto mpp_left = tensor<device float, extents<int, 4096, 32>, tensor_inline>((&(arg0_ptr[(((((int)blockIdx) >> 6) * 262144) + ((((int)threadIdx) >> 6) * 131072))])), extents<int, 4096, 32>{}, array<int, 2>{1, 4096}); auto mpp_right = tensor<device float, extents<int, 32, 4096>, tensor_inline>((&(arg1_ptr[(((((int)blockIdx) & 63) * 64) + (((((int)threadIdx) & 63) >> 5) * 32))])), extents<int, 32, 4096>{}, array<int, 2>{1, 4096}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); };
  metal::threadgroup_barrier(metal::mem_flags(2));
  tile_i_10_mpp_c_fragment.store(tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>((&(tile_storage_0_shared[(((((int)threadIdx) >> 6) * 2048) + (((((int)threadIdx) & 63) >> 5) * 32))])), extents<int, 32, 32>{}, array<int, 2>{1, 64}));
  metal::threadgroup_barrier(metal::mem_flags(2));
  threadgroup float tile_storage_15_shared[4096];
  for (int tile_i_16_chunk = 0; tile_i_16_chunk < 32; ++tile_i_16_chunk) {
    int cse_v5 = ((tile_i_16_chunk * 128) + ((int)threadIdx));
    tile_storage_15_shared[((tile_i_16_chunk * 128) + ((int)threadIdx))] = ((1.250000e-01f * tile_storage_0_shared[((tile_i_16_chunk * 128) + ((int)threadIdx))]) + 2.500000e-01f);
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_18_chunk = 0; tile_i_18_chunk < 32; ++tile_i_18_chunk) {
    int cse_v6 = ((tile_i_18_chunk * 128) + ((int)threadIdx));
    arg2_ptr[((((((((int)blockIdx) >> 6) * 262144) + (tile_i_18_chunk * 8192)) + ((((int)threadIdx) >> 6) * 4096)) + ((((int)blockIdx) & 63) * 64)) + (((int)threadIdx) & 63))] = ((5.000000e-01f * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * (1.000000e+00f + select(((exp((2.000000e+00f * (7.978846e-01f * (tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))] + (((4.471500e-02f * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]))))) - 1.000000e+00f) / (exp((2.000000e+00f * (7.978846e-01f * (tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))] + (((4.471500e-02f * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]))))) + 1.000000e+00f)), ((1.000000e+00f - exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))] + (((4.471500e-02f * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))])))))) / (1.000000e+00f + exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))] + (((4.471500e-02f * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]))))))), ((7.978846e-01f * (tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))] + (((4.471500e-02f * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]) * tile_storage_15_shared[((tile_i_18_chunk * 128) + ((int)threadIdx))]))) >= 0.000000e+00f))));
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
}



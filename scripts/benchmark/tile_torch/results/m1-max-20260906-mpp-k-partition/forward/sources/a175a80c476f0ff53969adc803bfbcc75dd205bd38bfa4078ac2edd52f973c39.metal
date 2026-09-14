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
  constexpr auto mpp_descriptor = mpp::tensor_ops::matmul2d_descriptor(32, 32, mpp::tensor_ops::dynamic_length_v<int>, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  using mpp_operation = mpp::tensor_ops::matmul2d<mpp_descriptor, execution_simdgroups<1>>;
  using mpp_tensor = mpp_operation::cooperative_tensor_destination_t<mpp_operation::cooperative_tensor_left_input_t<float, float, float>, mpp_operation::cooperative_tensor_right_input_t<float, float, float>, float>;
  mpp_tensor tile_i_10_mpp_c_fragment;
  int cse_v1 = (((int)blockIdx) >> 5);
  int cse_v2 = (((int)threadIdx) >> 5);
  int cse_v3 = ((((int)blockIdx) & 31) * 32);
  { auto mpp_left = tensor<device float, extents<int, dynamic_extent, 32>, tensor_inline>((&(arg0_ptr[(((((int)blockIdx) >> 5) * 196736) + ((((int)threadIdx) >> 5) * 49184))])), extents<int, dynamic_extent, 32>{int(1537)}, array<int, 2>{1, 1537}); auto mpp_right = tensor<device float, extents<int, 32, dynamic_extent>, tensor_inline>((&(arg1_ptr[((((int)blockIdx) & 31) * 32)])), extents<int, 32, dynamic_extent>{int(1537)}, array<int, 2>{1, 1024}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); };
  metal::threadgroup_barrier(metal::mem_flags(2));
  tile_i_10_mpp_c_fragment.store(tensor<device float, extents<int, 32, 32>, tensor_inline>((&(arg2_ptr[((((((int)blockIdx) >> 5) * 131072) + ((((int)threadIdx) >> 5) * 32768)) + ((((int)blockIdx) & 31) * 32))])), extents<int, 32, 32>{}, array<int, 2>{1, 1024}));
  metal::threadgroup_barrier(metal::mem_flags(2));
}



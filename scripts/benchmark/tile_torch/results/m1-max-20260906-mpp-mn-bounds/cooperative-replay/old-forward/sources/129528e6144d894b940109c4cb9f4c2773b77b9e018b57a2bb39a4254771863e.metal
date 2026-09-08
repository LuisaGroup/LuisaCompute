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
  int cse_v1 = (((int)threadIdx) >> 5);
  int cse_v7 = ((((int)threadIdx) >> 5) * 1024);
  tile_i_10_mpp_c_fragment.load(tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>((&(tile_storage_0_shared[((((int)threadIdx) >> 5) * 1024)])), extents<int, 32, 32>{}, array<int, 2>{1, 32}));
  int cse_v2 = (((int)blockIdx) / 9);
  int cse_v6 = (((int)threadIdx) & 31);
  int cse_v8 = ((((int)blockIdx) / 9) * 128);
  int cse_v9 = ((((int)blockIdx) % 9) * 32);
  int cse_v10 = ((((int)threadIdx) >> 5) * 257);
  int cse_v11 = (((((int)blockIdx) % 9) * 32) + (((int)threadIdx) & 31));
  for (int pipeline_6 = 0; pipeline_6 < 4; ++pipeline_6) {
    threadgroup float tile_storage_3_shared[2048];
    threadgroup float tile_storage_6_shared[512];
    int cse_v4 = (pipeline_6 * 16);
    for (int tile_i_4_chunk = 0; tile_i_4_chunk < 16; ++tile_i_4_chunk) {
      int cse_v3 = (((int)threadIdx) >> 4);
      int cse_v5 = (((int)threadIdx) & 15);
      float condval;
      if (((((((((int)blockIdx) / 9) * 128) + (tile_i_4_chunk * 8)) + (((int)threadIdx) >> 4)) < 129) && (((pipeline_6 * 16) + (((int)threadIdx) & 15)) < 61))) {
        condval = arg0_ptr[((((((((int)blockIdx) / 9) * 7808) + (tile_i_4_chunk * 488)) + ((((int)threadIdx) >> 4) * 61)) + (pipeline_6 * 16)) + (((int)threadIdx) & 15))];
      } else {
        condval = 0.000000e+00f;
      }
      tile_storage_3_shared[((tile_i_4_chunk * 128) + ((int)threadIdx))] = condval;
    }
    for (int tile_i_7_chunk = 0; tile_i_7_chunk < 4; ++tile_i_7_chunk) {
      float condval_1;
      if ((((((pipeline_6 * 16) + (tile_i_7_chunk * 4)) + (((int)threadIdx) >> 5)) < 61) && ((((((int)blockIdx) % 9) * 32) + (((int)threadIdx) & 31)) < 257))) {
        condval_1 = arg1_ptr[(((((pipeline_6 * 4112) + (tile_i_7_chunk * 1028)) + ((((int)threadIdx) >> 5) * 257)) + ((((int)blockIdx) % 9) * 32)) + (((int)threadIdx) & 31))];
      } else {
        condval_1 = 0.000000e+00f;
      }
      tile_storage_6_shared[((tile_i_7_chunk * 128) + ((int)threadIdx))] = condval_1;
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    { auto mpp_left = tensor<threadgroup float, extents<int, 16, 32>, tensor_inline>((&(tile_storage_3_shared[((((int)threadIdx) >> 5) * 512)])), extents<int, 16, 32>{}, array<int, 2>{1, 16}); auto mpp_right = tensor<threadgroup float, extents<int, 32, 16>, tensor_inline>((&(tile_storage_6_shared[0])), extents<int, 32, 16>{}, array<int, 2>{1, 32}); mpp_operation{}.run(mpp_left, mpp_right, tile_i_10_mpp_c_fragment); };
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  tile_i_10_mpp_c_fragment.store(tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>((&(tile_storage_0_shared[((((int)threadIdx) >> 5) * 1024)])), extents<int, 32, 32>{}, array<int, 2>{1, 32}));
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_15_chunk = 0; tile_i_15_chunk < 32; ++tile_i_15_chunk) {
    if ((((((((int)blockIdx) / 9) * 128) + (tile_i_15_chunk * 4)) + (((int)threadIdx) >> 5)) < 129) && ((((((int)blockIdx) % 9) * 32) + (((int)threadIdx) & 31)) < 257)) {
      arg2_ptr[((((((((int)blockIdx) / 9) * 32896) + (tile_i_15_chunk * 1028)) + ((((int)threadIdx) >> 5) * 257)) + ((((int)blockIdx) % 9) * 32)) + (((int)threadIdx) & 31))] = tile_storage_0_shared[((tile_i_15_chunk * 128) + ((int)threadIdx))];
    }
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
}



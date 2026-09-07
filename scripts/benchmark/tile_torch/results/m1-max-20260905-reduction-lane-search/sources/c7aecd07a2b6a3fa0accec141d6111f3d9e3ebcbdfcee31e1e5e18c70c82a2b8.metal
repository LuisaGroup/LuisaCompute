// Function: benchmark_rmsnorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_rmsnorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)blockIdx) * 4096);
  for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 128; ++n_6_0_subgroup_chunk) {
    int cse_v3 = (((((int)blockIdx) * 4096) + (n_6_0_subgroup_chunk * 32)) + ((int)threadIdx));
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 4096) + (n_6_0_subgroup_chunk * 32)) + ((int)threadIdx))] * arg0_ptr[(((((int)blockIdx) * 4096) + (n_6_0_subgroup_chunk * 32)) + ((int)threadIdx))]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  for (int tile_i_10_subgroup_chunk = 0; tile_i_10_subgroup_chunk < 128; ++tile_i_10_subgroup_chunk) {
    int cse_v2 = (tile_i_10_subgroup_chunk * 32);
    int cse_v4 = (((((int)blockIdx) * 4096) + (tile_i_10_subgroup_chunk * 32)) + ((int)threadIdx));
    arg2_ptr[(((((int)blockIdx) * 4096) + (tile_i_10_subgroup_chunk * 32)) + ((int)threadIdx))] = ((arg0_ptr[(((((int)blockIdx) * 4096) + (tile_i_10_subgroup_chunk * 32)) + ((int)threadIdx))] / sqrt(((tile_storage_3[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[((tile_i_10_subgroup_chunk * 32) + ((int)threadIdx))]);
  }
}



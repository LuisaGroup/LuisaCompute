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
  for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 3; ++n_6_0_subgroup_chunk) {
    int cse_v2 = ((n_6_0_subgroup_chunk * 32) + ((int)threadIdx));
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((n_6_0_subgroup_chunk * 32) + ((int)threadIdx))] * arg0_ptr[((n_6_0_subgroup_chunk * 32) + ((int)threadIdx))]));
  }
  int cse_v1 = (((int)threadIdx) + 96);
  if (((int)threadIdx) < 31) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((int)threadIdx) + 96)] * arg0_ptr[(((int)threadIdx) + 96)]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  for (int tile_i_10_subgroup_chunk = 0; tile_i_10_subgroup_chunk < 3; ++tile_i_10_subgroup_chunk) {
    int cse_v3 = ((tile_i_10_subgroup_chunk * 32) + ((int)threadIdx));
    arg2_ptr[((tile_i_10_subgroup_chunk * 32) + ((int)threadIdx))] = ((arg0_ptr[((tile_i_10_subgroup_chunk * 32) + ((int)threadIdx))] / sqrt(((tile_storage_3[0] / 1.270000e+02f) + 1.000000e-05f))) * arg1_ptr[((tile_i_10_subgroup_chunk * 32) + ((int)threadIdx))]);
  }
  if (((int)threadIdx) < 31) {
    arg2_ptr[(((int)threadIdx) + 96)] = ((arg0_ptr[(((int)threadIdx) + 96)] / sqrt(((tile_storage_3[0] / 1.270000e+02f) + 1.000000e-05f))) * arg1_ptr[(((int)threadIdx) + 96)]);
  }
}



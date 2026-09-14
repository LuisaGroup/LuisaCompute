// Function: benchmark_sum_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_sum_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float parallel_0_subgroup_partials_0[4];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)threadIdx) * 4);
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 16; ++n_5_0_subgroup_chunk) {
    int cse_v3 = (((((int)blockIdx) * 8192) + (n_5_0_subgroup_chunk * 512)) + (((int)threadIdx) * 4));
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 8192) + (n_5_0_subgroup_chunk * 512)) + (((int)threadIdx) * 4))]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 8192) + (n_5_0_subgroup_chunk * 512)) + (((int)threadIdx) * 4)) + 1)]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 8192) + (n_5_0_subgroup_chunk * 512)) + (((int)threadIdx) * 4)) + 2)]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 8192) + (n_5_0_subgroup_chunk * 512)) + (((int)threadIdx) * 4)) + 3)]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v2 = (((int)threadIdx) & 31);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 4)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  if (((int)threadIdx) < 1) {
    arg1_ptr[((((int)threadIdx) * 4) + ((int)blockIdx))] = tile_storage_3[(((int)threadIdx) * 4)];
  }
}



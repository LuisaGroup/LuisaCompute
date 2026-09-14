// Function: benchmark_layernorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_layernorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float parallel_0_subgroup_partials_0[4];
  threadgroup float parallel_0_subgroup_partials_1[4];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)blockIdx) * 1024);
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 8; ++n_5_0_subgroup_chunk) {
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 1024) + (n_5_0_subgroup_chunk * 128)) + ((int)threadIdx))]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v2 = (((int)threadIdx) & 31);
  int cse_v3 = (((int)threadIdx) >> 5);
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
  thread float tile_storage_7[1];
  thread float tile_storage_9[1];
  tile_storage_9[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 8; ++n_20_0_subgroup_chunk) {
    int cse_v6 = (((((int)blockIdx) * 1024) + (n_20_0_subgroup_chunk * 128)) + ((int)threadIdx));
    tile_storage_9[0] = (tile_storage_9[0] + ((arg0_ptr[(((((int)blockIdx) * 1024) + (n_20_0_subgroup_chunk * 128)) + ((int)threadIdx))] - (tile_storage_3[0] / 1.024000e+03f)) * (arg0_ptr[(((((int)blockIdx) * 1024) + (n_20_0_subgroup_chunk * 128)) + ((int)threadIdx))] - (tile_storage_3[0] / 1.024000e+03f))));
  }
  tile_storage_9[0] = simd_sum(tile_storage_9[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_9[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 4)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_9[0] = simd_sum(condval_1);
  tile_storage_7[0] = tile_storage_9[0];
  for (int tile_i_17_subgroup_chunk = 0; tile_i_17_subgroup_chunk < 8; ++tile_i_17_subgroup_chunk) {
    int cse_v4 = (tile_i_17_subgroup_chunk * 128);
    int cse_v5 = ((tile_i_17_subgroup_chunk * 128) + ((int)threadIdx));
    int cse_v7 = (((((int)blockIdx) * 1024) + (tile_i_17_subgroup_chunk * 128)) + ((int)threadIdx));
    arg2_ptr[(((((int)blockIdx) * 1024) + (tile_i_17_subgroup_chunk * 128)) + ((int)threadIdx))] = ((((arg0_ptr[(((((int)blockIdx) * 1024) + (tile_i_17_subgroup_chunk * 128)) + ((int)threadIdx))] - (tile_storage_3[0] / 1.024000e+03f)) / sqrt(((tile_storage_7[0] / 1.024000e+03f) + 1.000000e-05f))) * arg1_ptr[((tile_i_17_subgroup_chunk * 128) + ((int)threadIdx))]) + arg1_ptr[(((tile_i_17_subgroup_chunk * 128) + ((int)threadIdx)) + 1024)]);
  }
}



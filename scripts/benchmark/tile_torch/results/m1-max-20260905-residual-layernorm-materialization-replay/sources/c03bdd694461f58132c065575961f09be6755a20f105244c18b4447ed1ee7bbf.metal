// Function: benchmark_residual_layernorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_residual_layernorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float parallel_0_subgroup_partials_0[2];
  threadgroup float parallel_0_subgroup_partials_1[2];
  thread float tile_storage_6[1];
  thread float tile_storage_8[1];
  tile_storage_8[0] = 0.000000e+00f;
  for (int n_7_0_subgroup_chunk = 0; n_7_0_subgroup_chunk < 2; ++n_7_0_subgroup_chunk) {
    int cse_v3 = ((n_7_0_subgroup_chunk * 64) + ((int)threadIdx));
    if (((n_7_0_subgroup_chunk * 64) + ((int)threadIdx)) < 127) {
      tile_storage_8[0] = (tile_storage_8[0] + (arg0_ptr[((n_7_0_subgroup_chunk * 64) + ((int)threadIdx))] + arg1_ptr[((n_7_0_subgroup_chunk * 64) + ((int)threadIdx))]));
    }
  }
  tile_storage_8[0] = simd_sum(tile_storage_8[0]);
  int cse_v1 = (((int)threadIdx) & 31);
  int cse_v2 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_8[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 2)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_8[0] = simd_sum(condval);
  tile_storage_6[0] = tile_storage_8[0];
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_22_0_subgroup_chunk = 0; n_22_0_subgroup_chunk < 2; ++n_22_0_subgroup_chunk) {
    int cse_v4 = ((n_22_0_subgroup_chunk * 64) + ((int)threadIdx));
    if (((n_22_0_subgroup_chunk * 64) + ((int)threadIdx)) < 127) {
      tile_storage_12[0] = (tile_storage_12[0] + (((arg0_ptr[((n_22_0_subgroup_chunk * 64) + ((int)threadIdx))] + arg1_ptr[((n_22_0_subgroup_chunk * 64) + ((int)threadIdx))]) - (tile_storage_6[0] / 1.270000e+02f)) * ((arg0_ptr[((n_22_0_subgroup_chunk * 64) + ((int)threadIdx))] + arg1_ptr[((n_22_0_subgroup_chunk * 64) + ((int)threadIdx))]) - (tile_storage_6[0] / 1.270000e+02f))));
    }
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 2)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_14_subgroup_chunk = 0; tile_i_14_subgroup_chunk < 2; ++tile_i_14_subgroup_chunk) {
    int cse_v5 = ((tile_i_14_subgroup_chunk * 64) + ((int)threadIdx));
    if (((tile_i_14_subgroup_chunk * 64) + ((int)threadIdx)) < 127) {
      arg2_ptr[((tile_i_14_subgroup_chunk * 64) + ((int)threadIdx))] = (((arg0_ptr[((tile_i_14_subgroup_chunk * 64) + ((int)threadIdx))] + arg1_ptr[((tile_i_14_subgroup_chunk * 64) + ((int)threadIdx))]) - (tile_storage_6[0] / 1.270000e+02f)) / sqrt(((tile_storage_10[0] / 1.270000e+02f) + 1.000000e-05f)));
    }
  }
}



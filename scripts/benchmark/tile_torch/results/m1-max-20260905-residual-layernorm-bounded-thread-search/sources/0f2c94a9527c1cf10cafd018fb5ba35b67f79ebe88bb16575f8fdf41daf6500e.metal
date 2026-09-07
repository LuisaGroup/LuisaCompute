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
  thread float tile_storage_6_worker_stripe[4];
  for (int tile_i_7_subgroup_chunk = 0; tile_i_7_subgroup_chunk < 4; ++tile_i_7_subgroup_chunk) {
    int cse_v1 = ((tile_i_7_subgroup_chunk * 32) + ((int)threadIdx));
    if (((tile_i_7_subgroup_chunk * 32) + ((int)threadIdx)) < 127) {
      tile_storage_6_worker_stripe[tile_i_7_subgroup_chunk] = (arg0_ptr[((tile_i_7_subgroup_chunk * 32) + ((int)threadIdx))] + arg1_ptr[((tile_i_7_subgroup_chunk * 32) + ((int)threadIdx))]);
    }
  }
  thread float tile_storage_9[1];
  thread float tile_storage_11[1];
  tile_storage_11[0] = 0.000000e+00f;
  for (int n_7_0_subgroup_chunk = 0; n_7_0_subgroup_chunk < 4; ++n_7_0_subgroup_chunk) {
    if (((n_7_0_subgroup_chunk * 32) + ((int)threadIdx)) < 127) {
      tile_storage_11[0] = (tile_storage_11[0] + tile_storage_6_worker_stripe[n_7_0_subgroup_chunk]);
    }
  }
  tile_storage_11[0] = simd_sum(tile_storage_11[0]);
  tile_storage_9[0] = tile_storage_11[0];
  thread float tile_storage_13_worker_stripe[4];
  for (int tile_i_14_subgroup_chunk = 0; tile_i_14_subgroup_chunk < 4; ++tile_i_14_subgroup_chunk) {
    if (((tile_i_14_subgroup_chunk * 32) + ((int)threadIdx)) < 127) {
      tile_storage_13_worker_stripe[tile_i_14_subgroup_chunk] = (tile_storage_6_worker_stripe[tile_i_14_subgroup_chunk] - (tile_storage_9[0] / 1.270000e+02f));
    }
  }
  thread float tile_storage_16[1];
  thread float tile_storage_18[1];
  tile_storage_18[0] = 0.000000e+00f;
  for (int n_22_0_subgroup_chunk = 0; n_22_0_subgroup_chunk < 4; ++n_22_0_subgroup_chunk) {
    if (((n_22_0_subgroup_chunk * 32) + ((int)threadIdx)) < 127) {
      tile_storage_18[0] = (tile_storage_18[0] + (tile_storage_13_worker_stripe[n_22_0_subgroup_chunk] * tile_storage_13_worker_stripe[n_22_0_subgroup_chunk]));
    }
  }
  tile_storage_18[0] = simd_sum(tile_storage_18[0]);
  tile_storage_16[0] = tile_storage_18[0];
  for (int tile_i_20_subgroup_chunk = 0; tile_i_20_subgroup_chunk < 4; ++tile_i_20_subgroup_chunk) {
    int cse_v2 = ((tile_i_20_subgroup_chunk * 32) + ((int)threadIdx));
    if (((tile_i_20_subgroup_chunk * 32) + ((int)threadIdx)) < 127) {
      arg2_ptr[((tile_i_20_subgroup_chunk * 32) + ((int)threadIdx))] = (tile_storage_13_worker_stripe[tile_i_20_subgroup_chunk] / sqrt(((tile_storage_16[0] / 1.270000e+02f) + 1.000000e-05f)));
    }
  }
}



// Function: benchmark_softmax_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_softmax_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = -INFINITY;
  int cse_v2 = (((int)threadIdx) & 31);
  int cse_v5 = ((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 5) * 257));
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 9; ++n_5_0_subgroup_chunk) {
    int cse_v1 = (n_5_0_subgroup_chunk * 32);
    if (((n_5_0_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
      tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 32)) + (((int)threadIdx) & 31))]);
    }
  }
  tile_storage_5[0] = simd_max(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[9];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 9; ++tile_i_8_subgroup_chunk) {
    int cse_v3 = (tile_i_8_subgroup_chunk * 32);
    if (((tile_i_8_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
      tile_storage_7_worker_stripe[tile_i_8_subgroup_chunk] = exp((arg0_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk * 32)) + (((int)threadIdx) & 31))] - tile_storage_3[0]));
    }
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_18_0_subgroup_chunk = 0; n_18_0_subgroup_chunk < 9; ++n_18_0_subgroup_chunk) {
    if (((n_18_0_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
      tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[n_18_0_subgroup_chunk]);
    }
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_14_subgroup_chunk = 0; tile_i_14_subgroup_chunk < 9; ++tile_i_14_subgroup_chunk) {
    int cse_v4 = (tile_i_14_subgroup_chunk * 32);
    if (((tile_i_14_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
      arg1_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_14_subgroup_chunk * 32)) + (((int)threadIdx) & 31))] = (tile_storage_7_worker_stripe[tile_i_14_subgroup_chunk] / tile_storage_10[0]);
    }
  }
}



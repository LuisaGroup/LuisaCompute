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
  int cse_v1 = (((int)blockIdx) * 2048);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 16; ++n_5_0_subgroup_chunk) {
    int cse_v6 = (((((int)blockIdx) * 2048) + (n_5_0_subgroup_chunk * 128)) + (((int)threadIdx) * 4));
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[(((((int)blockIdx) * 2048) + (n_5_0_subgroup_chunk * 128)) + (((int)threadIdx) * 4))]);
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 2048) + (n_5_0_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 1)]);
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 2048) + (n_5_0_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 2)]);
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 2048) + (n_5_0_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 3)]);
  }
  tile_storage_5[0] = simd_max(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[64];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 16; ++tile_i_8_subgroup_chunk) {
    int cse_v3 = (tile_i_8_subgroup_chunk * 4);
    int cse_v7 = (((((int)blockIdx) * 2048) + (tile_i_8_subgroup_chunk * 128)) + (((int)threadIdx) * 4));
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = exp((arg0_ptr[(((((int)blockIdx) * 2048) + (tile_i_8_subgroup_chunk * 128)) + (((int)threadIdx) * 4))] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = exp((arg0_ptr[((((((int)blockIdx) * 2048) + (tile_i_8_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 1)] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = exp((arg0_ptr[((((((int)blockIdx) * 2048) + (tile_i_8_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 2)] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = exp((arg0_ptr[((((((int)blockIdx) * 2048) + (tile_i_8_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 3)] - tile_storage_3[0]));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_18_0_subgroup_chunk = 0; n_18_0_subgroup_chunk < 16; ++n_18_0_subgroup_chunk) {
    int cse_v4 = (n_18_0_subgroup_chunk * 4);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[(n_18_0_subgroup_chunk * 4)]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk * 4) + 1)]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk * 4) + 2)]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk * 4) + 3)]);
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_14_subgroup_chunk = 0; tile_i_14_subgroup_chunk < 16; ++tile_i_14_subgroup_chunk) {
    int cse_v5 = (tile_i_14_subgroup_chunk * 4);
    int cse_v8 = (((((int)blockIdx) * 2048) + (tile_i_14_subgroup_chunk * 128)) + (((int)threadIdx) * 4));
    arg1_ptr[(((((int)blockIdx) * 2048) + (tile_i_14_subgroup_chunk * 128)) + (((int)threadIdx) * 4))] = (tile_storage_7_worker_stripe[(tile_i_14_subgroup_chunk * 4)] / tile_storage_10[0]);
    arg1_ptr[((((((int)blockIdx) * 2048) + (tile_i_14_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 1)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk * 4) + 1)] / tile_storage_10[0]);
    arg1_ptr[((((((int)blockIdx) * 2048) + (tile_i_14_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 2)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk * 4) + 2)] / tile_storage_10[0]);
    arg1_ptr[((((((int)blockIdx) * 2048) + (tile_i_14_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 3)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk * 4) + 3)] / tile_storage_10[0]);
  }
}



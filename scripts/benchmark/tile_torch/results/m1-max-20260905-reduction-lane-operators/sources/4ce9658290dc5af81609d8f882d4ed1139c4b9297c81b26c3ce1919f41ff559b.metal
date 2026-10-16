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
  int cse_v1 = (((int)threadIdx) >> 5);
  if (((((int)blockIdx) * 8) + (((int)threadIdx) >> 5)) < 17) {
    thread float tile_storage_3[1];
    thread float tile_storage_5[1];
    tile_storage_5[0] = -INFINITY;
    int cse_v2 = (((int)threadIdx) & 31);
    int cse_v6 = ((((int)threadIdx) & 31) * 4);
    int cse_v7 = ((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257));
    for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 2; ++n_5_0_subgroup_chunk) {
      int cse_v8 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4));
      tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4))]);
      tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 1)]);
      tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 2)]);
      tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 3)]);
    }
    int cse_v9 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + ((((int)threadIdx) & 31) * 4)) + 256);
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + ((((int)threadIdx) & 31) * 4)) + 256)]);
    }
    tile_storage_5[0] = simd_max(tile_storage_5[0]);
    tile_storage_3[0] = tile_storage_5[0];
    thread float tile_storage_7_worker_stripe[9];
    for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 2; ++tile_i_8_subgroup_chunk) {
      int cse_v3 = (tile_i_8_subgroup_chunk * 4);
      int cse_v10 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4));
      tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = exp((arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4))] - tile_storage_3[0]));
      tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = exp((arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 1)] - tile_storage_3[0]));
      tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = exp((arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 2)] - tile_storage_3[0]));
      tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = exp((arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 3)] - tile_storage_3[0]));
    }
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_7_worker_stripe[8] = exp((arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + ((((int)threadIdx) & 31) * 4)) + 256)] - tile_storage_3[0]));
    }
    thread float tile_storage_10[1];
    thread float tile_storage_12[1];
    tile_storage_12[0] = 0.000000e+00f;
    for (int n_18_0_subgroup_chunk = 0; n_18_0_subgroup_chunk < 2; ++n_18_0_subgroup_chunk) {
      int cse_v4 = (n_18_0_subgroup_chunk * 4);
      tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[(n_18_0_subgroup_chunk * 4)]);
      tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk * 4) + 1)]);
      tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk * 4) + 2)]);
      tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk * 4) + 3)]);
    }
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[8]);
    }
    tile_storage_12[0] = simd_sum(tile_storage_12[0]);
    tile_storage_10[0] = tile_storage_12[0];
    for (int tile_i_14_subgroup_chunk = 0; tile_i_14_subgroup_chunk < 2; ++tile_i_14_subgroup_chunk) {
      int cse_v5 = (tile_i_14_subgroup_chunk * 4);
      int cse_v11 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_14_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4));
      arg1_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_14_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4))] = (tile_storage_7_worker_stripe[(tile_i_14_subgroup_chunk * 4)] / tile_storage_10[0]);
      arg1_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_14_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 1)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk * 4) + 1)] / tile_storage_10[0]);
      arg1_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_14_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 2)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk * 4) + 2)] / tile_storage_10[0]);
      arg1_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_14_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 3)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk * 4) + 3)] / tile_storage_10[0]);
    }
    if ((((int)threadIdx) & 31) < 1) {
      arg1_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + ((((int)threadIdx) & 31) * 4)) + 256)] = (tile_storage_7_worker_stripe[8] / tile_storage_10[0]);
    }
  }
}



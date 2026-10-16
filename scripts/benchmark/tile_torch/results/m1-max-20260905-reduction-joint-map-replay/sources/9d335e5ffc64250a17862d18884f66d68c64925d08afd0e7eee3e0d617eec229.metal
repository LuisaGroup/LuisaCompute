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
  threadgroup float parallel_0_subgroup_partials_0[4];
  threadgroup float parallel_0_subgroup_partials_1[4];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = -INFINITY;
  int cse_v1 = (((int)blockIdx) * 4096);
  for (int n_5_0_subgroup_chunk_pack = 0; n_5_0_subgroup_chunk_pack < 8; ++n_5_0_subgroup_chunk_pack) {
    int cse_v7 = (((((int)blockIdx) * 4096) + (n_5_0_subgroup_chunk_pack * 512)) + ((int)threadIdx));
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[(((((int)blockIdx) * 4096) + (n_5_0_subgroup_chunk_pack * 512)) + ((int)threadIdx))]);
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 4096) + (n_5_0_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 128)]);
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 4096) + (n_5_0_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 256)]);
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((((int)blockIdx) * 4096) + (n_5_0_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 384)]);
  }
  tile_storage_5[0] = simd_max(tile_storage_5[0]);
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
    condval = -INFINITY;
  }
  tile_storage_5[0] = simd_max(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[32];
  for (int tile_i_8_subgroup_chunk_pack = 0; tile_i_8_subgroup_chunk_pack < 8; ++tile_i_8_subgroup_chunk_pack) {
    int cse_v4 = (tile_i_8_subgroup_chunk_pack * 4);
    int cse_v8 = (((((int)blockIdx) * 4096) + (tile_i_8_subgroup_chunk_pack * 512)) + ((int)threadIdx));
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk_pack * 4)] = exp((arg0_ptr[(((((int)blockIdx) * 4096) + (tile_i_8_subgroup_chunk_pack * 512)) + ((int)threadIdx))] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk_pack * 4) + 1)] = exp((arg0_ptr[((((((int)blockIdx) * 4096) + (tile_i_8_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 128)] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk_pack * 4) + 2)] = exp((arg0_ptr[((((((int)blockIdx) * 4096) + (tile_i_8_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 256)] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk_pack * 4) + 3)] = exp((arg0_ptr[((((((int)blockIdx) * 4096) + (tile_i_8_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 384)] - tile_storage_3[0]));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_18_0_subgroup_chunk_pack = 0; n_18_0_subgroup_chunk_pack < 8; ++n_18_0_subgroup_chunk_pack) {
    int cse_v5 = (n_18_0_subgroup_chunk_pack * 4);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[(n_18_0_subgroup_chunk_pack * 4)]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk_pack * 4) + 1)]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk_pack * 4) + 2)]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[((n_18_0_subgroup_chunk_pack * 4) + 3)]);
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 4)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_14_subgroup_chunk_pack = 0; tile_i_14_subgroup_chunk_pack < 8; ++tile_i_14_subgroup_chunk_pack) {
    int cse_v6 = (tile_i_14_subgroup_chunk_pack * 4);
    int cse_v9 = (((((int)blockIdx) * 4096) + (tile_i_14_subgroup_chunk_pack * 512)) + ((int)threadIdx));
    arg1_ptr[(((((int)blockIdx) * 4096) + (tile_i_14_subgroup_chunk_pack * 512)) + ((int)threadIdx))] = (tile_storage_7_worker_stripe[(tile_i_14_subgroup_chunk_pack * 4)] / tile_storage_10[0]);
    arg1_ptr[((((((int)blockIdx) * 4096) + (tile_i_14_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 128)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk_pack * 4) + 1)] / tile_storage_10[0]);
    arg1_ptr[((((((int)blockIdx) * 4096) + (tile_i_14_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 256)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk_pack * 4) + 2)] / tile_storage_10[0]);
    arg1_ptr[((((((int)blockIdx) * 4096) + (tile_i_14_subgroup_chunk_pack * 512)) + ((int)threadIdx)) + 384)] = (tile_storage_7_worker_stripe[((tile_i_14_subgroup_chunk_pack * 4) + 3)] / tile_storage_10[0]);
  }
}



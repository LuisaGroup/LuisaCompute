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
  threadgroup float parallel_0_subgroup_partials_0[16];
  threadgroup float parallel_0_subgroup_partials_1[16];
  thread float tile_storage_0_worker_stripe[24];
  int cse_v1 = (((int)threadIdx) >> 8);
  int cse_v10 = ((((int)threadIdx) & 255) * 4);
  int cse_v17 = ((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144));
  for (int tile_i_1_subgroup_chunk = 0; tile_i_1_subgroup_chunk < 6; ++tile_i_1_subgroup_chunk) {
    int cse_v2 = (tile_i_1_subgroup_chunk * 4);
    int cse_v20 = ((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_1_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4));
    tile_storage_0_worker_stripe[(tile_i_1_subgroup_chunk * 4)] = arg0_ptr[((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_1_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4))];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 1)] = arg0_ptr[(((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_1_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4)) + 1)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 2)] = arg0_ptr[(((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_1_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4)) + 2)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 3)] = arg0_ptr[(((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_1_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4)) + 3)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 6; ++n_5_0_subgroup_chunk) {
    int cse_v3 = (n_5_0_subgroup_chunk * 4);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[(n_5_0_subgroup_chunk * 4)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 1)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 2)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 3)]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v4 = (((int)threadIdx) & 31);
  int cse_v5 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int cse_v18 = (((((int)threadIdx) >> 8) * 8) + (((int)threadIdx) & 31));
  float condval;
  if (((((int)threadIdx) & 31) < 8)) {
    condval = parallel_0_subgroup_partials_0[(((((int)threadIdx) >> 8) * 8) + (((int)threadIdx) & 31))];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[24];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 6; ++tile_i_8_subgroup_chunk) {
    int cse_v6 = (tile_i_8_subgroup_chunk * 4);
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = (tile_storage_0_worker_stripe[(tile_i_8_subgroup_chunk * 4)] - (tile_storage_3[0] / 6.144000e+03f));
    int cse_v11 = ((tile_i_8_subgroup_chunk * 4) + 1);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] - (tile_storage_3[0] / 6.144000e+03f));
    int cse_v12 = ((tile_i_8_subgroup_chunk * 4) + 2);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] - (tile_storage_3[0] / 6.144000e+03f));
    int cse_v13 = ((tile_i_8_subgroup_chunk * 4) + 3);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] - (tile_storage_3[0] / 6.144000e+03f));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 6; ++n_20_0_subgroup_chunk) {
    int cse_v7 = (n_20_0_subgroup_chunk * 4);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)] * tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)]));
    int cse_v14 = ((n_20_0_subgroup_chunk * 4) + 1);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)]));
    int cse_v15 = ((n_20_0_subgroup_chunk * 4) + 2);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)]));
    int cse_v16 = ((n_20_0_subgroup_chunk * 4) + 3);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)]));
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 8)) {
    condval_1 = parallel_0_subgroup_partials_1[(((((int)threadIdx) >> 8) * 8) + (((int)threadIdx) & 31))];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_20_subgroup_chunk = 0; tile_i_20_subgroup_chunk < 6; ++tile_i_20_subgroup_chunk) {
    int cse_v8 = (tile_i_20_subgroup_chunk * 4);
    int cse_v9 = (tile_i_20_subgroup_chunk * 1024);
    int cse_v19 = ((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4));
    int cse_v21 = ((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_20_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4));
    arg2_ptr[((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_20_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4))] = (((tile_storage_7_worker_stripe[(tile_i_20_subgroup_chunk * 4)] / sqrt(((tile_storage_10[0] / 6.144000e+03f) + 1.000000e-05f))) * arg1_ptr[((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4))]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 6144)]);
    arg2_ptr[(((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_20_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4)) + 1)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_10[0] / 6.144000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 1)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 6145)]);
    arg2_ptr[(((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_20_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4)) + 2)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_10[0] / 6.144000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 2)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 6146)]);
    arg2_ptr[(((((((int)blockIdx) * 12288) + ((((int)threadIdx) >> 8) * 6144)) + (tile_i_20_subgroup_chunk * 1024)) + ((((int)threadIdx) & 255) * 4)) + 3)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_10[0] / 6.144000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 3)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1024) + ((((int)threadIdx) & 255) * 4)) + 6147)]);
  }
}



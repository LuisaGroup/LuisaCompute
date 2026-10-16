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
  thread float tile_storage_0_worker_stripe[16];
  int cse_v1 = (((int)blockIdx) * 8191);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int tile_i_1_subgroup_chunk = 0; tile_i_1_subgroup_chunk < 3; ++tile_i_1_subgroup_chunk) {
    int cse_v3 = (tile_i_1_subgroup_chunk * 4);
    int cse_v25 = (((((int)blockIdx) * 8191) + (tile_i_1_subgroup_chunk * 2048)) + (((int)threadIdx) * 4));
    tile_storage_0_worker_stripe[(tile_i_1_subgroup_chunk * 4)] = arg0_ptr[(((((int)blockIdx) * 8191) + (tile_i_1_subgroup_chunk * 2048)) + (((int)threadIdx) * 4))];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 1)] = arg0_ptr[((((((int)blockIdx) * 8191) + (tile_i_1_subgroup_chunk * 2048)) + (((int)threadIdx) * 4)) + 1)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 2)] = arg0_ptr[((((((int)blockIdx) * 8191) + (tile_i_1_subgroup_chunk * 2048)) + (((int)threadIdx) * 4)) + 2)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 3)] = arg0_ptr[((((((int)blockIdx) * 8191) + (tile_i_1_subgroup_chunk * 2048)) + (((int)threadIdx) * 4)) + 3)];
  }
  int cse_v11 = ((((int)blockIdx) * 8191) + (((int)threadIdx) * 4));
  int cse_v26 = (((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6144);
  int cse_v27 = (((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6145);
  int cse_v28 = (((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6146);
  int cse_v29 = (((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6147);
  if (((int)threadIdx) < 511) {
    tile_storage_0_worker_stripe[12] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6144)];
    tile_storage_0_worker_stripe[13] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6145)];
    tile_storage_0_worker_stripe[14] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6146)];
    tile_storage_0_worker_stripe[15] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6147)];
  }
  if (((int)threadIdx) == 511) {
    tile_storage_0_worker_stripe[12] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6144)];
    tile_storage_0_worker_stripe[13] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6145)];
    tile_storage_0_worker_stripe[14] = arg0_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6146)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 3; ++n_5_0_subgroup_chunk) {
    int cse_v4 = (n_5_0_subgroup_chunk * 4);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[(n_5_0_subgroup_chunk * 4)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 1)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 2)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 3)]);
  }
  if (((int)threadIdx) < 511) {
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[12]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[13]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[14]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[15]);
  }
  if (((int)threadIdx) == 511) {
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[12]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[13]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[14]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v5 = (((int)threadIdx) & 31);
  int cse_v6 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 16)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[16];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 3; ++tile_i_8_subgroup_chunk) {
    int cse_v7 = (tile_i_8_subgroup_chunk * 4);
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = (tile_storage_0_worker_stripe[(tile_i_8_subgroup_chunk * 4)] - (tile_storage_3[0] / 8.191000e+03f));
    int cse_v12 = ((tile_i_8_subgroup_chunk * 4) + 1);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] - (tile_storage_3[0] / 8.191000e+03f));
    int cse_v13 = ((tile_i_8_subgroup_chunk * 4) + 2);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] - (tile_storage_3[0] / 8.191000e+03f));
    int cse_v14 = ((tile_i_8_subgroup_chunk * 4) + 3);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] - (tile_storage_3[0] / 8.191000e+03f));
  }
  if (((int)threadIdx) < 511) {
    tile_storage_7_worker_stripe[12] = (tile_storage_0_worker_stripe[12] - (tile_storage_3[0] / 8.191000e+03f));
    tile_storage_7_worker_stripe[13] = (tile_storage_0_worker_stripe[13] - (tile_storage_3[0] / 8.191000e+03f));
    tile_storage_7_worker_stripe[14] = (tile_storage_0_worker_stripe[14] - (tile_storage_3[0] / 8.191000e+03f));
    tile_storage_7_worker_stripe[15] = (tile_storage_0_worker_stripe[15] - (tile_storage_3[0] / 8.191000e+03f));
  }
  if (((int)threadIdx) == 511) {
    tile_storage_7_worker_stripe[12] = (tile_storage_0_worker_stripe[12] - (tile_storage_3[0] / 8.191000e+03f));
    tile_storage_7_worker_stripe[13] = (tile_storage_0_worker_stripe[13] - (tile_storage_3[0] / 8.191000e+03f));
    tile_storage_7_worker_stripe[14] = (tile_storage_0_worker_stripe[14] - (tile_storage_3[0] / 8.191000e+03f));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 3; ++n_20_0_subgroup_chunk) {
    int cse_v8 = (n_20_0_subgroup_chunk * 4);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)] * tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)]));
    int cse_v15 = ((n_20_0_subgroup_chunk * 4) + 1);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)]));
    int cse_v16 = ((n_20_0_subgroup_chunk * 4) + 2);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)]));
    int cse_v17 = ((n_20_0_subgroup_chunk * 4) + 3);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)]));
  }
  if (((int)threadIdx) < 511) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[12] * tile_storage_7_worker_stripe[12]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[13] * tile_storage_7_worker_stripe[13]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[14] * tile_storage_7_worker_stripe[14]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[15] * tile_storage_7_worker_stripe[15]));
  }
  if (((int)threadIdx) == 511) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[12] * tile_storage_7_worker_stripe[12]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[13] * tile_storage_7_worker_stripe[13]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[14] * tile_storage_7_worker_stripe[14]));
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 16)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_20_subgroup_chunk = 0; tile_i_20_subgroup_chunk < 3; ++tile_i_20_subgroup_chunk) {
    int cse_v9 = (tile_i_20_subgroup_chunk * 4);
    int cse_v10 = (tile_i_20_subgroup_chunk * 2048);
    int cse_v18 = ((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4));
    int cse_v30 = (((((int)blockIdx) * 8191) + (tile_i_20_subgroup_chunk * 2048)) + (((int)threadIdx) * 4));
    arg2_ptr[(((((int)blockIdx) * 8191) + (tile_i_20_subgroup_chunk * 2048)) + (((int)threadIdx) * 4))] = (((tile_storage_7_worker_stripe[(tile_i_20_subgroup_chunk * 4)] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4))]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 8191)]);
    arg2_ptr[((((((int)blockIdx) * 8191) + (tile_i_20_subgroup_chunk * 2048)) + (((int)threadIdx) * 4)) + 1)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 1)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 8192)]);
    arg2_ptr[((((((int)blockIdx) * 8191) + (tile_i_20_subgroup_chunk * 2048)) + (((int)threadIdx) * 4)) + 2)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 2)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 8193)]);
    arg2_ptr[((((((int)blockIdx) * 8191) + (tile_i_20_subgroup_chunk * 2048)) + (((int)threadIdx) * 4)) + 3)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 3)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 2048) + (((int)threadIdx) * 4)) + 8194)]);
  }
  int cse_v19 = ((((int)threadIdx) * 4) + 6144);
  int cse_v20 = ((((int)threadIdx) * 4) + 14335);
  int cse_v21 = ((((int)threadIdx) * 4) + 6145);
  int cse_v22 = ((((int)threadIdx) * 4) + 14336);
  int cse_v23 = ((((int)threadIdx) * 4) + 6146);
  int cse_v24 = ((((int)threadIdx) * 4) + 14337);
  if (((int)threadIdx) < 511) {
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6144)] = (((tile_storage_7_worker_stripe[12] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6144)]) + arg1_ptr[((((int)threadIdx) * 4) + 14335)]);
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6145)] = (((tile_storage_7_worker_stripe[13] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6145)]) + arg1_ptr[((((int)threadIdx) * 4) + 14336)]);
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6146)] = (((tile_storage_7_worker_stripe[14] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6146)]) + arg1_ptr[((((int)threadIdx) * 4) + 14337)]);
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6147)] = (((tile_storage_7_worker_stripe[15] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6147)]) + arg1_ptr[((((int)threadIdx) * 4) + 14338)]);
  }
  if (((int)threadIdx) == 511) {
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6144)] = (((tile_storage_7_worker_stripe[12] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6144)]) + arg1_ptr[((((int)threadIdx) * 4) + 14335)]);
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6145)] = (((tile_storage_7_worker_stripe[13] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6145)]) + arg1_ptr[((((int)threadIdx) * 4) + 14336)]);
    arg2_ptr[(((((int)blockIdx) * 8191) + (((int)threadIdx) * 4)) + 6146)] = (((tile_storage_7_worker_stripe[14] / sqrt(((tile_storage_10[0] / 8.191000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 6146)]) + arg1_ptr[((((int)threadIdx) * 4) + 14337)]);
  }
}



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
  threadgroup float parallel_0_subgroup_partials_0[25];
  threadgroup float parallel_0_subgroup_partials_1[25];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)blockIdx) * 12289);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 3; ++n_5_0_subgroup_chunk) {
    int cse_v14 = (((((int)blockIdx) * 12289) + (n_5_0_subgroup_chunk * 3200)) + (((int)threadIdx) * 4));
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 12289) + (n_5_0_subgroup_chunk * 3200)) + (((int)threadIdx) * 4))]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 12289) + (n_5_0_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 1)]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 12289) + (n_5_0_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 2)]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 12289) + (n_5_0_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 3)]);
  }
  int cse_v9 = ((((int)blockIdx) * 12289) + (((int)threadIdx) * 4));
  int cse_v15 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9600);
  if (((int)threadIdx) < 673) {
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9600)]);
  }
  int cse_v16 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9601);
  if (((int)threadIdx) < 672) {
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9601)]);
  }
  int cse_v17 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9602);
  if (((int)threadIdx) < 672) {
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9602)]);
  }
  int cse_v18 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9603);
  if (((int)threadIdx) < 672) {
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9603)]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v3 = (((int)threadIdx) & 31);
  int cse_v4 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 25)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[16];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 3; ++tile_i_8_subgroup_chunk) {
    int cse_v5 = (tile_i_8_subgroup_chunk * 4);
    int cse_v19 = (((((int)blockIdx) * 12289) + (tile_i_8_subgroup_chunk * 3200)) + (((int)threadIdx) * 4));
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = (arg0_ptr[(((((int)blockIdx) * 12289) + (tile_i_8_subgroup_chunk * 3200)) + (((int)threadIdx) * 4))] - (tile_storage_3[0] / 1.228900e+04f));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = (arg0_ptr[((((((int)blockIdx) * 12289) + (tile_i_8_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 1)] - (tile_storage_3[0] / 1.228900e+04f));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = (arg0_ptr[((((((int)blockIdx) * 12289) + (tile_i_8_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 2)] - (tile_storage_3[0] / 1.228900e+04f));
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = (arg0_ptr[((((((int)blockIdx) * 12289) + (tile_i_8_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 3)] - (tile_storage_3[0] / 1.228900e+04f));
  }
  if (((int)threadIdx) < 673) {
    tile_storage_7_worker_stripe[12] = (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9600)] - (tile_storage_3[0] / 1.228900e+04f));
  }
  if (((int)threadIdx) < 672) {
    tile_storage_7_worker_stripe[13] = (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9601)] - (tile_storage_3[0] / 1.228900e+04f));
  }
  if (((int)threadIdx) < 672) {
    tile_storage_7_worker_stripe[14] = (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9602)] - (tile_storage_3[0] / 1.228900e+04f));
  }
  if (((int)threadIdx) < 672) {
    tile_storage_7_worker_stripe[15] = (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9603)] - (tile_storage_3[0] / 1.228900e+04f));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 3; ++n_20_0_subgroup_chunk) {
    int cse_v6 = (n_20_0_subgroup_chunk * 4);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)] * tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)]));
    int cse_v10 = ((n_20_0_subgroup_chunk * 4) + 1);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)]));
    int cse_v11 = ((n_20_0_subgroup_chunk * 4) + 2);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)]));
    int cse_v12 = ((n_20_0_subgroup_chunk * 4) + 3);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)]));
  }
  if (((int)threadIdx) < 673) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[12] * tile_storage_7_worker_stripe[12]));
  }
  if (((int)threadIdx) < 672) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[13] * tile_storage_7_worker_stripe[13]));
  }
  if (((int)threadIdx) < 672) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[14] * tile_storage_7_worker_stripe[14]));
  }
  if (((int)threadIdx) < 672) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[15] * tile_storage_7_worker_stripe[15]));
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 25)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_20_subgroup_chunk = 0; tile_i_20_subgroup_chunk < 3; ++tile_i_20_subgroup_chunk) {
    int cse_v7 = (tile_i_20_subgroup_chunk * 4);
    int cse_v8 = (tile_i_20_subgroup_chunk * 3200);
    int cse_v13 = ((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4));
    int cse_v20 = (((((int)blockIdx) * 12289) + (tile_i_20_subgroup_chunk * 3200)) + (((int)threadIdx) * 4));
    arg2_ptr[(((((int)blockIdx) * 12289) + (tile_i_20_subgroup_chunk * 3200)) + (((int)threadIdx) * 4))] = (((tile_storage_7_worker_stripe[(tile_i_20_subgroup_chunk * 4)] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4))]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 12289)]);
    arg2_ptr[((((((int)blockIdx) * 12289) + (tile_i_20_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 1)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 1)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 12290)]);
    arg2_ptr[((((((int)blockIdx) * 12289) + (tile_i_20_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 2)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 2)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 12291)]);
    arg2_ptr[((((((int)blockIdx) * 12289) + (tile_i_20_subgroup_chunk * 3200)) + (((int)threadIdx) * 4)) + 3)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 3)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 3200) + (((int)threadIdx) * 4)) + 12292)]);
  }
  if (((int)threadIdx) < 673) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9600)] = (((tile_storage_7_worker_stripe[12] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 9600)]) + arg1_ptr[((((int)threadIdx) * 4) + 21889)]);
  }
  if (((int)threadIdx) < 672) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9601)] = (((tile_storage_7_worker_stripe[13] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 9601)]) + arg1_ptr[((((int)threadIdx) * 4) + 21890)]);
  }
  if (((int)threadIdx) < 672) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9602)] = (((tile_storage_7_worker_stripe[14] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 9602)]) + arg1_ptr[((((int)threadIdx) * 4) + 21891)]);
  }
  if (((int)threadIdx) < 672) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 9603)] = (((tile_storage_7_worker_stripe[15] / sqrt(((tile_storage_10[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 9603)]) + arg1_ptr[((((int)threadIdx) * 4) + 21892)]);
  }
}



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
  threadgroup float parallel_0_subgroup_partials_0[11];
  threadgroup float parallel_0_subgroup_partials_1[11];
  thread float tile_storage_0_worker_stripe[12];
  int cse_v1 = (((int)blockIdx) * 4096);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int tile_i_1_subgroup_chunk = 0; tile_i_1_subgroup_chunk < 2; ++tile_i_1_subgroup_chunk) {
    int cse_v3 = (tile_i_1_subgroup_chunk * 4);
    int cse_v19 = (((((int)blockIdx) * 4096) + (tile_i_1_subgroup_chunk * 1408)) + (((int)threadIdx) * 4));
    tile_storage_0_worker_stripe[(tile_i_1_subgroup_chunk * 4)] = arg0_ptr[(((((int)blockIdx) * 4096) + (tile_i_1_subgroup_chunk * 1408)) + (((int)threadIdx) * 4))];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 1)] = arg0_ptr[((((((int)blockIdx) * 4096) + (tile_i_1_subgroup_chunk * 1408)) + (((int)threadIdx) * 4)) + 1)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 2)] = arg0_ptr[((((((int)blockIdx) * 4096) + (tile_i_1_subgroup_chunk * 1408)) + (((int)threadIdx) * 4)) + 2)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 3)] = arg0_ptr[((((((int)blockIdx) * 4096) + (tile_i_1_subgroup_chunk * 1408)) + (((int)threadIdx) * 4)) + 3)];
  }
  int cse_v11 = ((((int)blockIdx) * 4096) + (((int)threadIdx) * 4));
  int cse_v20 = (((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2816);
  int cse_v21 = (((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2817);
  int cse_v22 = (((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2818);
  int cse_v23 = (((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2819);
  if (((int)threadIdx) < 320) {
    tile_storage_0_worker_stripe[8] = arg0_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2816)];
    tile_storage_0_worker_stripe[9] = arg0_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2817)];
    tile_storage_0_worker_stripe[10] = arg0_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2818)];
    tile_storage_0_worker_stripe[11] = arg0_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2819)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 2; ++n_5_0_subgroup_chunk) {
    int cse_v4 = (n_5_0_subgroup_chunk * 4);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[(n_5_0_subgroup_chunk * 4)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 1)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 2)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 3)]);
  }
  if (((int)threadIdx) < 320) {
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[8]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[9]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[10]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[11]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v5 = (((int)threadIdx) & 31);
  int cse_v6 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 11)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[12];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 2; ++tile_i_8_subgroup_chunk) {
    int cse_v7 = (tile_i_8_subgroup_chunk * 4);
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = (tile_storage_0_worker_stripe[(tile_i_8_subgroup_chunk * 4)] - (tile_storage_3[0] / 4.096000e+03f));
    int cse_v12 = ((tile_i_8_subgroup_chunk * 4) + 1);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] - (tile_storage_3[0] / 4.096000e+03f));
    int cse_v13 = ((tile_i_8_subgroup_chunk * 4) + 2);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] - (tile_storage_3[0] / 4.096000e+03f));
    int cse_v14 = ((tile_i_8_subgroup_chunk * 4) + 3);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] - (tile_storage_3[0] / 4.096000e+03f));
  }
  if (((int)threadIdx) < 320) {
    tile_storage_7_worker_stripe[8] = (tile_storage_0_worker_stripe[8] - (tile_storage_3[0] / 4.096000e+03f));
    tile_storage_7_worker_stripe[9] = (tile_storage_0_worker_stripe[9] - (tile_storage_3[0] / 4.096000e+03f));
    tile_storage_7_worker_stripe[10] = (tile_storage_0_worker_stripe[10] - (tile_storage_3[0] / 4.096000e+03f));
    tile_storage_7_worker_stripe[11] = (tile_storage_0_worker_stripe[11] - (tile_storage_3[0] / 4.096000e+03f));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 2; ++n_20_0_subgroup_chunk) {
    int cse_v8 = (n_20_0_subgroup_chunk * 4);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)] * tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)]));
    int cse_v15 = ((n_20_0_subgroup_chunk * 4) + 1);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)]));
    int cse_v16 = ((n_20_0_subgroup_chunk * 4) + 2);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)]));
    int cse_v17 = ((n_20_0_subgroup_chunk * 4) + 3);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)]));
  }
  if (((int)threadIdx) < 320) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[8] * tile_storage_7_worker_stripe[8]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[9] * tile_storage_7_worker_stripe[9]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[10] * tile_storage_7_worker_stripe[10]));
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[11] * tile_storage_7_worker_stripe[11]));
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 11)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_20_subgroup_chunk = 0; tile_i_20_subgroup_chunk < 2; ++tile_i_20_subgroup_chunk) {
    int cse_v9 = (tile_i_20_subgroup_chunk * 4);
    int cse_v10 = (tile_i_20_subgroup_chunk * 1408);
    int cse_v18 = ((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4));
    int cse_v24 = (((((int)blockIdx) * 4096) + (tile_i_20_subgroup_chunk * 1408)) + (((int)threadIdx) * 4));
    arg2_ptr[(((((int)blockIdx) * 4096) + (tile_i_20_subgroup_chunk * 1408)) + (((int)threadIdx) * 4))] = (((tile_storage_7_worker_stripe[(tile_i_20_subgroup_chunk * 4)] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4))]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 4096)]);
    arg2_ptr[((((((int)blockIdx) * 4096) + (tile_i_20_subgroup_chunk * 1408)) + (((int)threadIdx) * 4)) + 1)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 1)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 4097)]);
    arg2_ptr[((((((int)blockIdx) * 4096) + (tile_i_20_subgroup_chunk * 1408)) + (((int)threadIdx) * 4)) + 2)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 2)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 4098)]);
    arg2_ptr[((((((int)blockIdx) * 4096) + (tile_i_20_subgroup_chunk * 1408)) + (((int)threadIdx) * 4)) + 3)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 3)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 1408) + (((int)threadIdx) * 4)) + 4099)]);
  }
  if (((int)threadIdx) < 320) {
    arg2_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2816)] = (((tile_storage_7_worker_stripe[8] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 2816)]) + arg1_ptr[((((int)threadIdx) * 4) + 6912)]);
    arg2_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2817)] = (((tile_storage_7_worker_stripe[9] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 2817)]) + arg1_ptr[((((int)threadIdx) * 4) + 6913)]);
    arg2_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2818)] = (((tile_storage_7_worker_stripe[10] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 2818)]) + arg1_ptr[((((int)threadIdx) * 4) + 6914)]);
    arg2_ptr[(((((int)blockIdx) * 4096) + (((int)threadIdx) * 4)) + 2819)] = (((tile_storage_7_worker_stripe[11] / sqrt(((tile_storage_10[0] / 4.096000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 2819)]) + arg1_ptr[((((int)threadIdx) * 4) + 6915)]);
  }
}



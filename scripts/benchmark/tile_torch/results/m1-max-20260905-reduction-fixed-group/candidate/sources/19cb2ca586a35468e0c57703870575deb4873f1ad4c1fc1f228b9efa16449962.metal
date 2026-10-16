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
  threadgroup float parallel_0_subgroup_partials_0[8];
  threadgroup float parallel_0_subgroup_partials_1[8];
  thread float tile_storage_0_worker_stripe[5];
  int cse_v1 = (((int)threadIdx) >> 6);
  int cse_v2 = (((int)threadIdx) & 63);
  int cse_v5 = ((((int)threadIdx) & 63) * 4);
  int cse_v6 = ((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257));
  int cse_v8 = (((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4));
  tile_storage_0_worker_stripe[0] = arg0_ptr[(((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4))];
  int cse_v10 = ((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 1);
  tile_storage_0_worker_stripe[1] = arg0_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 1)];
  int cse_v11 = ((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 2);
  tile_storage_0_worker_stripe[2] = arg0_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 2)];
  int cse_v12 = ((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 3);
  tile_storage_0_worker_stripe[3] = arg0_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 3)];
  int cse_v9 = (((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + 256);
  if ((((int)threadIdx) % 64) == 0) {
    tile_storage_0_worker_stripe[4] = arg0_ptr[(((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + 256)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[0]);
  tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[1]);
  tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[2]);
  tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[3]);
  if ((((int)threadIdx) % 64) == 0) {
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[4]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v3 = (((int)threadIdx) & 31);
  int cse_v4 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int cse_v7 = (((((int)threadIdx) >> 6) * 2) + (((int)threadIdx) & 31));
  float condval;
  if (((((int)threadIdx) & 31) < 2)) {
    condval = parallel_0_subgroup_partials_0[(((((int)threadIdx) >> 6) * 2) + (((int)threadIdx) & 31))];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[5];
  tile_storage_7_worker_stripe[0] = (tile_storage_0_worker_stripe[0] - (tile_storage_3[0] / 2.570000e+02f));
  tile_storage_7_worker_stripe[1] = (tile_storage_0_worker_stripe[1] - (tile_storage_3[0] / 2.570000e+02f));
  tile_storage_7_worker_stripe[2] = (tile_storage_0_worker_stripe[2] - (tile_storage_3[0] / 2.570000e+02f));
  tile_storage_7_worker_stripe[3] = (tile_storage_0_worker_stripe[3] - (tile_storage_3[0] / 2.570000e+02f));
  if ((((int)threadIdx) % 64) == 0) {
    tile_storage_7_worker_stripe[4] = (tile_storage_0_worker_stripe[4] - (tile_storage_3[0] / 2.570000e+02f));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[0] * tile_storage_7_worker_stripe[0]));
  tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[1] * tile_storage_7_worker_stripe[1]));
  tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[2] * tile_storage_7_worker_stripe[2]));
  tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[3] * tile_storage_7_worker_stripe[3]));
  if ((((int)threadIdx) % 64) == 0) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[4] * tile_storage_7_worker_stripe[4]));
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 2)) {
    condval_1 = parallel_0_subgroup_partials_1[(((((int)threadIdx) >> 6) * 2) + (((int)threadIdx) & 31))];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  arg2_ptr[(((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4))] = (((tile_storage_7_worker_stripe[0] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) & 63) * 4)]) + arg1_ptr[(((((int)threadIdx) & 63) * 4) + 257)]);
  arg2_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 1)] = (((tile_storage_7_worker_stripe[1] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((((int)threadIdx) & 63) * 4) + 1)]) + arg1_ptr[(((((int)threadIdx) & 63) * 4) + 258)]);
  arg2_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 2)] = (((tile_storage_7_worker_stripe[2] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((((int)threadIdx) & 63) * 4) + 2)]) + arg1_ptr[(((((int)threadIdx) & 63) * 4) + 259)]);
  arg2_ptr[((((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + ((((int)threadIdx) & 63) * 4)) + 3)] = (((tile_storage_7_worker_stripe[3] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((((int)threadIdx) & 63) * 4) + 3)]) + arg1_ptr[(((((int)threadIdx) & 63) * 4) + 260)]);
  if ((((int)threadIdx) % 64) == 0) {
    arg2_ptr[(((((int)blockIdx) * 1028) + ((((int)threadIdx) >> 6) * 257)) + 256)] = (((tile_storage_7_worker_stripe[4] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[256]) + arg1_ptr[513]);
  }
}



// Function: benchmark_rmsnorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_rmsnorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float parallel_0_subgroup_partials_0[26];
  thread float tile_storage_0_worker_stripe[20];
  int cse_v1 = (((int)blockIdx) * 16384);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int tile_i_1_subgroup_chunk = 0; tile_i_1_subgroup_chunk < 4; ++tile_i_1_subgroup_chunk) {
    int cse_v3 = (tile_i_1_subgroup_chunk * 4);
    int cse_v13 = (((((int)blockIdx) * 16384) + (tile_i_1_subgroup_chunk * 3328)) + (((int)threadIdx) * 4));
    tile_storage_0_worker_stripe[(tile_i_1_subgroup_chunk * 4)] = arg0_ptr[(((((int)blockIdx) * 16384) + (tile_i_1_subgroup_chunk * 3328)) + (((int)threadIdx) * 4))];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 1)] = arg0_ptr[((((((int)blockIdx) * 16384) + (tile_i_1_subgroup_chunk * 3328)) + (((int)threadIdx) * 4)) + 1)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 2)] = arg0_ptr[((((((int)blockIdx) * 16384) + (tile_i_1_subgroup_chunk * 3328)) + (((int)threadIdx) * 4)) + 2)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 3)] = arg0_ptr[((((((int)blockIdx) * 16384) + (tile_i_1_subgroup_chunk * 3328)) + (((int)threadIdx) * 4)) + 3)];
  }
  int cse_v8 = ((((int)blockIdx) * 16384) + (((int)threadIdx) * 4));
  int cse_v14 = (((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13312);
  int cse_v15 = (((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13313);
  int cse_v16 = (((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13314);
  int cse_v17 = (((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13315);
  if (((int)threadIdx) < 768) {
    tile_storage_0_worker_stripe[16] = arg0_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13312)];
    tile_storage_0_worker_stripe[17] = arg0_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13313)];
    tile_storage_0_worker_stripe[18] = arg0_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13314)];
    tile_storage_0_worker_stripe[19] = arg0_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13315)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 4; ++n_6_0_subgroup_chunk) {
    int cse_v4 = (n_6_0_subgroup_chunk * 4);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[(n_6_0_subgroup_chunk * 4)] * tile_storage_0_worker_stripe[(n_6_0_subgroup_chunk * 4)]));
    int cse_v9 = ((n_6_0_subgroup_chunk * 4) + 1);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 1)] * tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 1)]));
    int cse_v10 = ((n_6_0_subgroup_chunk * 4) + 2);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 2)] * tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 2)]));
    int cse_v11 = ((n_6_0_subgroup_chunk * 4) + 3);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 3)] * tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 3)]));
  }
  if (((int)threadIdx) < 768) {
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[16] * tile_storage_0_worker_stripe[16]));
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[17] * tile_storage_0_worker_stripe[17]));
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[18] * tile_storage_0_worker_stripe[18]));
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[19] * tile_storage_0_worker_stripe[19]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v5 = (((int)threadIdx) & 31);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 26)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  for (int tile_i_10_subgroup_chunk = 0; tile_i_10_subgroup_chunk < 4; ++tile_i_10_subgroup_chunk) {
    int cse_v6 = (tile_i_10_subgroup_chunk * 4);
    int cse_v7 = (tile_i_10_subgroup_chunk * 3328);
    int cse_v12 = ((tile_i_10_subgroup_chunk * 3328) + (((int)threadIdx) * 4));
    int cse_v18 = (((((int)blockIdx) * 16384) + (tile_i_10_subgroup_chunk * 3328)) + (((int)threadIdx) * 4));
    arg2_ptr[(((((int)blockIdx) * 16384) + (tile_i_10_subgroup_chunk * 3328)) + (((int)threadIdx) * 4))] = ((tile_storage_0_worker_stripe[(tile_i_10_subgroup_chunk * 4)] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[((tile_i_10_subgroup_chunk * 3328) + (((int)threadIdx) * 4))]);
    arg2_ptr[((((((int)blockIdx) * 16384) + (tile_i_10_subgroup_chunk * 3328)) + (((int)threadIdx) * 4)) + 1)] = ((tile_storage_0_worker_stripe[((tile_i_10_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 3328) + (((int)threadIdx) * 4)) + 1)]);
    arg2_ptr[((((((int)blockIdx) * 16384) + (tile_i_10_subgroup_chunk * 3328)) + (((int)threadIdx) * 4)) + 2)] = ((tile_storage_0_worker_stripe[((tile_i_10_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 3328) + (((int)threadIdx) * 4)) + 2)]);
    arg2_ptr[((((((int)blockIdx) * 16384) + (tile_i_10_subgroup_chunk * 3328)) + (((int)threadIdx) * 4)) + 3)] = ((tile_storage_0_worker_stripe[((tile_i_10_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 3328) + (((int)threadIdx) * 4)) + 3)]);
  }
  if (((int)threadIdx) < 768) {
    arg2_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13312)] = ((tile_storage_0_worker_stripe[16] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 13312)]);
    arg2_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13313)] = ((tile_storage_0_worker_stripe[17] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 13313)]);
    arg2_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13314)] = ((tile_storage_0_worker_stripe[18] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 13314)]);
    arg2_ptr[(((((int)blockIdx) * 16384) + (((int)threadIdx) * 4)) + 13315)] = ((tile_storage_0_worker_stripe[19] / sqrt(((tile_storage_3[0] / 1.638400e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 13315)]);
  }
}



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
  threadgroup float parallel_0_subgroup_partials_0[16];
  threadgroup float parallel_0_subgroup_partials_1[16];
  thread float tile_storage_0_worker_stripe[8];
  int cse_v1 = (((int)threadIdx) >> 8);
  int cse_v2 = (((int)threadIdx) & 255);
  int cse_v5 = ((((int)blockIdx) * 2) + (((int)threadIdx) >> 8));
  int cse_v6 = ((((int)threadIdx) & 255) * 4);
  int cse_v9 = (min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537);
  int cse_v11 = ((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4));
  tile_storage_0_worker_stripe[0] = arg0_ptr[((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4))];
  tile_storage_0_worker_stripe[1] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 1)];
  tile_storage_0_worker_stripe[2] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 2)];
  tile_storage_0_worker_stripe[3] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 3)];
  if ((((int)threadIdx) & 255) < 128) {
    tile_storage_0_worker_stripe[4] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 1024)];
    tile_storage_0_worker_stripe[5] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 1025)];
    tile_storage_0_worker_stripe[6] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 1026)];
    tile_storage_0_worker_stripe[7] = arg0_ptr[(((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + ((((int)threadIdx) & 255) * 4)) + 1027)];
  }
  if ((((int)threadIdx) & 255) == 128) {
    tile_storage_0_worker_stripe[4] = arg0_ptr[((min(((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)), 36) * 1537) + 1536)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = -INFINITY;
  tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[0]);
  tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[1]);
  tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[2]);
  tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[3]);
  if ((((int)threadIdx) & 255) < 128) {
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[4]);
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[5]);
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[6]);
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[7]);
  }
  if ((((int)threadIdx) & 255) == 128) {
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[4]);
  }
  tile_storage_5[0] = simd_max(tile_storage_5[0]);
  int cse_v3 = (((int)threadIdx) & 31);
  int cse_v4 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int cse_v7 = (((((int)threadIdx) >> 8) * 8) + (((int)threadIdx) & 31));
  float condval;
  if (((((int)threadIdx) & 31) < 8)) {
    condval = parallel_0_subgroup_partials_0[(((((int)threadIdx) >> 8) * 8) + (((int)threadIdx) & 31))];
  } else {
    condval = -INFINITY;
  }
  tile_storage_5[0] = simd_max(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[8];
  tile_storage_7_worker_stripe[0] = exp((tile_storage_0_worker_stripe[0] - tile_storage_3[0]));
  tile_storage_7_worker_stripe[1] = exp((tile_storage_0_worker_stripe[1] - tile_storage_3[0]));
  tile_storage_7_worker_stripe[2] = exp((tile_storage_0_worker_stripe[2] - tile_storage_3[0]));
  tile_storage_7_worker_stripe[3] = exp((tile_storage_0_worker_stripe[3] - tile_storage_3[0]));
  if ((((int)threadIdx) & 255) < 128) {
    tile_storage_7_worker_stripe[4] = exp((tile_storage_0_worker_stripe[4] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[5] = exp((tile_storage_0_worker_stripe[5] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[6] = exp((tile_storage_0_worker_stripe[6] - tile_storage_3[0]));
    tile_storage_7_worker_stripe[7] = exp((tile_storage_0_worker_stripe[7] - tile_storage_3[0]));
  }
  if ((((int)threadIdx) & 255) == 128) {
    tile_storage_7_worker_stripe[4] = exp((tile_storage_0_worker_stripe[4] - tile_storage_3[0]));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[0]);
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[1]);
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[2]);
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[3]);
  if ((((int)threadIdx) & 255) < 128) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[4]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[5]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[6]);
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[7]);
  }
  if ((((int)threadIdx) & 255) == 128) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[4]);
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
  int cse_v8 = ((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537));
  int cse_v10 = (((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4));
  if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
    arg1_ptr[(((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4))] = (tile_storage_7_worker_stripe[0] / tile_storage_10[0]);
  }
  if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
    arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 1)] = (tile_storage_7_worker_stripe[1] / tile_storage_10[0]);
  }
  if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
    arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 2)] = (tile_storage_7_worker_stripe[2] / tile_storage_10[0]);
  }
  if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
    arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 3)] = (tile_storage_7_worker_stripe[3] / tile_storage_10[0]);
  }
  if ((((int)threadIdx) & 255) < 128) {
    if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
      arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 1024)] = (tile_storage_7_worker_stripe[4] / tile_storage_10[0]);
    }
    if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
      arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 1025)] = (tile_storage_7_worker_stripe[5] / tile_storage_10[0]);
    }
    if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
      arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 1026)] = (tile_storage_7_worker_stripe[6] / tile_storage_10[0]);
    }
    if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
      arg1_ptr[((((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + ((((int)threadIdx) & 255) * 4)) + 1027)] = (tile_storage_7_worker_stripe[7] / tile_storage_10[0]);
    }
  }
  if ((((int)threadIdx) & 255) == 128) {
    if (((((int)blockIdx) * 2) + (((int)threadIdx) >> 8)) < 37) {
      arg1_ptr[(((((int)blockIdx) * 3074) + ((((int)threadIdx) >> 8) * 1537)) + 1536)] = (tile_storage_7_worker_stripe[4] / tile_storage_10[0]);
    }
  }
}



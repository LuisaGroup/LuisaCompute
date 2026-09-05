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
  threadgroup float parallel_0_subgroup_partials_0[13];
  threadgroup float parallel_0_subgroup_partials_1[13];
  thread float tile_storage_0_worker_stripe[4];
  int cse_v3 = ((((int)blockIdx) * 1537) + (((int)threadIdx) * 4));
  if (((int)threadIdx) < 385) {
    tile_storage_0_worker_stripe[0] = arg0_ptr[((((int)blockIdx) * 1537) + (((int)threadIdx) * 4))];
  }
  int cse_v4 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1);
  if (((int)threadIdx) < 384) {
    tile_storage_0_worker_stripe[1] = arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1)];
  }
  int cse_v5 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2);
  if (((int)threadIdx) < 384) {
    tile_storage_0_worker_stripe[2] = arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2)];
  }
  int cse_v6 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3);
  if (((int)threadIdx) < 384) {
    tile_storage_0_worker_stripe[3] = arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = -INFINITY;
  if (((int)threadIdx) < 385) {
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[0]);
  }
  if (((int)threadIdx) < 384) {
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[1]);
  }
  if (((int)threadIdx) < 384) {
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[2]);
  }
  if (((int)threadIdx) < 384) {
    tile_storage_5[0] = max(tile_storage_5[0], tile_storage_0_worker_stripe[3]);
  }
  tile_storage_5[0] = simd_max(tile_storage_5[0]);
  int cse_v1 = (((int)threadIdx) & 31);
  int cse_v2 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 13)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = -INFINITY;
  }
  tile_storage_5[0] = simd_max(condval);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[4];
  if (((int)threadIdx) < 385) {
    tile_storage_7_worker_stripe[0] = exp((tile_storage_0_worker_stripe[0] - tile_storage_3[0]));
  }
  if (((int)threadIdx) < 384) {
    tile_storage_7_worker_stripe[1] = exp((tile_storage_0_worker_stripe[1] - tile_storage_3[0]));
  }
  if (((int)threadIdx) < 384) {
    tile_storage_7_worker_stripe[2] = exp((tile_storage_0_worker_stripe[2] - tile_storage_3[0]));
  }
  if (((int)threadIdx) < 384) {
    tile_storage_7_worker_stripe[3] = exp((tile_storage_0_worker_stripe[3] - tile_storage_3[0]));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  if (((int)threadIdx) < 385) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[0]);
  }
  if (((int)threadIdx) < 384) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[1]);
  }
  if (((int)threadIdx) < 384) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[2]);
  }
  if (((int)threadIdx) < 384) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[3]);
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_12[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 13)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_12[0] = simd_sum(condval_1);
  tile_storage_10[0] = tile_storage_12[0];
  if (((int)threadIdx) < 385) {
    arg1_ptr[((((int)blockIdx) * 1537) + (((int)threadIdx) * 4))] = (tile_storage_7_worker_stripe[0] / tile_storage_10[0]);
  }
  if (((int)threadIdx) < 384) {
    arg1_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1)] = (tile_storage_7_worker_stripe[1] / tile_storage_10[0]);
  }
  if (((int)threadIdx) < 384) {
    arg1_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2)] = (tile_storage_7_worker_stripe[2] / tile_storage_10[0]);
  }
  if (((int)threadIdx) < 384) {
    arg1_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3)] = (tile_storage_7_worker_stripe[3] / tile_storage_10[0]);
  }
}



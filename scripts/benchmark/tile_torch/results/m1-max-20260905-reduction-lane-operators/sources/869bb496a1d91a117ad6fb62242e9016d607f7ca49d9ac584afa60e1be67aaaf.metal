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
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = -INFINITY;
  int cse_v1 = (((int)threadIdx) * 4);
  tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[(((int)threadIdx) * 4)]);
  int cse_v2 = ((((int)threadIdx) * 4) + 1);
  tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((int)threadIdx) * 4) + 1)]);
  int cse_v3 = ((((int)threadIdx) * 4) + 2);
  tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((int)threadIdx) * 4) + 2)]);
  int cse_v4 = ((((int)threadIdx) * 4) + 3);
  if (((int)threadIdx) < 31) {
    tile_storage_5[0] = max(tile_storage_5[0], arg0_ptr[((((int)threadIdx) * 4) + 3)]);
  }
  tile_storage_5[0] = simd_max(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[4];
  tile_storage_7_worker_stripe[0] = exp((arg0_ptr[(((int)threadIdx) * 4)] - tile_storage_3[0]));
  tile_storage_7_worker_stripe[1] = exp((arg0_ptr[((((int)threadIdx) * 4) + 1)] - tile_storage_3[0]));
  tile_storage_7_worker_stripe[2] = exp((arg0_ptr[((((int)threadIdx) * 4) + 2)] - tile_storage_3[0]));
  if (((int)threadIdx) < 31) {
    tile_storage_7_worker_stripe[3] = exp((arg0_ptr[((((int)threadIdx) * 4) + 3)] - tile_storage_3[0]));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[0]);
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[1]);
  tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[2]);
  if (((int)threadIdx) < 31) {
    tile_storage_12[0] = (tile_storage_12[0] + tile_storage_7_worker_stripe[3]);
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  tile_storage_10[0] = tile_storage_12[0];
  arg1_ptr[(((int)threadIdx) * 4)] = (tile_storage_7_worker_stripe[0] / tile_storage_10[0]);
  arg1_ptr[((((int)threadIdx) * 4) + 1)] = (tile_storage_7_worker_stripe[1] / tile_storage_10[0]);
  arg1_ptr[((((int)threadIdx) * 4) + 2)] = (tile_storage_7_worker_stripe[2] / tile_storage_10[0]);
  if (((int)threadIdx) < 31) {
    arg1_ptr[((((int)threadIdx) * 4) + 3)] = (tile_storage_7_worker_stripe[3] / tile_storage_10[0]);
  }
}



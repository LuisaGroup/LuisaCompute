// Function: benchmark_residual_layernorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_residual_layernorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float parallel_0_subgroup_partials_0[4];
  threadgroup float parallel_0_subgroup_partials_1[4];
  thread float tile_storage_6_worker_stripe[1];
  if (((int)threadIdx) < 127) {
    tile_storage_6_worker_stripe[0] = (arg0_ptr[((int)threadIdx)] + arg1_ptr[((int)threadIdx)]);
  }
  thread float tile_storage_9[1];
  thread float tile_storage_11[1];
  tile_storage_11[0] = 0.000000e+00f;
  if (((int)threadIdx) < 127) {
    tile_storage_11[0] = (tile_storage_11[0] + tile_storage_6_worker_stripe[0]);
  }
  tile_storage_11[0] = simd_sum(tile_storage_11[0]);
  int cse_v1 = (((int)threadIdx) & 31);
  int cse_v2 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_11[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 4)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_11[0] = simd_sum(condval);
  tile_storage_9[0] = tile_storage_11[0];
  thread float tile_storage_13_worker_stripe[1];
  if (((int)threadIdx) < 127) {
    tile_storage_13_worker_stripe[0] = (tile_storage_6_worker_stripe[0] - (tile_storage_9[0] / 1.270000e+02f));
  }
  thread float tile_storage_16[1];
  thread float tile_storage_18[1];
  tile_storage_18[0] = 0.000000e+00f;
  if (((int)threadIdx) < 127) {
    tile_storage_18[0] = (tile_storage_18[0] + (tile_storage_13_worker_stripe[0] * tile_storage_13_worker_stripe[0]));
  }
  tile_storage_18[0] = simd_sum(tile_storage_18[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_18[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 4)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_18[0] = simd_sum(condval_1);
  tile_storage_16[0] = tile_storage_18[0];
  if (((int)threadIdx) < 127) {
    arg2_ptr[((int)threadIdx)] = (tile_storage_13_worker_stripe[0] / sqrt(((tile_storage_16[0] / 1.270000e+02f) + 1.000000e-05f)));
  }
}



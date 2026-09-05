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
  threadgroup float parallel_0_subgroup_partials_0[8];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)blockIdx) * 257);
  int cse_v3 = ((((int)blockIdx) * 257) + ((int)threadIdx));
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)blockIdx) * 257) + ((int)threadIdx))] * arg0_ptr[((((int)blockIdx) * 257) + ((int)threadIdx))]));
  int cse_v4 = ((((int)blockIdx) * 257) + 256);
  if (((int)threadIdx) < 1) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)blockIdx) * 257) + 256)] * arg0_ptr[((((int)blockIdx) * 257) + 256)]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v2 = (((int)threadIdx) & 31);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 8)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  arg2_ptr[((((int)blockIdx) * 257) + ((int)threadIdx))] = ((arg0_ptr[((((int)blockIdx) * 257) + ((int)threadIdx))] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[((int)threadIdx)]);
  if (((int)threadIdx) < 1) {
    arg2_ptr[((((int)blockIdx) * 257) + 256)] = ((arg0_ptr[((((int)blockIdx) * 257) + 256)] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[256]);
  }
}



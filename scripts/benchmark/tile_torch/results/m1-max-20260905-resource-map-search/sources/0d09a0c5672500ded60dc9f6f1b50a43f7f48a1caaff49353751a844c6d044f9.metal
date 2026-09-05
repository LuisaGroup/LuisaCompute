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
  threadgroup float parallel_0_subgroup_partials_0[4];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)threadIdx) * 4);
  int cse_v3 = ((((int)blockIdx) * 769) + (((int)threadIdx) * 4));
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)blockIdx) * 769) + (((int)threadIdx) * 4))] * arg0_ptr[((((int)blockIdx) * 769) + (((int)threadIdx) * 4))]));
  int cse_v4 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 1);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 1)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 1)]));
  int cse_v5 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 2);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 2)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 2)]));
  int cse_v6 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 3);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 3)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 3)]));
  int cse_v7 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 512);
  if (((int)threadIdx) < 65) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 512)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 512)]));
  }
  int cse_v8 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 513);
  if (((int)threadIdx) < 64) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 513)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 513)]));
  }
  int cse_v9 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 514);
  if (((int)threadIdx) < 64) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 514)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 514)]));
  }
  int cse_v10 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 515);
  if (((int)threadIdx) < 64) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 515)] * arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 515)]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v2 = (((int)threadIdx) & 31);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 4)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  arg2_ptr[((((int)blockIdx) * 769) + (((int)threadIdx) * 4))] = ((arg0_ptr[((((int)blockIdx) * 769) + (((int)threadIdx) * 4))] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[(((int)threadIdx) * 4)]);
  arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 1)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 1)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 1)]);
  arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 2)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 2)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 2)]);
  arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 3)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 3)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 3)]);
  if (((int)threadIdx) < 65) {
    arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 512)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 512)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 512)]);
  }
  if (((int)threadIdx) < 64) {
    arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 513)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 513)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 513)]);
  }
  if (((int)threadIdx) < 64) {
    arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 514)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 514)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 514)]);
  }
  if (((int)threadIdx) < 64) {
    arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 515)] = ((arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 515)] / sqrt(((tile_storage_3[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 515)]);
  }
}



// Function: benchmark_add_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_add_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_0[256];
  int cse_v1 = ((((int)threadIdx) & 1) * 256);
  int cse_v2 = (((((int)threadIdx) >> 1) * 257) + ((((int)threadIdx) & 1) * 256));
  for (int tile_i_2 = 0; tile_i_2 < 256; ++tile_i_2) {
    float condval;
    if (((((((int)threadIdx) & 1) * 256) + tile_i_2) < 257)) {
      condval = arg0_ptr[((((((int)threadIdx) >> 1) * 257) + ((((int)threadIdx) & 1) * 256)) + tile_i_2)];
    } else {
      condval = 0.000000e+00f;
    }
    tile_storage_0[tile_i_2] = condval;
  }
  thread float tile_storage_3[256];
  for (int tile_i_5 = 0; tile_i_5 < 256; ++tile_i_5) {
    float condval_1;
    if (((((((int)threadIdx) & 1) * 256) + tile_i_5) < 257)) {
      condval_1 = arg1_ptr[((((((int)threadIdx) >> 1) * 257) + ((((int)threadIdx) & 1) * 256)) + tile_i_5)];
    } else {
      condval_1 = 0.000000e+00f;
    }
    tile_storage_3[tile_i_5] = condval_1;
  }
  for (int tile_i_7 = 0; tile_i_7 < 256; ++tile_i_7) {
    if ((((((int)threadIdx) & 1) * 256) + tile_i_7) < 257) {
      arg2_ptr[((((((int)threadIdx) >> 1) * 257) + ((((int)threadIdx) & 1) * 256)) + tile_i_7)] = (tile_storage_0[tile_i_7] + tile_storage_3[tile_i_7]);
    }
  }
}



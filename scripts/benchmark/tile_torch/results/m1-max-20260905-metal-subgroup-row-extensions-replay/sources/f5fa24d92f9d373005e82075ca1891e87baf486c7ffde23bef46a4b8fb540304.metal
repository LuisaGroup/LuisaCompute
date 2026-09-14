// Function: benchmark_cross_entropy_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_cross_entropy_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device long* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_0[127];
  for (int tile_i_2 = 0; tile_i_2 < 127; ++tile_i_2) {
    tile_storage_0[tile_i_2] = arg0_ptr[tile_i_2];
  }
  thread long tile_storage_3[1];
  tile_storage_3[0] = arg1_ptr[0];
  thread float tile_storage_5[1];
  thread float tile_storage_7[1];
  tile_storage_7[0] = -INFINITY;
  for (int n_6_0 = 0; n_6_0 < 127; ++n_6_0) {
    thread float tile_storage_8[1];
    tile_storage_8[0] = max(tile_storage_7[0], tile_storage_0[n_6_0]);
    tile_storage_7[0] = tile_storage_8[0];
  }
  tile_storage_5[0] = tile_storage_7[0];
  thread float tile_storage_9[1];
  thread float tile_storage_11[1];
  tile_storage_11[0] = 0.000000e+00f;
  for (int n_20_0 = 0; n_20_0 < 127; ++n_20_0) {
    thread float tile_storage_12[1];
    tile_storage_12[0] = (tile_storage_11[0] + exp((tile_storage_0[n_20_0] - tile_storage_5[0])));
    tile_storage_11[0] = tile_storage_12[0];
  }
  tile_storage_9[0] = tile_storage_11[0];
  thread float tile_storage_13[1];
  float condval;
  if ((((long)0 <= tile_storage_3[0]) && (tile_storage_3[0] < (long)127))) {
    condval = tile_storage_0[tile_storage_3[0]];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_13[0] = condval;
  arg2_ptr[0] = ((log(tile_storage_9[0]) + tile_storage_5[0]) - tile_storage_13[0]);
}



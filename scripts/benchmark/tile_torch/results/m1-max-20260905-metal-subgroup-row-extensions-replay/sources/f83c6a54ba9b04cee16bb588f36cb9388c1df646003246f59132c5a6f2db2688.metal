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
  thread float tile_storage_0[1024];
  int cse_v1 = (((int)threadIdx) * 1024);
  for (int tile_i_2 = 0; tile_i_2 < 1024; ++tile_i_2) {
    tile_storage_0[tile_i_2] = arg0_ptr[((((int)threadIdx) * 1024) + tile_i_2)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_5_0 = 0; n_5_0 < 1024; ++n_5_0) {
    thread float tile_storage_6[1];
    tile_storage_6[0] = (tile_storage_5[0] + tile_storage_0[n_5_0]);
    tile_storage_5[0] = tile_storage_6[0];
  }
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7[1];
  thread float tile_storage_9[1];
  tile_storage_9[0] = 0.000000e+00f;
  for (int n_20_0 = 0; n_20_0 < 1024; ++n_20_0) {
    thread float tile_storage_10[1];
    tile_storage_10[0] = (tile_storage_9[0] + ((tile_storage_0[n_20_0] - (tile_storage_3[0] / 1.024000e+03f)) * (tile_storage_0[n_20_0] - (tile_storage_3[0] / 1.024000e+03f))));
    tile_storage_9[0] = tile_storage_10[0];
  }
  tile_storage_7[0] = tile_storage_9[0];
  thread float tile_storage_11[1024];
  for (int tile_i_13 = 0; tile_i_13 < 1024; ++tile_i_13) {
    tile_storage_11[tile_i_13] = arg1_ptr[tile_i_13];
  }
  thread float tile_storage_14[1024];
  for (int tile_i_16 = 0; tile_i_16 < 1024; ++tile_i_16) {
    tile_storage_14[tile_i_16] = arg1_ptr[(tile_i_16 + 1024)];
  }
  for (int tile_i_18 = 0; tile_i_18 < 1024; ++tile_i_18) {
    arg2_ptr[((((int)threadIdx) * 1024) + tile_i_18)] = ((((tile_storage_0[tile_i_18] - (tile_storage_3[0] / 1.024000e+03f)) / sqrt(((tile_storage_7[0] / 1.024000e+03f) + 1.000000e-05f))) * tile_storage_11[tile_i_18]) + tile_storage_14[tile_i_18]);
  }
}



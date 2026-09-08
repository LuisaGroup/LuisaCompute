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
  thread float tile_storage_6[4096];
  int cse_v1 = (((int)threadIdx) * 4096);
  for (int tile_i_8 = 0; tile_i_8 < 4096; ++tile_i_8) {
    int cse_v2 = ((((int)threadIdx) * 4096) + tile_i_8);
    tile_storage_6[tile_i_8] = (arg0_ptr[((((int)threadIdx) * 4096) + tile_i_8)] + arg1_ptr[((((int)threadIdx) * 4096) + tile_i_8)]);
  }
  thread float tile_storage_9[1];
  thread float tile_storage_11[1];
  tile_storage_11[0] = 0.000000e+00f;
  for (int n_7_0 = 0; n_7_0 < 4096; ++n_7_0) {
    thread float tile_storage_12[1];
    tile_storage_12[0] = (tile_storage_11[0] + tile_storage_6[n_7_0]);
    tile_storage_11[0] = tile_storage_12[0];
  }
  tile_storage_9[0] = tile_storage_11[0];
  thread float tile_storage_13[4096];
  for (int tile_i_15 = 0; tile_i_15 < 4096; ++tile_i_15) {
    tile_storage_13[tile_i_15] = (tile_storage_6[tile_i_15] - (tile_storage_9[0] / 4.096000e+03f));
  }
  thread float tile_storage_16[1];
  thread float tile_storage_18[1];
  tile_storage_18[0] = 0.000000e+00f;
  for (int n_22_0 = 0; n_22_0 < 4096; ++n_22_0) {
    thread float tile_storage_19[1];
    tile_storage_19[0] = (tile_storage_18[0] + (tile_storage_13[n_22_0] * tile_storage_13[n_22_0]));
    tile_storage_18[0] = tile_storage_19[0];
  }
  tile_storage_16[0] = tile_storage_18[0];
  for (int tile_i_21 = 0; tile_i_21 < 4096; ++tile_i_21) {
    arg2_ptr[((((int)threadIdx) * 4096) + tile_i_21)] = (tile_storage_13[tile_i_21] / sqrt(((tile_storage_16[0] / 4.096000e+03f) + 1.000000e-05f)));
  }
}



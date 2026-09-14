// Function: benchmark_gelu_add_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_gelu_add_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_6[256];
  int cse_v1 = ((((int)blockIdx) * 65536) + (((int)threadIdx) * 256));
  for (int tile_i_8 = 0; tile_i_8 < 256; ++tile_i_8) {
    int cse_v2 = (((((int)blockIdx) * 65536) + (((int)threadIdx) * 256)) + tile_i_8);
    tile_storage_6[tile_i_8] = (arg0_ptr[(((((int)blockIdx) * 65536) + (((int)threadIdx) * 256)) + tile_i_8)] + arg1_ptr[(((((int)blockIdx) * 65536) + (((int)threadIdx) * 256)) + tile_i_8)]);
  }
  for (int tile_i_10 = 0; tile_i_10 < 256; ++tile_i_10) {
    arg2_ptr[(((((int)blockIdx) * 65536) + (((int)threadIdx) * 256)) + tile_i_10)] = ((5.000000e-01f * tile_storage_6[tile_i_10]) * (1.000000e+00f + select(((exp((2.000000e+00f * (7.978846e-01f * (tile_storage_6[tile_i_10] + (((4.471500e-02f * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]))))) - 1.000000e+00f) / (exp((2.000000e+00f * (7.978846e-01f * (tile_storage_6[tile_i_10] + (((4.471500e-02f * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]))))) + 1.000000e+00f)), ((1.000000e+00f - exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_6[tile_i_10] + (((4.471500e-02f * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10])))))) / (1.000000e+00f + exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_6[tile_i_10] + (((4.471500e-02f * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]))))))), ((7.978846e-01f * (tile_storage_6[tile_i_10] + (((4.471500e-02f * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]) * tile_storage_6[tile_i_10]))) >= 0.000000e+00f))));
  }
}



// Function: benchmark_activation_pair_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_activation_pair_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  int cse_v1 = ((((int)blockIdx) * 256) + ((int)threadIdx));
  if (((((int)blockIdx) * 256) + ((int)threadIdx)) < 259) {
    thread float tile_storage_0[256];
    int cse_v2 = ((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256);
    int cse_v3 = (((((((int)blockIdx) * 256) + ((int)threadIdx)) / 7) * 1537) + ((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256));
    for (int tile_i_2 = 0; tile_i_2 < 256; ++tile_i_2) {
      float condval;
      if (((((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256) + tile_i_2) < 1537)) {
        condval = arg0_ptr[((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 7) * 1537) + ((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256)) + tile_i_2)];
      } else {
        condval = 0.000000e+00f;
      }
      tile_storage_0[tile_i_2] = condval;
    }
    thread float tile_storage_3[256];
    for (int tile_i_5 = 0; tile_i_5 < 256; ++tile_i_5) {
      tile_storage_3[tile_i_5] = select(((exp((2.000000e+00f * (7.978846e-01f * (tile_storage_0[tile_i_5] + (4.471500e-02f * ((tile_storage_0[tile_i_5] * tile_storage_0[tile_i_5]) * tile_storage_0[tile_i_5])))))) - 1.000000e+00f) / (exp((2.000000e+00f * (7.978846e-01f * (tile_storage_0[tile_i_5] + (4.471500e-02f * ((tile_storage_0[tile_i_5] * tile_storage_0[tile_i_5]) * tile_storage_0[tile_i_5])))))) + 1.000000e+00f)), ((1.000000e+00f - exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_0[tile_i_5] + (4.471500e-02f * ((tile_storage_0[tile_i_5] * tile_storage_0[tile_i_5]) * tile_storage_0[tile_i_5]))))))) / (1.000000e+00f + exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_0[tile_i_5] + (4.471500e-02f * ((tile_storage_0[tile_i_5] * tile_storage_0[tile_i_5]) * tile_storage_0[tile_i_5])))))))), ((7.978846e-01f * (tile_storage_0[tile_i_5] + (4.471500e-02f * ((tile_storage_0[tile_i_5] * tile_storage_0[tile_i_5]) * tile_storage_0[tile_i_5])))) >= 0.000000e+00f));
    }
    for (int tile_i_7 = 0; tile_i_7 < 256; ++tile_i_7) {
      if ((((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256) + tile_i_7) < 1537) {
        arg1_ptr[((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 7) * 1537) + ((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256)) + tile_i_7)] = ((5.000000e-01f * tile_storage_0[tile_i_7]) * (1.000000e+00f + tile_storage_3[tile_i_7]));
      }
    }
    for (int tile_i_9 = 0; tile_i_9 < 256; ++tile_i_9) {
      if ((((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256) + tile_i_9) < 1537) {
        arg2_ptr[((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 7) * 1537) + ((((((int)blockIdx) * 256) + ((int)threadIdx)) % 7) * 256)) + tile_i_9)] = ((5.000000e-01f * (1.000000e+00f + tile_storage_3[tile_i_9])) + ((((5.000000e-01f * tile_storage_0[tile_i_9]) * (1.000000e+00f - (tile_storage_3[tile_i_9] * tile_storage_3[tile_i_9]))) * 7.978846e-01f) * (1.000000e+00f + ((1.341450e-01f * tile_storage_0[tile_i_9]) * tile_storage_0[tile_i_9]))));
      }
    }
  }
}



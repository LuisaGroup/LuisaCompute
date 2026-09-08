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
  int cse_v1 = ((((int)blockIdx) & 1) * 256);
  int cse_v2 = (((((int)blockIdx) & 1) * 256) + ((int)threadIdx));
  int cse_v3 = ((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx));
  float condval;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval = 0.000000e+00f;
  }
  float condval_1;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_1 = arg1_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_1 = 0.000000e+00f;
  }
  float tile_storage_6_element = (condval + condval_1);
  if ((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257) {
    arg2_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))] = ((5.000000e-01f * tile_storage_6_element) * (1.000000e+00f + select(((exp((2.000000e+00f * (7.978846e-01f * (tile_storage_6_element + (((4.471500e-02f * tile_storage_6_element) * tile_storage_6_element) * tile_storage_6_element))))) - 1.000000e+00f) / (exp((2.000000e+00f * (7.978846e-01f * (tile_storage_6_element + (((4.471500e-02f * tile_storage_6_element) * tile_storage_6_element) * tile_storage_6_element))))) + 1.000000e+00f)), ((1.000000e+00f - exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_6_element + (((4.471500e-02f * tile_storage_6_element) * tile_storage_6_element) * tile_storage_6_element)))))) / (1.000000e+00f + exp((-2.000000e+00f * (7.978846e-01f * (tile_storage_6_element + (((4.471500e-02f * tile_storage_6_element) * tile_storage_6_element) * tile_storage_6_element))))))), ((7.978846e-01f * (tile_storage_6_element + (((4.471500e-02f * tile_storage_6_element) * tile_storage_6_element) * tile_storage_6_element))) >= 0.000000e+00f))));
  }
}



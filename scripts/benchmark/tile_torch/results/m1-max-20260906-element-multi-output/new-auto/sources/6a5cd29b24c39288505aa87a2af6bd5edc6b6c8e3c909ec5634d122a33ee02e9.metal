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
  int cse_v2 = ((((int)blockIdx) * 256) + ((int)threadIdx));
  float tile_storage_3_element = select(((exp((2.000000e+00f * (7.978846e-01f * (arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] + (4.471500e-02f * ((arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))])))))) - 1.000000e+00f) / (exp((2.000000e+00f * (7.978846e-01f * (arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] + (4.471500e-02f * ((arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))])))))) + 1.000000e+00f)), ((1.000000e+00f - exp((-2.000000e+00f * (7.978846e-01f * (arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] + (4.471500e-02f * ((arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]))))))) / (1.000000e+00f + exp((-2.000000e+00f * (7.978846e-01f * (arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] + (4.471500e-02f * ((arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))])))))))), ((7.978846e-01f * (arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] + (4.471500e-02f * ((arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))])))) >= 0.000000e+00f));
  float cse_v1 = (1.000000e+00f + tile_storage_3_element);
  arg1_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] = ((5.000000e-01f * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * (1.000000e+00f + tile_storage_3_element));
  arg2_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))] = ((5.000000e-01f * (1.000000e+00f + tile_storage_3_element)) + ((((5.000000e-01f * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * (1.000000e+00f - (tile_storage_3_element * tile_storage_3_element))) * 7.978846e-01f) * (1.000000e+00f + ((1.341450e-01f * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) * arg0_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]))));
}



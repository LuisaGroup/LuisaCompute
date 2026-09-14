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
  float condval;
  if ((((int)threadIdx) < 127)) {
    condval = arg0_ptr[((int)threadIdx)];
  } else {
    condval = 0.000000e+00f;
  }
  float tile_storage_3_element = (1.000000e+00f / (1.000000e+00f + exp((0.000000e+00f - condval))));
  if (((int)threadIdx) < 127) {
    arg1_ptr[((int)threadIdx)] = tile_storage_3_element;
  }
  if (((int)threadIdx) < 127) {
    arg2_ptr[((int)threadIdx)] = (tile_storage_3_element * (1.000000e+00f - tile_storage_3_element));
  }
}



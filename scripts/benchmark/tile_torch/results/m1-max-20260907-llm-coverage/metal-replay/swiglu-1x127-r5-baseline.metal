// Function: llm_rows_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void llm_rows_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg3_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  arg3_ptr[((int)threadIdx)] = ((arg0_ptr[((int)threadIdx)] / (1.000000e+00f + exp((0.000000e+00f - arg0_ptr[((int)threadIdx)])))) * arg1_ptr[((int)threadIdx)]);
}



// Function: llm_rows_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void llm_rows_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  device float* arg3_ptr [[ buffer(3) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  int cse_v1 = ((((int)blockIdx) * 256) + ((int)threadIdx));
  if (((((int)blockIdx) * 256) + ((int)threadIdx)) < 28453) {
    int cse_v2 = (((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769));
    int cse_v3 = ((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769)) + 769);
    arg3_ptr[(((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769))] = ((arg0_ptr[(((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769))] * arg1_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) - (arg0_ptr[((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769)) + 769)] * arg2_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]));
    arg3_ptr[((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769)) + 769)] = ((arg0_ptr[(((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769))] * arg2_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]) + (arg0_ptr[((((((((int)blockIdx) * 256) + ((int)threadIdx)) / 769) * 1538) + (((((int)blockIdx) * 256) + ((int)threadIdx)) % 769)) + 769)] * arg1_ptr[((((int)blockIdx) * 256) + ((int)threadIdx))]));
  }
}



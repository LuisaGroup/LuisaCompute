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
  thread float tile_storage_0[769];
  int cse_v1 = (((int)threadIdx) * 1538);
  for (int tile_i_2 = 0; tile_i_2 < 769; ++tile_i_2) {
    tile_storage_0[tile_i_2] = arg0_ptr[((((int)threadIdx) * 1538) + tile_i_2)];
  }
  thread float tile_storage_3[769];
  for (int tile_i_5 = 0; tile_i_5 < 769; ++tile_i_5) {
    tile_storage_3[tile_i_5] = arg0_ptr[(((((int)threadIdx) * 1538) + tile_i_5) + 769)];
  }
  thread float tile_storage_6[769];
  int cse_v2 = (((int)threadIdx) * 769);
  for (int tile_i_8 = 0; tile_i_8 < 769; ++tile_i_8) {
    tile_storage_6[tile_i_8] = arg1_ptr[((((int)threadIdx) * 769) + tile_i_8)];
  }
  thread float tile_storage_9[769];
  for (int tile_i_11 = 0; tile_i_11 < 769; ++tile_i_11) {
    tile_storage_9[tile_i_11] = arg2_ptr[((((int)threadIdx) * 769) + tile_i_11)];
  }
  for (int tile_i_13 = 0; tile_i_13 < 769; ++tile_i_13) {
    arg3_ptr[((((int)threadIdx) * 1538) + tile_i_13)] = ((tile_storage_0[tile_i_13] * tile_storage_6[tile_i_13]) - (tile_storage_3[tile_i_13] * tile_storage_9[tile_i_13]));
  }
  for (int tile_i_15 = 0; tile_i_15 < 769; ++tile_i_15) {
    arg3_ptr[(((((int)threadIdx) * 1538) + tile_i_15) + 769)] = ((tile_storage_0[tile_i_15] * tile_storage_9[tile_i_15]) + (tile_storage_3[tile_i_15] * tile_storage_6[tile_i_15]));
  }
}



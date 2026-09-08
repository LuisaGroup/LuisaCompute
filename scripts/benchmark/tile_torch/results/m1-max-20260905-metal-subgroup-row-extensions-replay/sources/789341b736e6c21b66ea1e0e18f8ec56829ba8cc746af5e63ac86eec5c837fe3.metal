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
  int cse_v1 = (((int)threadIdx) >> 5);
  if (((((int)blockIdx) * 8) + (((int)threadIdx) >> 5)) < 17) {
    thread float tile_storage_3[1];
    thread float tile_storage_5[1];
    tile_storage_5[0] = 0.000000e+00f;
    int cse_v3 = (((int)threadIdx) & 31);
    int cse_v7 = ((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257));
    for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 9; ++n_5_0_subgroup_chunk) {
      int cse_v2 = (n_5_0_subgroup_chunk * 32);
      if (((n_5_0_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
        tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 32)) + (((int)threadIdx) & 31))]);
      }
    }
    tile_storage_5[0] = simd_sum(tile_storage_5[0]);
    tile_storage_3[0] = tile_storage_5[0];
    thread float tile_storage_7[1];
    thread float tile_storage_9[1];
    tile_storage_9[0] = 0.000000e+00f;
    for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 9; ++n_20_0_subgroup_chunk) {
      int cse_v4 = (n_20_0_subgroup_chunk * 32);
      if (((n_20_0_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
        int cse_v8 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_20_0_subgroup_chunk * 32)) + (((int)threadIdx) & 31));
        tile_storage_9[0] = (tile_storage_9[0] + ((arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_20_0_subgroup_chunk * 32)) + (((int)threadIdx) & 31))] - (tile_storage_3[0] / 2.570000e+02f)) * (arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_20_0_subgroup_chunk * 32)) + (((int)threadIdx) & 31))] - (tile_storage_3[0] / 2.570000e+02f))));
      }
    }
    tile_storage_9[0] = simd_sum(tile_storage_9[0]);
    tile_storage_7[0] = tile_storage_9[0];
    for (int tile_i_17_subgroup_chunk = 0; tile_i_17_subgroup_chunk < 9; ++tile_i_17_subgroup_chunk) {
      int cse_v5 = (tile_i_17_subgroup_chunk * 32);
      int cse_v6 = ((tile_i_17_subgroup_chunk * 32) + (((int)threadIdx) & 31));
      if (((tile_i_17_subgroup_chunk * 32) + (((int)threadIdx) & 31)) < 257) {
        int cse_v9 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_17_subgroup_chunk * 32)) + (((int)threadIdx) & 31));
        arg2_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_17_subgroup_chunk * 32)) + (((int)threadIdx) & 31))] = ((((arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_17_subgroup_chunk * 32)) + (((int)threadIdx) & 31))] - (tile_storage_3[0] / 2.570000e+02f)) / sqrt(((tile_storage_7[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[((tile_i_17_subgroup_chunk * 32) + (((int)threadIdx) & 31))]) + arg1_ptr[(((tile_i_17_subgroup_chunk * 32) + (((int)threadIdx) & 31)) + 257)]);
      }
    }
  }
}



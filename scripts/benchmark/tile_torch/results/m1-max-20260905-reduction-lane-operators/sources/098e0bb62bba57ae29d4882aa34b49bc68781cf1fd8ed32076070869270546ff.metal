// Function: benchmark_sum_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_sum_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  int cse_v1 = (((int)blockIdx) * 8);
  int cse_v2 = (((int)threadIdx) >> 5);
  if (((((int)blockIdx) * 8) + (((int)threadIdx) >> 5)) < 17) {
    thread float tile_storage_3[1];
    thread float tile_storage_5[1];
    tile_storage_5[0] = 0.000000e+00f;
    int cse_v3 = (((int)threadIdx) & 31);
    int cse_v4 = ((((int)threadIdx) & 31) * 4);
    int cse_v5 = ((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257));
    for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 2; ++n_5_0_subgroup_chunk) {
      int cse_v7 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4));
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4))]);
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 1)]);
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 2)]);
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 3)]);
    }
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + ((((int)threadIdx) & 31) * 4)) + 256)]);
    }
    tile_storage_5[0] = simd_sum(tile_storage_5[0]);
    tile_storage_3[0] = tile_storage_5[0];
    if ((((int)threadIdx) & 31) < 1) {
      int cse_v6 = (((((int)blockIdx) * 8) + ((((int)threadIdx) & 31) * 4)) + (((int)threadIdx) >> 5));
      if ((((((int)blockIdx) * 8) + ((((int)threadIdx) & 31) * 4)) + (((int)threadIdx) >> 5)) < 17) {
        arg1_ptr[(((((int)blockIdx) * 8) + ((((int)threadIdx) & 31) * 4)) + (((int)threadIdx) >> 5))] = tile_storage_3[((((int)threadIdx) & 31) * 4)];
      }
    }
  }
}



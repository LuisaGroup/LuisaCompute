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
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)threadIdx) >> 5);
  int cse_v2 = (((int)threadIdx) & 31);
  for (int n_5_0_subgroup_chunk_pack = 0; n_5_0_subgroup_chunk_pack < 32; ++n_5_0_subgroup_chunk_pack) {
    int cse_v3 = ((((((int)blockIdx) * 16384) + ((((int)threadIdx) >> 5) * 4096)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31));
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 16384) + ((((int)threadIdx) >> 5) * 4096)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31))]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 16384) + ((((int)threadIdx) >> 5) * 4096)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 32)]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 16384) + ((((int)threadIdx) >> 5) * 4096)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 64)]);
    tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 16384) + ((((int)threadIdx) >> 5) * 4096)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 96)]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  if ((((int)threadIdx) & 31) < 1) {
    arg1_ptr[(((((int)blockIdx) * 4) + (((int)threadIdx) >> 5)) + (((int)threadIdx) & 31))] = tile_storage_3[(((int)threadIdx) & 31)];
  }
}



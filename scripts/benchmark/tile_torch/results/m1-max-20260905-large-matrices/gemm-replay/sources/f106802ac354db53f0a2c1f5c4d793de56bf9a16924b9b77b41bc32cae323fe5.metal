// Function: benchmark_gemm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_gemm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float tile_storage_0_shared[1024];
  for (int tile_i_1_chunk = 0; tile_i_1_chunk < 8; ++tile_i_1_chunk) {
    tile_storage_0_shared[((tile_i_1_chunk * 128) + ((int)threadIdx))] = 0.000000e+00f;
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  simdgroup_float8x8 tile_i_10_mma_c[4];
  int cse_v5 = ((((int)threadIdx) >> 6) * 512);
  int cse_v9 = (((((int)threadIdx) & 63) >> 5) * 16);
  int cse_v12 = (((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16));
  simdgroup_load(tile_i_10_mma_c[0], (&(tile_storage_0_shared[(((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
  int cse_v14 = ((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8);
  simdgroup_load(tile_i_10_mma_c[1], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
  int cse_v15 = ((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256);
  simdgroup_load(tile_i_10_mma_c[2], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256)])), 32, 0, (bool)0);
  int cse_v16 = ((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264);
  simdgroup_load(tile_i_10_mma_c[3], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264)])), 32, 0, (bool)0);
  int cse_v1 = (((int)blockIdx) / 129);
  int cse_v2 = (((int)threadIdx) >> 5);
  int cse_v4 = (((int)threadIdx) & 31);
  int cse_v6 = ((((int)blockIdx) / 129) * 32);
  int cse_v7 = ((((int)blockIdx) % 129) * 32);
  int cse_v8 = ((((int)threadIdx) >> 5) * 4097);
  int cse_v10 = (((((int)blockIdx) % 129) * 32) + (((int)threadIdx) & 31));
  for (int pipeline_6 = 0; pipeline_6 < 33; ++pipeline_6) {
    threadgroup float tile_storage_3_shared[1024];
    threadgroup float tile_storage_6_shared[1024];
    int cse_v3 = (pipeline_6 * 32);
    for (int tile_i_4_chunk = 0; tile_i_4_chunk < 8; ++tile_i_4_chunk) {
      float condval;
      if (((((((((int)blockIdx) / 129) * 32) + (tile_i_4_chunk * 4)) + (((int)threadIdx) >> 5)) < 2049) && (((pipeline_6 * 32) + (((int)threadIdx) & 31)) < 1025))) {
        condval = arg0_ptr[((((((((int)blockIdx) / 129) * 32800) + (tile_i_4_chunk * 4100)) + ((((int)threadIdx) >> 5) * 1025)) + (pipeline_6 * 32)) + (((int)threadIdx) & 31))];
      } else {
        condval = 0.000000e+00f;
      }
      tile_storage_3_shared[((tile_i_4_chunk * 128) + ((int)threadIdx))] = condval;
    }
    for (int tile_i_7_chunk = 0; tile_i_7_chunk < 8; ++tile_i_7_chunk) {
      float condval_1;
      if ((((((pipeline_6 * 32) + (tile_i_7_chunk * 4)) + (((int)threadIdx) >> 5)) < 1025) && ((((((int)blockIdx) % 129) * 32) + (((int)threadIdx) & 31)) < 4097))) {
        condval_1 = arg1_ptr[(((((pipeline_6 * 131104) + (tile_i_7_chunk * 16388)) + ((((int)threadIdx) >> 5) * 4097)) + ((((int)blockIdx) % 129) * 32)) + (((int)threadIdx) & 31))];
      } else {
        condval_1 = 0.000000e+00f;
      }
      tile_storage_6_shared[((tile_i_7_chunk * 128) + ((int)threadIdx))] = condval_1;
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    simdgroup_float8x8 tile_i_10_mma_a[2];
    simdgroup_float8x8 tile_i_10_mma_b[2];
    for (int tile_i_10_mma_k = 0; tile_i_10_mma_k < 4; ++tile_i_10_mma_k) {
      int cse_v11 = (((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k * 8));
      simdgroup_load(tile_i_10_mma_a[0], (&(tile_storage_3_shared[(((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k * 8))])), 32, 0, (bool)0);
      simdgroup_load(tile_i_10_mma_a[1], (&(tile_storage_3_shared[((((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k * 8)) + 256)])), 32, 0, (bool)0);
      int cse_v13 = ((tile_i_10_mma_k * 256) + (((((int)threadIdx) & 63) >> 5) * 16));
      simdgroup_load(tile_i_10_mma_b[0], (&(tile_storage_6_shared[((tile_i_10_mma_k * 256) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
      simdgroup_load(tile_i_10_mma_b[1], (&(tile_storage_6_shared[(((tile_i_10_mma_k * 256) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[0], tile_i_10_mma_a[0], tile_i_10_mma_b[0], tile_i_10_mma_c[0]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[1], tile_i_10_mma_a[0], tile_i_10_mma_b[1], tile_i_10_mma_c[1]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[2], tile_i_10_mma_a[1], tile_i_10_mma_b[0], tile_i_10_mma_c[2]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[3], tile_i_10_mma_a[1], tile_i_10_mma_b[1], tile_i_10_mma_c[3]);
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  simdgroup_store(tile_i_10_mma_c[0], (&(tile_storage_0_shared[(((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c[1], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c[2], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256)])), 32, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c[3], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264)])), 32, 0, (bool)0);
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_15_chunk = 0; tile_i_15_chunk < 8; ++tile_i_15_chunk) {
    if ((((((((int)blockIdx) / 129) * 32) + (tile_i_15_chunk * 4)) + (((int)threadIdx) >> 5)) < 2049) && ((((((int)blockIdx) % 129) * 32) + (((int)threadIdx) & 31)) < 4097)) {
      arg2_ptr[((((((((int)blockIdx) / 129) * 131104) + (tile_i_15_chunk * 16388)) + ((((int)threadIdx) >> 5) * 4097)) + ((((int)blockIdx) % 129) * 32)) + (((int)threadIdx) & 31))] = tile_storage_0_shared[((tile_i_15_chunk * 128) + ((int)threadIdx))];
    }
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
}



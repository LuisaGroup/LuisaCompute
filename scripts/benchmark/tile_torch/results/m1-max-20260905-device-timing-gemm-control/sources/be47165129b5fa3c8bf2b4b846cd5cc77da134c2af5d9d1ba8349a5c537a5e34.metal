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
  threadgroup float tile_storage_3_shared[2048];
  threadgroup float tile_storage_6_shared[2048];
  threadgroup float tile_storage_9_shared[1024];
  int cse_v1 = (((int)threadIdx) & 31);
  int cse_v2 = ((((int)blockIdx) >> 2) * 4096);
  int cse_v3 = ((((int)threadIdx) >> 5) * 128);
  for (int tile_i_4_chunk = 0; tile_i_4_chunk < 8; ++tile_i_4_chunk) {
    tile_storage_3_shared[((tile_i_4_chunk * 128) + ((int)threadIdx))] = arg0_ptr[(((((((int)blockIdx) >> 2) * 4096) + (tile_i_4_chunk * 512)) + ((((int)threadIdx) >> 5) * 128)) + (((int)threadIdx) & 31))];
  }
  int cse_v4 = ((((int)blockIdx) & 3) * 32);
  for (int tile_i_7_chunk = 0; tile_i_7_chunk < 8; ++tile_i_7_chunk) {
    tile_storage_6_shared[((tile_i_7_chunk * 128) + ((int)threadIdx))] = arg1_ptr[((((tile_i_7_chunk * 512) + ((((int)threadIdx) >> 5) * 128)) + ((((int)blockIdx) & 3) * 32)) + (((int)threadIdx) & 31))];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  int cse_v5 = ((((int)threadIdx) >> 6) * 512);
  int cse_v10 = (((((int)threadIdx) & 63) >> 5) * 16);
  int cse_v12 = (((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16));
  int cse_v16 = ((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8);
  int cse_v17 = ((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256);
  int cse_v18 = ((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264);
  for (int pipeline_6 = 0; pipeline_6 < 3; ++pipeline_6) {
    int cse_v9 = (((pipeline_6 + 1) & 1) * 1024);
    for (int tile_i_4_chunk_1 = 0; tile_i_4_chunk_1 < 8; ++tile_i_4_chunk_1) {
      tile_storage_3_shared[(((((pipeline_6 + 1) & 1) * 1024) + (tile_i_4_chunk_1 * 128)) + ((int)threadIdx))] = arg0_ptr[(((((((((int)blockIdx) >> 2) * 4096) + (tile_i_4_chunk_1 * 512)) + ((((int)threadIdx) >> 5) * 128)) + (pipeline_6 * 32)) + (((int)threadIdx) & 31)) + 32)];
    }
    for (int tile_i_7_chunk_1 = 0; tile_i_7_chunk_1 < 8; ++tile_i_7_chunk_1) {
      tile_storage_6_shared[(((((pipeline_6 + 1) & 1) * 1024) + (tile_i_7_chunk_1 * 128)) + ((int)threadIdx))] = arg1_ptr[((((((pipeline_6 * 4096) + (tile_i_7_chunk_1 * 512)) + ((((int)threadIdx) >> 5) * 128)) + ((((int)blockIdx) & 3) * 32)) + (((int)threadIdx) & 31)) + 4096)];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    simdgroup_float8x8 tile_i_10_mma_c[4];
    simdgroup_load(tile_i_10_mma_c[0], (&(tile_storage_0_shared[(((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
    simdgroup_load(tile_i_10_mma_c[1], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
    simdgroup_load(tile_i_10_mma_c[2], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256)])), 32, 0, (bool)0);
    simdgroup_load(tile_i_10_mma_c[3], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264)])), 32, 0, (bool)0);
    simdgroup_float8x8 tile_i_10_mma_a[2];
    simdgroup_float8x8 tile_i_10_mma_b[2];
    for (int tile_i_10_mma_k = 0; tile_i_10_mma_k < 4; ++tile_i_10_mma_k) {
      int cse_v6 = ((pipeline_6 & 1) * 1024);
      int cse_v13 = ((((pipeline_6 & 1) * 1024) + ((((int)threadIdx) >> 6) * 512)) + (tile_i_10_mma_k * 8));
      simdgroup_load(tile_i_10_mma_a[0], (&(tile_storage_3_shared[((((pipeline_6 & 1) * 1024) + ((((int)threadIdx) >> 6) * 512)) + (tile_i_10_mma_k * 8))])), 32, 0, (bool)0);
      simdgroup_load(tile_i_10_mma_a[1], (&(tile_storage_3_shared[(((((pipeline_6 & 1) * 1024) + ((((int)threadIdx) >> 6) * 512)) + (tile_i_10_mma_k * 8)) + 256)])), 32, 0, (bool)0);
      int cse_v14 = ((((pipeline_6 & 1) * 1024) + (tile_i_10_mma_k * 256)) + (((((int)threadIdx) & 63) >> 5) * 16));
      simdgroup_load(tile_i_10_mma_b[0], (&(tile_storage_6_shared[((((pipeline_6 & 1) * 1024) + (tile_i_10_mma_k * 256)) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
      simdgroup_load(tile_i_10_mma_b[1], (&(tile_storage_6_shared[(((((pipeline_6 & 1) * 1024) + (tile_i_10_mma_k * 256)) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[0], tile_i_10_mma_a[0], tile_i_10_mma_b[0], tile_i_10_mma_c[0]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[1], tile_i_10_mma_a[0], tile_i_10_mma_b[1], tile_i_10_mma_c[1]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[2], tile_i_10_mma_a[1], tile_i_10_mma_b[0], tile_i_10_mma_c[2]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[3], tile_i_10_mma_a[1], tile_i_10_mma_b[1], tile_i_10_mma_c[3]);
    }
    simdgroup_store(tile_i_10_mma_c[0], (&(tile_storage_9_shared[(((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
    simdgroup_store(tile_i_10_mma_c[1], (&(tile_storage_9_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
    simdgroup_store(tile_i_10_mma_c[2], (&(tile_storage_9_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256)])), 32, 0, (bool)0);
    simdgroup_store(tile_i_10_mma_c[3], (&(tile_storage_9_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264)])), 32, 0, (bool)0);
    metal::threadgroup_barrier(metal::mem_flags(2));
    for (int tile_i_13_chunk = 0; tile_i_13_chunk < 8; ++tile_i_13_chunk) {
      int cse_v7 = ((tile_i_13_chunk * 128) + ((int)threadIdx));
      tile_storage_0_shared[((tile_i_13_chunk * 128) + ((int)threadIdx))] = tile_storage_9_shared[((tile_i_13_chunk * 128) + ((int)threadIdx))];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  simdgroup_float8x8 tile_i_10_mma_c_1[4];
  simdgroup_load(tile_i_10_mma_c_1[0], (&(tile_storage_0_shared[(((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
  simdgroup_load(tile_i_10_mma_c_1[1], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
  simdgroup_load(tile_i_10_mma_c_1[2], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256)])), 32, 0, (bool)0);
  simdgroup_load(tile_i_10_mma_c_1[3], (&(tile_storage_0_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264)])), 32, 0, (bool)0);
  simdgroup_float8x8 tile_i_10_mma_a_1[2];
  simdgroup_float8x8 tile_i_10_mma_b_1[2];
  for (int tile_i_10_mma_k_1 = 0; tile_i_10_mma_k_1 < 4; ++tile_i_10_mma_k_1) {
    int cse_v11 = (((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k_1 * 8));
    simdgroup_load(tile_i_10_mma_a_1[0], (&(tile_storage_3_shared[((((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k_1 * 8)) + 1024)])), 32, 0, (bool)0);
    simdgroup_load(tile_i_10_mma_a_1[1], (&(tile_storage_3_shared[((((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k_1 * 8)) + 1280)])), 32, 0, (bool)0);
    int cse_v15 = ((tile_i_10_mma_k_1 * 256) + (((((int)threadIdx) & 63) >> 5) * 16));
    simdgroup_load(tile_i_10_mma_b_1[0], (&(tile_storage_6_shared[(((tile_i_10_mma_k_1 * 256) + (((((int)threadIdx) & 63) >> 5) * 16)) + 1024)])), 32, 0, (bool)0);
    simdgroup_load(tile_i_10_mma_b_1[1], (&(tile_storage_6_shared[(((tile_i_10_mma_k_1 * 256) + (((((int)threadIdx) & 63) >> 5) * 16)) + 1032)])), 32, 0, (bool)0);
    simdgroup_multiply_accumulate(tile_i_10_mma_c_1[0], tile_i_10_mma_a_1[0], tile_i_10_mma_b_1[0], tile_i_10_mma_c_1[0]);
    simdgroup_multiply_accumulate(tile_i_10_mma_c_1[1], tile_i_10_mma_a_1[0], tile_i_10_mma_b_1[1], tile_i_10_mma_c_1[1]);
    simdgroup_multiply_accumulate(tile_i_10_mma_c_1[2], tile_i_10_mma_a_1[1], tile_i_10_mma_b_1[0], tile_i_10_mma_c_1[2]);
    simdgroup_multiply_accumulate(tile_i_10_mma_c_1[3], tile_i_10_mma_a_1[1], tile_i_10_mma_b_1[1], tile_i_10_mma_c_1[3]);
  }
  simdgroup_store(tile_i_10_mma_c_1[0], (&(tile_storage_9_shared[(((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c_1[1], (&(tile_storage_9_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c_1[2], (&(tile_storage_9_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 256)])), 32, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c_1[3], (&(tile_storage_9_shared[((((((int)threadIdx) >> 6) * 512) + (((((int)threadIdx) & 63) >> 5) * 16)) + 264)])), 32, 0, (bool)0);
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_13_chunk_1 = 0; tile_i_13_chunk_1 < 8; ++tile_i_13_chunk_1) {
    int cse_v8 = ((tile_i_13_chunk_1 * 128) + ((int)threadIdx));
    tile_storage_0_shared[((tile_i_13_chunk_1 * 128) + ((int)threadIdx))] = tile_storage_9_shared[((tile_i_13_chunk_1 * 128) + ((int)threadIdx))];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_15_chunk = 0; tile_i_15_chunk < 8; ++tile_i_15_chunk) {
    arg2_ptr[((((((((int)blockIdx) >> 2) * 4096) + (tile_i_15_chunk * 512)) + ((((int)threadIdx) >> 5) * 128)) + ((((int)blockIdx) & 3) * 32)) + (((int)threadIdx) & 31))] = tile_storage_0_shared[((tile_i_15_chunk * 128) + ((int)threadIdx))];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
}



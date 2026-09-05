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
  simdgroup_float8x8 tile_i_10_mma_c[4];
  tile_i_10_mma_c[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  tile_i_10_mma_c[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  tile_i_10_mma_c[2] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  tile_i_10_mma_c[3] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  int cse_v1 = (((int)blockIdx) >> 7);
  int cse_v4 = (((int)threadIdx) >> 6);
  int cse_v5 = ((((int)blockIdx) & 127) * 32);
  int cse_v7 = (((((int)threadIdx) & 63) >> 5) * 16);
  for (int pipeline_6 = 0; pipeline_6 < 344; ++pipeline_6) {
    threadgroup float tile_storage_3_shared[1024];
    threadgroup float tile_storage_6_shared[1024];
    int cse_v2 = (((int)threadIdx) >> 5);
    int cse_v3 = (((int)threadIdx) & 31);
    for (int tile_i_4_chunk = 0; tile_i_4_chunk < 8; ++tile_i_4_chunk) {
      tile_storage_3_shared[((tile_i_4_chunk * 128) + ((int)threadIdx))] = arg0_ptr[((((((((int)blockIdx) >> 7) * 352256) + (tile_i_4_chunk * 44032)) + ((((int)threadIdx) >> 5) * 11008)) + (pipeline_6 * 32)) + (((int)threadIdx) & 31))];
    }
    for (int tile_i_7_chunk = 0; tile_i_7_chunk < 8; ++tile_i_7_chunk) {
      tile_storage_6_shared[((tile_i_7_chunk * 128) + ((int)threadIdx))] = arg1_ptr[(((((pipeline_6 * 131072) + (tile_i_7_chunk * 16384)) + ((((int)threadIdx) >> 5) * 4096)) + ((((int)blockIdx) & 127) * 32)) + (((int)threadIdx) & 31))];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    simdgroup_float8x8 tile_i_10_mma_a[2];
    simdgroup_float8x8 tile_i_10_mma_b[2];
    for (int tile_i_10_mma_k = 0; tile_i_10_mma_k < 4; ++tile_i_10_mma_k) {
      int cse_v6 = (((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k * 8));
      simdgroup_load(tile_i_10_mma_a[0], (&(tile_storage_3_shared[(((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k * 8))])), 32, 0, (bool)0);
      simdgroup_load(tile_i_10_mma_a[1], (&(tile_storage_3_shared[((((((int)threadIdx) >> 6) * 512) + (tile_i_10_mma_k * 8)) + 256)])), 32, 0, (bool)0);
      int cse_v8 = ((tile_i_10_mma_k * 256) + (((((int)threadIdx) & 63) >> 5) * 16));
      simdgroup_load(tile_i_10_mma_b[0], (&(tile_storage_6_shared[((tile_i_10_mma_k * 256) + (((((int)threadIdx) & 63) >> 5) * 16))])), 32, 0, (bool)0);
      simdgroup_load(tile_i_10_mma_b[1], (&(tile_storage_6_shared[(((tile_i_10_mma_k * 256) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 32, 0, (bool)0);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[0], tile_i_10_mma_a[0], tile_i_10_mma_b[0], tile_i_10_mma_c[0]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[1], tile_i_10_mma_a[0], tile_i_10_mma_b[1], tile_i_10_mma_c[1]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[2], tile_i_10_mma_a[1], tile_i_10_mma_b[0], tile_i_10_mma_c[2]);
      simdgroup_multiply_accumulate(tile_i_10_mma_c[3], tile_i_10_mma_a[1], tile_i_10_mma_b[1], tile_i_10_mma_c[3]);
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  int cse_v9 = (((((((int)blockIdx) >> 7) * 131072) + ((((int)threadIdx) >> 6) * 65536)) + ((((int)blockIdx) & 127) * 32)) + (((((int)threadIdx) & 63) >> 5) * 16));
  simdgroup_store(tile_i_10_mma_c[0], (&(arg2_ptr[(((((((int)blockIdx) >> 7) * 131072) + ((((int)threadIdx) >> 6) * 65536)) + ((((int)blockIdx) & 127) * 32)) + (((((int)threadIdx) & 63) >> 5) * 16))])), 4096, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c[1], (&(arg2_ptr[((((((((int)blockIdx) >> 7) * 131072) + ((((int)threadIdx) >> 6) * 65536)) + ((((int)blockIdx) & 127) * 32)) + (((((int)threadIdx) & 63) >> 5) * 16)) + 8)])), 4096, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c[2], (&(arg2_ptr[((((((((int)blockIdx) >> 7) * 131072) + ((((int)threadIdx) >> 6) * 65536)) + ((((int)blockIdx) & 127) * 32)) + (((((int)threadIdx) & 63) >> 5) * 16)) + 32768)])), 4096, 0, (bool)0);
  simdgroup_store(tile_i_10_mma_c[3], (&(arg2_ptr[((((((((int)blockIdx) >> 7) * 131072) + ((((int)threadIdx) >> 6) * 65536)) + ((((int)blockIdx) & 127) * 32)) + (((((int)threadIdx) & 63) >> 5) * 16)) + 32776)])), 4096, 0, (bool)0);
  metal::threadgroup_barrier(metal::mem_flags(2));
}



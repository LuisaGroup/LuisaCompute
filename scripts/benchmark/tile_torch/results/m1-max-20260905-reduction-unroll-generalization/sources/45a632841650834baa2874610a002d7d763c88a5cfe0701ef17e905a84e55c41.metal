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
    int cse_v2 = (((int)threadIdx) & 31);
    int cse_v11 = ((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257));
    for (int n_5_0_subgroup_chunk_pack = 0; n_5_0_subgroup_chunk_pack < 2; ++n_5_0_subgroup_chunk_pack) {
      int cse_v12 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31));
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31))]);
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 32)]);
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 64)]);
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (n_5_0_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 96)]);
    }
    int cse_v13 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (((int)threadIdx) & 31)) + 256);
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_5[0] = (tile_storage_5[0] + arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (((int)threadIdx) & 31)) + 256)]);
    }
    tile_storage_5[0] = simd_sum(tile_storage_5[0]);
    tile_storage_3[0] = tile_storage_5[0];
    thread float tile_storage_7_worker_stripe[9];
    for (int tile_i_8_subgroup_chunk_pack = 0; tile_i_8_subgroup_chunk_pack < 2; ++tile_i_8_subgroup_chunk_pack) {
      int cse_v3 = (tile_i_8_subgroup_chunk_pack * 4);
      int cse_v14 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31));
      tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk_pack * 4)] = (arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31))] - (tile_storage_3[0] / 2.570000e+02f));
      tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk_pack * 4) + 1)] = (arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 32)] - (tile_storage_3[0] / 2.570000e+02f));
      tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk_pack * 4) + 2)] = (arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 64)] - (tile_storage_3[0] / 2.570000e+02f));
      tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk_pack * 4) + 3)] = (arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_8_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 96)] - (tile_storage_3[0] / 2.570000e+02f));
    }
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_7_worker_stripe[8] = (arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (((int)threadIdx) & 31)) + 256)] - (tile_storage_3[0] / 2.570000e+02f));
    }
    thread float tile_storage_10[1];
    thread float tile_storage_12[1];
    tile_storage_12[0] = 0.000000e+00f;
    for (int n_20_0_subgroup_chunk_pack = 0; n_20_0_subgroup_chunk_pack < 2; ++n_20_0_subgroup_chunk_pack) {
      int cse_v4 = (n_20_0_subgroup_chunk_pack * 4);
      tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk_pack * 4)] * tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk_pack * 4)]));
      int cse_v7 = ((n_20_0_subgroup_chunk_pack * 4) + 1);
      tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk_pack * 4) + 1)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk_pack * 4) + 1)]));
      int cse_v8 = ((n_20_0_subgroup_chunk_pack * 4) + 2);
      tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk_pack * 4) + 2)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk_pack * 4) + 2)]));
      int cse_v9 = ((n_20_0_subgroup_chunk_pack * 4) + 3);
      tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk_pack * 4) + 3)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk_pack * 4) + 3)]));
    }
    if ((((int)threadIdx) & 31) < 1) {
      tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[8] * tile_storage_7_worker_stripe[8]));
    }
    tile_storage_12[0] = simd_sum(tile_storage_12[0]);
    tile_storage_10[0] = tile_storage_12[0];
    for (int tile_i_20_subgroup_chunk_pack = 0; tile_i_20_subgroup_chunk_pack < 2; ++tile_i_20_subgroup_chunk_pack) {
      int cse_v5 = (tile_i_20_subgroup_chunk_pack * 4);
      int cse_v6 = (tile_i_20_subgroup_chunk_pack * 128);
      int cse_v10 = ((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31));
      int cse_v15 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_20_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31));
      arg2_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_20_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31))] = (((tile_storage_7_worker_stripe[(tile_i_20_subgroup_chunk_pack * 4)] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31))]) + arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 257)]);
      arg2_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_20_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 32)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk_pack * 4) + 1)] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 32)]) + arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 289)]);
      arg2_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_20_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 64)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk_pack * 4) + 2)] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 64)]) + arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 321)]);
      arg2_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_20_subgroup_chunk_pack * 128)) + (((int)threadIdx) & 31)) + 96)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk_pack * 4) + 3)] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 96)]) + arg1_ptr[(((tile_i_20_subgroup_chunk_pack * 128) + (((int)threadIdx) & 31)) + 353)]);
    }
    if ((((int)threadIdx) & 31) < 1) {
      arg2_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (((int)threadIdx) & 31)) + 256)] = (((tile_storage_7_worker_stripe[8] / sqrt(((tile_storage_10[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) & 31) + 256)]) + arg1_ptr[((((int)threadIdx) & 31) + 513)]);
    }
  }
}



// Function: benchmark_rmsnorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_rmsnorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_0_worker_stripe[9];
  int cse_v1 = (((int)threadIdx) & 31);
  int cse_v6 = ((((int)threadIdx) & 31) * 4);
  int cse_v10 = ((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257));
  for (int tile_i_1_subgroup_chunk = 0; tile_i_1_subgroup_chunk < 2; ++tile_i_1_subgroup_chunk) {
    int cse_v2 = (tile_i_1_subgroup_chunk * 4);
    int cse_v13 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_1_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4));
    tile_storage_0_worker_stripe[(tile_i_1_subgroup_chunk * 4)] = arg0_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_1_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4))];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 1)] = arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_1_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 1)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 2)] = arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_1_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 2)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 3)] = arg0_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_1_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 3)];
  }
  int cse_v12 = (((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + 256);
  if ((((int)threadIdx) % 32) == 0) {
    tile_storage_0_worker_stripe[8] = arg0_ptr[(((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + 256)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 2; ++n_6_0_subgroup_chunk) {
    int cse_v3 = (n_6_0_subgroup_chunk * 4);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[(n_6_0_subgroup_chunk * 4)] * tile_storage_0_worker_stripe[(n_6_0_subgroup_chunk * 4)]));
    int cse_v7 = ((n_6_0_subgroup_chunk * 4) + 1);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 1)] * tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 1)]));
    int cse_v8 = ((n_6_0_subgroup_chunk * 4) + 2);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 2)] * tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 2)]));
    int cse_v9 = ((n_6_0_subgroup_chunk * 4) + 3);
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 3)] * tile_storage_0_worker_stripe[((n_6_0_subgroup_chunk * 4) + 3)]));
  }
  if ((((int)threadIdx) % 32) == 0) {
    tile_storage_5[0] = (tile_storage_5[0] + (tile_storage_0_worker_stripe[8] * tile_storage_0_worker_stripe[8]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  for (int tile_i_10_subgroup_chunk = 0; tile_i_10_subgroup_chunk < 2; ++tile_i_10_subgroup_chunk) {
    int cse_v4 = (tile_i_10_subgroup_chunk * 4);
    int cse_v5 = (tile_i_10_subgroup_chunk * 128);
    int cse_v11 = ((tile_i_10_subgroup_chunk * 128) + ((((int)threadIdx) & 31) * 4));
    int cse_v14 = ((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_10_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4));
    arg2_ptr[((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_10_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4))] = ((tile_storage_0_worker_stripe[(tile_i_10_subgroup_chunk * 4)] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[((tile_i_10_subgroup_chunk * 128) + ((((int)threadIdx) & 31) * 4))]);
    arg2_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_10_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 1)] = ((tile_storage_0_worker_stripe[((tile_i_10_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 128) + ((((int)threadIdx) & 31) * 4)) + 1)]);
    arg2_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_10_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 2)] = ((tile_storage_0_worker_stripe[((tile_i_10_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 128) + ((((int)threadIdx) & 31) * 4)) + 2)]);
    arg2_ptr[(((((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + (tile_i_10_subgroup_chunk * 128)) + ((((int)threadIdx) & 31) * 4)) + 3)] = ((tile_storage_0_worker_stripe[((tile_i_10_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 128) + ((((int)threadIdx) & 31) * 4)) + 3)]);
  }
  if ((((int)threadIdx) % 32) == 0) {
    arg2_ptr[(((((int)blockIdx) * 2056) + ((((int)threadIdx) >> 5) * 257)) + 256)] = ((tile_storage_0_worker_stripe[8] / sqrt(((tile_storage_3[0] / 2.570000e+02f) + 1.000000e-05f))) * arg1_ptr[256]);
  }
}



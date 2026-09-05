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
  thread float tile_storage_0_worker_stripe[25];
  int cse_v1 = (((int)blockIdx) * 769);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int tile_i_1_subgroup_chunk = 0; tile_i_1_subgroup_chunk < 6; ++tile_i_1_subgroup_chunk) {
    int cse_v3 = (tile_i_1_subgroup_chunk * 4);
    int cse_v16 = (((((int)blockIdx) * 769) + (tile_i_1_subgroup_chunk * 128)) + (((int)threadIdx) * 4));
    tile_storage_0_worker_stripe[(tile_i_1_subgroup_chunk * 4)] = arg0_ptr[(((((int)blockIdx) * 769) + (tile_i_1_subgroup_chunk * 128)) + (((int)threadIdx) * 4))];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 1)] = arg0_ptr[((((((int)blockIdx) * 769) + (tile_i_1_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 1)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 2)] = arg0_ptr[((((((int)blockIdx) * 769) + (tile_i_1_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 2)];
    tile_storage_0_worker_stripe[((tile_i_1_subgroup_chunk * 4) + 3)] = arg0_ptr[((((((int)blockIdx) * 769) + (tile_i_1_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 3)];
  }
  int cse_v17 = (((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 768);
  if (((int)threadIdx) < 1) {
    tile_storage_0_worker_stripe[24] = arg0_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 768)];
  }
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  for (int n_5_0_subgroup_chunk = 0; n_5_0_subgroup_chunk < 6; ++n_5_0_subgroup_chunk) {
    int cse_v4 = (n_5_0_subgroup_chunk * 4);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[(n_5_0_subgroup_chunk * 4)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 1)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 2)]);
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[((n_5_0_subgroup_chunk * 4) + 3)]);
  }
  if (((int)threadIdx) < 1) {
    tile_storage_5[0] = (tile_storage_5[0] + tile_storage_0_worker_stripe[24]);
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  thread float tile_storage_7_worker_stripe[25];
  for (int tile_i_8_subgroup_chunk = 0; tile_i_8_subgroup_chunk < 6; ++tile_i_8_subgroup_chunk) {
    int cse_v5 = (tile_i_8_subgroup_chunk * 4);
    tile_storage_7_worker_stripe[(tile_i_8_subgroup_chunk * 4)] = (tile_storage_0_worker_stripe[(tile_i_8_subgroup_chunk * 4)] - (tile_storage_3[0] / 7.690000e+02f));
    int cse_v9 = ((tile_i_8_subgroup_chunk * 4) + 1);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 1)] - (tile_storage_3[0] / 7.690000e+02f));
    int cse_v10 = ((tile_i_8_subgroup_chunk * 4) + 2);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 2)] - (tile_storage_3[0] / 7.690000e+02f));
    int cse_v11 = ((tile_i_8_subgroup_chunk * 4) + 3);
    tile_storage_7_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] = (tile_storage_0_worker_stripe[((tile_i_8_subgroup_chunk * 4) + 3)] - (tile_storage_3[0] / 7.690000e+02f));
  }
  if (((int)threadIdx) < 1) {
    tile_storage_7_worker_stripe[24] = (tile_storage_0_worker_stripe[24] - (tile_storage_3[0] / 7.690000e+02f));
  }
  thread float tile_storage_10[1];
  thread float tile_storage_12[1];
  tile_storage_12[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 6; ++n_20_0_subgroup_chunk) {
    int cse_v6 = (n_20_0_subgroup_chunk * 4);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)] * tile_storage_7_worker_stripe[(n_20_0_subgroup_chunk * 4)]));
    int cse_v12 = ((n_20_0_subgroup_chunk * 4) + 1);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 1)]));
    int cse_v13 = ((n_20_0_subgroup_chunk * 4) + 2);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 2)]));
    int cse_v14 = ((n_20_0_subgroup_chunk * 4) + 3);
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)] * tile_storage_7_worker_stripe[((n_20_0_subgroup_chunk * 4) + 3)]));
  }
  if (((int)threadIdx) < 1) {
    tile_storage_12[0] = (tile_storage_12[0] + (tile_storage_7_worker_stripe[24] * tile_storage_7_worker_stripe[24]));
  }
  tile_storage_12[0] = simd_sum(tile_storage_12[0]);
  tile_storage_10[0] = tile_storage_12[0];
  for (int tile_i_20_subgroup_chunk = 0; tile_i_20_subgroup_chunk < 6; ++tile_i_20_subgroup_chunk) {
    int cse_v7 = (tile_i_20_subgroup_chunk * 4);
    int cse_v8 = (tile_i_20_subgroup_chunk * 128);
    int cse_v15 = ((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4));
    int cse_v18 = (((((int)blockIdx) * 769) + (tile_i_20_subgroup_chunk * 128)) + (((int)threadIdx) * 4));
    arg2_ptr[(((((int)blockIdx) * 769) + (tile_i_20_subgroup_chunk * 128)) + (((int)threadIdx) * 4))] = (((tile_storage_7_worker_stripe[(tile_i_20_subgroup_chunk * 4)] / sqrt(((tile_storage_10[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4))]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 769)]);
    arg2_ptr[((((((int)blockIdx) * 769) + (tile_i_20_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 1)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 1)] / sqrt(((tile_storage_10[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 1)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 770)]);
    arg2_ptr[((((((int)blockIdx) * 769) + (tile_i_20_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 2)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 2)] / sqrt(((tile_storage_10[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 2)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 771)]);
    arg2_ptr[((((((int)blockIdx) * 769) + (tile_i_20_subgroup_chunk * 128)) + (((int)threadIdx) * 4)) + 3)] = (((tile_storage_7_worker_stripe[((tile_i_20_subgroup_chunk * 4) + 3)] / sqrt(((tile_storage_10[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 3)]) + arg1_ptr[(((tile_i_20_subgroup_chunk * 128) + (((int)threadIdx) * 4)) + 772)]);
  }
  if (((int)threadIdx) < 1) {
    arg2_ptr[(((((int)blockIdx) * 769) + (((int)threadIdx) * 4)) + 768)] = (((tile_storage_7_worker_stripe[24] / sqrt(((tile_storage_10[0] / 7.690000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 768)]) + arg1_ptr[((((int)threadIdx) * 4) + 1537)]);
  }
}



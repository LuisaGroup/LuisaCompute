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
  threadgroup float parallel_0_subgroup_partials_0[20];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)blockIdx) * 12289);
  int cse_v2 = (((int)threadIdx) * 4);
  for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 4; ++n_6_0_subgroup_chunk) {
    int cse_v7 = (((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4));
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4))] * arg0_ptr[(((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4))]));
    int cse_v13 = ((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 1);
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 1)] * arg0_ptr[((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 1)]));
    int cse_v14 = ((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 2);
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 2)] * arg0_ptr[((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 2)]));
    int cse_v15 = ((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 3);
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 3)] * arg0_ptr[((((((int)blockIdx) * 12289) + (n_6_0_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 3)]));
  }
  int cse_v5 = ((((int)blockIdx) * 12289) + (((int)threadIdx) * 4));
  int cse_v8 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10240);
  if (((int)threadIdx) < 513) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10240)] * arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10240)]));
  }
  int cse_v9 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10241);
  if (((int)threadIdx) < 512) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10241)] * arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10241)]));
  }
  int cse_v10 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10242);
  if (((int)threadIdx) < 512) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10242)] * arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10242)]));
  }
  int cse_v11 = (((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10243);
  if (((int)threadIdx) < 512) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10243)] * arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10243)]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v3 = (((int)threadIdx) & 31);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 20)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  for (int tile_i_10_subgroup_chunk = 0; tile_i_10_subgroup_chunk < 4; ++tile_i_10_subgroup_chunk) {
    int cse_v4 = (tile_i_10_subgroup_chunk * 2560);
    int cse_v6 = ((tile_i_10_subgroup_chunk * 2560) + (((int)threadIdx) * 4));
    int cse_v12 = (((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4));
    arg2_ptr[(((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4))] = ((arg0_ptr[(((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4))] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((tile_i_10_subgroup_chunk * 2560) + (((int)threadIdx) * 4))]);
    int cse_v16 = ((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 1);
    arg2_ptr[((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 1)] = ((arg0_ptr[((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 1)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 2560) + (((int)threadIdx) * 4)) + 1)]);
    int cse_v17 = ((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 2);
    arg2_ptr[((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 2)] = ((arg0_ptr[((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 2)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 2560) + (((int)threadIdx) * 4)) + 2)]);
    int cse_v18 = ((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 3);
    arg2_ptr[((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 3)] = ((arg0_ptr[((((((int)blockIdx) * 12289) + (tile_i_10_subgroup_chunk * 2560)) + (((int)threadIdx) * 4)) + 3)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[(((tile_i_10_subgroup_chunk * 2560) + (((int)threadIdx) * 4)) + 3)]);
  }
  if (((int)threadIdx) < 513) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10240)] = ((arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10240)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 10240)]);
  }
  if (((int)threadIdx) < 512) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10241)] = ((arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10241)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 10241)]);
  }
  if (((int)threadIdx) < 512) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10242)] = ((arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10242)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 10242)]);
  }
  if (((int)threadIdx) < 512) {
    arg2_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10243)] = ((arg0_ptr[(((((int)blockIdx) * 12289) + (((int)threadIdx) * 4)) + 10243)] / sqrt(((tile_storage_3[0] / 1.228900e+04f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 10243)]);
  }
}



// Function: benchmark_cross_entropy_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_cross_entropy_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device long* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float parallel_0_subgroup_partials_0[8];
  threadgroup float parallel_0_subgroup_partials_1[8];
  thread long tile_storage_3[1];
  if (((int)threadIdx) < 1) {
    tile_storage_3[0] = arg1_ptr[((long)blockIdx)];
  }
  thread float tile_storage_5[1];
  thread float tile_storage_7[1];
  tile_storage_7[0] = -INFINITY;
  long cse_v1 = (((long)blockIdx) * (long)4096);
  long cse_v2 = ((long)((int)threadIdx));
  for (int n_6_0_subgroup_chunk_pack = 0; n_6_0_subgroup_chunk_pack < 4; ++n_6_0_subgroup_chunk_pack) {
    long cse_v5 = (((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx)));
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[(((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx)))]);
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx))) + (long)256)]);
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx))) + (long)512)]);
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx))) + (long)768)]);
  }
  tile_storage_7[0] = simd_max(tile_storage_7[0]);
  int cse_v3 = (((int)threadIdx) & 31);
  int cse_v4 = (((int)threadIdx) >> 5);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_7[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 8)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = -INFINITY;
  }
  tile_storage_7[0] = simd_max(condval);
  tile_storage_5[0] = tile_storage_7[0];
  thread float tile_storage_9[1];
  thread float tile_storage_11[1];
  tile_storage_11[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk_pack = 0; n_20_0_subgroup_chunk_pack < 4; ++n_20_0_subgroup_chunk_pack) {
    long cse_v6 = (((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx)));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[(((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx)))] - tile_storage_5[0])));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx))) + (long)256)] - tile_storage_5[0])));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx))) + (long)512)] - tile_storage_5[0])));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk_pack) * (long)1024)) + ((long)((int)threadIdx))) + (long)768)] - tile_storage_5[0])));
  }
  tile_storage_11[0] = simd_sum(tile_storage_11[0]);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_1[(((int)threadIdx) >> 5)] = tile_storage_11[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((int)threadIdx) & 31) < 8)) {
    condval_1 = parallel_0_subgroup_partials_1[(((int)threadIdx) & 31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_11[0] = simd_sum(condval_1);
  tile_storage_9[0] = tile_storage_11[0];
  thread float tile_storage_13[1];
  if (((int)threadIdx) < 1) {
    float condval_2;
    if ((((long)0 <= tile_storage_3[0]) && (tile_storage_3[0] < (long)4096))) {
      condval_2 = arg0_ptr[((((long)blockIdx) * (long)4096) + tile_storage_3[0])];
    } else {
      condval_2 = 0.000000e+00f;
    }
    tile_storage_13[0] = condval_2;
  }
  if (((int)threadIdx) < 1) {
    arg2_ptr[((long)blockIdx)] = ((log(tile_storage_9[0]) + tile_storage_5[0]) - tile_storage_13[0]);
  }
}



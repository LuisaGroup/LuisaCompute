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
  long cse_v1 = (((long)threadIdx) * (long)4);
  long cse_v5 = ((((long)threadIdx) * (long)4) + ((long)blockIdx));
  if (((long)threadIdx) < (long)1) {
    tile_storage_3[(((long)threadIdx) * (long)4)] = arg1_ptr[((((long)threadIdx) * (long)4) + ((long)blockIdx))];
  }
  thread float tile_storage_5[1];
  thread float tile_storage_7[1];
  tile_storage_7[0] = -INFINITY;
  long cse_v2 = (((long)blockIdx) * (long)4096);
  for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 4; ++n_6_0_subgroup_chunk) {
    long cse_v6 = (((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4));
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[(((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4))]);
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4)) + (long)1)]);
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4)) + (long)2)]);
    tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_6_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4)) + (long)3)]);
  }
  tile_storage_7[0] = simd_max(tile_storage_7[0]);
  long cse_v3 = (((long)threadIdx) & (long)31);
  long cse_v4 = (((long)threadIdx) >> (long)5);
  if ((((long)threadIdx) % (long)32) == (long)0) {
    parallel_0_subgroup_partials_0[(((long)threadIdx) >> (long)5)] = tile_storage_7[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((long)threadIdx) & (long)31) < (long)8)) {
    condval = parallel_0_subgroup_partials_0[(((long)threadIdx) & (long)31)];
  } else {
    condval = -INFINITY;
  }
  tile_storage_7[0] = simd_max(condval);
  tile_storage_5[0] = tile_storage_7[0];
  thread float tile_storage_9[1];
  thread float tile_storage_11[1];
  tile_storage_11[0] = 0.000000e+00f;
  for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 4; ++n_20_0_subgroup_chunk) {
    long cse_v7 = (((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[(((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4))] - tile_storage_5[0])));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4)) + (long)1)] - tile_storage_5[0])));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4)) + (long)2)] - tile_storage_5[0])));
    tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)4096) + (((long)n_20_0_subgroup_chunk) * (long)1024)) + (((long)threadIdx) * (long)4)) + (long)3)] - tile_storage_5[0])));
  }
  tile_storage_11[0] = simd_sum(tile_storage_11[0]);
  if ((((long)threadIdx) % (long)32) == (long)0) {
    parallel_0_subgroup_partials_1[(((long)threadIdx) >> (long)5)] = tile_storage_11[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval_1;
  if (((((long)threadIdx) & (long)31) < (long)8)) {
    condval_1 = parallel_0_subgroup_partials_1[(((long)threadIdx) & (long)31)];
  } else {
    condval_1 = 0.000000e+00f;
  }
  tile_storage_11[0] = simd_sum(condval_1);
  tile_storage_9[0] = tile_storage_11[0];
  thread float tile_storage_13[1];
  if (((long)threadIdx) < (long)1) {
    float condval_2;
    if ((((long)0 <= tile_storage_3[(((long)threadIdx) * (long)4)]) && (tile_storage_3[(((long)threadIdx) * (long)4)] < (long)4096))) {
      condval_2 = arg0_ptr[(((((long)threadIdx) * (long)16384) + (((long)blockIdx) * (long)4096)) + tile_storage_3[(((long)threadIdx) * (long)4)])];
    } else {
      condval_2 = 0.000000e+00f;
    }
    tile_storage_13[(((long)threadIdx) * (long)4)] = condval_2;
  }
  if (((long)threadIdx) < (long)1) {
    arg2_ptr[((((long)threadIdx) * (long)4) + ((long)blockIdx))] = ((log(tile_storage_9[0]) + tile_storage_5[0]) - tile_storage_13[0]);
  }
}



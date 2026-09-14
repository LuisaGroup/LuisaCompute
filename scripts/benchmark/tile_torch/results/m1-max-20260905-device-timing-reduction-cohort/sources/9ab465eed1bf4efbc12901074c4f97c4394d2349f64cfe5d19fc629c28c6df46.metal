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
  long cse_v1 = (((long)threadIdx) >> (long)5);
  long cse_v3 = ((((long)blockIdx) * (long)8) + (((long)threadIdx) >> (long)5));
  if (((((long)blockIdx) * (long)8) + (((long)threadIdx) >> (long)5)) < (long)17) {
    thread long tile_storage_3[1];
    long cse_v2 = (((long)threadIdx) & (long)31);
    long cse_v6 = (((((long)blockIdx) * (long)8) + (((long)threadIdx) >> (long)5)) + (((long)threadIdx) & (long)31));
    if ((((long)threadIdx) & (long)31) < (long)1) {
      tile_storage_3[(((long)threadIdx) & (long)31)] = arg1_ptr[(((((long)blockIdx) * (long)8) + (((long)threadIdx) >> (long)5)) + (((long)threadIdx) & (long)31))];
    }
    thread float tile_storage_5[1];
    thread float tile_storage_7[1];
    tile_storage_7[0] = -INFINITY;
    long cse_v7 = ((((long)blockIdx) * (long)2056) + ((((long)threadIdx) >> (long)5) * (long)257));
    for (int n_6_0_subgroup_chunk = 0; n_6_0_subgroup_chunk < 9; ++n_6_0_subgroup_chunk) {
      long cse_v4 = (((long)n_6_0_subgroup_chunk) * (long)32);
      if (((((long)n_6_0_subgroup_chunk) * (long)32) + (((long)threadIdx) & (long)31)) < (long)257) {
        tile_storage_7[0] = max(tile_storage_7[0], arg0_ptr[((((((long)blockIdx) * (long)2056) + ((((long)threadIdx) >> (long)5) * (long)257)) + (((long)n_6_0_subgroup_chunk) * (long)32)) + (((long)threadIdx) & (long)31))]);
      }
    }
    tile_storage_7[0] = simd_max(tile_storage_7[0]);
    tile_storage_5[0] = tile_storage_7[0];
    thread float tile_storage_9[1];
    thread float tile_storage_11[1];
    tile_storage_11[0] = 0.000000e+00f;
    for (int n_20_0_subgroup_chunk = 0; n_20_0_subgroup_chunk < 9; ++n_20_0_subgroup_chunk) {
      long cse_v5 = (((long)n_20_0_subgroup_chunk) * (long)32);
      if (((((long)n_20_0_subgroup_chunk) * (long)32) + (((long)threadIdx) & (long)31)) < (long)257) {
        tile_storage_11[0] = (tile_storage_11[0] + exp((arg0_ptr[((((((long)blockIdx) * (long)2056) + ((((long)threadIdx) >> (long)5) * (long)257)) + (((long)n_20_0_subgroup_chunk) * (long)32)) + (((long)threadIdx) & (long)31))] - tile_storage_5[0])));
      }
    }
    tile_storage_11[0] = simd_sum(tile_storage_11[0]);
    tile_storage_9[0] = tile_storage_11[0];
    thread float tile_storage_13[1];
    if ((((long)threadIdx) & (long)31) < (long)1) {
      float condval;
      if ((((long)0 <= tile_storage_3[(((long)threadIdx) & (long)31)]) && (tile_storage_3[(((long)threadIdx) & (long)31)] < (long)257))) {
        condval = arg0_ptr[((((((long)blockIdx) * (long)2056) + ((((long)threadIdx) >> (long)5) * (long)257)) + ((((long)threadIdx) & (long)31) * (long)257)) + tile_storage_3[(((long)threadIdx) & (long)31)])];
      } else {
        condval = 0.000000e+00f;
      }
      tile_storage_13[(((long)threadIdx) & (long)31)] = condval;
    }
    if ((((long)threadIdx) & (long)31) < (long)1) {
      arg2_ptr[(((((long)blockIdx) * (long)8) + (((long)threadIdx) >> (long)5)) + (((long)threadIdx) & (long)31))] = ((log(tile_storage_9[0]) + tile_storage_5[0]) - tile_storage_13[0]);
    }
  }
}



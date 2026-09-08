// Function: llm_attention_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void llm_attention_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  device float* arg3_ptr [[ buffer(3) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float tile_storage_0_shared[64];
  int cse_v3 = ((((int)blockIdx) * 64) + ((int)threadIdx));
  tile_storage_0_shared[((int)threadIdx)] = arg0_ptr[((((int)blockIdx) * 64) + ((int)threadIdx))];
  threadgroup float tile_storage_5_shared[1];
  if (((int)threadIdx) < 1) {
    tile_storage_5_shared[0] = -1.000000e+30f;
  }
  threadgroup float tile_storage_9_shared[1];
  if (((int)threadIdx) < 1) {
    tile_storage_9_shared[0] = 0.000000e+00f;
  }
  threadgroup float tile_storage_13_shared[64];
  tile_storage_13_shared[((int)threadIdx)] = 0.000000e+00f;
  metal::threadgroup_barrier(metal::mem_flags(3));
  for (long pipeline_10 = (long)0; pipeline_10 < (long)64; ++pipeline_10) {
    threadgroup float tile_storage_18_shared[2048];
    threadgroup float tile_storage_23_shared[2048];
    threadgroup float tile_storage_28_shared[32];
    threadgroup long tile_storage_34_shared[32];
    threadgroup long tile_storage_36_shared[32];
    threadgroup long tile_storage_38_shared[1];
    threadgroup bool tile_storage_40_shared[32];
    threadgroup float tile_storage_43_shared[32];
    threadgroup float tile_storage_48_shared[1];
    threadgroup float tile_storage_54_shared[1];
    threadgroup float tile_storage_58_shared[1];
    threadgroup float tile_storage_62_shared[32];
    threadgroup float tile_storage_67_shared[1];
    threadgroup float tile_storage_73_shared[64];
    threadgroup float tile_storage_85_shared[1];
    long cse_v1 = ((long)((int)threadIdx));
    long cse_v4 = (((((long)((int)blockIdx)) / (long)4) * (long)131072) + (pipeline_10 * (long)2048));
    for (int tile_i_19_chunk = 0; tile_i_19_chunk < 32; ++tile_i_19_chunk) {
      tile_storage_18_shared[((tile_i_19_chunk * 64) + ((int)threadIdx))] = arg1_ptr[(((((((long)((int)blockIdx)) / (long)4) * (long)131072) + (pipeline_10 * (long)2048)) + (((long)tile_i_19_chunk) * (long)64)) + ((long)((int)threadIdx)))];
    }
    for (int tile_i_24_chunk = 0; tile_i_24_chunk < 32; ++tile_i_24_chunk) {
      tile_storage_23_shared[((tile_i_24_chunk * 64) + ((int)threadIdx))] = arg2_ptr[(((((((long)((int)blockIdx)) / (long)4) * (long)131072) + (pipeline_10 * (long)2048)) + (((long)tile_i_24_chunk) * (long)64)) + ((long)((int)threadIdx)))];
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      tile_storage_28_shared[((int)threadIdx)] = 0.000000e+00f;
      for (int tile_i_33 = 0; tile_i_33 < 64; ++tile_i_33) {
        tile_storage_28_shared[((int)threadIdx)] = (tile_storage_28_shared[((int)threadIdx)] + (tile_storage_0_shared[tile_i_33] * tile_storage_18_shared[((((int)threadIdx) * 64) + tile_i_33)]));
      }
    }
    if (((int)threadIdx) < 32) {
      tile_storage_34_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    if (((int)threadIdx) < 32) {
      tile_storage_36_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    if (((int)threadIdx) < 1) {
      tile_storage_38_shared[0] = ((long)((int)threadIdx));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      long cse_v2 = (pipeline_10 * (long)32);
      tile_storage_40_shared[((int)threadIdx)] = ((((pipeline_10 * (long)32) + tile_storage_34_shared[((int)threadIdx)]) < (long)2048) && (((pipeline_10 * (long)32) + tile_storage_36_shared[((int)threadIdx)]) <= (tile_storage_38_shared[0] + (long)2047)));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      float condval;
      if (tile_storage_40_shared[((int)threadIdx)]) {
        condval = (tile_storage_28_shared[((int)threadIdx)] * 1.250000e-01f);
      } else {
        condval = -1.000000e+30f;
      }
      tile_storage_43_shared[((int)threadIdx)] = condval;
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      thread float tile_storage_52[1];
      tile_storage_52[0] = -INFINITY;
      for (int n_55_0 = 0; n_55_0 < 32; ++n_55_0) {
        thread float tile_storage_53[1];
        tile_storage_53[0] = max(tile_storage_52[0], tile_storage_43_shared[n_55_0]);
        tile_storage_52[0] = tile_storage_53[0];
      }
      tile_storage_48_shared[0] = tile_storage_52[0];
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      tile_storage_54_shared[0] = max(tile_storage_5_shared[0], tile_storage_48_shared[0]);
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      tile_storage_58_shared[0] = exp((tile_storage_5_shared[0] - tile_storage_54_shared[0]));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      float condval_1;
      if (tile_storage_40_shared[((int)threadIdx)]) {
        condval_1 = exp((tile_storage_43_shared[((int)threadIdx)] - tile_storage_54_shared[0]));
      } else {
        condval_1 = 0.000000e+00f;
      }
      tile_storage_62_shared[((int)threadIdx)] = condval_1;
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      thread float tile_storage_71[1];
      tile_storage_71[0] = 0.000000e+00f;
      for (int n_86_0 = 0; n_86_0 < 32; ++n_86_0) {
        thread float tile_storage_72[1];
        tile_storage_72[0] = (tile_storage_71[0] + tile_storage_62_shared[n_86_0]);
        tile_storage_71[0] = tile_storage_72[0];
      }
      tile_storage_67_shared[0] = tile_storage_71[0];
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    tile_storage_73_shared[((int)threadIdx)] = (tile_storage_13_shared[((int)threadIdx)] * tile_storage_58_shared[0]);
    for (int tile_i_78 = 0; tile_i_78 < 32; ++tile_i_78) {
      tile_storage_73_shared[((int)threadIdx)] = (tile_storage_73_shared[((int)threadIdx)] + (tile_storage_62_shared[tile_i_78] * tile_storage_23_shared[((tile_i_78 * 64) + ((int)threadIdx))]));
    }
    if (((int)threadIdx) < 1) {
      tile_storage_85_shared[0] = ((tile_storage_9_shared[0] * tile_storage_58_shared[0]) + tile_storage_67_shared[0]);
    }
    if (((int)threadIdx) < 1) {
      tile_storage_5_shared[0] = tile_storage_54_shared[0];
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      tile_storage_9_shared[0] = tile_storage_85_shared[0];
    }
    tile_storage_13_shared[((int)threadIdx)] = tile_storage_73_shared[((int)threadIdx)];
    metal::threadgroup_barrier(metal::mem_flags(3));
  }
  arg3_ptr[((((int)blockIdx) * 64) + ((int)threadIdx))] = (tile_storage_13_shared[((int)threadIdx)] / tile_storage_9_shared[0]);
  metal::threadgroup_barrier(metal::mem_flags(3));
}



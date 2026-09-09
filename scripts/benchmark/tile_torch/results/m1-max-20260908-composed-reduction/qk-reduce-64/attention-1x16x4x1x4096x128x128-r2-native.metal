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
  threadgroup float tile_storage_5_shared[1];
  if (((int)threadIdx) < 1) {
    tile_storage_5_shared[0] = -1.000000e+30f;
  }
  threadgroup float tile_storage_9_shared[1];
  if (((int)threadIdx) < 1) {
    tile_storage_9_shared[0] = 0.000000e+00f;
  }
  threadgroup float tile_storage_13_shared[128];
  for (int tile_i_14_chunk = 0; tile_i_14_chunk < 2; ++tile_i_14_chunk) {
    tile_storage_13_shared[((tile_i_14_chunk * 64) + ((int)threadIdx))] = 0.000000e+00f;
  }
  metal::threadgroup_barrier(metal::mem_flags(3));
  int cse_v1 = (((int)blockIdx) * 128);
  for (long pipeline_10 = (long)0; pipeline_10 < (long)128; ++pipeline_10) {
    threadgroup float tile_storage_28_shared[32];
    threadgroup long tile_storage_35_shared[32];
    threadgroup long tile_storage_37_shared[32];
    threadgroup long tile_storage_39_shared[1];
    threadgroup bool tile_storage_41_shared[32];
    threadgroup float tile_storage_44_shared[32];
    threadgroup float tile_storage_49_shared[1];
    threadgroup float tile_storage_55_shared[1];
    threadgroup float tile_storage_59_shared[1];
    threadgroup float tile_storage_63_shared[32];
    threadgroup float tile_storage_68_shared[1];
    threadgroup float tile_storage_74_shared[128];
    threadgroup float tile_storage_86_shared[1];
    long cse_v3 = ((long)((int)threadIdx));
    long cse_v8 = (((((long)((int)blockIdx)) / (long)4) * (long)524288) + (pipeline_10 * (long)4096));
    for (int tile_i_29_reduction_batch = 0; tile_i_29_reduction_batch < 16; ++tile_i_29_reduction_batch) {
      thread float d_30_0_lane_carry[1];
      d_30_0_lane_carry[0] = 0.000000e+00f;
      int cse_v2 = (((int)threadIdx) & 31);
      for (int d_30_0_reduction_chunk = 0; d_30_0_reduction_chunk < 4; ++d_30_0_reduction_chunk) {
        d_30_0_lane_carry[0] = (d_30_0_lane_carry[0] + (arg0_ptr[(((((int)blockIdx) * 128) + (d_30_0_reduction_chunk * 32)) + (((int)threadIdx) & 31))] * arg1_ptr[(((((((((long)((int)blockIdx)) / (long)4) * (long)524288) + (pipeline_10 * (long)4096)) + (((long)tile_i_29_reduction_batch) * (long)256)) + ((((long)((int)threadIdx)) >> (long)5) * (long)128)) + (((long)d_30_0_reduction_chunk) * (long)32)) + (((long)((int)threadIdx)) & (long)31))]));
      }
      float d_30_0_subgroup_value = simd_sum(d_30_0_lane_carry[0]);
      if ((((int)threadIdx) % 32) == 0) {
        tile_storage_28_shared[((tile_i_29_reduction_batch * 2) + (((int)threadIdx) >> 5))] = d_30_0_subgroup_value;
      }
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      tile_storage_35_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    if (((int)threadIdx) < 32) {
      tile_storage_37_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    if (((int)threadIdx) < 1) {
      tile_storage_39_shared[0] = ((long)((int)threadIdx));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      long cse_v4 = (pipeline_10 * (long)32);
      tile_storage_41_shared[((int)threadIdx)] = ((((pipeline_10 * (long)32) + tile_storage_35_shared[((int)threadIdx)]) < (long)4096) && (((pipeline_10 * (long)32) + tile_storage_37_shared[((int)threadIdx)]) <= (tile_storage_39_shared[0] + (long)4095)));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      float condval;
      if (tile_storage_41_shared[((int)threadIdx)]) {
        condval = (tile_storage_28_shared[((int)threadIdx)] * 8.838835e-02f);
      } else {
        condval = -1.000000e+30f;
      }
      tile_storage_44_shared[((int)threadIdx)] = condval;
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      thread float n_74_0_lane_carry[1];
      n_74_0_lane_carry[0] = -INFINITY;
      n_74_0_lane_carry[0] = max(n_74_0_lane_carry[0], tile_storage_44_shared[((int)threadIdx)]);
      float n_74_0_subgroup_value = simd_max(n_74_0_lane_carry[0]);
      if (((int)threadIdx) == 0) {
        tile_storage_49_shared[0] = n_74_0_subgroup_value;
      }
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      tile_storage_55_shared[0] = max(tile_storage_5_shared[0], tile_storage_49_shared[0]);
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      tile_storage_59_shared[0] = exp((tile_storage_5_shared[0] - tile_storage_55_shared[0]));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      float condval_1;
      if (tile_storage_41_shared[((int)threadIdx)]) {
        condval_1 = exp((tile_storage_44_shared[((int)threadIdx)] - tile_storage_55_shared[0]));
      } else {
        condval_1 = 0.000000e+00f;
      }
      tile_storage_63_shared[((int)threadIdx)] = condval_1;
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      thread float n_106_0_lane_carry[1];
      n_106_0_lane_carry[0] = 0.000000e+00f;
      n_106_0_lane_carry[0] = (n_106_0_lane_carry[0] + tile_storage_63_shared[((int)threadIdx)]);
      float n_106_0_subgroup_value = simd_sum(n_106_0_lane_carry[0]);
      if (((int)threadIdx) == 0) {
        tile_storage_68_shared[0] = n_106_0_subgroup_value;
      }
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    for (int tile_i_75_chunk = 0; tile_i_75_chunk < 2; ++tile_i_75_chunk) {
      int cse_v6 = ((tile_i_75_chunk * 64) + ((int)threadIdx));
      tile_storage_74_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] = (tile_storage_13_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] * tile_storage_59_shared[0]);
      for (int tile_i_79 = 0; tile_i_79 < 32; ++tile_i_79) {
        tile_storage_74_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] = (tile_storage_74_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] + (tile_storage_63_shared[tile_i_79] * arg2_ptr[((((((((long)((int)blockIdx)) / (long)4) * (long)524288) + (pipeline_10 * (long)4096)) + (((long)tile_i_79) * (long)128)) + (((long)tile_i_75_chunk) * (long)64)) + ((long)((int)threadIdx)))]));
      }
    }
    if (((int)threadIdx) < 1) {
      tile_storage_86_shared[0] = ((tile_storage_9_shared[0] * tile_storage_59_shared[0]) + tile_storage_68_shared[0]);
    }
    if (((int)threadIdx) < 1) {
      tile_storage_5_shared[0] = tile_storage_55_shared[0];
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 1) {
      tile_storage_9_shared[0] = tile_storage_86_shared[0];
    }
    for (int tile_i_93_chunk = 0; tile_i_93_chunk < 2; ++tile_i_93_chunk) {
      int cse_v7 = ((tile_i_93_chunk * 64) + ((int)threadIdx));
      tile_storage_13_shared[((tile_i_93_chunk * 64) + ((int)threadIdx))] = tile_storage_74_shared[((tile_i_93_chunk * 64) + ((int)threadIdx))];
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
  }
  for (int tile_i_97_chunk = 0; tile_i_97_chunk < 2; ++tile_i_97_chunk) {
    int cse_v5 = (tile_i_97_chunk * 64);
    arg3_ptr[(((((int)blockIdx) * 128) + (tile_i_97_chunk * 64)) + ((int)threadIdx))] = (tile_storage_13_shared[((tile_i_97_chunk * 64) + ((int)threadIdx))] / tile_storage_9_shared[0]);
  }
  metal::threadgroup_barrier(metal::mem_flags(3));
}



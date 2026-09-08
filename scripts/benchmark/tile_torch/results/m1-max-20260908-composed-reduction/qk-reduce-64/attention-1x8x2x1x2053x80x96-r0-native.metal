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
  threadgroup float tile_storage_13_shared[96];
  for (int tile_i_14_chunk = 0; tile_i_14_chunk < 2; ++tile_i_14_chunk) {
    int cse_v7 = ((tile_i_14_chunk * 64) + ((int)threadIdx));
    if (((tile_i_14_chunk * 64) + ((int)threadIdx)) < 96) {
      tile_storage_13_shared[((tile_i_14_chunk * 64) + ((int)threadIdx))] = 0.000000e+00f;
    }
  }
  metal::threadgroup_barrier(metal::mem_flags(3));
  for (long pipeline_10 = (long)0; pipeline_10 < (long)65; ++pipeline_10) {
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
    threadgroup float tile_storage_74_shared[96];
    threadgroup float tile_storage_86_shared[1];
    long cse_v2 = (pipeline_10 * (long)32);
    long cse_v4 = ((long)((int)threadIdx));
    long cse_v9 = (((long)((int)blockIdx)) / (long)4);
    for (int tile_i_29_reduction_batch = 0; tile_i_29_reduction_batch < 16; ++tile_i_29_reduction_batch) {
      thread float d_30_0_lane_carry[1];
      d_30_0_lane_carry[0] = 0.000000e+00f;
      int cse_v1 = (((int)threadIdx) & 31);
      for (int d_30_0_reduction_chunk = 0; d_30_0_reduction_chunk < 3; ++d_30_0_reduction_chunk) {
        if (((d_30_0_reduction_chunk * 2) + ((((int)threadIdx) & 31) >> 4)) < 5) {
          long cse_v3 = ((long)tile_i_29_reduction_batch);
          long cse_v8 = (((long)((int)threadIdx)) >> (long)5);
          float condval;
          if (((((pipeline_10 * (long)32) + (((long)tile_i_29_reduction_batch) * (long)2)) + (((long)((int)threadIdx)) >> (long)5)) < (long)2053)) {
            condval = arg1_ptr[(((((((((long)((int)blockIdx)) / (long)4) * (long)164240) + (pipeline_10 * (long)2560)) + (((long)tile_i_29_reduction_batch) * (long)160)) + ((((long)((int)threadIdx)) >> (long)5) * (long)80)) + (((long)d_30_0_reduction_chunk) * (long)32)) + (((long)((int)threadIdx)) & (long)31))];
          } else {
            condval = 0.000000e+00f;
          }
          d_30_0_lane_carry[0] = (d_30_0_lane_carry[0] + (arg0_ptr[(((((int)blockIdx) * 80) + (d_30_0_reduction_chunk * 32)) + (((int)threadIdx) & 31))] * condval));
        }
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
      tile_storage_41_shared[((int)threadIdx)] = ((((pipeline_10 * (long)32) + tile_storage_35_shared[((int)threadIdx)]) < (long)2053) && (((pipeline_10 * (long)32) + tile_storage_37_shared[((int)threadIdx)]) <= (tile_storage_39_shared[0] + (long)2052)));
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
    if (((int)threadIdx) < 32) {
      float condval_1;
      if (tile_storage_41_shared[((int)threadIdx)]) {
        condval_1 = (tile_storage_28_shared[((int)threadIdx)] * 1.118034e-01f);
      } else {
        condval_1 = -1.000000e+30f;
      }
      tile_storage_44_shared[((int)threadIdx)] = condval_1;
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
      float condval_2;
      if (tile_storage_41_shared[((int)threadIdx)]) {
        condval_2 = exp((tile_storage_44_shared[((int)threadIdx)] - tile_storage_55_shared[0]));
      } else {
        condval_2 = 0.000000e+00f;
      }
      tile_storage_63_shared[((int)threadIdx)] = condval_2;
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
      int cse_v10 = ((tile_i_75_chunk * 64) + ((int)threadIdx));
      if (((tile_i_75_chunk * 64) + ((int)threadIdx)) < 96) {
        tile_storage_74_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] = (tile_storage_13_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] * tile_storage_59_shared[0]);
        for (int tile_i_79 = 0; tile_i_79 < 32; ++tile_i_79) {
          long cse_v5 = ((long)tile_i_79);
          float condval_3;
          if ((((pipeline_10 * (long)32) + ((long)tile_i_79)) < (long)2053)) {
            condval_3 = arg2_ptr[((((((((long)((int)blockIdx)) / (long)4) * (long)197088) + (pipeline_10 * (long)3072)) + (((long)tile_i_79) * (long)96)) + (((long)tile_i_75_chunk) * (long)64)) + ((long)((int)threadIdx)))];
          } else {
            condval_3 = 0.000000e+00f;
          }
          tile_storage_74_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] = (tile_storage_74_shared[((tile_i_75_chunk * 64) + ((int)threadIdx))] + (tile_storage_63_shared[tile_i_79] * condval_3));
        }
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
      int cse_v11 = ((tile_i_93_chunk * 64) + ((int)threadIdx));
      if (((tile_i_93_chunk * 64) + ((int)threadIdx)) < 96) {
        tile_storage_13_shared[((tile_i_93_chunk * 64) + ((int)threadIdx))] = tile_storage_74_shared[((tile_i_93_chunk * 64) + ((int)threadIdx))];
      }
    }
    metal::threadgroup_barrier(metal::mem_flags(3));
  }
  for (int tile_i_97_chunk = 0; tile_i_97_chunk < 2; ++tile_i_97_chunk) {
    int cse_v6 = (tile_i_97_chunk * 64);
    int cse_v12 = ((tile_i_97_chunk * 64) + ((int)threadIdx));
    if (((tile_i_97_chunk * 64) + ((int)threadIdx)) < 96) {
      arg3_ptr[(((((int)blockIdx) * 96) + (tile_i_97_chunk * 64)) + ((int)threadIdx))] = (tile_storage_13_shared[((tile_i_97_chunk * 64) + ((int)threadIdx))] / tile_storage_9_shared[0]);
    }
  }
  metal::threadgroup_barrier(metal::mem_flags(3));
}



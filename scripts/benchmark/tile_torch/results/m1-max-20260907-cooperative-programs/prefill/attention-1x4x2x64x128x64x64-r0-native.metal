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
  threadgroup float tile_storage_0_shared[512];
  long cse_v1 = (((long)blockIdx) * (long)512);
  long cse_v2 = ((long)((int)threadIdx));
  for (int tile_i_1_chunk = 0; tile_i_1_chunk < 8; ++tile_i_1_chunk) {
    tile_storage_0_shared[((tile_i_1_chunk * 64) + ((int)threadIdx))] = arg0_ptr[(((((long)blockIdx) * (long)512) + (((long)tile_i_1_chunk) * (long)64)) + ((long)((int)threadIdx)))];
  }
  threadgroup float tile_storage_5_shared[8];
  if (((int)threadIdx) < 8) {
    tile_storage_5_shared[((int)threadIdx)] = -1.000000e+30f;
  }
  threadgroup float tile_storage_9_shared[8];
  if (((int)threadIdx) < 8) {
    tile_storage_9_shared[((int)threadIdx)] = 0.000000e+00f;
  }
  threadgroup float tile_storage_13_shared[512];
  for (int tile_i_14_chunk = 0; tile_i_14_chunk < 8; ++tile_i_14_chunk) {
    tile_storage_13_shared[((tile_i_14_chunk * 64) + ((int)threadIdx))] = 0.000000e+00f;
  }
  threadgroup float tile_storage_18_shared[2048];
  threadgroup float tile_storage_23_shared[2048];
  threadgroup float tile_storage_28_shared[128];
  threadgroup long tile_storage_34_shared[16];
  threadgroup long tile_storage_36_shared[16];
  threadgroup long tile_storage_38_shared[8];
  threadgroup bool tile_storage_40_shared[128];
  threadgroup float tile_storage_43_shared[128];
  threadgroup float tile_storage_48_shared[8];
  threadgroup float tile_storage_54_shared[8];
  threadgroup float tile_storage_58_shared[8];
  threadgroup float tile_storage_62_shared[128];
  threadgroup float tile_storage_67_shared[8];
  threadgroup float tile_storage_73_shared[512];
  threadgroup float tile_storage_85_shared[8];
  long cse_v28 = (((((long)blockIdx) >> (long)3) / (long)2) * (long)8192);
  for (int tile_i_19_chunk = 0; tile_i_19_chunk < 16; ++tile_i_19_chunk) {
    tile_storage_18_shared[((tile_i_19_chunk * 64) + ((int)threadIdx))] = arg1_ptr[(((((((long)blockIdx) >> (long)3) / (long)2) * (long)8192) + (((long)tile_i_19_chunk) * (long)64)) + ((long)((int)threadIdx)))];
  }
  for (int tile_i_24_chunk = 0; tile_i_24_chunk < 16; ++tile_i_24_chunk) {
    tile_storage_23_shared[((tile_i_24_chunk * 64) + ((int)threadIdx))] = arg2_ptr[(((((((long)blockIdx) >> (long)3) / (long)2) * (long)8192) + (((long)tile_i_24_chunk) * (long)64)) + ((long)((int)threadIdx)))];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  int cse_v3 = (((int)threadIdx) >> 5);
  int cse_v5 = (((int)threadIdx) >> 3);
  int cse_v6 = (((int)threadIdx) & 7);
  int cse_v7 = (((int)threadIdx) >> 4);
  int cse_v8 = (((int)threadIdx) * 16);
  int cse_v15 = ((((int)threadIdx) >> 5) * 8);
  long cse_v17 = ((((long)blockIdx) & (long)7) * (long)8);
  int cse_v18 = ((((int)threadIdx) & 15) * 8);
  for (long pipeline_10 = (long)0; pipeline_10 < (long)7; ++pipeline_10) {
    long cse_v29 = (((pipeline_10 + (long)1) & (long)1) * (long)1024);
    long cse_v30 = ((((((long)blockIdx) >> (long)3) / (long)2) * (long)8192) + (pipeline_10 * (long)1024));
    for (int tile_i_19_chunk_1 = 0; tile_i_19_chunk_1 < 16; ++tile_i_19_chunk_1) {
      long cse_v12 = (((long)tile_i_19_chunk_1) * (long)64);
      tile_storage_18_shared[(((((pipeline_10 + (long)1) & (long)1) * (long)1024) + (((long)tile_i_19_chunk_1) * (long)64)) + ((long)((int)threadIdx)))] = arg1_ptr[(((((((((long)blockIdx) >> (long)3) / (long)2) * (long)8192) + (pipeline_10 * (long)1024)) + (((long)tile_i_19_chunk_1) * (long)64)) + ((long)((int)threadIdx))) + (long)1024)];
    }
    for (int tile_i_24_chunk_1 = 0; tile_i_24_chunk_1 < 16; ++tile_i_24_chunk_1) {
      long cse_v13 = (((long)tile_i_24_chunk_1) * (long)64);
      tile_storage_23_shared[(((((pipeline_10 + (long)1) & (long)1) * (long)1024) + (((long)tile_i_24_chunk_1) * (long)64)) + ((long)((int)threadIdx)))] = arg2_ptr[(((((((((long)blockIdx) >> (long)3) / (long)2) * (long)8192) + (pipeline_10 * (long)1024)) + (((long)tile_i_24_chunk_1) * (long)64)) + ((long)((int)threadIdx))) + (long)1024)];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    simdgroup_float8x8 tile_i_31_mma_c[1];
    tile_i_31_mma_c[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
    simdgroup_float8x8 tile_i_31_mma_a[1];
    simdgroup_float8x8 tile_i_31_mma_b[1];
    long cse_v14 = ((pipeline_10 & (long)1) * (long)1024);
    for (int tile_i_31_mma_k = 0; tile_i_31_mma_k < 8; ++tile_i_31_mma_k) {
      simdgroup_load(tile_i_31_mma_a[0], (&(tile_storage_0_shared[(tile_i_31_mma_k * 8)])), 64, 0, (bool)0);
      simdgroup_load(tile_i_31_mma_b[0], (&(tile_storage_18_shared[((((pipeline_10 & (long)1) * (long)1024) + ((((long)((int)threadIdx)) >> (long)5) * (long)512)) + (((long)tile_i_31_mma_k) * (long)8))])), 64, 0, (bool)1);
      simdgroup_multiply_accumulate(tile_i_31_mma_c[0], tile_i_31_mma_a[0], tile_i_31_mma_b[0], tile_i_31_mma_c[0]);
    }
    simdgroup_store(tile_i_31_mma_c[0], (&(tile_storage_28_shared[((((int)threadIdx) >> 5) * 8)])), 16, 0, (bool)0);
    metal::threadgroup_barrier(metal::mem_flags(2));
    if (((int)threadIdx) < 16) {
      tile_storage_34_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    if (((int)threadIdx) < 16) {
      tile_storage_36_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    if (((int)threadIdx) < 8) {
      tile_storage_38_shared[((int)threadIdx)] = ((long)((int)threadIdx));
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    for (int tile_i_41_chunk = 0; tile_i_41_chunk < 2; ++tile_i_41_chunk) {
      long cse_v4 = (pipeline_10 * (long)16);
      int cse_v16 = ((tile_i_41_chunk * 8) + (((int)threadIdx) >> 3));
      tile_storage_40_shared[((tile_i_41_chunk * 64) + ((int)threadIdx))] = ((((pipeline_10 * (long)16) + tile_storage_34_shared[((tile_i_41_chunk * 8) + (((int)threadIdx) >> 3))]) < (long)128) && (((pipeline_10 * (long)16) + tile_storage_36_shared[((tile_i_41_chunk * 8) + (((int)threadIdx) >> 3))]) <= ((((((long)blockIdx) & (long)7) * (long)8) + tile_storage_38_shared[(((int)threadIdx) & 7)]) + (long)64)));
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    for (int tile_i_44_chunk = 0; tile_i_44_chunk < 2; ++tile_i_44_chunk) {
      int cse_v19 = ((tile_i_44_chunk * 64) + ((int)threadIdx));
      float condval;
      if (tile_storage_40_shared[((((((int)threadIdx) & 15) * 8) + (tile_i_44_chunk * 4)) + (((int)threadIdx) >> 4))]) {
        condval = (tile_storage_28_shared[((tile_i_44_chunk * 64) + ((int)threadIdx))] * 1.250000e-01f);
      } else {
        condval = -1.000000e+30f;
      }
      tile_storage_43_shared[((tile_i_44_chunk * 64) + ((int)threadIdx))] = condval;
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    if (((int)threadIdx) < 8) {
      thread float tile_storage_52[1];
      tile_storage_52[0] = -INFINITY;
      for (int n_55_0 = 0; n_55_0 < 16; ++n_55_0) {
        thread float tile_storage_53[1];
        tile_storage_53[0] = max(tile_storage_52[0], tile_storage_43_shared[((((int)threadIdx) * 16) + n_55_0)]);
        tile_storage_52[0] = tile_storage_53[0];
      }
      tile_storage_48_shared[((int)threadIdx)] = tile_storage_52[0];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    if (((int)threadIdx) < 8) {
      tile_storage_54_shared[((int)threadIdx)] = max(tile_storage_5_shared[((int)threadIdx)], tile_storage_48_shared[((int)threadIdx)]);
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    if (((int)threadIdx) < 8) {
      tile_storage_58_shared[((int)threadIdx)] = exp((tile_storage_5_shared[((int)threadIdx)] - tile_storage_54_shared[((int)threadIdx)]));
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    for (int tile_i_63_chunk = 0; tile_i_63_chunk < 2; ++tile_i_63_chunk) {
      int cse_v9 = (tile_i_63_chunk * 4);
      int cse_v20 = ((tile_i_63_chunk * 64) + ((int)threadIdx));
      float condval_1;
      if (tile_storage_40_shared[((((((int)threadIdx) & 15) * 8) + (tile_i_63_chunk * 4)) + (((int)threadIdx) >> 4))]) {
        condval_1 = exp((tile_storage_43_shared[((tile_i_63_chunk * 64) + ((int)threadIdx))] - tile_storage_54_shared[((tile_i_63_chunk * 4) + (((int)threadIdx) >> 4))]));
      } else {
        condval_1 = 0.000000e+00f;
      }
      tile_storage_62_shared[((tile_i_63_chunk * 64) + ((int)threadIdx))] = condval_1;
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    if (((int)threadIdx) < 8) {
      thread float tile_storage_71[1];
      tile_storage_71[0] = 0.000000e+00f;
      for (int n_86_0 = 0; n_86_0 < 16; ++n_86_0) {
        thread float tile_storage_72[1];
        tile_storage_72[0] = (tile_storage_71[0] + tile_storage_62_shared[((((int)threadIdx) * 16) + n_86_0)]);
        tile_storage_71[0] = tile_storage_72[0];
      }
      tile_storage_67_shared[((int)threadIdx)] = tile_storage_71[0];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    for (int tile_i_74_chunk = 0; tile_i_74_chunk < 8; ++tile_i_74_chunk) {
      int cse_v21 = ((tile_i_74_chunk * 64) + ((int)threadIdx));
      tile_storage_73_shared[((tile_i_74_chunk * 64) + ((int)threadIdx))] = (tile_storage_13_shared[((tile_i_74_chunk * 64) + ((int)threadIdx))] * tile_storage_58_shared[tile_i_74_chunk]);
      for (int tile_i_78 = 0; tile_i_78 < 16; ++tile_i_78) {
        tile_storage_73_shared[((tile_i_74_chunk * 64) + ((int)threadIdx))] = (tile_storage_73_shared[((tile_i_74_chunk * 64) + ((int)threadIdx))] + (tile_storage_62_shared[((tile_i_74_chunk * 16) + tile_i_78)] * tile_storage_23_shared[((((pipeline_10 & (long)1) * (long)1024) + (((long)tile_i_78) * (long)64)) + ((long)((int)threadIdx)))]));
      }
    }
    if (((int)threadIdx) < 8) {
      tile_storage_85_shared[((int)threadIdx)] = ((tile_storage_9_shared[((int)threadIdx)] * tile_storage_58_shared[((int)threadIdx)]) + tile_storage_67_shared[((int)threadIdx)]);
    }
    if (((int)threadIdx) < 8) {
      tile_storage_5_shared[((int)threadIdx)] = tile_storage_54_shared[((int)threadIdx)];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
    if (((int)threadIdx) < 8) {
      tile_storage_9_shared[((int)threadIdx)] = tile_storage_85_shared[((int)threadIdx)];
    }
    for (int tile_i_92_chunk = 0; tile_i_92_chunk < 8; ++tile_i_92_chunk) {
      int cse_v22 = ((tile_i_92_chunk * 64) + ((int)threadIdx));
      tile_storage_13_shared[((tile_i_92_chunk * 64) + ((int)threadIdx))] = tile_storage_73_shared[((tile_i_92_chunk * 64) + ((int)threadIdx))];
    }
    metal::threadgroup_barrier(metal::mem_flags(2));
  }
  simdgroup_float8x8 tile_i_31_mma_c_1[1];
  tile_i_31_mma_c_1[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  simdgroup_float8x8 tile_i_31_mma_a_1[1];
  simdgroup_float8x8 tile_i_31_mma_b_1[1];
  for (int tile_i_31_mma_k_1 = 0; tile_i_31_mma_k_1 < 8; ++tile_i_31_mma_k_1) {
    int cse_v10 = (tile_i_31_mma_k_1 * 8);
    simdgroup_load(tile_i_31_mma_a_1[0], (&(tile_storage_0_shared[(tile_i_31_mma_k_1 * 8)])), 64, 0, (bool)0);
    simdgroup_load(tile_i_31_mma_b_1[0], (&(tile_storage_18_shared[((((((int)threadIdx) >> 5) * 512) + (tile_i_31_mma_k_1 * 8)) + 1024)])), 64, 0, (bool)1);
    simdgroup_multiply_accumulate(tile_i_31_mma_c_1[0], tile_i_31_mma_a_1[0], tile_i_31_mma_b_1[0], tile_i_31_mma_c_1[0]);
  }
  simdgroup_store(tile_i_31_mma_c_1[0], (&(tile_storage_28_shared[((((int)threadIdx) >> 5) * 8)])), 16, 0, (bool)0);
  metal::threadgroup_barrier(metal::mem_flags(2));
  if (((int)threadIdx) < 16) {
    tile_storage_34_shared[((int)threadIdx)] = ((long)((int)threadIdx));
  }
  if (((int)threadIdx) < 16) {
    tile_storage_36_shared[((int)threadIdx)] = ((long)((int)threadIdx));
  }
  if (((int)threadIdx) < 8) {
    tile_storage_38_shared[((int)threadIdx)] = ((long)((int)threadIdx));
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_41_chunk_1 = 0; tile_i_41_chunk_1 < 2; ++tile_i_41_chunk_1) {
    int cse_v23 = ((tile_i_41_chunk_1 * 8) + (((int)threadIdx) >> 3));
    tile_storage_40_shared[((tile_i_41_chunk_1 * 64) + ((int)threadIdx))] = ((tile_storage_34_shared[((tile_i_41_chunk_1 * 8) + (((int)threadIdx) >> 3))] < (long)16) && ((tile_storage_36_shared[((tile_i_41_chunk_1 * 8) + (((int)threadIdx) >> 3))] + (long)48) <= (((((long)blockIdx) & (long)7) * (long)8) + tile_storage_38_shared[(((int)threadIdx) & 7)])));
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_44_chunk_1 = 0; tile_i_44_chunk_1 < 2; ++tile_i_44_chunk_1) {
    int cse_v24 = ((tile_i_44_chunk_1 * 64) + ((int)threadIdx));
    float condval_2;
    if (tile_storage_40_shared[((((((int)threadIdx) & 15) * 8) + (tile_i_44_chunk_1 * 4)) + (((int)threadIdx) >> 4))]) {
      condval_2 = (tile_storage_28_shared[((tile_i_44_chunk_1 * 64) + ((int)threadIdx))] * 1.250000e-01f);
    } else {
      condval_2 = -1.000000e+30f;
    }
    tile_storage_43_shared[((tile_i_44_chunk_1 * 64) + ((int)threadIdx))] = condval_2;
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  if (((int)threadIdx) < 8) {
    thread float tile_storage_52_1[1];
    tile_storage_52_1[0] = -INFINITY;
    for (int n_55_0_1 = 0; n_55_0_1 < 16; ++n_55_0_1) {
      thread float tile_storage_53_1[1];
      tile_storage_53_1[0] = max(tile_storage_52_1[0], tile_storage_43_shared[((((int)threadIdx) * 16) + n_55_0_1)]);
      tile_storage_52_1[0] = tile_storage_53_1[0];
    }
    tile_storage_48_shared[((int)threadIdx)] = tile_storage_52_1[0];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  if (((int)threadIdx) < 8) {
    tile_storage_54_shared[((int)threadIdx)] = max(tile_storage_5_shared[((int)threadIdx)], tile_storage_48_shared[((int)threadIdx)]);
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  if (((int)threadIdx) < 8) {
    tile_storage_58_shared[((int)threadIdx)] = exp((tile_storage_5_shared[((int)threadIdx)] - tile_storage_54_shared[((int)threadIdx)]));
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_63_chunk_1 = 0; tile_i_63_chunk_1 < 2; ++tile_i_63_chunk_1) {
    int cse_v11 = (tile_i_63_chunk_1 * 4);
    int cse_v25 = ((tile_i_63_chunk_1 * 64) + ((int)threadIdx));
    float condval_3;
    if (tile_storage_40_shared[((((((int)threadIdx) & 15) * 8) + (tile_i_63_chunk_1 * 4)) + (((int)threadIdx) >> 4))]) {
      condval_3 = exp((tile_storage_43_shared[((tile_i_63_chunk_1 * 64) + ((int)threadIdx))] - tile_storage_54_shared[((tile_i_63_chunk_1 * 4) + (((int)threadIdx) >> 4))]));
    } else {
      condval_3 = 0.000000e+00f;
    }
    tile_storage_62_shared[((tile_i_63_chunk_1 * 64) + ((int)threadIdx))] = condval_3;
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  if (((int)threadIdx) < 8) {
    thread float tile_storage_71_1[1];
    tile_storage_71_1[0] = 0.000000e+00f;
    for (int n_86_0_1 = 0; n_86_0_1 < 16; ++n_86_0_1) {
      thread float tile_storage_72_1[1];
      tile_storage_72_1[0] = (tile_storage_71_1[0] + tile_storage_62_shared[((((int)threadIdx) * 16) + n_86_0_1)]);
      tile_storage_71_1[0] = tile_storage_72_1[0];
    }
    tile_storage_67_shared[((int)threadIdx)] = tile_storage_71_1[0];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_74_chunk_1 = 0; tile_i_74_chunk_1 < 8; ++tile_i_74_chunk_1) {
    int cse_v26 = ((tile_i_74_chunk_1 * 64) + ((int)threadIdx));
    tile_storage_73_shared[((tile_i_74_chunk_1 * 64) + ((int)threadIdx))] = (tile_storage_13_shared[((tile_i_74_chunk_1 * 64) + ((int)threadIdx))] * tile_storage_58_shared[tile_i_74_chunk_1]);
    for (int tile_i_78_1 = 0; tile_i_78_1 < 16; ++tile_i_78_1) {
      tile_storage_73_shared[((tile_i_74_chunk_1 * 64) + ((int)threadIdx))] = (tile_storage_73_shared[((tile_i_74_chunk_1 * 64) + ((int)threadIdx))] + (tile_storage_62_shared[((tile_i_74_chunk_1 * 16) + tile_i_78_1)] * tile_storage_23_shared[(((tile_i_78_1 * 64) + ((int)threadIdx)) + 1024)]));
    }
  }
  if (((int)threadIdx) < 8) {
    tile_storage_85_shared[((int)threadIdx)] = ((tile_storage_9_shared[((int)threadIdx)] * tile_storage_58_shared[((int)threadIdx)]) + tile_storage_67_shared[((int)threadIdx)]);
  }
  if (((int)threadIdx) < 8) {
    tile_storage_5_shared[((int)threadIdx)] = tile_storage_54_shared[((int)threadIdx)];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  if (((int)threadIdx) < 8) {
    tile_storage_9_shared[((int)threadIdx)] = tile_storage_85_shared[((int)threadIdx)];
  }
  for (int tile_i_92_chunk_1 = 0; tile_i_92_chunk_1 < 8; ++tile_i_92_chunk_1) {
    int cse_v27 = ((tile_i_92_chunk_1 * 64) + ((int)threadIdx));
    tile_storage_13_shared[((tile_i_92_chunk_1 * 64) + ((int)threadIdx))] = tile_storage_73_shared[((tile_i_92_chunk_1 * 64) + ((int)threadIdx))];
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
  for (int tile_i_96_chunk = 0; tile_i_96_chunk < 8; ++tile_i_96_chunk) {
    arg3_ptr[(((((long)blockIdx) * (long)512) + (((long)tile_i_96_chunk) * (long)64)) + ((long)((int)threadIdx)))] = (tile_storage_13_shared[((tile_i_96_chunk * 64) + ((int)threadIdx))] / tile_storage_9_shared[tile_i_96_chunk]);
  }
  metal::threadgroup_barrier(metal::mem_flags(2));
}



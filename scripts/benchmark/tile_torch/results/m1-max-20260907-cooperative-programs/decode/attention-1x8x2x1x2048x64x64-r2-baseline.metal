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
  thread float tile_storage_0[64];
  int cse_v1 = (((int)threadIdx) * 64);
  for (int tile_i_4 = 0; tile_i_4 < 64; ++tile_i_4) {
    tile_storage_0[tile_i_4] = arg0_ptr[((((int)threadIdx) * 64) + tile_i_4)];
  }
  thread float tile_storage_5[1];
  tile_storage_5[0] = -1.000000e+30f;
  thread float tile_storage_9[1];
  tile_storage_9[0] = 0.000000e+00f;
  thread float tile_storage_13[64];
  for (int tile_i_17 = 0; tile_i_17 < 64; ++tile_i_17) {
    tile_storage_13[tile_i_17] = 0.000000e+00f;
  }
  thread float tile_storage_18[4096];
  thread float tile_storage_23[4096];
  thread float tile_storage_28[32];
  thread long tile_storage_34[32];
  thread long tile_storage_36[32];
  thread long tile_storage_38[1];
  thread bool tile_storage_40[32];
  thread float tile_storage_43[32];
  thread float tile_storage_48[1];
  thread float tile_storage_54[1];
  thread float tile_storage_58[1];
  thread float tile_storage_62[32];
  thread float tile_storage_67[1];
  thread float tile_storage_73[64];
  thread float tile_storage_85[1];
  int cse_v7 = ((((int)threadIdx) / 4) * 131072);
  for (int tile_i_21 = 0; tile_i_21 < 32; ++tile_i_21) {
    for (int tile_i_22 = 0; tile_i_22 < 64; ++tile_i_22) {
      int cse_v2 = (tile_i_21 * 64);
      tile_storage_18[((tile_i_21 * 64) + tile_i_22)] = arg1_ptr[((((((int)threadIdx) / 4) * 131072) + (tile_i_21 * 64)) + tile_i_22)];
    }
  }
  for (int tile_i_26 = 0; tile_i_26 < 32; ++tile_i_26) {
    for (int tile_i_27 = 0; tile_i_27 < 64; ++tile_i_27) {
      int cse_v3 = (tile_i_26 * 64);
      tile_storage_23[((tile_i_26 * 64) + tile_i_27)] = arg2_ptr[((((((int)threadIdx) / 4) * 131072) + (tile_i_26 * 64)) + tile_i_27)];
    }
  }
  for (long pipeline_10 = (long)0; pipeline_10 < (long)63; ++pipeline_10) {
    long cse_v11 = (((pipeline_10 + (long)1) & (long)1) * (long)2048);
    long cse_v12 = (((((long)((int)threadIdx)) / (long)4) * (long)131072) + (pipeline_10 * (long)2048));
    for (int tile_i_21_1 = 0; tile_i_21_1 < 32; ++tile_i_21_1) {
      for (int tile_i_22_1 = 0; tile_i_22_1 < 64; ++tile_i_22_1) {
        long cse_v4 = ((long)tile_i_22_1);
        long cse_v8 = (((long)tile_i_21_1) * (long)64);
        tile_storage_18[(((((pipeline_10 + (long)1) & (long)1) * (long)2048) + (((long)tile_i_21_1) * (long)64)) + ((long)tile_i_22_1))] = arg1_ptr[((((((((long)((int)threadIdx)) / (long)4) * (long)131072) + (pipeline_10 * (long)2048)) + (((long)tile_i_21_1) * (long)64)) + ((long)tile_i_22_1)) + (long)2048)];
      }
    }
    for (int tile_i_26_1 = 0; tile_i_26_1 < 32; ++tile_i_26_1) {
      for (int tile_i_27_1 = 0; tile_i_27_1 < 64; ++tile_i_27_1) {
        long cse_v5 = ((long)tile_i_27_1);
        long cse_v9 = (((long)tile_i_26_1) * (long)64);
        tile_storage_23[(((((pipeline_10 + (long)1) & (long)1) * (long)2048) + (((long)tile_i_26_1) * (long)64)) + ((long)tile_i_27_1))] = arg2_ptr[((((((((long)((int)threadIdx)) / (long)4) * (long)131072) + (pipeline_10 * (long)2048)) + (((long)tile_i_26_1) * (long)64)) + ((long)tile_i_27_1)) + (long)2048)];
      }
    }
    long cse_v10 = ((pipeline_10 & (long)1) * (long)2048);
    for (int tile_i_32 = 0; tile_i_32 < 32; ++tile_i_32) {
      tile_storage_28[tile_i_32] = 0.000000e+00f;
      for (int tile_i_33 = 0; tile_i_33 < 64; ++tile_i_33) {
        tile_storage_28[tile_i_32] = (tile_storage_28[tile_i_32] + (tile_storage_0[tile_i_33] * tile_storage_18[((((pipeline_10 & (long)1) * (long)2048) + (((long)tile_i_32) * (long)64)) + ((long)tile_i_33))]));
      }
    }
    for (int tile_i_35 = 0; tile_i_35 < 32; ++tile_i_35) {
      tile_storage_34[tile_i_35] = ((long)tile_i_35);
    }
    for (int tile_i_37 = 0; tile_i_37 < 32; ++tile_i_37) {
      tile_storage_36[tile_i_37] = ((long)tile_i_37);
    }
    tile_storage_38[0] = (long)0;
    for (int tile_i_41 = 0; tile_i_41 < 32; ++tile_i_41) {
      long cse_v6 = (pipeline_10 * (long)32);
      tile_storage_40[tile_i_41] = ((((pipeline_10 * (long)32) + tile_storage_34[tile_i_41]) < (long)2048) && (((pipeline_10 * (long)32) + tile_storage_36[tile_i_41]) <= (tile_storage_38[0] + (long)2047)));
    }
    for (int tile_i_47 = 0; tile_i_47 < 32; ++tile_i_47) {
      float condval;
      if (tile_storage_40[tile_i_47]) {
        condval = (tile_storage_28[tile_i_47] * 1.250000e-01f);
      } else {
        condval = -1.000000e+30f;
      }
      tile_storage_43[tile_i_47] = condval;
    }
    thread float tile_storage_52[1];
    tile_storage_52[0] = -INFINITY;
    for (int n_55_0 = 0; n_55_0 < 32; ++n_55_0) {
      thread float tile_storage_53[1];
      tile_storage_53[0] = max(tile_storage_52[0], tile_storage_43[n_55_0]);
      tile_storage_52[0] = tile_storage_53[0];
    }
    tile_storage_48[0] = tile_storage_52[0];
    tile_storage_54[0] = max(tile_storage_5[0], tile_storage_48[0]);
    tile_storage_58[0] = exp((tile_storage_5[0] - tile_storage_54[0]));
    for (int tile_i_66 = 0; tile_i_66 < 32; ++tile_i_66) {
      float condval_1;
      if (tile_storage_40[tile_i_66]) {
        condval_1 = exp((tile_storage_43[tile_i_66] - tile_storage_54[0]));
      } else {
        condval_1 = 0.000000e+00f;
      }
      tile_storage_62[tile_i_66] = condval_1;
    }
    thread float tile_storage_71[1];
    tile_storage_71[0] = 0.000000e+00f;
    for (int n_86_0 = 0; n_86_0 < 32; ++n_86_0) {
      thread float tile_storage_72[1];
      tile_storage_72[0] = (tile_storage_71[0] + tile_storage_62[n_86_0]);
      tile_storage_71[0] = tile_storage_72[0];
    }
    tile_storage_67[0] = tile_storage_71[0];
    for (int tile_i_77 = 0; tile_i_77 < 64; ++tile_i_77) {
      tile_storage_73[tile_i_77] = (tile_storage_13[tile_i_77] * tile_storage_58[0]);
      for (int tile_i_78 = 0; tile_i_78 < 32; ++tile_i_78) {
        tile_storage_73[tile_i_77] = (tile_storage_73[tile_i_77] + (tile_storage_62[tile_i_78] * tile_storage_23[((((pipeline_10 & (long)1) * (long)2048) + (((long)tile_i_78) * (long)64)) + ((long)tile_i_77))]));
      }
    }
    tile_storage_85[0] = ((tile_storage_9[0] * tile_storage_58[0]) + tile_storage_67[0]);
    tile_storage_5[0] = tile_storage_54[0];
    tile_storage_9[0] = tile_storage_85[0];
    for (int tile_i_95 = 0; tile_i_95 < 64; ++tile_i_95) {
      tile_storage_13[tile_i_95] = tile_storage_73[tile_i_95];
    }
  }
  for (int tile_i_32_1 = 0; tile_i_32_1 < 32; ++tile_i_32_1) {
    tile_storage_28[tile_i_32_1] = 0.000000e+00f;
    for (int tile_i_33_1 = 0; tile_i_33_1 < 64; ++tile_i_33_1) {
      tile_storage_28[tile_i_32_1] = (tile_storage_28[tile_i_32_1] + (tile_storage_0[tile_i_33_1] * tile_storage_18[(((tile_i_32_1 * 64) + tile_i_33_1) + 2048)]));
    }
  }
  for (int tile_i_35_1 = 0; tile_i_35_1 < 32; ++tile_i_35_1) {
    tile_storage_34[tile_i_35_1] = ((long)tile_i_35_1);
  }
  for (int tile_i_37_1 = 0; tile_i_37_1 < 32; ++tile_i_37_1) {
    tile_storage_36[tile_i_37_1] = ((long)tile_i_37_1);
  }
  tile_storage_38[0] = (long)0;
  for (int tile_i_41_1 = 0; tile_i_41_1 < 32; ++tile_i_41_1) {
    tile_storage_40[tile_i_41_1] = ((tile_storage_34[tile_i_41_1] < (long)32) && (tile_storage_36[tile_i_41_1] <= (tile_storage_38[0] + (long)31)));
  }
  for (int tile_i_47_1 = 0; tile_i_47_1 < 32; ++tile_i_47_1) {
    float condval_2;
    if (tile_storage_40[tile_i_47_1]) {
      condval_2 = (tile_storage_28[tile_i_47_1] * 1.250000e-01f);
    } else {
      condval_2 = -1.000000e+30f;
    }
    tile_storage_43[tile_i_47_1] = condval_2;
  }
  thread float tile_storage_52_1[1];
  tile_storage_52_1[0] = -INFINITY;
  for (int n_55_0_1 = 0; n_55_0_1 < 32; ++n_55_0_1) {
    thread float tile_storage_53_1[1];
    tile_storage_53_1[0] = max(tile_storage_52_1[0], tile_storage_43[n_55_0_1]);
    tile_storage_52_1[0] = tile_storage_53_1[0];
  }
  tile_storage_48[0] = tile_storage_52_1[0];
  tile_storage_54[0] = max(tile_storage_5[0], tile_storage_48[0]);
  tile_storage_58[0] = exp((tile_storage_5[0] - tile_storage_54[0]));
  for (int tile_i_66_1 = 0; tile_i_66_1 < 32; ++tile_i_66_1) {
    float condval_3;
    if (tile_storage_40[tile_i_66_1]) {
      condval_3 = exp((tile_storage_43[tile_i_66_1] - tile_storage_54[0]));
    } else {
      condval_3 = 0.000000e+00f;
    }
    tile_storage_62[tile_i_66_1] = condval_3;
  }
  thread float tile_storage_71_1[1];
  tile_storage_71_1[0] = 0.000000e+00f;
  for (int n_86_0_1 = 0; n_86_0_1 < 32; ++n_86_0_1) {
    thread float tile_storage_72_1[1];
    tile_storage_72_1[0] = (tile_storage_71_1[0] + tile_storage_62[n_86_0_1]);
    tile_storage_71_1[0] = tile_storage_72_1[0];
  }
  tile_storage_67[0] = tile_storage_71_1[0];
  for (int tile_i_77_1 = 0; tile_i_77_1 < 64; ++tile_i_77_1) {
    tile_storage_73[tile_i_77_1] = (tile_storage_13[tile_i_77_1] * tile_storage_58[0]);
    for (int tile_i_78_1 = 0; tile_i_78_1 < 32; ++tile_i_78_1) {
      tile_storage_73[tile_i_77_1] = (tile_storage_73[tile_i_77_1] + (tile_storage_62[tile_i_78_1] * tile_storage_23[(((tile_i_78_1 * 64) + tile_i_77_1) + 2048)]));
    }
  }
  tile_storage_85[0] = ((tile_storage_9[0] * tile_storage_58[0]) + tile_storage_67[0]);
  tile_storage_5[0] = tile_storage_54[0];
  tile_storage_9[0] = tile_storage_85[0];
  for (int tile_i_95_1 = 0; tile_i_95_1 < 64; ++tile_i_95_1) {
    tile_storage_13[tile_i_95_1] = tile_storage_73[tile_i_95_1];
  }
  for (int tile_i_99 = 0; tile_i_99 < 64; ++tile_i_99) {
    arg3_ptr[((((int)threadIdx) * 64) + tile_i_99)] = (tile_storage_13[tile_i_99] / tile_storage_9[0]);
  }
}



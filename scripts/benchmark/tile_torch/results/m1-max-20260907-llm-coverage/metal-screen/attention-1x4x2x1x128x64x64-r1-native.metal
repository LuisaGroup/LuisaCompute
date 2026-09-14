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
  thread float tile_storage_0[128];
  int cse_v1 = (((int)threadIdx) * 64);
  for (int tile_i_3 = 0; tile_i_3 < 2; ++tile_i_3) {
    for (int tile_i_4 = 0; tile_i_4 < 64; ++tile_i_4) {
      float condval;
      if ((tile_i_3 < 1)) {
        condval = arg0_ptr[((((int)threadIdx) * 64) + tile_i_4)];
      } else {
        condval = 0.000000e+00f;
      }
      tile_storage_0[((tile_i_3 * 64) + tile_i_4)] = condval;
    }
  }
  thread float tile_storage_5[2];
  for (int tile_i_8 = 0; tile_i_8 < 2; ++tile_i_8) {
    tile_storage_5[tile_i_8] = -1.000000e+30f;
  }
  thread float tile_storage_9[2];
  for (int tile_i_12 = 0; tile_i_12 < 2; ++tile_i_12) {
    tile_storage_9[tile_i_12] = 0.000000e+00f;
  }
  thread float tile_storage_13[128];
  for (int tile_i_16 = 0; tile_i_16 < 2; ++tile_i_16) {
    for (int tile_i_17 = 0; tile_i_17 < 64; ++tile_i_17) {
      tile_storage_13[((tile_i_16 * 64) + tile_i_17)] = 0.000000e+00f;
    }
  }
  thread float tile_storage_18[384];
  thread float tile_storage_23[384];
  thread float tile_storage_28[6];
  thread long tile_storage_34[3];
  thread long tile_storage_36[3];
  thread long tile_storage_38[2];
  thread bool tile_storage_40[6];
  thread float tile_storage_43[6];
  thread float tile_storage_48[2];
  thread float tile_storage_54[2];
  thread float tile_storage_58[2];
  thread float tile_storage_62[6];
  thread float tile_storage_67[2];
  thread float tile_storage_73[128];
  thread float tile_storage_85[2];
  int cse_v9 = ((((int)threadIdx) / 2) * 8192);
  for (int tile_i_21 = 0; tile_i_21 < 3; ++tile_i_21) {
    for (int tile_i_22 = 0; tile_i_22 < 64; ++tile_i_22) {
      int cse_v2 = (tile_i_21 * 64);
      tile_storage_18[((tile_i_21 * 64) + tile_i_22)] = arg1_ptr[((((((int)threadIdx) / 2) * 8192) + (tile_i_21 * 64)) + tile_i_22)];
    }
  }
  for (int tile_i_26 = 0; tile_i_26 < 3; ++tile_i_26) {
    for (int tile_i_27 = 0; tile_i_27 < 64; ++tile_i_27) {
      int cse_v3 = (tile_i_26 * 64);
      tile_storage_23[((tile_i_26 * 64) + tile_i_27)] = arg2_ptr[((((((int)threadIdx) / 2) * 8192) + (tile_i_26 * 64)) + tile_i_27)];
    }
  }
  for (long pipeline_10 = (long)0; pipeline_10 < (long)42; ++pipeline_10) {
    long cse_v4 = (pipeline_10 * (long)3);
    long cse_v23 = (((pipeline_10 + (long)1) & (long)1) * (long)192);
    long cse_v24 = (((((long)((int)threadIdx)) / (long)2) * (long)8192) + (pipeline_10 * (long)192));
    for (int tile_i_21_1 = 0; tile_i_21_1 < 3; ++tile_i_21_1) {
      for (int tile_i_22_1 = 0; tile_i_22_1 < 64; ++tile_i_22_1) {
        long cse_v5 = ((long)tile_i_21_1);
        long cse_v6 = ((long)tile_i_22_1);
        long cse_v10 = (((long)tile_i_21_1) * (long)64);
        float condval_1;
        if ((((pipeline_10 * (long)3) + ((long)tile_i_21_1)) < (long)125)) {
          condval_1 = arg1_ptr[((((((((long)((int)threadIdx)) / (long)2) * (long)8192) + (pipeline_10 * (long)192)) + (((long)tile_i_21_1) * (long)64)) + ((long)tile_i_22_1)) + (long)192)];
        } else {
          condval_1 = 0.000000e+00f;
        }
        tile_storage_18[(((((pipeline_10 + (long)1) & (long)1) * (long)192) + (((long)tile_i_21_1) * (long)64)) + ((long)tile_i_22_1))] = condval_1;
      }
    }
    for (int tile_i_26_1 = 0; tile_i_26_1 < 3; ++tile_i_26_1) {
      for (int tile_i_27_1 = 0; tile_i_27_1 < 64; ++tile_i_27_1) {
        long cse_v7 = ((long)tile_i_26_1);
        long cse_v8 = ((long)tile_i_27_1);
        long cse_v11 = (((long)tile_i_26_1) * (long)64);
        float condval_2;
        if ((((pipeline_10 * (long)3) + ((long)tile_i_26_1)) < (long)125)) {
          condval_2 = arg2_ptr[((((((((long)((int)threadIdx)) / (long)2) * (long)8192) + (pipeline_10 * (long)192)) + (((long)tile_i_26_1) * (long)64)) + ((long)tile_i_27_1)) + (long)192)];
        } else {
          condval_2 = 0.000000e+00f;
        }
        tile_storage_23[(((((pipeline_10 + (long)1) & (long)1) * (long)192) + (((long)tile_i_26_1) * (long)64)) + ((long)tile_i_27_1))] = condval_2;
      }
    }
    long cse_v13 = ((pipeline_10 & (long)1) * (long)192);
    for (int tile_i_31 = 0; tile_i_31 < 2; ++tile_i_31) {
      for (int tile_i_32 = 0; tile_i_32 < 3; ++tile_i_32) {
        int cse_v12 = ((tile_i_31 * 3) + tile_i_32);
        tile_storage_28[((tile_i_31 * 3) + tile_i_32)] = 0.000000e+00f;
        for (int tile_i_33 = 0; tile_i_33 < 64; ++tile_i_33) {
          tile_storage_28[((tile_i_31 * 3) + tile_i_32)] = (tile_storage_28[((tile_i_31 * 3) + tile_i_32)] + (tile_storage_0[((tile_i_31 * 64) + tile_i_33)] * tile_storage_18[((((pipeline_10 & (long)1) * (long)192) + (((long)tile_i_32) * (long)64)) + ((long)tile_i_33))]));
        }
      }
    }
    for (int tile_i_35 = 0; tile_i_35 < 3; ++tile_i_35) {
      tile_storage_34[tile_i_35] = ((long)tile_i_35);
    }
    for (int tile_i_37 = 0; tile_i_37 < 3; ++tile_i_37) {
      tile_storage_36[tile_i_37] = ((long)tile_i_37);
    }
    for (int tile_i_39 = 0; tile_i_39 < 2; ++tile_i_39) {
      tile_storage_38[tile_i_39] = ((long)tile_i_39);
    }
    for (int tile_i_41 = 0; tile_i_41 < 3; ++tile_i_41) {
      for (int tile_i_42 = 0; tile_i_42 < 2; ++tile_i_42) {
        tile_storage_40[((tile_i_41 * 2) + tile_i_42)] = ((((pipeline_10 * (long)3) + tile_storage_34[tile_i_41]) < (long)128) && (((pipeline_10 * (long)3) + tile_storage_36[tile_i_41]) <= (tile_storage_38[tile_i_42] + (long)127)));
      }
    }
    for (int tile_i_46 = 0; tile_i_46 < 2; ++tile_i_46) {
      for (int tile_i_47 = 0; tile_i_47 < 3; ++tile_i_47) {
        int cse_v14 = ((tile_i_46 * 3) + tile_i_47);
        float condval_3;
        if (tile_storage_40[((tile_i_47 * 2) + tile_i_46)]) {
          condval_3 = (tile_storage_28[((tile_i_46 * 3) + tile_i_47)] * 1.250000e-01f);
        } else {
          condval_3 = -1.000000e+30f;
        }
        tile_storage_43[((tile_i_46 * 3) + tile_i_47)] = condval_3;
      }
    }
    for (int tile_i_51 = 0; tile_i_51 < 2; ++tile_i_51) {
      thread float tile_storage_52[1];
      tile_storage_52[0] = -INFINITY;
      for (int n_55_0 = 0; n_55_0 < 3; ++n_55_0) {
        thread float tile_storage_53[1];
        tile_storage_53[0] = max(tile_storage_52[0], tile_storage_43[((tile_i_51 * 3) + n_55_0)]);
        tile_storage_52[0] = tile_storage_53[0];
      }
      tile_storage_48[tile_i_51] = tile_storage_52[0];
    }
    for (int tile_i_57 = 0; tile_i_57 < 2; ++tile_i_57) {
      tile_storage_54[tile_i_57] = max(tile_storage_5[tile_i_57], tile_storage_48[tile_i_57]);
    }
    for (int tile_i_61 = 0; tile_i_61 < 2; ++tile_i_61) {
      tile_storage_58[tile_i_61] = exp((tile_storage_5[tile_i_61] - tile_storage_54[tile_i_61]));
    }
    for (int tile_i_65 = 0; tile_i_65 < 2; ++tile_i_65) {
      for (int tile_i_66 = 0; tile_i_66 < 3; ++tile_i_66) {
        int cse_v15 = ((tile_i_65 * 3) + tile_i_66);
        float condval_4;
        if (tile_storage_40[((tile_i_66 * 2) + tile_i_65)]) {
          condval_4 = exp((tile_storage_43[((tile_i_65 * 3) + tile_i_66)] - tile_storage_54[tile_i_65]));
        } else {
          condval_4 = 0.000000e+00f;
        }
        tile_storage_62[((tile_i_65 * 3) + tile_i_66)] = condval_4;
      }
    }
    for (int tile_i_70 = 0; tile_i_70 < 2; ++tile_i_70) {
      thread float tile_storage_71[1];
      tile_storage_71[0] = 0.000000e+00f;
      for (int n_86_0 = 0; n_86_0 < 3; ++n_86_0) {
        thread float tile_storage_72[1];
        tile_storage_72[0] = (tile_storage_71[0] + tile_storage_62[((tile_i_70 * 3) + n_86_0)]);
        tile_storage_71[0] = tile_storage_72[0];
      }
      tile_storage_67[tile_i_70] = tile_storage_71[0];
    }
    for (int tile_i_76 = 0; tile_i_76 < 2; ++tile_i_76) {
      for (int tile_i_77 = 0; tile_i_77 < 64; ++tile_i_77) {
        int cse_v16 = ((tile_i_76 * 64) + tile_i_77);
        tile_storage_73[((tile_i_76 * 64) + tile_i_77)] = (tile_storage_13[((tile_i_76 * 64) + tile_i_77)] * tile_storage_58[tile_i_76]);
        for (int tile_i_78 = 0; tile_i_78 < 3; ++tile_i_78) {
          tile_storage_73[((tile_i_76 * 64) + tile_i_77)] = (tile_storage_73[((tile_i_76 * 64) + tile_i_77)] + (tile_storage_62[((tile_i_76 * 3) + tile_i_78)] * tile_storage_23[((((pipeline_10 & (long)1) * (long)192) + (((long)tile_i_78) * (long)64)) + ((long)tile_i_77))]));
        }
      }
    }
    for (int tile_i_88 = 0; tile_i_88 < 2; ++tile_i_88) {
      tile_storage_85[tile_i_88] = ((tile_storage_9[tile_i_88] * tile_storage_58[tile_i_88]) + tile_storage_67[tile_i_88]);
    }
    for (int tile_i_81 = 0; tile_i_81 < 2; ++tile_i_81) {
      tile_storage_5[tile_i_81] = tile_storage_54[tile_i_81];
    }
    for (int tile_i_91 = 0; tile_i_91 < 2; ++tile_i_91) {
      tile_storage_9[tile_i_91] = tile_storage_85[tile_i_91];
    }
    for (int tile_i_94 = 0; tile_i_94 < 2; ++tile_i_94) {
      for (int tile_i_95 = 0; tile_i_95 < 64; ++tile_i_95) {
        int cse_v17 = ((tile_i_94 * 64) + tile_i_95);
        tile_storage_13[((tile_i_94 * 64) + tile_i_95)] = tile_storage_73[((tile_i_94 * 64) + tile_i_95)];
      }
    }
  }
  for (int tile_i_31_1 = 0; tile_i_31_1 < 2; ++tile_i_31_1) {
    for (int tile_i_32_1 = 0; tile_i_32_1 < 3; ++tile_i_32_1) {
      int cse_v18 = ((tile_i_31_1 * 3) + tile_i_32_1);
      tile_storage_28[((tile_i_31_1 * 3) + tile_i_32_1)] = 0.000000e+00f;
      for (int tile_i_33_1 = 0; tile_i_33_1 < 64; ++tile_i_33_1) {
        tile_storage_28[((tile_i_31_1 * 3) + tile_i_32_1)] = (tile_storage_28[((tile_i_31_1 * 3) + tile_i_32_1)] + (tile_storage_0[((tile_i_31_1 * 64) + tile_i_33_1)] * tile_storage_18[((tile_i_32_1 * 64) + tile_i_33_1)]));
      }
    }
  }
  for (int tile_i_35_1 = 0; tile_i_35_1 < 3; ++tile_i_35_1) {
    tile_storage_34[tile_i_35_1] = ((long)tile_i_35_1);
  }
  for (int tile_i_37_1 = 0; tile_i_37_1 < 3; ++tile_i_37_1) {
    tile_storage_36[tile_i_37_1] = ((long)tile_i_37_1);
  }
  for (int tile_i_39_1 = 0; tile_i_39_1 < 2; ++tile_i_39_1) {
    tile_storage_38[tile_i_39_1] = ((long)tile_i_39_1);
  }
  for (int tile_i_41_1 = 0; tile_i_41_1 < 3; ++tile_i_41_1) {
    for (int tile_i_42_1 = 0; tile_i_42_1 < 2; ++tile_i_42_1) {
      tile_storage_40[((tile_i_41_1 * 2) + tile_i_42_1)] = ((tile_storage_34[tile_i_41_1] < (long)2) && (tile_storage_36[tile_i_41_1] <= (tile_storage_38[tile_i_42_1] + (long)1)));
    }
  }
  for (int tile_i_46_1 = 0; tile_i_46_1 < 2; ++tile_i_46_1) {
    for (int tile_i_47_1 = 0; tile_i_47_1 < 3; ++tile_i_47_1) {
      int cse_v19 = ((tile_i_46_1 * 3) + tile_i_47_1);
      float condval_5;
      if (tile_storage_40[((tile_i_47_1 * 2) + tile_i_46_1)]) {
        condval_5 = (tile_storage_28[((tile_i_46_1 * 3) + tile_i_47_1)] * 1.250000e-01f);
      } else {
        condval_5 = -1.000000e+30f;
      }
      tile_storage_43[((tile_i_46_1 * 3) + tile_i_47_1)] = condval_5;
    }
  }
  for (int tile_i_51_1 = 0; tile_i_51_1 < 2; ++tile_i_51_1) {
    thread float tile_storage_52_1[1];
    tile_storage_52_1[0] = -INFINITY;
    for (int n_55_0_1 = 0; n_55_0_1 < 3; ++n_55_0_1) {
      thread float tile_storage_53_1[1];
      tile_storage_53_1[0] = max(tile_storage_52_1[0], tile_storage_43[((tile_i_51_1 * 3) + n_55_0_1)]);
      tile_storage_52_1[0] = tile_storage_53_1[0];
    }
    tile_storage_48[tile_i_51_1] = tile_storage_52_1[0];
  }
  for (int tile_i_57_1 = 0; tile_i_57_1 < 2; ++tile_i_57_1) {
    tile_storage_54[tile_i_57_1] = max(tile_storage_5[tile_i_57_1], tile_storage_48[tile_i_57_1]);
  }
  for (int tile_i_61_1 = 0; tile_i_61_1 < 2; ++tile_i_61_1) {
    tile_storage_58[tile_i_61_1] = exp((tile_storage_5[tile_i_61_1] - tile_storage_54[tile_i_61_1]));
  }
  for (int tile_i_65_1 = 0; tile_i_65_1 < 2; ++tile_i_65_1) {
    for (int tile_i_66_1 = 0; tile_i_66_1 < 3; ++tile_i_66_1) {
      int cse_v20 = ((tile_i_65_1 * 3) + tile_i_66_1);
      float condval_6;
      if (tile_storage_40[((tile_i_66_1 * 2) + tile_i_65_1)]) {
        condval_6 = exp((tile_storage_43[((tile_i_65_1 * 3) + tile_i_66_1)] - tile_storage_54[tile_i_65_1]));
      } else {
        condval_6 = 0.000000e+00f;
      }
      tile_storage_62[((tile_i_65_1 * 3) + tile_i_66_1)] = condval_6;
    }
  }
  for (int tile_i_70_1 = 0; tile_i_70_1 < 2; ++tile_i_70_1) {
    thread float tile_storage_71_1[1];
    tile_storage_71_1[0] = 0.000000e+00f;
    for (int n_86_0_1 = 0; n_86_0_1 < 3; ++n_86_0_1) {
      thread float tile_storage_72_1[1];
      tile_storage_72_1[0] = (tile_storage_71_1[0] + tile_storage_62[((tile_i_70_1 * 3) + n_86_0_1)]);
      tile_storage_71_1[0] = tile_storage_72_1[0];
    }
    tile_storage_67[tile_i_70_1] = tile_storage_71_1[0];
  }
  for (int tile_i_76_1 = 0; tile_i_76_1 < 2; ++tile_i_76_1) {
    for (int tile_i_77_1 = 0; tile_i_77_1 < 64; ++tile_i_77_1) {
      int cse_v21 = ((tile_i_76_1 * 64) + tile_i_77_1);
      tile_storage_73[((tile_i_76_1 * 64) + tile_i_77_1)] = (tile_storage_13[((tile_i_76_1 * 64) + tile_i_77_1)] * tile_storage_58[tile_i_76_1]);
      for (int tile_i_78_1 = 0; tile_i_78_1 < 3; ++tile_i_78_1) {
        tile_storage_73[((tile_i_76_1 * 64) + tile_i_77_1)] = (tile_storage_73[((tile_i_76_1 * 64) + tile_i_77_1)] + (tile_storage_62[((tile_i_76_1 * 3) + tile_i_78_1)] * tile_storage_23[((tile_i_78_1 * 64) + tile_i_77_1)]));
      }
    }
  }
  for (int tile_i_88_1 = 0; tile_i_88_1 < 2; ++tile_i_88_1) {
    tile_storage_85[tile_i_88_1] = ((tile_storage_9[tile_i_88_1] * tile_storage_58[tile_i_88_1]) + tile_storage_67[tile_i_88_1]);
  }
  for (int tile_i_81_1 = 0; tile_i_81_1 < 2; ++tile_i_81_1) {
    tile_storage_5[tile_i_81_1] = tile_storage_54[tile_i_81_1];
  }
  for (int tile_i_91_1 = 0; tile_i_91_1 < 2; ++tile_i_91_1) {
    tile_storage_9[tile_i_91_1] = tile_storage_85[tile_i_91_1];
  }
  for (int tile_i_94_1 = 0; tile_i_94_1 < 2; ++tile_i_94_1) {
    for (int tile_i_95_1 = 0; tile_i_95_1 < 64; ++tile_i_95_1) {
      int cse_v22 = ((tile_i_94_1 * 64) + tile_i_95_1);
      tile_storage_13[((tile_i_94_1 * 64) + tile_i_95_1)] = tile_storage_73[((tile_i_94_1 * 64) + tile_i_95_1)];
    }
  }
  for (int tile_i_98 = 0; tile_i_98 < 2; ++tile_i_98) {
    for (int tile_i_99 = 0; tile_i_99 < 64; ++tile_i_99) {
      if (tile_i_98 < 1) {
        arg3_ptr[((((int)threadIdx) * 64) + tile_i_99)] = (tile_storage_13[tile_i_99] / tile_storage_9[0]);
      }
    }
  }
}



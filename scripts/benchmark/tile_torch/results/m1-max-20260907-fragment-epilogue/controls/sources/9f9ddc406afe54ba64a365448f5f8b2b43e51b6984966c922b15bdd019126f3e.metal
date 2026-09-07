// Function: benchmark_activation_pair_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_activation_pair_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  int cse_v2 = ((((int)blockIdx) & 1) * 256);
  int cse_v3 = (((((int)blockIdx) & 1) * 256) + ((int)threadIdx));
  int cse_v4 = ((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx));
  float condval;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval = 0.000000e+00f;
  }
  float condval_1;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_1 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_1 = 0.000000e+00f;
  }
  float condval_2;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_2 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_2 = 0.000000e+00f;
  }
  float condval_3;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_3 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_3 = 0.000000e+00f;
  }
  float condval_4;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_4 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_4 = 0.000000e+00f;
  }
  float condval_5;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_5 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_5 = 0.000000e+00f;
  }
  float condval_6;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_6 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_6 = 0.000000e+00f;
  }
  float condval_7;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_7 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_7 = 0.000000e+00f;
  }
  float condval_8;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_8 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_8 = 0.000000e+00f;
  }
  float condval_9;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_9 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_9 = 0.000000e+00f;
  }
  float condval_10;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_10 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_10 = 0.000000e+00f;
  }
  float condval_11;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_11 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_11 = 0.000000e+00f;
  }
  float condval_12;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_12 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_12 = 0.000000e+00f;
  }
  float condval_13;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_13 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_13 = 0.000000e+00f;
  }
  float condval_14;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_14 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_14 = 0.000000e+00f;
  }
  float condval_15;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_15 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_15 = 0.000000e+00f;
  }
  float condval_16;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_16 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_16 = 0.000000e+00f;
  }
  float condval_17;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_17 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_17 = 0.000000e+00f;
  }
  float condval_18;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_18 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_18 = 0.000000e+00f;
  }
  float condval_19;
  if (((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257)) {
    condval_19 = arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))];
  } else {
    condval_19 = 0.000000e+00f;
  }
  float tile_storage_3_element = select(((exp((2.000000e+00f * (7.978846e-01f * (condval + (4.471500e-02f * ((condval_1 * condval_2) * condval_3)))))) - 1.000000e+00f) / (exp((2.000000e+00f * (7.978846e-01f * (condval_4 + (4.471500e-02f * ((condval_5 * condval_6) * condval_7)))))) + 1.000000e+00f)), ((1.000000e+00f - exp((-2.000000e+00f * (7.978846e-01f * (condval_8 + (4.471500e-02f * ((condval_9 * condval_10) * condval_11))))))) / (1.000000e+00f + exp((-2.000000e+00f * (7.978846e-01f * (condval_12 + (4.471500e-02f * ((condval_13 * condval_14) * condval_15)))))))), ((7.978846e-01f * (condval_16 + (4.471500e-02f * ((condval_17 * condval_18) * condval_19)))) >= 0.000000e+00f));
  float cse_v1 = (1.000000e+00f + tile_storage_3_element);
  if ((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257) {
    arg1_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))] = ((5.000000e-01f * arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))]) * (1.000000e+00f + tile_storage_3_element));
  }
  if ((((((int)blockIdx) & 1) * 256) + ((int)threadIdx)) < 257) {
    arg2_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))] = ((5.000000e-01f * (1.000000e+00f + tile_storage_3_element)) + ((((5.000000e-01f * arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))]) * (1.000000e+00f - (tile_storage_3_element * tile_storage_3_element))) * 7.978846e-01f) * (1.000000e+00f + ((1.341450e-01f * arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))]) * arg0_ptr[((((((int)blockIdx) >> 1) * 257) + ((((int)blockIdx) & 1) * 256)) + ((int)threadIdx))]))));
  }
}



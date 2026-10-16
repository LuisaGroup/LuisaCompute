// Function: benchmark_rmsnorm_kernel
#include <metal_stdlib>
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

kernel void benchmark_rmsnorm_kernel(  device float* arg0_ptr [[ buffer(0) ]],
  device float* arg1_ptr [[ buffer(1) ]],
  device float* arg2_ptr [[ buffer(2) ]],
  uint blockIdx [[threadgroup_position_in_grid]],
  uint threadIdx [[thread_position_in_threadgroup]]
) {
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)threadIdx) * 2);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((int)threadIdx) * 2)] * arg0_ptr[(((int)threadIdx) * 2)]));
  int cse_v2 = ((((int)threadIdx) * 2) + 1);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)threadIdx) * 2) + 1)] * arg0_ptr[((((int)threadIdx) * 2) + 1)]));
  int cse_v3 = ((((int)threadIdx) * 2) + 64);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)threadIdx) * 2) + 64)] * arg0_ptr[((((int)threadIdx) * 2) + 64)]));
  int cse_v4 = ((((int)threadIdx) * 2) + 65);
  if (((int)threadIdx) < 31) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)threadIdx) * 2) + 65)] * arg0_ptr[((((int)threadIdx) * 2) + 65)]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  tile_storage_3[0] = tile_storage_5[0];
  arg2_ptr[(((int)threadIdx) * 2)] = ((arg0_ptr[(((int)threadIdx) * 2)] / sqrt(((tile_storage_3[0] / 1.270000e+02f) + 1.000000e-05f))) * arg1_ptr[(((int)threadIdx) * 2)]);
  arg2_ptr[((((int)threadIdx) * 2) + 1)] = ((arg0_ptr[((((int)threadIdx) * 2) + 1)] / sqrt(((tile_storage_3[0] / 1.270000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 2) + 1)]);
  arg2_ptr[((((int)threadIdx) * 2) + 64)] = ((arg0_ptr[((((int)threadIdx) * 2) + 64)] / sqrt(((tile_storage_3[0] / 1.270000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 2) + 64)]);
  if (((int)threadIdx) < 31) {
    arg2_ptr[((((int)threadIdx) * 2) + 65)] = ((arg0_ptr[((((int)threadIdx) * 2) + 65)] / sqrt(((tile_storage_3[0] / 1.270000e+02f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 2) + 65)]);
  }
}



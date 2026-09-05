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
  threadgroup float parallel_0_subgroup_partials_0[8];
  thread float tile_storage_3[1];
  thread float tile_storage_5[1];
  tile_storage_5[0] = 0.000000e+00f;
  int cse_v1 = (((int)threadIdx) * 4);
  int cse_v3 = ((((int)blockIdx) * 1537) + (((int)threadIdx) * 4));
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[((((int)blockIdx) * 1537) + (((int)threadIdx) * 4))] * arg0_ptr[((((int)blockIdx) * 1537) + (((int)threadIdx) * 4))]));
  int cse_v4 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1)]));
  int cse_v5 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2)]));
  int cse_v6 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3);
  tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3)]));
  int cse_v7 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1024);
  if (((int)threadIdx) < 129) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1024)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1024)]));
  }
  int cse_v8 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1025);
  if (((int)threadIdx) < 128) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1025)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1025)]));
  }
  int cse_v9 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1026);
  if (((int)threadIdx) < 128) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1026)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1026)]));
  }
  int cse_v10 = (((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1027);
  if (((int)threadIdx) < 128) {
    tile_storage_5[0] = (tile_storage_5[0] + (arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1027)] * arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1027)]));
  }
  tile_storage_5[0] = simd_sum(tile_storage_5[0]);
  int cse_v2 = (((int)threadIdx) & 31);
  if ((((int)threadIdx) % 32) == 0) {
    parallel_0_subgroup_partials_0[(((int)threadIdx) >> 5)] = tile_storage_5[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float condval;
  if (((((int)threadIdx) & 31) < 8)) {
    condval = parallel_0_subgroup_partials_0[(((int)threadIdx) & 31)];
  } else {
    condval = 0.000000e+00f;
  }
  tile_storage_5[0] = simd_sum(condval);
  tile_storage_3[0] = tile_storage_5[0];
  arg2_ptr[((((int)blockIdx) * 1537) + (((int)threadIdx) * 4))] = ((arg0_ptr[((((int)blockIdx) * 1537) + (((int)threadIdx) * 4))] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[(((int)threadIdx) * 4)]);
  arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 1)]);
  arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 2)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 2)]);
  arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 3)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 3)]);
  if (((int)threadIdx) < 129) {
    arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1024)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1024)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 1024)]);
  }
  if (((int)threadIdx) < 128) {
    arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1025)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1025)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 1025)]);
  }
  if (((int)threadIdx) < 128) {
    arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1026)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1026)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 1026)]);
  }
  if (((int)threadIdx) < 128) {
    arg2_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1027)] = ((arg0_ptr[(((((int)blockIdx) * 1537) + (((int)threadIdx) * 4)) + 1027)] / sqrt(((tile_storage_3[0] / 1.537000e+03f) + 1.000000e-05f))) * arg1_ptr[((((int)threadIdx) * 4) + 1027)]);
  }
}



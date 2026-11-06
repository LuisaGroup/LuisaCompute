#pragma once
#include <Shader/ComputeShader.h>
namespace lc::dx {
class BuiltinKernel {
public:
    static ComputeShader *load_bindless_set_kernel(Device *device);
    static ComputeShader *load_accel_set_kernel(Device *device);
    static ComputeShader *load_cluster_build_info_kernel(Device *device);
    static ComputeShader *load_bc6_try_mode_g10cs_kernel(Device *device);
    static ComputeShader *load_bc6_try_mode_le10cs_kernel(Device *device);
    static ComputeShader *load_bc6_encode_block_cs_kernel(Device *device);
    static ComputeShader *load_bc7_try_mode_456cs_kernel(Device *device);
    static ComputeShader *load_bc7_try_mode_137cs_kernel(Device *device);
    static ComputeShader *load_bc7_try_mode_02cs_kernel(Device *device);
    static ComputeShader *load_bc7_encode_block_cs_kernel(Device *device);
    BuiltinKernel() = delete;
    ~BuiltinKernel() = delete;
};
}// namespace lc::dx


#pragma once
#include "compute_shader.h"
namespace lc::vk {
class BuiltinKernel {
public:
    static ComputeShader *load_bindless_set_kernel(Device *device);
    static ComputeShader *load_accel_set_kernel(Device *device);
    static ComputeShader *load_indirect_prepare_kernel(Device *device);
    BuiltinKernel() = delete;
    ~BuiltinKernel() = delete;
};
}// namespace lc::vk

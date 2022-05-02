#pragma once
#include <Shader/ComputeShader.h>
namespace toolhub::directx {
class BuiltinKernel {
public:
    static ComputeShader const *LoadAccelSetKernel(Device* device);
};
}// namespace toolhub::directx
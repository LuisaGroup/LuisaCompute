#pragma once
#include <Shader/ComputeShader.h>
namespace lc::dx {
class BuiltinKernel {
public:
    static ComputeShader *LoadBindlessSetKernel(Device *device);
    static ComputeShader *LoadAccelSetKernel(Device *device);
    static ComputeShader *LoadBC6TryModeG10CSKernel(Device *device);
    static ComputeShader *LoadBC6TryModeLE10CSKernel(Device *device);
    static ComputeShader *LoadBC6EncodeBlockCSKernel(Device *device);
    static ComputeShader *LoadBC7TryMode456CSKernel(Device *device);
    static ComputeShader *LoadBC7TryMode137CSKernel(Device *device);
    static ComputeShader *LoadBC7TryMode02CSKernel(Device *device);
    static ComputeShader *LoadBC7EncodeBlockCSKernel(Device *device);
};
}// namespace lc::dx


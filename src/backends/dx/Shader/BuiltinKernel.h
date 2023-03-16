#pragma once
#include <Shader/ComputeShader.h>
namespace toolhub::directx {
class BuiltinKernel {
public:
    static ComputeShader *LoadAccelSetKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC6TryModeG10CSKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC6TryModeLE10CSKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC6EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC7TryMode456CSKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC7TryMode137CSKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC7TryMode02CSKernel(Device *device, luisa::BinaryIO const *ctx);
    static ComputeShader *LoadBC7EncodeBlockCSKernel(Device *device, luisa::BinaryIO const *ctx);
};
}// namespace toolhub::directx

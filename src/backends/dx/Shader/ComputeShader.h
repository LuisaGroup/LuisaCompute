#pragma once
#include <Shader/Shader.h>
#include <vstl/VGuid.h>
namespace toolhub::directx {
class PipelineLibrary;
class ComputeShader final : public Shader {
protected:
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
    Device *device;
    uint3 blockSize;

public:
    Tag GetTag() const { return Tag::ComputeShader; }
    uint3 BlockSize() const { return blockSize; }
    ComputeShader(
        uint3 blockSize,
        vstd::span<std::pair<vstd::string, Property> const> prop,
        ComPtr<ID3D12RootSignature> &&rootSig,
        vstd::span<vbyte const> code,
        Device *device,
        vstd::Guid guid,
        PipelineLibrary *pipeLib);
    ComputeShader(
        uint3 blockSize,
        vstd::span<std::pair<vstd::string, Property> const> properties,
        vstd::span<vbyte> binData,
        Device *device,
        vstd::Guid guid);
    ID3D12PipelineState *Pso() const { return pso.Get(); }
    ~ComputeShader();
    ComputeShader(ComputeShader &&v) = default;
    KILL_COPY_CONSTRUCT(ComputeShader)
};
}// namespace toolhub::directx
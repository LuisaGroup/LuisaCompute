#pragma once
#include <Shader/Shader.h>
#include <vstl/VGuid.h>
namespace toolhub::directx {
struct CodegenResult;
class ShaderSerializer;
class ComputeShader final : public Shader {
    friend class ShaderSerializer;

protected:
    ComPtr<ID3D12PipelineState> pso;
    Device *device;
    uint3 blockSize;
    vstd::Guid guid;
    ComputeShader(
        uint3 blockSize,
        Device *device,
        vstd::span<std::pair<vstd::string, Property> const> prop,
        vstd::Guid guid,
        ComPtr<ID3D12RootSignature> &&rootSig,
        ComPtr<ID3D12PipelineState> &&pso);

public:
    vstd::Guid GetGuid() const { return guid; }
    Tag GetTag() const { return Tag::ComputeShader; }
    uint3 BlockSize() const { return blockSize; }
    static ComputeShader *CompileCompute(
        Device *device,
        CodegenResult &code,
        uint3 blockSize,
        uint shaderModel);
    ComputeShader(
        uint3 blockSize,
        vstd::span<std::pair<vstd::string, Property> const> properties,
        vstd::span<vbyte const> binData,
        Device *device,
        vstd::Guid guid);
    ID3D12PipelineState *Pso() const { return pso.Get(); }
    ~ComputeShader();
    KILL_COPY_CONSTRUCT(ComputeShader)
    KILL_MOVE_CONSTRUCT(ComputeShader)
};
}// namespace toolhub::directx
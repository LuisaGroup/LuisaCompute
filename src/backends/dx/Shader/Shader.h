#pragma once
#include <vstl/common.h>
#include <Windows.h>
#include <d3dx12.h>
#include <Shader/ShaderVariableType.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <ast/function.h>
using namespace luisa::compute;
namespace toolhub::directx {
enum class FileType {
    Cache,
    ByteCode,
    Internal
};
struct SavedArgument {
    Type::Tag tag;
    Usage varUsage;
    uint structSize;
    SavedArgument() {}
    SavedArgument(Function kernel, Variable const &var);
    SavedArgument(Usage usage, Variable const &var);
    SavedArgument(Type const *type);
};
class TopAccel;
class CommandBufferBuilder;
class Shader : public vstd::IOperatorNewBase {
public:
    enum class Tag : uint8_t {
        ComputeShader,
        RayTracingShader,
        RasterShader
    };
    virtual Tag GetTag() const = 0;

protected:
    ComPtr<ID3D12PipelineState> pso;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    vstd::vector<Property> properties;
    vstd::vector<SavedArgument> kernelArguments;
    uint bindlessCount;
    void SavePSO(vstd::string_view psoName, BinaryIO *fileStream, Device const *device) const;

public:
    static vstd::string PSOName(Device const *device, vstd::string_view fileName);
    virtual ~Shader() noexcept = default;
    uint BindlessCount() const { return bindlessCount; }
    vstd::span<Property const> Properties() const { return properties; }
    vstd::span<SavedArgument const> Args() const { return kernelArguments; }
    Shader(
        vstd::vector<Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        ID3D12Device *device,
        bool isRaster);
    Shader(
        vstd::vector<Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&rootSig);
    ID3D12RootSignature *RootSig() const { return rootSig.Get(); }
    ID3D12PipelineState *Pso() const { return pso.Get(); }

    void SetComputeResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        BufferView buffer) const;
    void SetComputeResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        DescriptorHeapView view) const;
    void SetComputeResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        TopAccel const *bAccel) const;
    void SetComputeResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        std::pair<uint, uint4> const &constValue) const;

    void SetRasterResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        BufferView buffer) const;
    void SetRasterResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        DescriptorHeapView view) const;
    void SetRasterResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        TopAccel const *bAccel) const;
    void SetRasterResource(
        uint propertyName,
        CommandBufferBuilder *cmdList,
        std::pair<uint, uint4> const &constValue) const;

    KILL_COPY_CONSTRUCT(Shader)
    KILL_MOVE_CONSTRUCT(Shader)
};
}// namespace toolhub::directx
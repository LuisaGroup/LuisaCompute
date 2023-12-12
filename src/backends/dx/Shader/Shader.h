#pragma once
#include <luisa/vstl/common.h>
#include <Windows.h>
#include <d3dx12.h>
#include <Shader/ShaderVariableType.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <luisa/ast/function.h>
#include <luisa/core/binary_io.h>
namespace lc::dx {
using namespace luisa;
using namespace luisa::compute;
struct SavedArgument {
    Type::Tag tag;
    Usage varUsage;
    uint structSize;
    SavedArgument() {}
    SavedArgument(Function kernel, Variable const &var);
    SavedArgument(Usage usage, Variable const &var);
    SavedArgument(Type const *type);
};
enum class CacheType : uint8_t {
    Internal,
    Cache,
    ByteCode
};
inline static auto ReadBinaryIO(CacheType type, luisa::BinaryIO const *binIo, luisa::string_view name) {
    switch (type) {
        case CacheType::ByteCode:
            return binIo->read_shader_bytecode(name);
        case CacheType::Cache:
            return binIo->read_shader_cache(name);
        case CacheType::Internal:
            return binIo->read_internal_shader(name);
    }
    return luisa::unique_ptr<luisa::BinaryStream>{};
}
inline static void WriteBinaryIO(CacheType type, luisa::BinaryIO const *binIo, luisa::string_view name, luisa::span<std::byte const> data) {
    switch (type) {
        case CacheType::ByteCode:
            static_cast<void>(binIo->write_shader_bytecode(name, data));
            return;
        case CacheType::Cache:
            static_cast<void>(binIo->write_shader_cache(name, data));
            return;
        case CacheType::Internal:
            static_cast<void>(binIo->write_internal_shader(name, data));
            return;
    }
}

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
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> kernelArguments;
    vstd::vector<std::pair<vstd::string, Type const *>> printers;
    uint bindlessCount;
    void SavePSO(ID3D12PipelineState *pso, vstd::string_view psoName, luisa::BinaryIO const *fileStream, Device const *device) const;

public:
    static vstd::string PSOName(Device const *device, vstd::string_view fileName);
    virtual ~Shader() noexcept = default;
    vstd::span<const std::pair<vstd::string, Type const *>> Printers() const { return printers; }
    uint BindlessCount() const { return bindlessCount; }
    vstd::span<hlsl::Property const> Properties() const { return properties; }
    vstd::span<SavedArgument const> Args() const { return kernelArguments; }
    Shader(
        vstd::vector<hlsl::Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        ID3D12Device *device,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        bool isRaster);
    Shader(
        vstd::vector<hlsl::Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&rootSig,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers);
    ID3D12RootSignature *RootSig() const { return rootSig.Get(); }

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
}// namespace lc::dx

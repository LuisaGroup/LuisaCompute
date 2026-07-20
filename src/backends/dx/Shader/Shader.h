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
    Usage var_usage;
    uint struct_size;
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
luisa::unique_ptr<luisa::BinaryStream> read_binary_io(CacheType type, luisa::BinaryIO const *bin_io, luisa::string_view name);
inline static void write_binary_io(CacheType type, luisa::BinaryIO const *bin_io, luisa::string_view name, luisa::span<std::byte const> data) {
    switch (type) {
        case CacheType::ByteCode:
            static_cast<void>(bin_io->write_shader_bytecode(name, data));
            return;
        case CacheType::Cache:
            static_cast<void>(bin_io->write_shader_cache(name, data));
            return;
        case CacheType::Internal:
            static_cast<void>(bin_io->write_internal_shader(name, data));
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
    virtual Tag get_tag() const = 0;

protected:
    Microsoft::WRL::ComPtr<ID3D12RootSignature> _root_sig;
    vstd::vector<hlsl::Property> _properties;
    vstd::vector<SavedArgument> _kernel_arguments;
    vstd::vector<std::pair<vstd::string, Type const *>> _printers;
    uint _bindless_count;
    uint _validation_count;
    void _save_pso(ID3D12PipelineState *pso, vstd::string_view pso_name, luisa::BinaryIO const *file_stream, Device const *device) const;

public:
    static vstd::string pso_name(Device const *device, vstd::string_view file_name);
    virtual ~Shader() noexcept = default;
    vstd::span<const std::pair<vstd::string, Type const *>> printers() const { return _printers; }
    uint bindless_count() const { return _bindless_count; }
    uint validation_count() const { return _validation_count; }
    vstd::span<hlsl::Property const> properties() const { return _properties; }
    vstd::span<SavedArgument const> args() const { return _kernel_arguments; }
    Shader(
        vstd::vector<hlsl::Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        ID3D12Device *device,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        uint validation_count,
        bool isRaster);
    Shader(
        vstd::vector<hlsl::Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&root_sig,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        uint validation_count);
    ID3D12RootSignature *root_sig() const { return _root_sig.Get(); }

    void set_compute_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        BufferView buffer) const;
    void set_compute_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        DescriptorHeapView view) const;
    void set_compute_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        TopAccel const *bAccel) const;
    void set_compute_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        std::pair<uint, uint4> const &const_value) const;

    void set_raster_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        BufferView buffer) const;
    void set_raster_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        DescriptorHeapView view) const;
    void set_raster_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        TopAccel const *bAccel) const;
    void set_raster_resource(
        uint property_name,
        CommandBufferBuilder *cmd_list,
        std::pair<uint, uint4> const &const_value) const;

    KILL_COPY_CONSTRUCT(Shader)
    KILL_MOVE_CONSTRUCT(Shader)
};
}// namespace lc::dx

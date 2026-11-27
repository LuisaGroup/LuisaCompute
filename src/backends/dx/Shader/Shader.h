#pragma once
#include <luisa/vstl/common.h>
#include <Windows.h>
#include <d3dx12.h>
#include <Shader/ShaderVariableType.h>
#include <Shader/RootSigLayout.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/BindProperty.h>
#include <luisa/ast/function.h>
#include <luisa/core/binary_io.h>
namespace lc::dx {
using namespace luisa;
using namespace luisa::compute;
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
    using RootSigBinding = lc::dx::RootSigBinding;
    [[nodiscard]] static bool needs_descriptor_table_spill(
        vstd::span<hlsl::Property const> properties,
        bool append_object_id = false) noexcept {
        return lc::dx::needs_descriptor_table_spill(properties, append_object_id);
    }
    [[nodiscard]] static bool is_spilled_srv(hlsl::Property const &var) noexcept {
        return lc::dx::is_spilled_srv(var);
    }
    [[nodiscard]] static bool is_spilled_uav(hlsl::Property const &var) noexcept {
        return lc::dx::is_spilled_uav(var);
    }
    virtual Tag get_tag() const = 0;

protected:
    Microsoft::WRL::ComPtr<ID3D12RootSignature> _root_sig;
    vstd::vector<hlsl::Property> _properties;
    vstd::vector<SavedArgument> _kernel_arguments;
    vstd::vector<std::pair<vstd::string, Type const *>> _printers;
    // Number of bindless descriptor tables the draw/dispatch path pushes. Neither
    // constructor initialised it, so a shader built outside
    // `RasterShader::compile_raster` / `RasterDeSerialize` (e.g. a native raster
    // shader) made the encode path push an indeterminate number of global-heap
    // bindings; the default keeps such a shader at zero.
    uint _bindless_count{0u};
    uint _validation_count;
    bool _spilled_tables{false};
    uint _spilled_srv_root_param{0u};
    uint _spilled_uav_root_param{0u};
    uint _spilled_srv_count{0u};
    uint _spilled_uav_count{0u};
    vstd::vector<RootSigBinding> _root_sig_bindings;
    uint64_t _shader_uid{0u};
    void _init_root_sig_layout(bool append_object_id = false);
    void _save_pso(ID3D12PipelineState *pso, vstd::string_view pso_name, luisa::BinaryIO const *file_stream, Device const *device) const;

public:
    static vstd::string pso_name(Device const *device, vstd::string_view file_name);
    virtual ~Shader() noexcept = default;
    vstd::span<const std::pair<vstd::string, Type const *>> printers() const { return _printers; }
    uint bindless_count() const { return _bindless_count; }
    uint validation_count() const { return _validation_count; }
    [[nodiscard]] uint64_t uid() const noexcept { return _shader_uid; }
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

    void bind_compute_resources(
        CommandBufferBuilder *cmd_list,
        vstd::span<const BindProperty> resources) const;
    void bind_raster_resources(
        CommandBufferBuilder *cmd_list,
        vstd::span<const BindProperty> resources) const;

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

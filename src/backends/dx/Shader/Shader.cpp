#include <luisa/core/magic_enum.h>
#include <Shader/Shader.h>
#include <d3dcompiler.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderSerializer.h>
#include <luisa/core/logging.h>

namespace lc::dx {
SavedArgument::SavedArgument(Usage usage, Variable const &var)
    : SavedArgument(var.type()) {
    var_usage = usage;
}

SavedArgument::SavedArgument(Function kernel, Variable const &var)
    : SavedArgument(var.type()) {
    var_usage = kernel.variable_usage(var.uid());
}
SavedArgument::SavedArgument(Type const *type) {
    tag = type->tag();
    if (type->tag() == Type::Tag::BUFFER) {
        if (auto ele = type->element()) {
            struct_size = static_cast<uint>(ele->size());
        } else {
            struct_size = 1u;// byte buffer
        }
    } else if (luisa::to_underlying(type->tag()) < luisa::to_underlying(Type::Tag::BUFFER)) {
        struct_size = static_cast<uint>(type->size());
    }
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&root_sig,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    uint validation_count)
    : _root_sig(std::move(root_sig)),
      _properties(std::move(prop)),
      _kernel_arguments(std::move(args)),
      _printers(std::move(printers)),
      _validation_count(validation_count) {
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ID3D12Device *device,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    uint validation_count,
    bool isRaster)
    : _properties(std::move(prop)),
      _kernel_arguments(std::move(args)),
      _printers(std::move(printers)),
      _validation_count(validation_count) {
    auto serializedRootSig = ShaderSerializer::SerializeRootSig(
        _properties,
        isRaster);
    ThrowIfFailed(device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(_root_sig.GetAddressOf())));
}

void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmd_list->SetComputeRootConstantBufferView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmd_list->SetComputeRootShaderResourceView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmd_list->SetComputeRootUnorderedAccessView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default:
            LUISA_ERROR("Invalid shader resource type {}.\n\n"
                        "This might be due to the change of shader cache.\n"
                        "Please try delete the cache folder (default: build_dir/bin/.cache) "
                        "and re-run the program.\n"
                        "If the problem persists, please report this issue to the developers.\n",
                        to_string(var.type));
    }
}
void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmd_list->SetComputeRootDescriptorTable(
                property_name,
                view.heap->hGPU(view.index));
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &const_value) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    LUISA_ASSUME(_properties[property_name].type == hlsl::ShaderVariableType::ConstantValue);
    cmd_list->SetComputeRoot32BitConstants(property_name, const_value.first, &const_value.second, 0);
}
void Shader::set_compute_resource(
    uint property_name,
    CommandBufferBuilder *cmd_list,
    TopAccel const *bAccel) const {
    return set_compute_resource(
        property_name,
        cmd_list,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmd_list->SetGraphicsRootConstantBufferView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmd_list->SetGraphicsRootShaderResourceView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmd_list->SetGraphicsRootUnorderedAccessView(
                property_name,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    auto &&var = _properties[property_name];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmd_list->SetGraphicsRootDescriptorTable(
                property_name,
                view.heap->hGPU(view.index));
        } break;
        default: LUISA_ASSUME(false); break;
    }
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cmd_list,
    TopAccel const *bAccel) const {
    return set_raster_resource(
        property_name,
        cmd_list,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::set_raster_resource(
    uint property_name,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &const_value) const {
    auto cmd_list = cb->get_cb()->cmd_list();
    LUISA_ASSUME(_properties[property_name].type == hlsl::ShaderVariableType::ConstantValue);
    cmd_list->SetGraphicsRoot32BitConstants(property_name, const_value.first, &const_value.second, 0);
}
void Shader::_save_pso(ID3D12PipelineState *pso, vstd::string_view pso_name, luisa::BinaryIO const *file_stream, Device const *device) const {
    LUISA_VERBOSE("Write Pipeline cache to {}.", pso_name);
    ComPtr<ID3DBlob> psoCache;
    pso->GetCachedBlob(&psoCache);
    static_cast<void>(file_stream->write_shader_cache(
        pso_name,
        {reinterpret_cast<std::byte const *>(psoCache->GetBufferPointer()),
         psoCache->GetBufferSize()}));
};
vstd::string Shader::pso_name(Device const *device, vstd::string_view file_name) {
    vstd::fixed_vector<uint8_t, 64> data;
    luisa::enlarge_by(data, 16 + file_name.size());
    std::memcpy(data.data(), &device->adapter_id, 16);
    std::memcpy(data.data() + 16, file_name.data(), file_name.size());
    vstd::MD5 hash{data};
    return hash.to_string(false) + ".dx";
}

}// namespace lc::dx

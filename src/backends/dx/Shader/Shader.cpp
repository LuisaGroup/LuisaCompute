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
    varUsage = usage;
}

SavedArgument::SavedArgument(Function kernel, Variable const &var)
    : SavedArgument(var.type()) {
    varUsage = kernel.variable_usage(var.uid());
}
SavedArgument::SavedArgument(Type const *type) {
    if (luisa::to_underlying(type->tag()) < luisa::to_underlying(Type::Tag::BUFFER)) {
        structSize = type->size();
    }
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&rootSig,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers)
    : rootSig(std::move(rootSig)),
      properties(std::move(prop)),
      kernelArguments(std::move(args)),
      printers(std::move(printers)) {
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ID3D12Device *device,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    bool isRaster)
    : properties(std::move(prop)),
      kernelArguments(std::move(args)),
      printers(std::move(printers)) {
    auto serializedRootSig = ShaderSerializer::SerializeRootSig(
        properties,
        isRaster);
    ThrowIfFailed(device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(rootSig.GetAddressOf())));
}

void Shader::SetComputeResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmdList = cb->GetCB()->CmdList();
    auto &&var = properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmdList->SetComputeRootConstantBufferView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmdList->SetComputeRootShaderResourceView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmdList->SetComputeRootUnorderedAccessView(
                propertyName,
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
void Shader::SetComputeResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmdList = cb->GetCB()->CmdList();
    auto &&var = properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmdList->SetComputeRootDescriptorTable(
                propertyName,
                view.heap->hGPU(view.index));
        } break;
        default: assert(false); break;
    }
}
void Shader::SetComputeResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &constValue) const {
    auto cmdList = cb->GetCB()->CmdList();
    assert(properties[propertyName].type == hlsl::ShaderVariableType::ConstantValue);
    cmdList->SetComputeRoot32BitConstants(propertyName, constValue.first, &constValue.second, 0);
}
void Shader::SetComputeResource(
    uint propertyName,
    CommandBufferBuilder *cmdList,
    TopAccel const *bAccel) const {
    return SetComputeResource(
        propertyName,
        cmdList,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::SetRasterResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmdList = cb->GetCB()->CmdList();
    auto &&var = properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::ConstantBuffer: {
            cmdList->SetGraphicsRootConstantBufferView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::StructuredBuffer: {
            cmdList->SetGraphicsRootShaderResourceView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case hlsl::ShaderVariableType::RWStructuredBuffer: {
            cmdList->SetGraphicsRootUnorderedAccessView(
                propertyName,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default: assert(false); break;
    }
}
void Shader::SetRasterResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmdList = cb->GetCB()->CmdList();
    auto &&var = properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::UAVTextureHeap:
        case hlsl::ShaderVariableType::CBVBufferHeap:
        case hlsl::ShaderVariableType::SamplerHeap:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::SRVTextureHeap: {
            cmdList->SetGraphicsRootDescriptorTable(
                propertyName,
                view.heap->hGPU(view.index));
        } break;
        default: assert(false); break;
    }
}
void Shader::SetRasterResource(
    uint propertyName,
    CommandBufferBuilder *cmdList,
    TopAccel const *bAccel) const {
    return SetRasterResource(
        propertyName,
        cmdList,
        BufferView(bAccel->GetAccelBuffer()));
}
void Shader::SetRasterResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    std::pair<uint, uint4> const &constValue) const {
    auto cmdList = cb->GetCB()->CmdList();
    assert(properties[propertyName].type == hlsl::ShaderVariableType::ConstantValue);
    cmdList->SetGraphicsRoot32BitConstants(propertyName, constValue.first, &constValue.second, 0);
}
void Shader::SavePSO(ID3D12PipelineState *pso, vstd::string_view psoName, luisa::BinaryIO const *fileStream, Device const *device) const {
    LUISA_VERBOSE("Write Pipeline cache to {}.", psoName);
    ComPtr<ID3DBlob> psoCache;
    pso->GetCachedBlob(&psoCache);
    static_cast<void>(fileStream->write_shader_cache(
        psoName,
        {reinterpret_cast<std::byte const *>(psoCache->GetBufferPointer()),
         psoCache->GetBufferSize()}));
};
vstd::string Shader::PSOName(Device const *device, vstd::string_view fileName) {
    vstd::fixed_vector<uint8_t, 64> data;
    data.push_back_uninitialized(16 + fileName.size());
    memcpy(data.data(), &device->adapterID, 16);
    memcpy(data.data() + 16, fileName.data(), fileName.size());
    vstd::MD5 hash{data};
    return hash.to_string(false);
}

}// namespace lc::dx

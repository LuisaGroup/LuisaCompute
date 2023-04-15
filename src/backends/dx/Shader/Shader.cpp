
#include <Shader/Shader.h>
#include <d3dcompiler.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderSerializer.h>

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
    tag = type->tag();
    switch (type->tag()) {
        case Type::Tag::BUFFER:
        case Type::Tag::TEXTURE:
            structSize = type->element()->size();
            break;
        case Type::Tag::BINDLESS_ARRAY:
        case Type::Tag::ACCEL:
            structSize = 0;
            break;
        default:
            structSize = type->size();
            break;
    }
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&rootSig)
    : rootSig(std::move(rootSig)), properties(std::move(prop)), kernelArguments(std::move(args)) {
}

Shader::Shader(
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ID3D12Device *device,
    bool isRaster) : properties(std::move(prop)), kernelArguments(std::move(args)) {
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
        default: assert(false); break;
    }
}
void Shader::SetComputeResource(
    uint propertyName,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmdList = cb->GetCB()->CmdList();
    auto &&var = properties[propertyName];
    switch (var.type) {
        case hlsl::ShaderVariableType::UAVDescriptorHeap:
        case hlsl::ShaderVariableType::CBVDescriptorHeap:
        case hlsl::ShaderVariableType::SampDescriptorHeap:
        case hlsl::ShaderVariableType::SRVDescriptorHeap: {
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
        case hlsl::ShaderVariableType::UAVDescriptorHeap:
        case hlsl::ShaderVariableType::CBVDescriptorHeap:
        case hlsl::ShaderVariableType::SampDescriptorHeap:
        case hlsl::ShaderVariableType::SRVDescriptorHeap: {
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
    LUISA_INFO("Write Pipeline cache to {}.", psoName);
    ComPtr<ID3DBlob> psoCache;
    pso->GetCachedBlob(&psoCache);
    fileStream->write_shader_cache(psoName, {reinterpret_cast<std::byte const *>(psoCache->GetBufferPointer()), psoCache->GetBufferSize()});
};
vstd::string Shader::PSOName(Device const *device, vstd::string_view fileName) {
    vstd::fixed_vector<uint8_t, 64> data;
    data.push_back_uninitialized(16 + fileName.size());
    memcpy(data.data(), &device->adapterID, 16);
    memcpy(data.data() + 16, fileName.data(), fileName.size());
    vstd::MD5 hash{data};
    return hash.to_string(false);
}

vstd::vector<Argument> Shader::BindingToArg(vstd::span<const Function::Binding> bindings) {
    vstd::vector<Argument> r;
    vstd::push_back_func(
        r, bindings.size(),
        [&](size_t i) {
            return luisa::visit(
                [&]<typename T>(T const &a) -> Argument {
                    Argument arg;
                    if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                        arg.tag = Argument::Tag::BUFFER;
                        arg.buffer = a;
                    } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                        arg.tag = Argument::Tag::TEXTURE;
                        arg.texture = a;
                    } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                        arg.tag = Argument::Tag::BINDLESS_ARRAY;
                        arg.bindless_array = a;
                    } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                        arg.tag = Argument::Tag::ACCEL;
                        arg.accel = a;
                    } else {
                        LUISA_ERROR("Binding Contain unwanted variable.");
                    }
                    return arg;
                },
                bindings[i]);
        });
    return r;
}
}// namespace lc::dx
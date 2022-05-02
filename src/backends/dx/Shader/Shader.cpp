
#include <Shader/Shader.h>
#include <d3dcompiler.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/GlobalSamplers.h>
#include <Resource/TopAccel.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderSerializer.h>

namespace toolhub::directx {
vstd::optional<Shader::InsideProperty> Shader::GetProperty(vstd::string_view str) const {
    auto ite = properties.Find(str);
    if (!ite) return {};
    return ite.Value();
}
Shader::Shader(
    vstd::span<std::pair<vstd::string, Property> const> prop,
    ComPtr<ID3D12RootSignature> &&rootSig)
    : rootSig(std::move(rootSig)) {
    size_t idx = 0;
    for (auto &&i : prop) {
        auto ite = properties.Emplace(std::move(i.first), i.second);
        ite.Value().rootSigPos = idx;
        ++idx;
    }
}

Shader::Shader(
    vstd::span<std::pair<vstd::string, Property> const> prop,
    ID3D12Device *device) {
    auto serializedRootSig = ShaderSerializer::SerializeRootSig(
        prop);
    properties.reserve(prop.size());
    size_t idx = 0;
    for (auto &&i : prop) {
        auto ite = properties.Emplace(std::move(i.first), i.second);
        ite.Value().rootSigPos = idx;
        ++idx;
    }
    ThrowIfFailed(device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(rootSig.GetAddressOf())));
}

bool Shader::SetComputeResource(
    vstd::string_view propertyName,
    CommandBufferBuilder *cb,
    BufferView buffer) const {
    auto cmdList = cb->CmdList();
    auto var = GetProperty(propertyName);
    if (!var) return false;
    switch (var->type) {
        case ShaderVariableType::ConstantBuffer: {
            cmdList->SetComputeRootConstantBufferView(
                var->rootSigPos,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case ShaderVariableType::StructuredBuffer: {
            cmdList->SetComputeRootShaderResourceView(
                var->rootSigPos,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        case ShaderVariableType::RWStructuredBuffer: {
            cmdList->SetComputeRootUnorderedAccessView(
                var->rootSigPos,
                buffer.buffer->GetAddress() + buffer.offset);
        } break;
        default:
            return false;
    }
    return true;
}
bool Shader::SetComputeResource(
    vstd::string_view propertyName,
    CommandBufferBuilder *cb,
    DescriptorHeapView view) const {
    auto cmdList = cb->CmdList();
    auto var = GetProperty(propertyName);
    if (!var) return false;
    switch (var->type) {
        case ShaderVariableType::UAVDescriptorHeap:
        case ShaderVariableType::CBVDescriptorHeap:
        case ShaderVariableType::SampDescriptorHeap:
        case ShaderVariableType::SRVDescriptorHeap: {
            cmdList->SetComputeRootDescriptorTable(
                var->rootSigPos,
                view.heap->hGPU(view.index));
        } break;
        default:
            return false;
    }
    return true;
}
bool Shader::SetComputeResource(
    vstd::string_view propertyName,
    CommandBufferBuilder *cmdList,
    TopAccel const *bAccel) const {
    return SetComputeResource(
        propertyName,
        cmdList,
        BufferView(bAccel->GetAccelBuffer()));
}
}// namespace toolhub::directx
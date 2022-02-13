#pragma vengine_package vengine_directx
#include <Shader/ComputeShader.h>
#include <Shader/PipelineLibrary.h>
namespace toolhub::directx {
ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::span<std::pair<vstd::string, Property> const> properties,
    vstd::span<vbyte> binData,
    Device *device,
    vstd::Guid guid)
    : Shader(std::move(properties), device->device.Get()),
      blockSize(blockSize),
      device(device) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.CS.pShaderBytecode = binData.data();
    psoDesc.CS.BytecodeLength = binData.size();
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    ThrowIfFailed(device->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf())));
    device->CreateShader(
        this, guid);
}
ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::span<std::pair<vstd::string, Property> const> prop,
    ComPtr<ID3D12RootSignature> &&rootSig,
    vstd::span<vbyte const> code,
    Device *device,
    vstd::Guid guid,
    PipelineLibrary *pipeLib)
    : Shader(prop, std::move(rootSig)),
      blockSize(blockSize),
      device(device) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = this->rootSig.Get();
    psoDesc.CS.pShaderBytecode = code.data();
    psoDesc.CS.BytecodeLength = code.size();
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    if (pipeLib) {
        pso = pipeLib->GetPipelineState(
            guid,
            psoDesc);
    } else {
        ThrowIfFailed(device->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf())));
    }
    device->CreateShader(
        this, guid);
}
ComputeShader::~ComputeShader() {
    device->DestroyShader(this);
}
}// namespace toolhub::directx
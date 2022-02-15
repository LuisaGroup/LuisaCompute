#pragma vengine_package vengine_directx
#include <Shader/ComputeShader.h>
#include <Shader/PipelineLibrary.h>
namespace toolhub::directx {
ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::span<std::pair<vstd::string, Property> const> properties,
    vstd::span<vbyte const> binData,
    Device *device,
    vstd::Guid guid)
    : Shader(std::move(properties), device->device.Get()),
      blockSize(blockSize),
      device(device),
      guid(guid) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.CS.pShaderBytecode = binData.data();
    psoDesc.CS.BytecodeLength = binData.size();
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    ThrowIfFailed(device->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf())));
}
ComputeShader::ComputeShader(
    uint3 blockSize,
    Device *device,
    vstd::span<std::pair<vstd::string, Property> const> prop,
    vstd::Guid guid,
    ComPtr<ID3D12RootSignature> &&rootSig,
    ComPtr<ID3D12PipelineState> &&pso)
    : device(device),
      blockSize(blockSize),
      Shader(prop, std::move(rootSig)),
      guid(guid) ,
      pso(std::move(pso)) {
}

ComputeShader::~ComputeShader() {
}
}// namespace toolhub::directx
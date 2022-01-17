#pragma vengine_package vengine_directx
#include <Shader/ComputeShader.h>
namespace toolhub::directx {
ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::span<std::pair<vstd::string, Property>> &&properties,
    vstd::span<vbyte> binData,
    ID3D12Device *device)
    : Shader(std::move(properties), device),
      blockSize(blockSize) {

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.CS.pShaderBytecode = binData.data();
    psoDesc.CS.BytecodeLength = binData.size();
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    ThrowIfFailed(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf())));
}
ComputeShader ::~ComputeShader() {
}
}// namespace toolhub::directx
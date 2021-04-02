//#endif
#include "CommandSignature.h"
#include "../Shader.h"
#include <Singleton/ShaderID.h>
CommandSignature::CommandSignature(GFXDevice* device, SignatureType sigType, Shader const* drawShader) : sigType(sigType) {
	switch (sigType) {
		case SignatureType::MultiDrawIndirect: {
			D3D12_COMMAND_SIGNATURE_DESC desc;
			D3D12_INDIRECT_ARGUMENT_DESC indDesc[4];
			ZeroMemory(&desc, sizeof(D3D12_COMMAND_SIGNATURE_DESC));
			ZeroMemory(indDesc, 4 * sizeof(D3D12_INDIRECT_ARGUMENT_DESC));
			indDesc[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
			indDesc[0].ConstantBufferView.RootParameterIndex = drawShader->GetPropertyRootSigPos(ShaderID::GetPerObjectBufferID());
			indDesc[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
			indDesc[1].VertexBuffer.Slot = 0;
			indDesc[2].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
			indDesc[3].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
			desc.ByteStride = sizeof(MultiDrawCommand);
			desc.NodeMask = 0;
			desc.NumArgumentDescs = 4;
			desc.pArgumentDescs = indDesc;
			ThrowIfFailed(device->CreateCommandSignature(&desc, drawShader->mRootSignature.Get(), IID_PPV_ARGS(&mCommandSignature)));
		} break;
		case SignatureType::DrawInstanceIndirect: {
			D3D12_COMMAND_SIGNATURE_DESC desc;
			D3D12_INDIRECT_ARGUMENT_DESC indDesc;
			ZeroMemory(&desc, sizeof(D3D12_COMMAND_SIGNATURE_DESC));
			ZeroMemory(&indDesc, sizeof(D3D12_INDIRECT_ARGUMENT_DESC));
			indDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
			desc.ByteStride = sizeof(InstanceIndirectCommand);
			desc.NodeMask = 0;
			desc.NumArgumentDescs = 1;
			desc.pArgumentDescs = &indDesc;
			ThrowIfFailed(device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&mCommandSignature)));
		} break;
		case SignatureType::DispatchComputeIndirect: {
			D3D12_COMMAND_SIGNATURE_DESC desc;
			D3D12_INDIRECT_ARGUMENT_DESC indDesc;
			ZeroMemory(&desc, sizeof(D3D12_COMMAND_SIGNATURE_DESC));
			ZeroMemory(&indDesc, sizeof(D3D12_INDIRECT_ARGUMENT_DESC));
			indDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
			desc.ByteStride = sizeof(uint) * 3;
			desc.NodeMask = 0;
			desc.NumArgumentDescs = 1;
			desc.pArgumentDescs = &indDesc;
			ThrowIfFailed(device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&mCommandSignature)));
		} break;
	}
}

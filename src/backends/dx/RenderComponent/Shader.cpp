//#endif
#include <RenderComponent/Shader.h>
#include <Singleton/ShaderID.h>
#include <RenderComponent/TextureBase.h>
#include <RenderComponent/DescriptorHeap.h>
#include <RenderComponent/UploadBuffer.h>
#include <RenderComponent/TextureBase.h>
#include <RenderComponent/RenderTexture.h>
#include <fstream>
#include <JobSystem/JobInclude.h>
#include <RenderComponent/Utility/ShaderIO.h>
#include <RenderComponent/StructuredBuffer.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/Mesh.h>
#include <PipelineComponent/ThreadCommand.h>
using Microsoft::WRL::ComPtr;
Shader::~Shader() {
}
void Shader::BindShader(ThreadCommand* commandList) const {
	if (!commandList->UpdateRegisterShader(this)) return;
	commandList->GetCmdList()->SetGraphicsRootSignature(mRootSignature.Get());
}
void Shader::BindShader(ThreadCommand* commandList, const DescriptorHeap* descHeap) const {
	if (!commandList->UpdateRegisterShader(this)) return;
	commandList->GetCmdList()->SetGraphicsRootSignature(mRootSignature.Get());
	if (descHeap)
		descHeap->SetDescriptorHeap(commandList);
}
void Shader::GetPassPSODesc(uint pass, D3D12_GRAPHICS_PIPELINE_STATE_DESC* targetPSO) const {
	const ShaderPass& p = allPasses[pass];
	if (p.vsShader) {
		targetPSO->VS =
			{
				reinterpret_cast<BYTE*>(p.vsShader->GetBufferPointer()),
				p.vsShader->GetBufferSize()};
	}
	if (p.psShader) {
		targetPSO->PS =
			{
				reinterpret_cast<BYTE*>(p.psShader->GetBufferPointer()),
				p.psShader->GetBufferSize()};
	}
	if (p.hsShader) {
		targetPSO->HS =
			{
				reinterpret_cast<BYTE*>(p.hsShader->GetBufferPointer()),
				p.hsShader->GetBufferSize()};
	}
	if (p.dsShader) {
		targetPSO->DS =
			{
				reinterpret_cast<BYTE*>(p.dsShader->GetBufferPointer()),
				p.dsShader->GetBufferSize()};
	}
	targetPSO->BlendState = p.blendState;
	targetPSO->RasterizerState = p.rasterizeState;
	targetPSO->pRootSignature = mRootSignature.Get();
	targetPSO->DepthStencilState = p.depthStencilState;
}
uint Shader::GetPassIndex(const vengine::string& name) const {
	auto ite = passName.Find(name);
	if (!ite) return -1;
	return ite.Value();
}
Shader::Shader(vengine::string const& name, GFXDevice* device, const vengine::string& csoFilePath) : name(name) {
	ShaderIO::DecodeShader(csoFilePath, mVariablesVector, allPasses, serObj);
	for (auto ite = allPasses.begin(); ite != allPasses.end(); ite++) {
		passName.Insert(ite->name, ite.GetIndex());
	}
	mVariablesDict.Reserve(mVariablesVector.size() + 2);
	for (int32_t i = 0; i < mVariablesVector.size(); ++i) {
		ShaderVariable& variable = mVariablesVector[i];
		mVariablesDict.Insert(ShaderID::PropertyToID(variable.name), i);
	}
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = ShaderIO::GetRootSignature(
		mVariablesVector,
		serializedRootSig,
		errorBlob,
		D3D_ROOT_SIGNATURE_VERSION_1_0);
	if (errorBlob != nullptr) {
		::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);
	ThrowIfFailed(device->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}

bool Shader::SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* targetDesc, uint64 indexOffset) const {
	ShaderVariable var;
	uint rootSigPos;
	if (!VariableReflection(id, targetDesc, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			ID3D12DescriptorHeap* heap = nullptr;
			if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::SRV)
				return false;
			heap = targetDesc->Get();
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				targetDesc->hGPU(indexOffset));
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::UAV)
				return false;
			ID3D12DescriptorHeap* heap = targetDesc->Get();
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				targetDesc->hGPU(indexOffset));
		} break;
		default:
			return false;
	}
	return true;
}
bool Shader::SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* targetObj, uint64 indexOffset) const {
	ShaderVariable var;
	uint rootSigPos;
	if (!VariableReflection(id, targetObj, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::ConstantBuffer: {
			commandList->GetCmdList()->SetGraphicsRootConstantBufferView(
				rootSigPos,
				targetObj->GetAddress(indexOffset).address);
		} break;
		case ShaderVariableType::StructuredBuffer: {
			commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
				rootSigPos,
				targetObj->GetAddress(indexOffset).address);
		} break;
		case ShaderVariableType::RWStructuredBuffer: {
			commandList->GetCmdList()->SetGraphicsRootUnorderedAccessView(
				rootSigPos,
				targetObj->GetAddress(indexOffset).address);
		} break;
		default:
			return false;
	}
	return true;
}
bool Shader::SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* sbufferPtr, uint64 indexOffset) const {
	ShaderVariable var;
	uint rootSigPos;
	if (!VariableReflection(id, sbufferPtr, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::ConstantBuffer: {
			commandList->GetCmdList()->SetGraphicsRootConstantBufferView(
				rootSigPos,
				sbufferPtr->GetAddress(0, indexOffset).address);
		} break;
		case ShaderVariableType::StructuredBuffer: {
			commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
				rootSigPos,
				sbufferPtr->GetAddress(0, indexOffset).address);
		} break;
		case ShaderVariableType::RWStructuredBuffer: {
			commandList->GetCmdList()->SetGraphicsRootUnorderedAccessView(
				rootSigPos,
				sbufferPtr->GetAddress(0, indexOffset).address);
		} break;
		default:
			return false;
	}
	return true;
}
bool Shader::SetResource(ThreadCommand* commandList, uint id, Mesh const* meshPtr, uint64 byteOffset) const {
	ShaderVariable var;
	uint rootSigPos;
	if (!VariableReflection(id, meshPtr, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::ConstantBuffer: {
			commandList->GetCmdList()->SetGraphicsRootConstantBufferView(
				rootSigPos,
				meshPtr->GetResource()->GetGPUVirtualAddress() + byteOffset);
		} break;
		case ShaderVariableType::StructuredBuffer: {
			commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
				rootSigPos,
				meshPtr->GetResource()->GetGPUVirtualAddress() + byteOffset);
		} break;
		case ShaderVariableType::RWStructuredBuffer: {
			commandList->GetCmdList()->SetGraphicsRootUnorderedAccessView(
				rootSigPos,
				meshPtr->GetResource()->GetGPUVirtualAddress() + byteOffset);
		} break;
		default:
			return false;
	}
	return true;
}
bool Shader::SetResource(ThreadCommand* commandList, uint id, TextureBase const* targetObj) const {
	ShaderVariable var;
	uint rootSigPos;
	if (!VariableReflection(id, targetObj, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalDescIndex()));
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalUAVDescIndex(0)));
		} break;
		default:
			return false;
	}
	return true;
}
bool Shader::SetResource(ThreadCommand* commandList, uint id, RenderTexture const* targetObj, uint64 uavMipLevel) const {
	ShaderVariable var;
	uint rootSigPos;
	if (!VariableReflection(id, targetObj, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalDescIndex()));
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalUAVDescIndex(uavMipLevel)));
		} break;
		default:
			return false;
	}
	return true;
}
bool Shader::SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const {
	auto ite = mVariablesDict.Find(id);
	if (!ite) return false;
	uint rootSigPos = ite.Value();
	const ShaderVariable& var = mVariablesVector[rootSigPos];
	switch (var.type) {
		case ShaderVariableType::ConstantBuffer:
			commandList->GetCmdList()->SetGraphicsRootConstantBufferView(
				rootSigPos,
				address.address);
			break;
		case ShaderVariableType::StructuredBuffer:
			commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
				rootSigPos,
				address.address);
			break;
		case ShaderVariableType::RWStructuredBuffer:
			commandList->GetCmdList()->SetGraphicsRootUnorderedAccessView(
				rootSigPos,
				address.address);
			break;
		default:
			return false;
	}
	return true;
}

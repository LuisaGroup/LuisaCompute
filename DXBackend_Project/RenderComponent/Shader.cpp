//#endif
#include "Shader.h"
#include "../Singleton/ShaderID.h"
#include "Texture.h"
#include "../RenderComponent/DescriptorHeap.h"
#include "UploadBuffer.h"
#include <fstream>
#include "../JobSystem/JobInclude.h"
#include "Utility/ShaderIO.h"
#include "StructuredBuffer.h"
#include "Mesh.h"
#include "../PipelineComponent/ThreadCommand.h"
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
int32_t Shader::GetPropertyRootSigPos(uint id) const {
	auto ite = mVariablesDict.Find(id);
	if (!ite) return -1;
	return (int32_t)ite.Value();
}
/*
bool Shader::TrySetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid)const
{
	if (!targetObj) return false;
	auto ite = mVariablesDict.Find(id);
	if (!ite) return false;
	SetResWithoutCheck(commandList, ite, targetObj, indexOffset, tyid);
	return true;
}*/
bool Shader::SetResWithoutCheck(ThreadCommand* commandList, HashMap<uint, uint>::Iterator& ite, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const {
	uint rootSigPos = ite.Value();
	const ShaderVariable& var = mVariablesVector[rootSigPos];
	const UploadBuffer* uploadBufferPtr;
	ID3D12DescriptorHeap* heap = nullptr;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap:
			if (tyid != typeid(DescriptorHeap)) {
				return false;
			}
			{
				const DescriptorHeap* targetDesc = static_cast<const DescriptorHeap*>(targetObj);
				if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::SRV)
					return false;
				heap = targetDesc->Get();
				commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
					rootSigPos,
					targetDesc->hGPU(indexOffset));
			}
			break;
		case ShaderVariableType::UAVDescriptorHeap: {
			if (tyid != typeid(DescriptorHeap)) {
				return false;
			}
			const DescriptorHeap* targetDesc = static_cast<const DescriptorHeap*>(targetObj);
			if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::UAV)
				return false;
			heap = targetDesc->Get();
			commandList->GetCmdList()->SetGraphicsRootDescriptorTable(
				rootSigPos,
				targetDesc->hGPU(indexOffset));
		} break;
		case ShaderVariableType::ConstantBuffer:
			if (tyid != typeid(UploadBuffer) && tyid != typeid(StructuredBuffer)) {
				return false;
			}
			if (tyid == typeid(UploadBuffer)) {
				uploadBufferPtr = static_cast<const UploadBuffer*>(targetObj);
				commandList->GetCmdList()->SetGraphicsRootConstantBufferView(
					rootSigPos,
					uploadBufferPtr->GetAddress(indexOffset).address);
			} else {
				const StructuredBuffer* sbufferPtr = static_cast<const StructuredBuffer*>(targetObj);
				commandList->GetCmdList()->SetGraphicsRootConstantBufferView(
					rootSigPos,
					sbufferPtr->GetAddress(0, indexOffset).address);
			}
			break;
		case ShaderVariableType::StructuredBuffer:
			if (tyid != typeid(UploadBuffer) && tyid != typeid(StructuredBuffer) && tyid != typeid(Mesh)) {
				return false;
			}
			if (tyid == typeid(UploadBuffer)) {
				uploadBufferPtr = static_cast<const UploadBuffer*>(targetObj);
				commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
					rootSigPos,
					uploadBufferPtr->GetAddress(indexOffset).address);
			} else if (tyid == typeid(Mesh)) {
				const GPUResourceBase* meshPtr = static_cast<const GPUResourceBase*>(targetObj);
				commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
					rootSigPos,
					meshPtr->GetResource()->GetGPUVirtualAddress() + indexOffset);
			} else {
				const StructuredBuffer* sbufferPtr = static_cast<const StructuredBuffer*>(targetObj);
				commandList->GetCmdList()->SetGraphicsRootShaderResourceView(
					rootSigPos,
					sbufferPtr->GetAddress(0, indexOffset).address);
			}
			break;
	}
	return true;
}
bool Shader::SetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const {
	if (targetObj == nullptr) return false;
	auto ite = mVariablesDict.Find(id);
	if (!ite)
		return false;
	return SetResWithoutCheck(commandList, ite, targetObj, indexOffset, tyid);
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
		default:
			return false;
	}
	return true;
}

//#endif
#include <RenderComponent/ComputeShader.h>
#include <Singleton/ShaderID.h>
#include <RenderComponent/UploadBuffer.h>
#include <RenderComponent/DescriptorHeap.h>
#include <Common/GFXUtil.h>
#include <fstream>
#include <RenderComponent/StructuredBuffer.h>
#include <RenderComponent/Mesh.h>
#include <JobSystem/JobInclude.h>
#include <RenderComponent/Utility/ShaderIO.h>
#include <PipelineComponent/ThreadCommand.h>

using Microsoft::WRL::ComPtr;
uint ComputeShader::GetKernelIndex(const vengine::string& str) const {
	auto ite = kernelNames.Find(str);
	if (!ite) return -1;
	return ite.Value();
}
ComputeShader::ComputeShader(
	vengine::string const& name,
	const vengine::string& csoFilePath,
	GFXDevice* device) : cmdSig(device, CommandSignature::SignatureType::DispatchComputeIndirect), name(name) {
	ShaderIO::DecodeComputeShader(csoFilePath, mVariablesVector, csShaders, serObj);
	for (size_t i = 0; i < csShaders.size(); ++i) {
		kernelNames.Insert(csShaders[i].name, i);
	}
	mVariablesDict.Reserve(mVariablesVector.size() + 2);
	for (int32_t i = 0; i < mVariablesVector.size(); ++i) {
		ShaderVariable& variable = mVariablesVector[i];
		mVariablesDict.Insert(ShaderID::PropertyToID(variable.varID), i);
	}
	ComPtr<ID3DBlob> serializedRootSig = nullptr;
	ComPtr<ID3DBlob> errorBlob = nullptr;
	HRESULT hr = ShaderIO::GetRootSignature(
		mVariablesVector,
		serializedRootSig,
		errorBlob,
		D3D_ROOT_SIGNATURE_VERSION_1_0);
	if (errorBlob != nullptr) {
		OutputDebugStringA((char*)errorBlob->GetBufferPointer());
	}
	ThrowIfFailed(hr);
	ThrowIfFailed(device->device()->CreateRootSignature(
		0,
		serializedRootSig->GetBufferPointer(),
		serializedRootSig->GetBufferSize(),
		IID_PPV_ARGS(mRootSignature.GetAddressOf())));
	pso.resize(csShaders.size());
	for (int32_t i = 0; i < csShaders.size(); ++i) {
		D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.pRootSignature = mRootSignature.Get();
		psoDesc.CS =
			{
				reinterpret_cast<BYTE*>(csShaders[i].datas->GetBufferPointer()),
				csShaders[i].datas->GetBufferSize()};
		psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
		ThrowIfFailed(device->device()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso[i])));
	}
}
ComputeShader::~ComputeShader() {}
void ComputeShader::BindShader(ThreadCommand* commandList) const {
	if (!commandList->UpdateRegisterShader(this)) return;
	commandList->GetCmdList()->SetComputeRootSignature(mRootSignature.Get());
}
void ComputeShader::BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const {
	if (!commandList->UpdateRegisterShader(this)) return;
	commandList->GetCmdList()->SetComputeRootSignature(mRootSignature.Get());
	heap->SetDescriptorHeap(commandList);
}
bool ComputeShader::SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const {
	return ShaderIO::SetComputeBufferByAddress(
		mVariablesDict,
		mVariablesVector,
		commandList,
		id,
		address);
}

bool ComputeShader::SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool ComputeShader::SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool ComputeShader::SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool ComputeShader::SetResource(ThreadCommand* commandList, uint id, Mesh const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool ComputeShader::SetResource(ThreadCommand* commandList, uint id, TextureBase const* obj) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj);
}
bool ComputeShader::SetResource(ThreadCommand* commandList, uint id, RenderTexture const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
void ComputeShader::Dispatch(ThreadCommand* commandList, uint kernel, uint x, uint y, uint z) const {
	commandList->ExecuteResBarrier();
	auto pso = this->pso[kernel].Get();
	commandList->UpdatePSO(pso);
	commandList->GetCmdList()->Dispatch(x, y, z);
}
void ComputeShader::DispatchIndirect(ThreadCommand* commandList, uint dispatchKernel, StructuredBuffer* indirectBuffer, uint bufferElement, uint bufferIndex) const {
	commandList->ExecuteResBarrier();
	auto pso = this->pso[dispatchKernel].Get();
	commandList->UpdatePSO(pso);
	commandList->GetCmdList()->ExecuteIndirect(cmdSig.GetSignature(), 1, indirectBuffer->GetResource(), indirectBuffer->GetAddressOffset(bufferElement, bufferIndex), nullptr, 0);
}

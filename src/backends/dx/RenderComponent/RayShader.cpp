#include "RayShader.h"
#include "../Singleton/ShaderID.h"
#include "Texture.h"
#include "../RenderComponent/DescriptorHeap.h"
#include "UploadBuffer.h"
#include <fstream>
#include "../JobSystem/JobInclude.h"
#include "Utility/ShaderIO.h"
#include "StructuredBuffer.h"
#include "../PipelineComponent/ThreadCommand.h"
#include "Mesh.h"
using Microsoft::WRL::ComPtr;
RayShader::RayShader(GFXDevice* device, vengine::string const& path) {
	vengine::vector<char> binaryData;
	uint64 raypayloadShader = 0;
	uint64 recursiveCount = 0;
	ShaderIO::DecodeDXRShader(
		path,
		mVariablesVector,
		hitGroups,
		binaryData,
		recursiveCount,
		raypayloadShader,
		serObj);
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
	CD3DX12_STATE_OBJECT_DESC raytracingPipeline{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE};
	// DXIL library
	// This contains the shaders and their entrypoints for the state object.
	// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
	auto lib = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
	D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE(binaryData.data(), binaryData.size());
	lib->SetDXILLibrary(&libdxil);
	{
		auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
		if (!hitGroups.functions[(uint8_t)HitGroupFunctionType::ClosestHit].empty()) {
			hitGroup->SetClosestHitShaderImport(vengine::wstring(hitGroups.functions[(uint8_t)HitGroupFunctionType::ClosestHit]).c_str());
		}
		if (!hitGroups.functions[(uint8_t)HitGroupFunctionType::AnyHit].empty()) {
			hitGroup->SetAnyHitShaderImport(vengine::wstring(hitGroups.functions[(uint8_t)HitGroupFunctionType::AnyHit]).c_str());
		}
		if (!hitGroups.functions[(uint8_t)HitGroupFunctionType::Intersect].empty()) {
			hitGroup->SetIntersectionShaderImport(vengine::wstring(hitGroups.functions[(uint8_t)HitGroupFunctionType::Intersect]).c_str());
		}
		hitGroup->SetHitGroupExport(vengine::wstring(hitGroups.name).c_str());
		hitGroup->SetHitGroupType((D3D12_HIT_GROUP_TYPE)hitGroups.shaderType);
	}
	identifierBuffer.New(
		device,
		3,
		false,
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

	auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
	uint payloadSize = raypayloadShader;// float4 color
	uint attributeSize = sizeof(float2);// float2 barycentrics
	shaderConfig->Config(payloadSize, attributeSize);
	auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
	globalRootSignature->SetRootSignature(mRootSignature.Get());
	auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	pipelineConfig->Config(recursiveCount);
	ThrowIfFailed(static_cast<ID3D12Device5*>(device)->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&mStateObj)));
	ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
	ThrowIfFailed(mStateObj.As(&stateObjectProperties));
	auto BindIdentifier = [&](uint64 bufferIndex, vengine::string const& name) -> std::pair<GpuAddress, uint64> {
		if (name.empty()) return {{0}, 0};
		void* ptr = stateObjectProperties->GetShaderIdentifier(vengine::wstring(name).c_str());
		if (!ptr) return {{0}, 0};
		identifierBuffer->CopyData(
			bufferIndex,
			ptr, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
		return {identifierBuffer->GetAddress(bufferIndex), D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
	};
	auto rayGen = BindIdentifier(0, hitGroups.functions[(uint8_t)HitGroupFunctionType::RayGeneration]);
	auto miss = BindIdentifier(1, hitGroups.functions[(uint8_t)HitGroupFunctionType::Miss]);
	auto hitGroup = BindIdentifier(2, hitGroups.name);
	hitGroups.rayGenVAddress = rayGen.first;
	hitGroups.rayGenSize = rayGen.second;
	hitGroups.missVAddress = miss.first;
	hitGroups.missSize = miss.second;
	hitGroups.hitGroupVAddress = hitGroup.first;
	hitGroups.hitGroupSize = hitGroup.second;
}
RayShader::~RayShader() {
}
void RayShader::DispatchRays(
	ThreadCommand* originCmdList,
	uint width,
	uint height,
	uint depth) const {
	originCmdList->ExecuteResBarrier();
	ID3D12GraphicsCommandList4* cmdList = static_cast<ID3D12GraphicsCommandList4*>(originCmdList->GetCmdList());
	D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
	dispatchDesc.Depth = depth;
	dispatchDesc.Height = height;
	dispatchDesc.Width = width;
	dispatchDesc.HitGroupTable.StartAddress = hitGroups.hitGroupVAddress.address;
	dispatchDesc.HitGroupTable.SizeInBytes = hitGroups.hitGroupSize;
	dispatchDesc.HitGroupTable.StrideInBytes = hitGroups.hitGroupSize;
	dispatchDesc.RayGenerationShaderRecord.SizeInBytes = hitGroups.rayGenSize;
	dispatchDesc.RayGenerationShaderRecord.StartAddress = hitGroups.rayGenVAddress.address;
	dispatchDesc.MissShaderTable.SizeInBytes = hitGroups.missSize;
	dispatchDesc.MissShaderTable.StrideInBytes = hitGroups.missSize;
	dispatchDesc.MissShaderTable.StartAddress = hitGroups.missVAddress.address;
	cmdList->SetPipelineState1(mStateObj.Get());
	cmdList->DispatchRays(
		&dispatchDesc);
}

void RayShader::BindShader(ThreadCommand* commandList) const {
	if (!commandList->UpdateRegisterShader(this)) return;
	commandList->GetCmdList()->SetComputeRootSignature(mRootSignature.Get());
}
void RayShader::BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const {
	if (!commandList->UpdateRegisterShader(this)) return;
	commandList->GetCmdList()->SetComputeRootSignature(mRootSignature.Get());
	heap->SetDescriptorHeap(commandList);
}
bool RayShader::SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const {
	return ShaderIO::SetComputeBufferByAddress(
		mVariablesDict,
		mVariablesVector,
		commandList,
		id,
		address);
}
bool RayShader::SetResource(ThreadCommand* commandList, uint id, DescriptorHeap const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool RayShader::SetResource(ThreadCommand* commandList, uint id, UploadBuffer const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool RayShader::SetResource(ThreadCommand* commandList, uint id, StructuredBuffer const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool RayShader::SetResource(ThreadCommand* commandList, uint id, Mesh const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
bool RayShader::SetResource(ThreadCommand* commandList, uint id, TextureBase const* obj) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj);
}
bool RayShader::SetResource(ThreadCommand* commandList, uint id, RenderTexture const* obj, uint64 offset) const {
	return ShaderIO::SetComputeResource(this, commandList, id, obj, offset);
}
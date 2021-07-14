//#endif
#include <RenderComponent/Utility/ShaderIO.h>
#include <fstream>
#include <RenderComponent/RenderComponentInclude.h>
#include <Utility/BinaryReader.h>
#include <PipelineComponent/ThreadCommand.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/UploadBuffer.h>
#include <RenderComponent/StructuredBuffer.h>
#include <RenderComponent/TextureBase.h>
#include <RenderComponent/RenderTexture.h>
namespace ShaderIOGlobal {
template<typename T>
void DragData(BinaryReader& ifs, T& data) {
	ifs.Read((char*)&data, sizeof(T));
}
template<>
void DragData<vstd::string>(BinaryReader& ifs, vstd::string& str) {
	uint32_t length = 0;
	DragData<uint32_t>(ifs, length);
	str.clear();
	if (length == 0) return;
	str.resize(length);
	ifs.Read(str.data(), length);
}
void GetShaderVariable(
	vstd::vector<ShaderVariable>& vars,
	BinaryReader& ifs) {
	uint varSize = 0;
	DragData<uint>(ifs, varSize);
	vars.resize(varSize);
	for (auto i = vars.begin(); i != vars.end(); ++i) {
		DragData<uint>(ifs, i->varID);
		DragData<ShaderVariableType>(ifs, i->type);
		DragData<uint>(ifs, i->tableSize);
		DragData<uint>(ifs, i->registerPos);
		DragData<uint>(ifs, i->space);
	}
}
void GetShaderSerializedObject(
	BinaryReader& ifs,
	StackObject<BinaryJson, true>& serObj) {
	uint64 serObjSize = 0;
	DragData<uint64>(ifs, serObjSize);
	if (serObjSize > 0) {
		vstd::vector<char> jsonObj(serObjSize);
		ifs.Read(jsonObj.data(), serObjSize);
		serObj.New(jsonObj);
	} else {
		serObj.Delete();
	}
}
}// namespace ShaderIOGlobal

void ShaderIO::DecodeComputeShader(
	const vstd::string& fileName,
	vstd::vector<ShaderVariable>& vars,
	vstd::vector<ComputeKernel>& datas,
	StackObject<BinaryJson, true>& serObj) {
	using namespace ShaderIOGlobal;
	vars.clear();
	datas.clear();
	BinaryReader ifs(fileName);
	if (!ifs) return;
	GetShaderSerializedObject(ifs, serObj);
	GetShaderVariable(vars, ifs);
	uint blobSize = 0;
	DragData<uint>(ifs, blobSize);
	vstd::vector<Microsoft::WRL::ComPtr<ID3DBlob>, VEngine_AllocType::Stack> kernelBlobs(blobSize);
	for (auto&& i : kernelBlobs) {
		uint64_t kernelSize = 0;
		DragData<uint64_t>(ifs, kernelSize);
		D3DCreateBlob(kernelSize, &i);
		ifs.Read((char*)i->GetBufferPointer(), kernelSize);
	}
	uint kernelSize = 0;
	DragData<uint>(ifs, kernelSize);
	datas.resize(kernelSize);
	for (auto i = datas.begin(); i != datas.end(); ++i) {
		uint64_t kernelSize = 0;
		DragData(ifs, i->name);
		uint index = 0;
		DragData<uint>(ifs, index);
		i->datas = kernelBlobs[index];
	}
}
HRESULT ShaderIO::GetRootSignature(
	vstd::vector<ShaderVariable> const& variables,
	Microsoft::WRL::ComPtr<ID3DBlob>& serializedRootSig,
	Microsoft::WRL::ComPtr<ID3DBlob>& errorBlob,
	D3D_ROOT_SIGNATURE_VERSION rootSigVersion) {
	vstd::vector<CD3DX12_ROOT_PARAMETER, VEngine_AllocType::Stack> allParameter;
	auto&& staticSamplers = GFXUtil::GetStaticSamplers();
	vstd::vector<CD3DX12_DESCRIPTOR_RANGE, VEngine_AllocType::Stack> allTexTable;
	allParameter.reserve(variables.size());
	allTexTable.reserve(variables.size());
	for (auto&& var : variables) {
		switch (var.type) {
			case ShaderVariableType::SRVDescriptorHeap: {
				allTexTable.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, var.tableSize, var.registerPos, var.space);
			} break;
			case ShaderVariableType::UAVDescriptorHeap: {
				allTexTable.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, var.tableSize, var.registerPos, var.space);
			} break;
			case ShaderVariableType::SamplerDescHeap: {
				allTexTable.emplace_back().Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, var.tableSize, var.registerPos, var.space);
			} break;
		}
	}
	uint offset = 0;
	[&]() {
	for (auto&& var : variables) {
		CD3DX12_ROOT_PARAMETER slotRootParameter;
		switch (var.type) {
			case ShaderVariableType::SRVDescriptorHeap:
			case ShaderVariableType::UAVDescriptorHeap:
			case ShaderVariableType::SamplerDescHeap:
				slotRootParameter.InitAsDescriptorTable(1, allTexTable.data() + offset);
				offset++;
				break;
			case ShaderVariableType::ConstantBuffer:
				slotRootParameter.InitAsConstantBufferView(var.registerPos, var.space);
				break;
			case ShaderVariableType::StructuredBuffer:
				slotRootParameter.InitAsShaderResourceView(var.registerPos, var.space);
				break;
			case ShaderVariableType::RWStructuredBuffer:
				slotRootParameter.InitAsUnorderedAccessView(var.registerPos, var.space);
				break;
			default:
				return;
		}
		allParameter.push_back(slotRootParameter);
	} }();
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(allParameter.size(), allParameter.data(),
											(uint)staticSamplers.size(), staticSamplers.data(),
											D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	return D3D12SerializeRootSignature(&rootSigDesc, rootSigVersion,
									   serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
}
bool ShaderIO::SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, DescriptorHeap const* targetDesc, uint64 indexOffset) {
	ShaderVariable var;
	uint rootSigPos;
	if (!shader->VariableReflection(id, targetDesc, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			ID3D12DescriptorHeap* heap = nullptr;
			if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::SRV)
				return false;
			heap = targetDesc->Get();
			commandList->GetCmdList()->SetComputeRootDescriptorTable(
				rootSigPos,
				targetDesc->hGPU(indexOffset));
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::UAV)
				return false;
			ID3D12DescriptorHeap* heap = targetDesc->Get();
			commandList->GetCmdList()->SetComputeRootDescriptorTable(
				rootSigPos,
				targetDesc->hGPU(indexOffset));
		} break;
		default:
			return false;
	}
	return true;
}
bool ShaderIO::SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, UploadBuffer const* targetObj, uint64 indexOffset) {
	ShaderVariable var;
	uint rootSigPos;
	if (!shader->VariableReflection(id, targetObj, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::ConstantBuffer: {
			commandList->GetCmdList()->SetComputeRootConstantBufferView(
				rootSigPos,
				targetObj->GetAddress(indexOffset).address);
		} break;
		case ShaderVariableType::StructuredBuffer: {
			commandList->GetCmdList()->SetComputeRootShaderResourceView(
				rootSigPos,
				targetObj->GetAddress(indexOffset).address);
		} break;
		case ShaderVariableType::RWStructuredBuffer: {
			commandList->GetCmdList()->SetComputeRootShaderResourceView(
				rootSigPos,
				targetObj->GetAddress(indexOffset).address);
		} break;
		default:
			return false;
	}
	return true;
}
bool ShaderIO::SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, StructuredBuffer const* sbufferPtr, uint64 indexOffset) {
	ShaderVariable var;
	uint rootSigPos;
	if (!shader->VariableReflection(id, sbufferPtr, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::ConstantBuffer: {
			commandList->GetCmdList()->SetComputeRootConstantBufferView(
				rootSigPos,
				sbufferPtr->GetAddress(0, indexOffset).address);
		} break;
		case ShaderVariableType::StructuredBuffer: {
			commandList->GetCmdList()->SetComputeRootShaderResourceView(
				rootSigPos,
				sbufferPtr->GetAddress(0, indexOffset).address);
		} break;
		case ShaderVariableType::RWStructuredBuffer: {
			commandList->GetCmdList()->SetComputeRootUnorderedAccessView(
				rootSigPos,
				sbufferPtr->GetAddress(0, indexOffset).address);
		} break;
		default:
			return false;
	}
	return true;
}
bool ShaderIO::SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, TextureBase const* targetObj) {
	ShaderVariable var;
	uint rootSigPos;
	if (!shader->VariableReflection(id, targetObj, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			commandList->GetCmdList()->SetComputeRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalDescIndex()));
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			commandList->GetCmdList()->SetComputeRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalUAVDescIndex(0)));
		} break;
		default:
			return false;
	}
	return true;
}
bool ShaderIO::SetComputeResource(IShader const* shader, ThreadCommand* commandList, uint id, RenderTexture const* targetObj, uint64 uavMipLevel) {
	ShaderVariable var;
	uint rootSigPos;
	if (!shader->VariableReflection(id, targetObj, rootSigPos, var)) return false;
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			commandList->GetCmdList()->SetComputeRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalDescIndex()));
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			commandList->GetCmdList()->SetComputeRootDescriptorTable(
				rootSigPos,
				Graphics::GetGlobalDescHeap()->hGPU(targetObj->GetGlobalUAVDescIndex(uavMipLevel)));
		} break;
		default:
			return false;
	}
	return true;
}
bool ShaderIO::SetComputeBufferByAddress(
	HashMap<uint, uint> const& varDict,
	vstd::vector<ShaderVariable> const& varVector,
	ThreadCommand* tCmd,
	uint id,
	GpuAddress address) {
	auto ite = varDict.Find(id);
	if (!ite) return false;
	uint rootSigPos = ite.Value();
	const ShaderVariable& var = varVector[rootSigPos];
	auto commandList = tCmd->GetCmdList();
	switch (var.type) {
		case ShaderVariableType::RWStructuredBuffer:
			commandList->SetComputeRootUnorderedAccessView(
				rootSigPos,
				address.address);
			break;
		case ShaderVariableType::StructuredBuffer:
			commandList->SetComputeRootShaderResourceView(
				rootSigPos,
				address.address);
			break;
		case ShaderVariableType::ConstantBuffer:
			commandList->SetComputeRootConstantBufferView(
				rootSigPos,
				address.address);
			break;
		default:
			return false;
	}
	return true;
}
void ShaderIO::DecodeDXRShader(
	const vstd::string& fileName,
	vstd::vector<ShaderVariable>& vars,
	DXRHitGroup& hitGroup,
	vstd::vector<char>& binaryData,
	uint64& recursiveCount,
	uint64& raypayloadSize,
	StackObject<BinaryJson, true>& serObj) {
	using namespace ShaderIOGlobal;
	vars.clear();
	binaryData.clear();
	binaryData.clear();
	BinaryReader ifs(fileName);
	if (!ifs) return;
	GetShaderSerializedObject(ifs, serObj);
	GetShaderVariable(vars, ifs);
	DragData(ifs, recursiveCount);
	DragData(ifs, raypayloadSize);
	DragData(ifs, hitGroup.name);
	DragData(ifs, hitGroup.shaderType);
	for (auto& f : hitGroup.functions) {
		DragData(ifs, f);
	}
	uint64 binaryDataSize = 0;
	DragData(ifs, binaryDataSize);
	binaryData.resize(binaryDataSize);
	ifs.Read(
		binaryData.data(),
		binaryDataSize);
}

//#endif
#include "ShaderIO.h"
#include <fstream>
#include "../RenderComponentInclude.h"
#include <Utility/BinaryReader.h>
#include <PipelineComponent/ThreadCommand.h>
#include <Singleton/Graphics.h>
namespace ShaderIOGlobal {
template<typename T>
void DragData(BinaryReader& ifs, T& data) {
	ifs.Read((char*)&data, sizeof(T));
}
template<>
void DragData<vengine::string>(BinaryReader& ifs, vengine::string& str) {
	uint32_t length = 0;
	DragData<uint32_t>(ifs, length);
	str.clear();
	if (length == 0) return;
	str.resize(length);
	ifs.Read(str.data(), length);
}
void GetShaderVariable(
	vengine::vector<ShaderVariable>& vars,
	BinaryReader& ifs) {
	uint varSize = 0;
	DragData<uint>(ifs, varSize);
	vars.resize(varSize);
	for (auto i = vars.begin(); i != vars.end(); ++i) {
		DragData<vengine::string>(ifs, i->name);
		DragData<ShaderVariableType>(ifs, i->type);
		DragData<uint>(ifs, i->tableSize);
		DragData<uint>(ifs, i->registerPos);
		DragData<uint>(ifs, i->space);
	}
}
void GetShaderSerializedObject(
	BinaryReader& ifs,
	StackObject<SerializedObject, true>& serObj) {
	uint64 serObjSize = 0;
	DragData<uint64>(ifs, serObjSize);
	if (serObjSize > 0) {
		vengine::vector<char> jsonObj(serObjSize);
		ifs.Read(jsonObj.data(), serObjSize);
		serObj.New(jsonObj);
	} else {
		serObj.Delete();
	}
}
}// namespace ShaderIOGlobal
void ShaderIO::DecodeShader(
	const vengine::string& fileName,
	vengine::vector<ShaderVariable>& vars,
	vengine::vector<ShaderPass>& passes,
	StackObject<SerializedObject, true>& serObj) {
	using namespace ShaderIOGlobal;
	vars.clear();
	passes.clear();
	BinaryReader ifs(fileName);
	if (!ifs) return;
	GetShaderSerializedObject(ifs, serObj);
	GetShaderVariable(vars, ifs);
	uint functionCount = 0;
	DragData<uint>(ifs, functionCount);
	vengine::vector<Microsoft::WRL::ComPtr<ID3DBlob>> functions(functionCount);
	for (uint i = 0; i < functionCount; ++i) {
		uint64_t codeSize = 0;
		DragData(ifs, codeSize);
		if (codeSize > 0) {
			D3DCreateBlob(codeSize, &functions[i]);
			ifs.Read((char*)functions[i]->GetBufferPointer(), codeSize);
		}
	}
	uint passSize = 0;
	DragData<uint>(ifs, passSize);
	passes.resize(passSize);
	for (auto i = passes.begin(); i != passes.end(); ++i) {
		DragData(ifs, i->name);
		auto& name = i->name;
		DragData(ifs, i->rasterizeState);
		DragData(ifs, i->depthStencilState);
		DragData(ifs, i->blendState);
		int32_t vertIndex = 0, fragIndex = 0, hullIndex = 0, domainIndex = 0;
		DragData(ifs, vertIndex);
		DragData(ifs, hullIndex);
		DragData(ifs, domainIndex);
		DragData(ifs, fragIndex);
		i->vsShader = vertIndex >= 0 ? functions[vertIndex] : nullptr;
		i->hsShader = hullIndex >= 0 ? functions[hullIndex] : nullptr;
		i->dsShader = domainIndex >= 0 ? functions[domainIndex] : nullptr;
		i->psShader = fragIndex >= 0 ? functions[fragIndex] : nullptr;
	}
}
void ShaderIO::DecodeComputeShader(
	const vengine::string& fileName,
	vengine::vector<ShaderVariable>& vars,
	vengine::vector<ComputeKernel>& datas,
	StackObject<SerializedObject, true>& serObj) {
	using namespace ShaderIOGlobal;
	vars.clear();
	datas.clear();
	BinaryReader ifs(fileName);
	if (!ifs) return;
	GetShaderSerializedObject(ifs, serObj);
	GetShaderVariable(vars, ifs);
	uint blobSize = 0;
	DragData<uint>(ifs, blobSize);
	vengine::vector<Microsoft::WRL::ComPtr<ID3DBlob>> kernelBlobs(blobSize);
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
	vengine::vector<ShaderVariable> const& variables,
	Microsoft::WRL::ComPtr<ID3DBlob>& serializedRootSig,
	Microsoft::WRL::ComPtr<ID3DBlob>& errorBlob,
	D3D_ROOT_SIGNATURE_VERSION rootSigVersion) {
	vengine::vector<CD3DX12_ROOT_PARAMETER> allParameter;
	auto&& staticSamplers = GFXUtil::GetStaticSamplers();
	vengine::vector<CD3DX12_DESCRIPTOR_RANGE> allTexTable;
	allParameter.reserve(variables.size());
	allTexTable.reserve(variables.size());
	for (auto&& var : variables) {
		if (var.type == ShaderVariableType::SRVDescriptorHeap) {
			CD3DX12_DESCRIPTOR_RANGE texTable;
			texTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, var.tableSize, var.registerPos, var.space);
			allTexTable.push_back(texTable);
		} else if (var.type == ShaderVariableType::UAVDescriptorHeap) {
			CD3DX12_DESCRIPTOR_RANGE texTable;
			texTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, var.tableSize, var.registerPos, var.space);
			allTexTable.push_back(texTable);
		}
	}
	uint offset = 0;
	for (auto&& var : variables) {
		CD3DX12_ROOT_PARAMETER slotRootParameter;
		switch (var.type) {
			case ShaderVariableType::SRVDescriptorHeap:
				slotRootParameter.InitAsDescriptorTable(1, allTexTable.data() + offset);
				offset++;
				break;
			case ShaderVariableType::UAVDescriptorHeap:
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
				goto CONTINUE;
		}
		allParameter.push_back(slotRootParameter);
	}
CONTINUE:
	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(allParameter.size(), allParameter.data(),
											(uint)staticSamplers.size(), staticSamplers.data(),
											D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	return D3D12SerializeRootSignature(&rootSigDesc, rootSigVersion,
									   serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
}
bool ShaderIO::SetComputeShaderResWithoutCheck(
	vengine::vector<ShaderVariable> const& varVector,
	ThreadCommand* tCmd,
	HashMap<uint, uint>::Iterator& ite,
	const VObject* targetObj,
	uint64 indexOffset,
	IShader::ResourceType tyid) {
	uint rootSigPos = ite.Value();
	auto commandList = tCmd->GetCmdList();
	const ShaderVariable& var = varVector[rootSigPos];
	switch (var.type) {
		case ShaderVariableType::SRVDescriptorHeap: {
			switch (tyid) {
				case IShader::ResourceType::DESCRIPTOR_HEAP : {
					const DescriptorHeap* targetDesc = static_cast<const DescriptorHeap*>(targetObj);
					if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::SRV)
						return false;
					ID3D12DescriptorHeap* heap = targetDesc->Get();
					commandList->SetComputeRootDescriptorTable(
						rootSigPos,
						targetDesc->hGPU(indexOffset));
				} break;
				case IShader::ResourceType::TEXTURE: {
#ifdef DEBUG
					if (tCmd->GetBindedHeap() != Graphics::GetGlobalDescHeap()) {
						VEngine_Log("Global Heap not binded!\n"_sv);
						VENGINE_EXIT;
					}
#endif
					const DescriptorHeap* targetDesc = Graphics::GetGlobalDescHeap();
					TextureBase const* tex = static_cast<TextureBase const*>(targetObj);
					commandList->SetComputeRootDescriptorTable(
						rootSigPos,
						targetDesc->hGPU(tex->GetGlobalDescIndex()));
				} break;
				default:
					return false;
			}
			
		} break;
		case ShaderVariableType::UAVDescriptorHeap: {
			switch (tyid) {
				case IShader::ResourceType::DESCRIPTOR_HEAP: {
					const DescriptorHeap* targetDesc = static_cast<const DescriptorHeap*>(targetObj);
					if ((var.tableSize != (uint)-1) && targetDesc->GetBindType(indexOffset) != BindType::UAV)
						return false;
					ID3D12DescriptorHeap* heap = targetDesc->Get();
					commandList->SetComputeRootDescriptorTable(
						rootSigPos,
						targetDesc->hGPU(indexOffset));
				} break;
				case IShader::ResourceType::TEXTURE: {
#ifdef DEBUG
					if (tCmd->GetBindedHeap() != Graphics::GetGlobalDescHeap()) {
						VEngine_Log("Global Heap not binded!\n"_sv);
						VENGINE_EXIT;
					}
#endif
					const DescriptorHeap* targetDesc = Graphics::GetGlobalDescHeap();
					TextureBase const* tex = static_cast<TextureBase const*>(targetObj);
					commandList->SetComputeRootDescriptorTable(
						rootSigPos,
						targetDesc->hGPU(tex->GetGlobalUAVDescIndex(indexOffset)));
				} break;
				default:
					return false;
			}
		} break;
		case ShaderVariableType::ConstantBuffer: {
			static constexpr uint BufferID = (uint)IShader::ResourceType::STRUCTURE_BUFFER | (uint)IShader::ResourceType::UPLOAD_BUFFER;
			if (((uint)tyid & BufferID) == 0) {
				return false;
			}
			if (tyid == IShader::ResourceType::UPLOAD_BUFFER) {
				const UploadBuffer* uploadBufferPtr = static_cast<const UploadBuffer*>(targetObj);
				commandList->SetComputeRootConstantBufferView(
					rootSigPos,
					uploadBufferPtr->GetAddress(indexOffset).address);
			} else {
				const StructuredBuffer* sbufferPtr = static_cast<const StructuredBuffer*>(targetObj);
				commandList->SetComputeRootConstantBufferView(
					rootSigPos,
					sbufferPtr->GetAddress(0, indexOffset).address);
			}
		} break;
		case ShaderVariableType::StructuredBuffer: {
			static constexpr uint BufferMeshID = (uint)IShader::ResourceType::STRUCTURE_BUFFER | (uint)IShader::ResourceType::UPLOAD_BUFFER | (uint)IShader::ResourceType::MESH;
			if (((uint)tyid & BufferMeshID) == 0) {
				return false;
			}
			switch (tyid) {
				case IShader::ResourceType::UPLOAD_BUFFER: {
					const UploadBuffer* uploadBufferPtr = static_cast<const UploadBuffer*>(targetObj);
					commandList->SetComputeRootShaderResourceView(
						rootSigPos,
						uploadBufferPtr->GetAddress(indexOffset).address);
				} break;
				case IShader::ResourceType::MESH: {
					const GPUResourceBase* meshPtr = static_cast<const GPUResourceBase*>(targetObj);
					commandList->SetComputeRootShaderResourceView(
						rootSigPos,
						meshPtr->GetResource()->GetGPUVirtualAddress() + indexOffset);
				} break;
			    default: {
					const StructuredBuffer* sbufferPtr = static_cast<const StructuredBuffer*>(targetObj);
					commandList->SetComputeRootShaderResourceView(
						rootSigPos,
						sbufferPtr->GetAddress(0, indexOffset).address);
				} break;
			}
			break;
		} break;
		case ShaderVariableType::RWStructuredBuffer: {
			static constexpr uint BufferMeshID = (uint)IShader::ResourceType::STRUCTURE_BUFFER | (uint)IShader::ResourceType::MESH;
			if (tyid == IShader::ResourceType::MESH) {
				const GPUResourceBase* meshPtr = static_cast<const GPUResourceBase*>(targetObj);
				commandList->SetComputeRootUnorderedAccessView(
					rootSigPos,
					meshPtr->GetResource()->GetGPUVirtualAddress() + indexOffset);
			} else {
				const StructuredBuffer* sbufferPtr = static_cast<const StructuredBuffer*>(targetObj);
				commandList->SetComputeRootUnorderedAccessView(
					rootSigPos,
					sbufferPtr->GetAddress(0, indexOffset).address);
			}
		} break;
	}
	return true;
}
bool ShaderIO::SetComputeBufferByAddress(
	HashMap<uint, uint> const& varDict,
	vengine::vector<ShaderVariable> const& varVector,
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
	const vengine::string& fileName,
	vengine::vector<ShaderVariable>& vars,
	DXRHitGroup& hitGroup,
	vengine::vector<char>& binaryData,
	uint64& recursiveCount,
	uint64& raypayloadSize,
	StackObject<SerializedObject, true>& serObj) {
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

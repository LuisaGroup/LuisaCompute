#include <RenderComponent/Mesh.h>
#include <RenderComponent/RenderCommand.h>
#include <Singleton/MeshLayout.h>
#include <Singleton/Graphics.h>
#include <Common/vector.h>
#include <Utility/BinaryReader.h>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/Utility/IBufferAllocator.h>
#include <RenderComponent/DescriptorHeap.h>
using namespace DirectX;
using Microsoft::WRL::ComPtr;
namespace MeshGlobal {
static constexpr D3D12_RESOURCE_STATES D3D12_MESH_GENERIC_READ_STATE =
	(D3D12_RESOURCE_STATES)(
		(uint)D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
		| (uint)D3D12_RESOURCE_STATE_INDEX_BUFFER
		| (uint)D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
		| (uint)D3D12_RESOURCE_STATE_COPY_SOURCE);
struct DeleteGuard {
	char* ptr = nullptr;
	~DeleteGuard() {
		if (ptr) vengine_delete(ptr);
	}
};
//CPP
void CreateDefaultBuffer(
	GFXDevice* device,
	UINT64 byteSize,
	ComPtr<GFXResource>& uploadBuffer,
	ComPtr<GFXResource>& defaultBuffer,
	IBufferAllocator* alloc = nullptr,
	Mesh* mesh = nullptr) {
	if (!defaultBuffer) {
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_NONE);
		if (alloc) {
			ID3D12Heap* heap;
			uint64 offset;
			alloc->AllocateTextureHeap(
				device,
				byteSize,
				D3D12_HEAP_TYPE_DEFAULT,
				&heap,
				&offset,
				(uint64)mesh);
			ThrowIfFailed(device->device()->CreatePlacedResource(
				heap,
				offset,
				&buffer,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(defaultBuffer.GetAddressOf())));
		} else {
			// Create the actual default buffer resource.
			auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
			ThrowIfFailed(device->device()->CreateCommittedResource(
				&heap,
				D3D12_HEAP_FLAG_NONE,
				&buffer,
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS(defaultBuffer.GetAddressOf())));
		}
	}
	// In order to copy CPU memory data into our default buffer, we need to create
	// an intermediate upload heap.
	if (!uploadBuffer) {
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
		if (alloc) {
			ID3D12Heap* heap;
			uint64 offset;
			alloc->AllocateTextureHeap(
				device,
				byteSize,
				D3D12_HEAP_TYPE_UPLOAD,
				&heap,
				&offset,
				(uint64)mesh + 1);
			ThrowIfFailed(device->device()->CreatePlacedResource(
				heap,
				offset,
				&buffer,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(uploadBuffer.GetAddressOf())));
		} else {
			auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
			ThrowIfFailed(device->device()->CreateCommittedResource(
				&prop,
				D3D12_HEAP_FLAG_NONE,
				&buffer,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(uploadBuffer.GetAddressOf())));
		}
	}
}
void CopyToBuffer(
	UINT64 byteSize,
	ThreadCommand* cmdList,
	ComPtr<GFXResource>& uploadBuffer,
	ComPtr<GFXResource>& defaultBuffer,
	uint64_t defaultOffset, uint64_t uploadOffset) {
	cmdList->ExecuteResBarrier();
	cmdList->GetCmdList()->CopyBufferRegion(defaultBuffer.Get(), defaultOffset, uploadBuffer.Get(), uploadOffset, byteSize);
}
class MeshLoadCommand : public RenderCommand {
public:
	ComPtr<GFXResource> uploadResource;
	ComPtr<GFXResource> defaultResource;
	UINT64 byteSize;
	uint64_t defaultOffset;
	uint64_t uploadOffset;
	IBufferAllocator* allocator;
	Mesh* mesh;
	ObjWeakPtr<bool> flagPtr;
	GPUResourceState* selfPtrTracker;
	MeshLoadCommand(
		GPUResourceState* stateTracker,
		ObjectPtr<bool> const& flagPtr,
		ComPtr<GFXResource>& uploadResource,
		ComPtr<GFXResource>& defaultResource,
		UINT64 byteSize,
		uint64_t defaultOffset,
		uint64_t uploadOffset,
		IBufferAllocator* allocator = nullptr,
		Mesh* mesh = nullptr)
		: byteSize(byteSize), uploadResource(uploadResource), defaultResource(defaultResource),
		  defaultOffset(defaultOffset),
		  uploadOffset(uploadOffset),
		  allocator(allocator),
		  mesh(mesh),
		  selfPtrTracker(stateTracker),
		  flagPtr(flagPtr) {}
	void Execute(
		GFXDevice* device,
		ThreadCommand* directCommandList,
		ThreadCommand* copyCommandList) {
		if (!flagPtr) return;
		auto allocator = this->allocator;
		auto mesh = this->mesh;
		CopyToBuffer(byteSize, copyCommandList, uploadResource, defaultResource, defaultOffset, uploadOffset);
		directCommandList->UpdateResState(GPUResourceState_CopyDest, GPUResourceState_GenericRead, mesh);
		*selfPtrTracker = GPUResourceState_Common;
		*flagPtr = true;
	}
};

struct MeshLoadData {
	StackObject<MeshData, true> meshData;
	uint64_t offsets = 0;
	bool decoded = false;
	void Delete() {
		if (decoded) {
			meshData.Delete();
			decoded = false;
		}
	}
	MeshLoadData() {}
	MeshLoadData(const MeshLoadData& data) : meshData(data.meshData),
											 decoded(data.decoded),
											 offsets(data.offsets) {
	}
	~MeshLoadData() {
		Delete();
	}
};
struct IndexSettings {
	enum IndexFormat {
		IndexFormat_16Bit = 0,
		IndexFormat_32Bit = 1
	};
	IndexFormat indexFormat;
	uint indexCount;
};
struct MeshHeader {
	enum MeshDataType {
		MeshDataType_Vertex = 0,
		MeshDataType_Index = 1,
		MeshDataType_Normal = 2,
		MeshDataType_Tangent = 3,
		MeshDataType_UV = 4,
		MeshDataType_UV2 = 5,
		MeshDataType_UV3 = 6,
		MeshDataType_UV4 = 7,
		MeshDataType_Color = 8,
		MeshDataType_BoneIndex = 9,
		MeshDataType_BoneWeight = 10,
		MeshDataType_BoundingBox = 11,
		MeshDataType_BindPoses = 12,
		MeshDataType_SubMesh = 13,
		MeshDataType_Num = 14
	};
	MeshDataType type;
	union {
		IndexSettings indexSettings;
		uint vertexCount;
		uint normalCount;
		uint tangentCount;
		uint uvCount;
		uint colorCount;
		uint boneCount;
		uint bindPosesCount;
		uint subMeshCount;
	};
};
bool DecodeMesh(
	const vengine::string& filePath,
	MeshLoadData& meshLoadData,
	ArrayList<char>* dataPtr) {
	uint64 len = dataPtr->size();
	BinaryReader ifs(filePath);
	if (!ifs) {
		return false;//File Read Error!
	}
	uint chunkCount = 0;
	ifs.Read((char*)&chunkCount, sizeof(uint));
	if (chunkCount >= MeshHeader::MeshDataType_Num) return false;//Too many types
	auto& meshData = meshLoadData.meshData;
	meshData.New();
	meshData->datas = dataPtr;
	meshLoadData.decoded = true;
	struct ReadCommand {
		size_t readStartPos;
		size_t readDataSize;
		size_t memoryBufferOffset;
	};
	ReadCommand* allCommands = (ReadCommand*)alloca(sizeof(ReadCommand) * chunkCount);
	uint commandIndex = 0;
	size_t fstreamOffset = ifs.GetPos();
	size_t memoryBufferSize = 0;
	auto addReadCommandFunc = [&](size_t size) -> void {
		allCommands[commandIndex] =
			{
				fstreamOffset,
				size,
				memoryBufferSize};
		fstreamOffset += size;
		memoryBufferSize += size;
		commandIndex++;
	};
	for (uint i = 0; i < chunkCount; ++i) {
		MeshHeader header;
		ifs.SetPos(fstreamOffset);
		ifs.Read((char*)&header, sizeof(MeshHeader));
		fstreamOffset += sizeof(MeshHeader);
		if (header.type >= MeshHeader::MeshDataType_Num) return false;//Illegal Data Type
		uint64_t indexSize;
		switch (header.type) {
			case MeshHeader::MeshDataType_Vertex:
				meshData->vertexOffset = memoryBufferSize;
				meshData->vertexCount = header.vertexCount;
				addReadCommandFunc(sizeof(float3) * header.vertexCount);
				break;
			case MeshHeader::MeshDataType_Normal:
				meshData->normalOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float3) * header.normalCount);
				break;
			case MeshHeader::MeshDataType_Tangent:
				meshData->tangentOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float4) * header.tangentCount);
				break;
			case MeshHeader::MeshDataType_UV:
				meshData->uvOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_UV2:
				meshData->uv2Offset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_UV3:
				meshData->uv3Offset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_UV4:
				meshData->uv4Offset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_BoneIndex:
				meshData->boneIndexOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(uint4) * header.boneCount);
				break;
			case MeshHeader::MeshDataType_BoneWeight:
				meshData->boneWeightOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float4) * header.boneCount);
				break;
			case MeshHeader::MeshDataType_BindPoses:
				meshData->bindPosesOffset = memoryBufferSize;
				meshData->bindPosesCount = header.bindPosesCount;
				addReadCommandFunc(sizeof(float4x4) * header.bindPosesCount);
				break;
			case MeshHeader::MeshDataType_Index:
				indexSize = header.indexSettings.indexFormat == IndexSettings::IndexFormat_16Bit ? 2 : 4;
				meshData->indexFormat = indexSize == 2 ? GFXFormat_R16_UInt : GFXFormat_R32_UInt;
				meshData->indexCount = header.indexSettings.indexCount;
				meshData->indexDataOffset = memoryBufferSize;
				addReadCommandFunc(indexSize * header.indexSettings.indexCount);
				break;
			case MeshHeader::MeshDataType_BoundingBox:
				ifs.SetPos(fstreamOffset);
				ifs.Read((char*)&meshData->boundingCenter, sizeof(float3) * 2);
				fstreamOffset += sizeof(float3) * 2;
				break;
			case MeshHeader::MeshDataType_SubMesh:
				ifs.SetPos(fstreamOffset);
				meshData->subMeshes.resize(header.subMeshCount);
				ifs.Read((char*)meshData->subMeshes.data(), sizeof(SubMesh) * header.subMeshCount);
				fstreamOffset += sizeof(SubMesh) * header.subMeshCount;
				break;
		}
		meshData->datas->resize(memoryBufferSize + len);
		for (uint i = 0; i < commandIndex; ++i) {
			ifs.SetPos(allCommands[i].readStartPos);
			ifs.Read(allCommands[i].memoryBufferOffset + meshData->datas->data() + len, allCommands[i].readDataSize);
		}
	}
	return true;
}
bool DecodeMesh(
	const vengine::string& filePath,
	MeshData& meshData) {
	BinaryReader ifs(filePath.data());
	if (!ifs) {
		return false;//File Read Error!
	}
	uint chunkCount = 0;
	ifs.Read((char*)&chunkCount, sizeof(uint));
	if (chunkCount >= MeshHeader::MeshDataType_Num) return false;//Too many types
	struct ReadCommand {
		size_t readStartPos;
		size_t readDataSize;
		size_t memoryBufferOffset;
	};
	ReadCommand* allCommands = (ReadCommand*)alloca(sizeof(ReadCommand) * chunkCount);
	uint commandIndex = 0;
	size_t fstreamOffset = ifs.GetPos();
	size_t memoryBufferSize = 0;
	auto addReadCommandFunc = [&](size_t size) -> void {
		allCommands[commandIndex] =
			{
				fstreamOffset,
				size,
				memoryBufferSize};
		fstreamOffset += size;
		memoryBufferSize += size;
		commandIndex++;
	};
	for (uint i = 0; i < chunkCount; ++i) {
		MeshHeader header;
		ifs.SetPos(fstreamOffset);
		ifs.Read((char*)&header, sizeof(MeshHeader));
		fstreamOffset += sizeof(MeshHeader);
		if (header.type >= MeshHeader::MeshDataType_Num) return false;//Illegal Data Type
		uint64_t indexSize;
		switch (header.type) {
			case MeshHeader::MeshDataType_Vertex:
				meshData.vertexOffset = memoryBufferSize;
				meshData.vertexCount = header.vertexCount;
				addReadCommandFunc(sizeof(float3) * header.vertexCount);
				break;
			case MeshHeader::MeshDataType_Normal:
				meshData.normalOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float3) * header.normalCount);
				break;
			case MeshHeader::MeshDataType_Tangent:
				meshData.tangentOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float4) * header.tangentCount);
				break;
			case MeshHeader::MeshDataType_UV:
				meshData.uvOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_UV2:
				meshData.uv2Offset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_UV3:
				meshData.uv3Offset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_UV4:
				meshData.uv4Offset = memoryBufferSize;
				addReadCommandFunc(sizeof(float2) * header.uvCount);
				break;
			case MeshHeader::MeshDataType_BoneIndex:
				meshData.boneIndexOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(uint4) * header.boneCount);
				break;
			case MeshHeader::MeshDataType_BoneWeight:
				meshData.boneWeightOffset = memoryBufferSize;
				addReadCommandFunc(sizeof(float4) * header.boneCount);
				break;
			case MeshHeader::MeshDataType_BindPoses:
				meshData.bindPosesOffset = memoryBufferSize;
				meshData.bindPosesCount = header.bindPosesCount;
				addReadCommandFunc(sizeof(float4x4) * header.bindPosesCount);
				break;
			case MeshHeader::MeshDataType_Index:
				indexSize = header.indexSettings.indexFormat == IndexSettings::IndexFormat_16Bit ? 2 : 4;
				meshData.indexFormat = indexSize == 2 ? GFXFormat_R16_UInt : GFXFormat_R32_UInt;
				meshData.indexCount = header.indexSettings.indexCount;
				meshData.indexDataOffset = memoryBufferSize;
				addReadCommandFunc(indexSize * header.indexSettings.indexCount);
				break;
			case MeshHeader::MeshDataType_BoundingBox:
				ifs.SetPos(fstreamOffset);
				ifs.Read((char*)&meshData.boundingCenter, sizeof(float3) * 2);
				fstreamOffset += sizeof(float3) * 2;
				break;
			case MeshHeader::MeshDataType_SubMesh: {
				ifs.SetPos(fstreamOffset);
				meshData.subMeshes.resize(header.subMeshCount);
				ifs.Read((char*)meshData.subMeshes.data(), sizeof(SubMesh) * header.subMeshCount);
				fstreamOffset += sizeof(SubMesh) * header.subMeshCount;
			} break;
		}
		meshData.datas->resize(memoryBufferSize);
		for (uint i = 0; i < commandIndex; ++i) {
			ifs.SetPos(allCommands[i].readStartPos);
			ifs.Read(allCommands[i].memoryBufferOffset + meshData.datas->data(), allCommands[i].readDataSize);
		}
	}
	return true;
}
uint64_t GetStride(
	bool positions,
	bool normals,
	bool tangents,
	bool colors,
	bool uv,
	bool uv1,
	bool uv2,
	bool uv3,
	bool boneIndex,
	bool boneWeight) {
	uint64_t stride = 0;
	auto cumulate = [&](bool ptr, uint64_t size) -> void {
		if (ptr) stride += size;
	};
	cumulate(positions, 12);
	cumulate(normals, 12);
	cumulate(tangents, 16);
	cumulate(colors, 16);
	cumulate(uv, 8);
	cumulate(uv1, 8);
	cumulate(uv2, 8);
	cumulate(uv3, 8);
	cumulate(boneIndex, 16);
	cumulate(boneWeight, 16);
	return stride;
}
char* InitMeshData(
	uint vertexCount,
	float3* positions,
	float3* normals,
	float4* tangents,
	float4* colors,
	float2* uv,
	float2* uv1,
	float2* uv2,
	float2* uv3,
	int4* boneIndex,
	float4* boneWeight,
	GFXFormat indexFormat,
	uint indexCount,
	void* indexArrayPtr,
	uint& meshLayoutIndex,
	uint& VertexByteStride,
	uint& VertexBufferByteSize,
	DeleteGuard& deleteGuard,
	uint64& indexSize) {
	//TODO: May need support multi vertexBuffer
	MeshLayoutKey key = {
		positions ? 0 : -1,
		normals ? 0 : -1,
		tangents ? 0 : -1,
		colors ? 0 : -1,
		uv ? 0 : -1,
		uv1 ? 0 : -1,
		uv2 ? 0 : -1,
		uv3 ? 0 : -1,
		boneIndex ? 0 : -1,
		boneWeight ? 0 : -1};
	meshLayoutIndex = MeshLayout::GetMeshLayoutIndex(key);
	ArrayList<D3D12_INPUT_ELEMENT_DESC>* meshLayouts = MeshLayout::GetMeshLayoutValue(meshLayoutIndex);
	uint64_t stride = GetStride(
		positions,
		normals,
		tangents,
		colors,
		uv,
		uv1,
		uv2,
		uv3,
		boneIndex,
		boneWeight);
	VertexByteStride = stride;
	VertexBufferByteSize = stride * vertexCount;
	//IndexBufferByteSize = (IndexFormat == GFXFormat_R16_UInt ? 2 : 4) * indexCount;
	indexSize = (uint64)indexCount * ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4);
	char* dataPtr = reinterpret_cast<char*>(vengine_malloc(VertexBufferByteSize + indexSize));
	deleteGuard.ptr = dataPtr;
	auto vertBufferCopy = [&](char* buffer, char* ptr, uint size, int32_t& offset) -> void {
		if ((uint64_t)ptr < 2048) {
			if (ptr) offset += size;
			return;
		}
		for (int32_t i = 0; i < vertexCount; ++i) {
			memcpy(buffer + i * stride + offset, ptr + (uint64)size * i, size);
		}
		offset += size;
	};
	int32_t offset = 0;
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(positions),
		12,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(normals),
		12,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(tangents),
		16,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(colors),
		16,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(uv),
		8,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(uv1),
		8,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(uv2),
		8,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(uv3),
		8,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(boneIndex),
		16,
		offset);
	vertBufferCopy(
		dataPtr,
		reinterpret_cast<char*>(boneWeight),
		16,
		offset);
	char* indexBufferStart = dataPtr + VertexBufferByteSize;
	memcpy(indexBufferStart, indexArrayPtr, (uint64)indexCount * ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4));
	return dataPtr;
}
}// namespace MeshGlobal
D3D12_RESOURCE_STATES Mesh::GetGFXResourceState(GPUResourceState gfxState) const {
	if (gfxState == GPUResourceState_GenericRead) {
		return MeshGlobal::D3D12_MESH_GENERIC_READ_STATE;
	} else {
		return (D3D12_RESOURCE_STATES)gfxState;
	}
}
// namespace MeshGlobal
uint64_t Mesh::GetMeshSize(
	uint vertexCount,
	bool positions,
	bool normals,
	bool tangents,
	bool colors,
	bool uv,
	bool uv1,
	bool uv2,
	bool uv3,
	bool boneIndex,
	bool boneWeight,
	uint indexCount,
	GFXFormat indexFormat) {
	return MeshGlobal::GetStride(
			   positions,
			   normals,
			   tangents,
			   colors,
			   uv,
			   uv1,
			   uv2,
			   uv3,
			   boneIndex,
			   boneWeight)
			   * vertexCount
		   + (uint64)indexCount * ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4);
}
SubMesh const& Mesh::GetSubMesh(uint i) const {
	return subMeshes[i];
}
Mesh::Mesh(
	uint vertexCount,
	float3* positions,
	float3* normals,
	float4* tangents,
	float4* colors,
	float2* uv,
	float2* uv1,
	float2* uv2,
	float2* uv3,
	int4* boneIndex,
	float4* boneWeight,
	GFXDevice* device,
	GFXFormat indexFormat,
	uint indexCount,
	void* indexArrayPtr,
	IBufferAllocator* allocator,
	SubMesh* subMeshesPtr,
	uint subMeshCount,
	float3 const& boundingCenter,
	float3 const& boundingExtent)
	: allocator(allocator),
	  mVertexCount(vertexCount),
	  indexFormat(indexFormat),
	  indexCount(indexCount),
	  indexArrayPtr(indexArrayPtr),
	  boundingCenter(boundingCenter),
	  boundingExtent(boundingExtent),
	  GPUResourceBase(GPUResourceType::Buffer) {
	MeshGlobal::DeleteGuard guard;
	uint64 indexSize;
	char* dataPtr = MeshGlobal::InitMeshData(
		vertexCount,
		positions,
		normals,
		tangents,
		colors,
		uv,
		uv1,
		uv2,
		uv3,
		boneIndex,
		boneWeight,
		indexFormat,
		indexCount,
		indexArrayPtr,
		meshLayoutIndex,
		VertexByteStride,
		VertexBufferByteSize,
		guard,
		indexSize);
	ComPtr<GFXResource> uploadBuffer;
	MeshGlobal::CreateDefaultBuffer(device, indexSize + VertexBufferByteSize, uploadBuffer, Resource, allocator, this);
	char* mappedPtr = nullptr;
	ThrowIfFailed(uploadBuffer->Map(0, nullptr, (void**)&mappedPtr));
	memcpy(mappedPtr, dataPtr, indexSize + VertexBufferByteSize);
	uploadBuffer->Unmap(0, nullptr);
	MeshGlobal::MeshLoadCommand* meshLoadCommand;
	flagPtr = ObjectPtr<bool>::MakePtrNoMemoryFree(&flag);
	meshLoadCommand = new MeshGlobal::MeshLoadCommand(
		&initState,
		flagPtr,
		uploadBuffer,
		Resource,
		indexSize + VertexBufferByteSize,
		0,
		0,
		allocator,
		this);
	RenderCommand::UpdateResState(meshLoadCommand);
	vertexView.BufferLocation = Resource->GetGPUVirtualAddress();
	vertexView.StrideInBytes = VertexByteStride;
	vertexView.SizeInBytes = VertexBufferByteSize;
	indexView.BufferLocation = Resource->GetGPUVirtualAddress() + VertexBufferByteSize;
	indexView.Format = (DXGI_FORMAT)indexFormat;
	indexView.SizeInBytes = indexCount * ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4);
	if (subMeshCount == 0) {
		SubMesh& m = subMeshes.emplace_back();
		m.boundingCenter = boundingCenter;
		m.boundingExtent = boundingExtent;
		m.indexCount = indexCount;
		m.indexOffset = 0;
		m.materialIndex = 0;
		m.vertexOffset = 0;
	} else {
		subMeshes.resize(subMeshCount);
		memcpy(subMeshes.data(), subMeshesPtr, subMeshes.size() * sizeof(SubMesh));
	}
}
Mesh::Mesh(
	uint vertexCount,
	float3* positions,
	float3* normals,
	float4* tangents,
	float4* colors,
	float2* uv,
	float2* uv1,
	float2* uv2,
	float2* uv3,
	int4* boneIndex,
	float4* boneWeight,
	GFXDevice* device,
	GFXFormat indexFormat,
	uint indexCount,
	void* indexArrayPtr,
	const Microsoft::WRL::ComPtr<GFXResource>& defaultBuffer,
	uint64_t defaultOffset,
	const Microsoft::WRL::ComPtr<GFXResource>& inputUploadBuffer,
	uint64_t uploadOffset,
	SubMesh* subMeshesPtr,
	uint subMeshCount,
	float3 const& boundingCenter,
	float3 const& boundingExtent)
	: mVertexCount(vertexCount),
	  indexFormat(indexFormat),
	  indexCount(indexCount),
	  indexArrayPtr(indexArrayPtr),
	  boundingCenter(boundingCenter),
	  boundingExtent(boundingExtent),
	  GPUResourceBase(GPUResourceType::Buffer) {
	MeshGlobal::DeleteGuard guard;
	uint64 indexSize;
	char* dataPtr = MeshGlobal::InitMeshData(
		vertexCount,
		positions,
		normals,
		tangents,
		colors,
		uv,
		uv1,
		uv2,
		uv3,
		boneIndex,
		boneWeight,
		indexFormat,
		indexCount,
		indexArrayPtr,
		meshLayoutIndex,
		VertexByteStride,
		VertexBufferByteSize,
		guard,
		indexSize);
	if (!defaultBuffer) defaultOffset = 0;
	if (!inputUploadBuffer) uploadOffset = 0;
	ComPtr<GFXResource> uploadBuffer = inputUploadBuffer;
	Resource = defaultBuffer;
	MeshGlobal::CreateDefaultBuffer(device, indexSize + VertexBufferByteSize, uploadBuffer, Resource);
	char* mappedPtr = nullptr;
	ThrowIfFailed(uploadBuffer->Map(0, nullptr, (void**)&mappedPtr));
	memcpy(mappedPtr + uploadOffset, dataPtr, indexSize + VertexBufferByteSize);
	uploadBuffer->Unmap(0, nullptr);
	MeshGlobal::MeshLoadCommand* meshLoadCommand;
	flagPtr = ObjectPtr<bool>::MakePtrNoMemoryFree(&flag);
	meshLoadCommand = new MeshGlobal::MeshLoadCommand(
		&initState,
		flagPtr,
		uploadBuffer,
		Resource,
		indexSize + VertexBufferByteSize,
		defaultOffset,
		uploadOffset,
		nullptr,
		this);
	RenderCommand::UpdateResState(meshLoadCommand);
	vertexView.BufferLocation = Resource->GetGPUVirtualAddress() + defaultOffset;
	vertexView.StrideInBytes = VertexByteStride;
	vertexView.SizeInBytes = VertexBufferByteSize;
	indexView.BufferLocation = Resource->GetGPUVirtualAddress() + VertexBufferByteSize + defaultOffset;
	indexView.Format = (DXGI_FORMAT)indexFormat;
	indexView.SizeInBytes = indexCount * ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4);
	if (subMeshCount == 0) {
		SubMesh& m = subMeshes.emplace_back();
		m.boundingCenter = boundingCenter;
		m.boundingExtent = boundingExtent;
		m.indexCount = indexCount;
		m.indexOffset = 0;
		m.materialIndex = 0;
		m.vertexOffset = 0;
	} else {
		subMeshes.resize(subMeshCount);
		memcpy(subMeshes.data(), subMeshesPtr, subMeshes.size() * sizeof(SubMesh));
	}
}
ObjectPtr<Mesh> Mesh::LoadMeshFromFile(
	MeshData& meshData,
	const vengine::string& str,
	GFXDevice* device,
	bool normals,
	bool tangents,
	bool colors,
	bool uv,
	bool uv1,
	bool uv2,
	bool uv3,
	bool bone,
	IBufferAllocator* allocator,
	ArrayList<char>* dataPtr) {
	ArrayList<char> data;
	if (!dataPtr) dataPtr = &data;
	else
		dataPtr->clear();
	ObjectPtr<Mesh> result = nullptr;
	meshData.datas = dataPtr;
	if (!MeshGlobal::DecodeMesh(str, meshData))
		return result;
	char* ptrStart = meshData.datas->data();
	if (allocator) {
		result = ObjectPtr<Mesh>::MakePtr(
			new Mesh(
				meshData.vertexCount,
				(float3*)(meshData.vertexOffset + ptrStart),
				normals ? ((meshData.normalOffset >= 0) ? (float3*)(meshData.normalOffset + ptrStart) : (float3*)1) : nullptr,
				tangents ? ((meshData.tangentOffset >= 0) ? (float4*)(meshData.tangentOffset + ptrStart) : (float4*)1) : nullptr,
				colors ? ((meshData.colorOffset >= 0) ? (float4*)(meshData.colorOffset + ptrStart) : (float4*)1) : nullptr,
				uv ? ((meshData.uvOffset >= 0) ? (float2*)(meshData.uvOffset + ptrStart) : (float2*)1) : nullptr,
				uv1 ? ((meshData.uv2Offset >= 0) ? (float2*)(meshData.uv2Offset + ptrStart) : (float2*)1) : nullptr,
				uv2 ? ((meshData.uv3Offset >= 0) ? (float2*)(meshData.uv3Offset + ptrStart) : (float2*)1) : nullptr,
				uv3 ? ((meshData.uv4Offset >= 0) ? (float2*)(meshData.uv4Offset + ptrStart) : (float2*)1) : nullptr,
				bone ? ((meshData.boneIndexOffset >= 0) ? (int4*)(meshData.boneIndexOffset + ptrStart) : (int4*)1) : nullptr,
				bone ? ((meshData.boneWeightOffset >= 0) ? (float4*)(meshData.boneWeightOffset + ptrStart) : (float4*)1) : nullptr,
				device,
				meshData.indexFormat,
				meshData.indexCount,
				ptrStart + meshData.indexDataOffset,
				allocator,
				meshData.subMeshes.data(),
				meshData.subMeshes.size(),
				meshData.boundingCenter,
				meshData.boundingExtent));
	} else {
		result = ObjectPtr<Mesh>::MakePtr(
			new Mesh(
				meshData.vertexCount,
				(float3*)(meshData.vertexOffset + ptrStart),
				normals ? ((meshData.normalOffset >= 0) ? (float3*)(meshData.normalOffset + ptrStart) : (float3*)1) : nullptr,
				tangents ? ((meshData.tangentOffset >= 0) ? (float4*)(meshData.tangentOffset + ptrStart) : (float4*)1) : nullptr,
				colors ? ((meshData.colorOffset >= 0) ? (float4*)(meshData.colorOffset + ptrStart) : (float4*)1) : nullptr,
				uv ? ((meshData.uvOffset >= 0) ? (float2*)(meshData.uvOffset + ptrStart) : (float2*)1) : nullptr,
				uv1 ? ((meshData.uv2Offset >= 0) ? (float2*)(meshData.uv2Offset + ptrStart) : (float2*)1) : nullptr,
				uv2 ? ((meshData.uv3Offset >= 0) ? (float2*)(meshData.uv3Offset + ptrStart) : (float2*)1) : nullptr,
				uv3 ? ((meshData.uv4Offset >= 0) ? (float2*)(meshData.uv4Offset + ptrStart) : (float2*)1) : nullptr,
				bone ? ((meshData.boneIndexOffset >= 0) ? (int4*)(meshData.boneIndexOffset + ptrStart) : (int4*)1) : nullptr,
				bone ? ((meshData.boneWeightOffset >= 0) ? (float4*)(meshData.boneWeightOffset + ptrStart) : (float4*)1) : nullptr,
				device,
				meshData.indexFormat,
				meshData.indexCount,
				ptrStart + meshData.indexDataOffset,
				nullptr, 0,
				nullptr, 0,
				meshData.subMeshes.data(),
				meshData.subMeshes.size(),
				meshData.boundingCenter,
				meshData.boundingExtent));
	}
	Mesh* ptr = result;
	if (meshData.bindPosesOffset >= 0) {
		ptr->bindPoses.resize(meshData.bindPosesCount);
		memcpy(ptr->bindPoses.data(), meshData.datas->data() + meshData.bindPosesOffset, sizeof(Math::Matrix4) * meshData.bindPosesCount);
	}
	return result;
}
ObjectPtr<Mesh> Mesh::LoadMeshFromFile(
	const vengine::string& str,
	GFXDevice* device,
	bool normals,
	bool tangents,
	bool colors,
	bool uv,
	bool uv1,
	bool uv2,
	bool uv3,
	bool bone,
	IBufferAllocator* allocator,
	ArrayList<char>* dataPtr) {
	MeshData meshData;
	return LoadMeshFromFile(
		meshData,
		str,
		device,
		normals,
		tangents,
		colors,
		uv,
		uv1,
		uv2,
		uv3,
		bone,
		allocator,
		dataPtr);
}
bool Mesh::LoadMeshToArray(
	const vengine::string& str,
	bool normals,
	bool tangents,
	bool colors,
	bool uv,
	bool uv1,
	bool uv2,
	bool uv3,
	bool bone,
	ArrayList<char>& dataPtr,
	MeshData& meshData) {
	meshData.datas = &dataPtr;
	return MeshGlobal::DecodeMesh(str, meshData);
}
uint Mesh::GetSubMeshCount() const {
	return subMeshes.size();
}
void Mesh::GenerateVBOSRVGlobalDescriptor(GFXDevice* device) const {
	{
		std::lock_guard lck(srvDescLock);
		if (vboSrvDescIndex != -1) return;
		vboSrvDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	VboBindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), vboSrvDescIndex, device);
}
void Mesh::GenerateIBOSRVGlobalDescriptor(GFXDevice* device) const {
	{
		std::lock_guard lck(srvDescLock);
		if (iboSrvDescIndex != -1) return;
		iboSrvDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	IboBindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), iboSrvDescIndex, device);
}
void Mesh::GenerateVBOUAVGlobalDescriptor(GFXDevice* device) const {
	{
		std::lock_guard lck(srvDescLock);
		if (vboUavDescIndex != -1) return;
		vboUavDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	VboBindUAVToHeap(Graphics::GetGlobalDescHeapNonConst(), vboUavDescIndex, device);
}
void Mesh::GenerateIBOUAVGlobalDescriptor(GFXDevice* device) const {
	{
		std::lock_guard lck(srvDescLock);
		if (iboUavDescIndex != -1) return;
		iboUavDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	IboBindUAVToHeap(Graphics::GetGlobalDescHeapNonConst(), iboUavDescIndex, device);
}
void Mesh::VboBindUAVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	uavDesc.Buffer.NumElements = mVertexCount;
	uavDesc.Buffer.CounterOffsetInBytes = 0;
	uavDesc.Buffer.StructureByteStride = VertexByteStride;
	targetHeap->CreateUAV(device, this, &uavDesc, index);
}
void Mesh::VboBindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	srvDesc.Buffer.NumElements = mVertexCount;
	srvDesc.Buffer.StructureByteStride = VertexByteStride;
	targetHeap->CreateSRV(device, this, &srvDesc, index);
}
void Mesh::IboBindUAVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uint64 stride = ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4);
	uavDesc.Buffer.FirstElement = VertexBufferByteSize / stride;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	uavDesc.Buffer.NumElements = indexCount;
	uavDesc.Buffer.CounterOffsetInBytes = 0;
	uavDesc.Buffer.StructureByteStride = stride;
	targetHeap->CreateUAV(device, this, &uavDesc, index);
}
void Mesh::IboBindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	uint64 stride = ((indexFormat == GFXFormat_R16_UInt) ? 2 : 4);
	srvDesc.Buffer.FirstElement = VertexBufferByteSize / stride;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	srvDesc.Buffer.NumElements = indexCount;
	srvDesc.Buffer.StructureByteStride = stride;
	targetHeap->CreateSRV(device, this, &srvDesc, index);
}
uint Mesh::GetVBOSRVDescIndex(GFXDevice* device) const {
	GenerateVBOSRVGlobalDescriptor(device);
	return vboSrvDescIndex;
}
uint Mesh::GetIBOSRVDescIndex(GFXDevice* device) const {
	GenerateIBOSRVGlobalDescriptor(device);
	return iboSrvDescIndex;
}
uint Mesh::GetVBOUAVDescIndex(GFXDevice* device) const {
	GenerateVBOUAVGlobalDescriptor(device);
	return vboUavDescIndex;
}
uint Mesh::GetIBOUAVDescIndex(GFXDevice* device) const {
	GenerateIBOUAVGlobalDescriptor(device);
	return iboUavDescIndex;
}
Mesh::~Mesh() {
	if (allocator) {
		allocator->ReturnBuffer((uint64)this);
	}
	if (vboSrvDescIndex != -1) {
		Graphics::ReturnDescHeapIndexToPool(vboSrvDescIndex);
	}
	if (vboUavDescIndex != -1) {
		Graphics::ReturnDescHeapIndexToPool(vboUavDescIndex);
	}
	if (iboSrvDescIndex != -1) {
		Graphics::ReturnDescHeapIndexToPool(iboSrvDescIndex);
	}
	if (iboUavDescIndex != -1) {
		Graphics::ReturnDescHeapIndexToPool(iboUavDescIndex);
	}
}
float3 Mesh::GetBoundingCenter() const {
	return boundingCenter;
}
float3 Mesh::GetBoundingExtent() const {
	return boundingExtent;
}
void Mesh::LoadMeshFromFiles(
	const vengine::vector<vengine::string>& str, GFXDevice* device,
	bool normals,
	bool tangents,
	bool colors,
	bool uv,
	bool uv1,
	bool uv2,
	bool uv3,
	bool bone,
	vengine::vector<ObjectPtr<Mesh>>& results,
	ArrayList<char>* dataPtr) {
	ArrayList<char> data;
	if (!dataPtr) dataPtr = &data;
	else
		dataPtr->clear();
	results.resize(str.size());
	vengine::vector<std::pair<MeshGlobal::MeshLoadData, uint64>> meshLoadDatas(str.size());
	uint64_t bufferSize = 0;
	for (uint64_t i = 0; i < str.size(); ++i) {
		const vengine::string& s = str[i];
		MeshGlobal::MeshLoadData& a = meshLoadDatas[i].first;
		StackObject<MeshData, true>& meshData = a.meshData;
		meshLoadDatas[i].second = dataPtr->size();
		if (MeshGlobal::DecodeMesh(s, a, dataPtr)) {
			char* ptrStart = meshData->datas->data();
			a.offsets = bufferSize;
			bufferSize += GetMeshSize(
				meshData->vertexCount,
				(float3*)(meshData->vertexOffset + ptrStart),
				normals ? ((meshData->normalOffset >= 0) ? (float3*)(meshData->normalOffset + ptrStart) : (float3*)1) : nullptr,
				tangents ? ((meshData->tangentOffset >= 0) ? (float4*)(meshData->tangentOffset + ptrStart) : (float4*)1) : nullptr,
				colors ? ((meshData->colorOffset >= 0) ? (float4*)(meshData->colorOffset + ptrStart) : (float4*)1) : nullptr,
				uv ? ((meshData->uvOffset >= 0) ? (float2*)(meshData->uvOffset + ptrStart) : (float2*)1) : nullptr,
				uv1 ? ((meshData->uv2Offset >= 0) ? (float2*)(meshData->uv2Offset + ptrStart) : (float2*)1) : nullptr,
				uv2 ? ((meshData->uv3Offset >= 0) ? (float2*)(meshData->uv3Offset + ptrStart) : (float2*)1) : nullptr,
				uv3 ? ((meshData->uv4Offset >= 0) ? (float2*)(meshData->uv4Offset + ptrStart) : (float2*)1) : nullptr,
				bone ? ((meshData->boneIndexOffset >= 0) ? (int4*)(meshData->boneIndexOffset + ptrStart) : (int4*)1) : nullptr,
				bone ? ((meshData->boneWeightOffset >= 0) ? (float4*)(meshData->boneWeightOffset + ptrStart) : (float4*)1) : nullptr,
				meshData->indexCount,
				meshData->indexFormat);
		} else {
			meshData.Delete();
		}
	}
	if (bufferSize == 0) return;
	ComPtr<GFXResource> defaultBuffer;
	ComPtr<GFXResource> uploadBuffer;
	// Create the actual default buffer resource.
	auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	auto buf = CD3DX12_RESOURCE_DESC::Buffer(bufferSize, D3D12_RESOURCE_FLAG_NONE);
	ThrowIfFailed(device->device()->CreateCommittedResource(
		&prop,
		D3D12_HEAP_FLAG_NONE,
		&buf,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(defaultBuffer.GetAddressOf())));
	// In order to copy CPU memory data into our default buffer, we need to create
	// an intermediate upload heap.
	prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	buf = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
	ThrowIfFailed(device->device()->CreateCommittedResource(
		&prop,
		D3D12_HEAP_FLAG_NONE,
		&buf,
		MeshGlobal::D3D12_MESH_GENERIC_READ_STATE,
		nullptr,
		IID_PPV_ARGS(uploadBuffer.GetAddressOf())));
	for (uint64_t i = 0; i < meshLoadDatas.size(); ++i) {
		MeshGlobal::MeshLoadData& a = meshLoadDatas[i].first;
		if (a.decoded) {
			StackObject<MeshData, true>& meshData = a.meshData;
			char* ptrStart = meshData->datas->data() + meshLoadDatas[i].second;
			results[i] = ObjectPtr<Mesh>::NewObject(
				meshData->vertexCount,
				(float3*)(meshData->vertexOffset + ptrStart),
				normals ? ((meshData->normalOffset >= 0) ? (float3*)(meshData->normalOffset + ptrStart) : (float3*)1) : nullptr,
				tangents ? ((meshData->tangentOffset >= 0) ? (float4*)(meshData->tangentOffset + ptrStart) : (float4*)1) : nullptr,
				colors ? ((meshData->colorOffset >= 0) ? (float4*)(meshData->colorOffset + ptrStart) : (float4*)1) : nullptr,
				uv ? ((meshData->uvOffset >= 0) ? (float2*)(meshData->uvOffset + ptrStart) : (float2*)1) : nullptr,
				uv1 ? ((meshData->uv2Offset >= 0) ? (float2*)(meshData->uv2Offset + ptrStart) : (float2*)1) : nullptr,
				uv2 ? ((meshData->uv3Offset >= 0) ? (float2*)(meshData->uv3Offset + ptrStart) : (float2*)1) : nullptr,
				uv3 ? ((meshData->uv4Offset >= 0) ? (float2*)(meshData->uv4Offset + ptrStart) : (float2*)1) : nullptr,
				bone ? ((meshData->boneIndexOffset >= 0) ? (int4*)(meshData->boneIndexOffset + ptrStart) : (int4*)1) : nullptr,
				bone ? ((meshData->boneWeightOffset >= 0) ? (float4*)(meshData->boneWeightOffset + ptrStart) : (float4*)1) : nullptr,
				device,
				meshData->indexFormat,
				meshData->indexCount,
				ptrStart + meshData->indexDataOffset,
				defaultBuffer,
				a.offsets,
				uploadBuffer,
				a.offsets,
				meshData->subMeshes.data(),
				meshData->subMeshes.size(),
				meshData->boundingCenter,
				meshData->boundingExtent);
			Mesh* result = results[i];
			if (meshData->bindPosesOffset >= 0) {
				result->bindPoses.resize(meshData->bindPosesCount);
				memcpy(result->bindPoses.data(), meshData->datas->data() + meshData->bindPosesOffset, sizeof(Math::Matrix4) * meshData->bindPosesCount);
			}
		} else
			results[i] = nullptr;
	}
}

#include "VirtualVoxelMap.h"
#include "../RenderComponentInclude.h"
#include "../../Singleton/ShaderCompiler.h"
#include "../../Singleton/ShaderID.h"
#include "../../Singleton/Graphics.h"
#include "../../PipelineComponent/ThreadCommand.h"

uint3 VirtualVoxelMap::RepeatIndex(int3 voxelIndex) const {
	auto repeat = [](int32 a, int32 size) -> int32 {
		int32 v = a % size;
		if (v < 0) v = size + v;
		return v;
	};
	voxelIndex.x = repeat(voxelIndex.x, indirectSize.x);
	voxelIndex.y = repeat(voxelIndex.y, indirectSize.y);
	voxelIndex.z = repeat(voxelIndex.z, indirectSize.z);
	return uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z);
}

uint VirtualVoxelMap::GetIndex(int3 voxelIndex) const {
	uint3 voxelIndexU = RepeatIndex(voxelIndex);
	uint a = voxelIndexU.x + voxelIndexU.y * indirectSize.x + voxelIndexU.z * indirectSize.x * indirectSize.y;
	return a;
}

uint VirtualVoxelMap::GetIndex(uint3 voxelIndexU) const {
	uint a = voxelIndexU.x + voxelIndexU.y * indirectSize.x + voxelIndexU.z * indirectSize.x * indirectSize.y;
	return a;
}

VirtualVoxelMap::VirtualVoxelMap(
	GFXDevice* device,
	ITextureAllocator* texAlloc,
	IBufferAllocator* buffAlloc,
	uint3 maxIndirectSize,
	uint3 voxelChunkResolution,
	GFXFormat format,
	bool processEdge,
	CustomDataType customDataType)
	: indirectSize(maxIndirectSize),
	  customDataType(customDataType),
	  generateEdge(processEdge),
	  voxelChunkResolution(voxelChunkResolution),
	  format(format),
	  indirectUpload(new UploadBuffer(device, maxIndirectSize.x * maxIndirectSize.y * maxIndirectSize.z, false, (uint)customDataType * sizeof(uint), buffAlloc)),
	  allChunks(maxIndirectSize.x * maxIndirectSize.y * maxIndirectSize.z),
	  indirectRT(new RenderTexture(
		  device,
		  texAlloc,
		  maxIndirectSize.x,
		  maxIndirectSize.y,
		  RenderTextureFormat::GetColorFormat(GetIndirectRTFormat(customDataType)),
		  TextureDimension::Tex3D,
		  maxIndirectSize.z,
		  1,
		  RenderTextureState::Common)) {
	if ((indirectSize.x & 7) != 0 || (indirectSize.z & 7) != 0) {
		VEngine_Log("Indirect xz size should be multiple of 8!\n");
		throw 0;
	}
	uint uploadCount = indirectUpload->GetElementCount() * (uint)customDataType;
	int32* ptr = (int32*)indirectUpload->GetMappedDataPtr(0);
	for (uint i = 0; i < uploadCount; ++i) {
		ptr[i] = 65535;
	}
	if ((voxelChunkResolution.x & 7) != 0
		|| (voxelChunkResolution.y & 7) != 0
		|| (voxelChunkResolution.z & 7) != 0) {
		VEngine_Log("Voxel Resolution should be multiple of 8!\n");
		throw 0;
	}
	props.shader = ShaderCompiler::GetComputeShader("VirtualVoxelSetter");
	props._SelfMap = ShaderID::PropertyToID("_SelfMap");
	props._IndirectTex = ShaderID::PropertyToID("_IndirectTex");
	props._IndirectBuffer = ShaderID::PropertyToID("_IndirectBuffer");
	props._SRV_IndirectTex = ShaderID::PropertyToID("_SRV_IndirectTex");
}

RenderTexture* VirtualVoxelMap::GetRenderTextureChunk(int3 indirectIndex) const {
	return allChunks[GetIndex(indirectIndex)].rt.get();
}

RenderTexture* VirtualVoxelMap::GetIndirectTex() const {
	return indirectRT.get();
}

bool VirtualVoxelMap::IsChunkExists(int3 indirectIndex) const {
	return static_cast<bool>(allChunks[GetIndex(indirectIndex)].rt);
}

void VirtualVoxelMap::TryCreateChunk(GFXDevice* device, ITextureAllocator* allocator, int3 indirectIndex, uint3 customData) {
	uint idx = GetIndex(indirectIndex);
	auto&& c = allChunks[idx];
	if (c.rt) {
		MarkChunkAsDirty(indirectIndex);
		return;
	}
	uint3 res = generateEdge ? (voxelChunkResolution + uint3(2, 2, 2)) : voxelChunkResolution;
	c.rt = std::unique_ptr<RenderTexture>(
		new RenderTexture(
			device,
			allocator,
			res.x,
			res.y,
			RenderTextureFormat::GetColorFormat(format),
			TextureDimension::Tex3D,
			res.z,
			1,
			RenderTextureState::Common));
	uint v = c.rt->GetGlobalDescIndex();
	switch (customDataType) {
		case CustomDataType::None:
			*(uint*)indirectUpload->GetMappedDataPtr(idx) = v;
			break;
		case CustomDataType::UInt:
			*(uint2*)indirectUpload->GetMappedDataPtr(idx) = uint2(v, customData.x);
			break;
		case CustomDataType::UInt3:
			*(uint4*)indirectUpload->GetMappedDataPtr(idx) = uint4(v, customData.x, customData.y, customData.z);
			break;
	}
	indirectIsDirty = true;
	if (generateEdge && c.dirtyMask != DirtyMaskType::ALL) {
		c.dirtyMask = DirtyMaskType::ALL;
		executeCommand.Push(RepeatIndex(indirectIndex));
	}
}

void VirtualVoxelMap::TryDestroyChunk(int3 indirectIndex) {
	uint idx = GetIndex(indirectIndex);
	auto&& c = allChunks[idx];
	if (c.rt) {
		c.rt = nullptr;
		switch (customDataType) {
			case CustomDataType::None:
				*(uint*)indirectUpload->GetMappedDataPtr(idx) = 65535;
				break;
			case CustomDataType::UInt:
				*(uint2*)indirectUpload->GetMappedDataPtr(idx) = uint2(65535, 65535);
				break;
			case CustomDataType::UInt3:
				*(uint4*)indirectUpload->GetMappedDataPtr(idx) = uint4(65535, 65535, 65535, 65535);
				break;
		}
		indirectIsDirty = true;
	}
	c.dirtyMask = DirtyMaskType::None;
}

void VirtualVoxelMap::UpdateTextureChunk(int3 indirectIndex) {
	if (!IsChunkExists(indirectIndex)) return;
	MarkChunkAsDirty(indirectIndex);
}

void VirtualVoxelMap::MarkChunkAsDirty(int3 indirectIndex) {
	if (!generateEdge) return;
	auto SetNeighbor = [&](int3 offset, DirtyMaskType type) -> void {
		indirectIndex += offset;
		uint idx = GetIndex(indirectIndex);
		auto&& c = allChunks[idx];
		if (c.rt && ((uint8_t)c.dirtyMask & (uint8_t)type) == 0) {
			c.dirtyMask = (DirtyMaskType)((uint8_t)c.dirtyMask | (uint8_t)type);
			executeCommand.Push(RepeatIndex(indirectIndex));
		}
	};
	SetNeighbor(int3(-1, 0, 0), DirtyMaskType::Right);
	SetNeighbor(int3(1, 0, 0), DirtyMaskType::Left);
	SetNeighbor(int3(0, -1, 0), DirtyMaskType::Top);
	SetNeighbor(int3(0, 1, 0), DirtyMaskType::Down);
	SetNeighbor(int3(0, 0, -1), DirtyMaskType::Forward);
	SetNeighbor(int3(0, 0, 1), DirtyMaskType::Back);
}

void VirtualVoxelMap::ExecuteCommands(RenderPackage const& package, Runnable<CBufferChunk(size_t)> const& getCBufferChunk) {
	//TODO
	Command cmd;
	auto heap = Graphics::GetGlobalDescHeap();
	props.shader->BindShader(package.tCmd, heap);
	package.tCmd->RegistInitState(
		indirectRT->GetInitState(),
		indirectRT.get());

	if (indirectIsDirty) {
		indirectIsDirty = false;
		package.tCmd->UpdateResState(
			indirectRT->GetInitState(),
			GFXResourceState_UnorderedAccess,
			indirectRT.get());
		CBufferChunk chunk = getCBufferChunk(sizeof(uint4));
		uint4 inputData = uint4(indirectSize.x, indirectSize.y, indirectSize.z, (uint)customDataType);
		chunk.CopyData(&inputData);
		props.shader->SetResource(
			package.tCmd,
			props._IndirectBuffer,
			indirectUpload.get(),
			0);
		props.shader->SetResource(package.tCmd, ShaderID::GetParams(), chunk.GetBuffer(), chunk.GetOffset());
		props.shader->SetResource(
			package.tCmd,
			props._IndirectTex,
			heap,
			indirectRT->GetGlobalUAVDescIndex(0));
		props.shader->Dispatch(package.tCmd, 3, indirectSize.x / 8, indirectSize.y, indirectSize.z / 8);
		package.tCmd->UpdateResState(
			GFXResourceState_NonPixelRead,
			indirectRT.get());
	}
	while (executeCommand.Pop(&cmd)) {
		if (cmd.type == CommandType::GenerateEdge) {
			auto&& c = allChunks[GetIndex(cmd.generateEdgeIndex)];
			if (!c.rt || c.dirtyMask == DirtyMaskType::None) continue;
			uint4 param(voxelChunkResolution.x, voxelChunkResolution.y, voxelChunkResolution.z, 0);
			package.tCmd->RegistInitState(
				c.rt->GetInitState(),
				c.rt.get());
			package.tCmd->UpdateResState(
				GFXResourceState_UnorderedAccess,
				c.rt.get());
			struct VoxelParam {
				uint3 _VoxelMapSize;//the origin voxel size
				uint _Mode;			//0 - 5: from adjacent, 6 - 11: from self
				uint3 _VoxelMapIndex;
				uint align_Value0;
				uint3 _IndirectMapSize;
			};
			VoxelParam voxelParam;
			voxelParam._VoxelMapSize = voxelChunkResolution;
			voxelParam._VoxelMapIndex = cmd.generateEdgeIndex;
			voxelParam._IndirectMapSize = indirectSize;

			CBufferChunk ck = getCBufferChunk(sizeof(VoxelParam));
			ck.CopyData(&voxelParam);
			props.shader->SetResource(package.tCmd, props._SelfMap, heap, c.rt->GetGlobalUAVDescIndex(0));
			props.shader->SetResource(package.tCmd, ShaderID::GetMainTex(), heap, 0);
			props.shader->SetResource(package.tCmd, props._SRV_IndirectTex, heap, indirectRT->GetGlobalDescIndex());
			props.shader->SetResource(package.tCmd, ShaderID::GetParams(), ck.GetBuffer(), ck.GetOffset());
			uint3 dispatchCount = voxelChunkResolution + uint3(2, 2, 2);
			props.shader->Dispatch(package.tCmd, 0, (7 + dispatchCount.x) / 8, (7 + dispatchCount.z) / 8, 1);
			props.shader->Dispatch(package.tCmd, 1, (7 + voxelChunkResolution.y) / 8, (7 + dispatchCount.z) / 8, 1);
			props.shader->Dispatch(package.tCmd, 2, (7 + voxelChunkResolution.x) / 8, (7 + voxelChunkResolution.y) / 8, 1);

			//TODO
			//Corner
			package.tCmd->UpdateResState(
				c.rt->GetInitState(),
				c.rt.get());
			c.dirtyMask = DirtyMaskType::None;
		}
	}
	package.tCmd->UpdateResState(
		indirectRT->GetInitState(),
		indirectRT.get());
}

VirtualVoxelMap::~VirtualVoxelMap() {
}

VirtualVoxelMap::Chunk::Chunk() {
}

VirtualVoxelMap::Chunk::Chunk(Chunk&& c) : rt(std::move(c.rt)) {
}

VirtualVoxelMap::Chunk::~Chunk() {
}

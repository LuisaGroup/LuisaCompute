#include <PipelineComponent/RayTracing/RayTracingManager.h>
#include <RenderComponent/RenderComponentInclude.h>
#include <Singleton/ShaderID.h>

#include <Common/GameTimer.h>
#include <PipelineComponent/DXAllocator.h>
#include <PipelineComponent/ThreadCommand.h>
namespace RTAccStructUtil {
/*
		RayTracingManager::Command meshDeleteCmd(
			RayTracingManager::Command::CommandType::DeleteMesh,
			obj->GetInstanceID());//Submesh not used
		current->commands.Push(meshDeleteCmd);
*/
}// namespace RTAccStructUtil
namespace luisa::compute {

void GetRayTransform(D3D12_RAYTRACING_INSTANCE_DESC& inst, float4x4 const& tr) {
	using namespace Math;
	float4 right = tr[0];
	float4 up = tr[1];
	float4 forward = tr[2];
	float4 position = tr[3];
	float4* x = (float4*)(&inst.Transform[0][0]);
	*x = float4(right.x, up.x, forward.x, position.x);
	float4* y = (float4*)(&inst.Transform[1][0]);
	*y = float4(right.y, up.y, forward.y, position.y);
	float4* z = (float4*)(&inst.Transform[2][0]);
	*z = float4(right.z, up.z, forward.z, position.z);
}
void SpreadSize(bool& update, int64& size, uint64 newSize) {
	update = false;
	static constexpr int64 ALIGN_SIZE = 65536;//64kb
	if (size < (int64)newSize) {
		update = true;
	}
	size = GFXUtil::CalcAlign(Max<int64>(1, size), ALIGN_SIZE);
	while (size < newSize) {
		size *= 2;
	}
};
void UpdateMeshObject(
	GFXDevice* device,
	RayRendererData::MeshObject& meshObj,
	IMesh const* mesh) {
	meshObj.vboDescIndex = mesh->GetVBOSRVDescIndex(device);
	meshObj.iboDescIndex = mesh->GetIBOSRVDescIndex(device);
	meshObj.vertexOffset = 0;
	meshObj.indexOffset = 0;
}
void GetStaticTriangleGeometryDesc(GFXDevice* device, D3D12_RAYTRACING_GEOMETRY_DESC* data, IMesh const* mesh) {
	auto ibv = mesh->IndexBufferView();
	auto vbv = mesh->VertexBufferViews();
	size_t indexSize;
	if (ibv->Format == GFXFormat_R16_SInt || ibv->Format == GFXFormat_R16_UInt)
		indexSize = 2;
	else
		indexSize = 4;
	D3D12_RAYTRACING_GEOMETRY_DESC& geometryDesc = *data;
	geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
	geometryDesc.Triangles.IndexFormat = ibv->Format;
	geometryDesc.Triangles.Transform3x4 = 0;
	geometryDesc.Triangles.VertexFormat = (DXGI_FORMAT)GFXFormat_R32G32B32_Float;
	geometryDesc.Triangles.VertexBuffer.StrideInBytes = vbv->StrideInBytes;
	geometryDesc.Triangles.IndexBuffer = ibv->BufferLocation;
	geometryDesc.Triangles.IndexCount = mesh->GetIndexCount();
	geometryDesc.Triangles.VertexBuffer.StartAddress = vbv->BufferLocation;
	geometryDesc.Triangles.VertexCount = mesh->GetVertexCount();
}
bool RayTracingManager::Avaliable() const {
	return sepManager.GetElementCount() > 0;
}
RayRendererData* RayTracingManager::AddRenderer(
	IMesh* meshPtr,
	uint shaderID,
	uint materialID,
	float4x4 localToWorldMat) {
	using namespace RTAccStructUtil;
	if (!allBottomLevel.Contains(meshPtr->GetVObjectPtr()->GetInstanceID()))
		return nullptr;
	RayRendererData* newRender;
	newRender = rayRenderDataPool.New_Lock(poolMtx, meshPtr);
	newRender->transformMatrix = localToWorldMat;
	auto&& inst = newRender->instanceDesc;
	inst.InstanceID = 0;
	inst.InstanceMask = 1;
	inst.InstanceContributionToHitGroupIndex = 0;
	inst.Flags = 0;
	RayRendererData::MeshObject& meshObj = newRender->meshObj;
	meshObj.materialID = materialID;
	meshObj.shaderID = shaderID;
	sepManager.AddRenderer(newRender, 0);
	return newRender;
}

void RayTracingManager::UpdateRenderer(
	uint shaderID,
	uint materialID,
	RayRendererData* renderer) {
	using namespace RTAccStructUtil;
	RayRendererData::MeshObject& meshObj = renderer->meshObj;
	IMesh* mm = renderer->mesh;
	UpdateMeshObject(
		device,
		meshObj,
		mm);
	GetRayTransform(renderer->instanceDesc, renderer->transformMatrix);
	if (materialID != -1)
		meshObj.materialID = materialID;
	if (shaderID != -1)
		meshObj.shaderID = shaderID;
	sepManager.UpdateRenderer(renderer, 0);
}

void RayTracingManager::RemoveRenderer(
	RayRendererData* renderer) {
	sepManager.DeleteRenderer(renderer, 0, true);
}

void RayTracingManager::BuildTopLevelRTStruct(
	RenderPackage const& pack) {
	if (!isTopLevelDirty) return;
	isTopLevelDirty = false;
	if (!Avaliable()) return;
	topLevelBuildDesc.Inputs.NumDescs = sepManager.GetElementCount();
	ID3D12GraphicsCommandList4* cmdList = static_cast<ID3D12GraphicsCommandList4*>(pack.tCmd->GetCmdList());
	ID3D12Device5* device = static_cast<ID3D12Device5*>(this->device->device());
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
	device->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelBuildDesc.Inputs, &topLevelPrebuildInfo);
	uint64 scratchSize = 0;
	//Can Update
	if (topLevelAccStruct) {
		topLevelBuildDesc.SourceAccelerationStructureData = topLevelAccStruct->GetAddress(0, 0).address;
		scratchSize = topLevelPrebuildInfo.UpdateScratchDataSizeInBytes;
		topLevelBuildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
	} else {
		//Can not Update
		scratchSize = topLevelPrebuildInfo.ScratchDataSizeInBytes;
		topLevelBuildDesc.SourceAccelerationStructureData = 0;
		topLevelBuildDesc.Inputs.Flags =
			(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
				((uint)topLevelBuildDesc.Inputs.Flags)
				& (~((uint)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)));
	}
	ReserveStructSize(pack, topLevelPrebuildInfo.ResultDataMaxSizeInBytes, scratchSize);
	topLevelBuildDesc.DestAccelerationStructureData = topLevelAccStruct->GetAddress(0, 0).address;
	topLevelBuildDesc.ScratchAccelerationStructureData = scratchStruct->GetAddress(0, 0).address;
	topLevelBuildDesc.Inputs.InstanceDescs = instanceStruct->GetAddress(0, 0).address;
	pack.tCmd->ExecuteResBarrier();
	cmdList->BuildRaytracingAccelerationStructure(
		&topLevelBuildDesc,
		0,
		nullptr);
	pack.tCmd->UAVBarrier(
		topLevelAccStruct.get());
	pack.tCmd->UAVBarrier(
		scratchStruct.get());
}

GpuAddress RayTracingManager::GetInstanceBufferAddress() const {
	return {instanceStruct->GetAddress(0, 0)};
}

GpuAddress RayTracingManager::GetMeshObjectAddress() const {
	return {instanceStruct->GetAddress(1, 0)};
}

RayTracingManager::RayTracingManager(
	GFXDevice* originDevice)
	: device(originDevice),
	  sbuffers(256),
	  instanceUploadPool(sizeof(D3D12_RAYTRACING_INSTANCE_DESC), 1024, false),
	  meshObjUploadPool(sizeof(RayRendererData::MeshObject), 1024, false),
	  rayRenderDataPool(256) {
	static constexpr uint64 ACC_ALIGN = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
	ID3D12Device5* device = static_cast<ID3D12Device5*>(originDevice->device());
	memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& topLevelInputs = topLevelBuildDesc.Inputs;

	topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	topLevelInputs.Flags =
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
		| D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
	topLevelInputs.NumDescs = 0;
	topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
	rendDisposer = [this](SeparableRenderer* renderer) -> void {
		rayRenderDataPool.Delete_Lock(poolMtx, renderer);
	};
	lastFrameUpdateFunction = [](GFXDevice* device, SeparableRenderer* renderer, uint custom) {};
	addFunction = [this](GFXDevice* device, SeparableRenderer* renderer, uint custom) -> bool {
		auto ptr = static_cast<RayRendererData*>(renderer);

		ReserveInstanceBuffer(
			*pack,
			sepManager.GetElementCount());
		pack->tCmd->UpdateResState(
			GPUResourceState_CopyDest,
			instanceStruct.get());
		updateFunction(device, renderer, custom);
		return false;
	};
	updateFunction = [this](GFXDevice* device, SeparableRenderer* renderer, uint custom) -> bool {
		auto ptr = static_cast<RayRendererData*>(renderer);
		//////// Set Mesh
		auto ite = allBottomLevel.Find(ptr->mesh->GetVObjectPtr()->GetInstanceID());
		if (!ite) {
			VEngine_Log("Ray Renderer Contains No Mesh!\n"_sv);
			VSTL_ABORT();
		}

		ptr->instanceDesc.AccelerationStructure = ite.Value().bottomBufferChunk->GetAddress(0, 0).address;
		CopyInstanceDescData(ptr);
		return false;
	};
	removeFunction = [this](GFXDevice* device, SeparableRenderer* renderer, SeparableRenderer* last, uint custom, bool isLast) -> void {
		if (!isLast) {
			CopyInstanceDescData(static_cast<RayRendererData*>(last), renderer->GetListIndex());
		}
	};
}

RayTracingManager::~RayTracingManager() {
}

void RayTracingManager::ReserveStructSize(RenderPackage const& package, uint64 newStrSize, uint64 newScratchSize) {

	/*uint64 instanceSize = sizeof(D3D12_RAYTRACING_INSTANCE_DESC)*/
	bool update;
	SpreadSize(update, topLevelRayStructSize, newStrSize);
	if (update) {
		if (topLevelAccStruct) {
			topLevelAccStruct->ReleaseAfterFrame(package.frameRes);
		}
		topLevelAccStruct = std::unique_ptr<StructuredBuffer>(
			new StructuredBuffer(
				device,
				{StructuredBufferElement::Get(1, topLevelRayStructSize)},
				GPUResourceState_RayTracingStruct,
				nullptr));
	}
	SpreadSize(update, topLevelScratchSize, newScratchSize);
	if (update) {
		if (scratchStruct) {
			scratchStruct->ReleaseAfterFrame(package.frameRes);
		}
		scratchStruct = std::unique_ptr<StructuredBuffer>(
			new StructuredBuffer(
				device,
				{StructuredBufferElement::Get(1, topLevelScratchSize)},
				GPUResourceState_UnorderedAccess,
				nullptr));
	}
}
void RayTracingManager::ReserveInstanceBuffer(RenderPackage const& package, uint64 newObjSize) {
	bool update;
	static constexpr size_t STRIDE = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) + sizeof(RayRendererData::MeshObject);
	uint64 instanceSize = STRIDE * newObjSize;
	SpreadSize(update, instanceBufferSize, instanceSize);
	if (update) {
		newObjSize = instanceBufferSize / STRIDE;
		instanceBufferSize = STRIDE * newObjSize;
		std::unique_ptr<StructuredBuffer> newBuffer(
			new StructuredBuffer(
				device,
				{StructuredBufferElement::Get(sizeof(D3D12_RAYTRACING_INSTANCE_DESC), newObjSize),
				 StructuredBufferElement::Get(sizeof(RayRendererData::MeshObject), newObjSize)},
				GPUResourceState_NonPixelShaderRes,
				nullptr));
		auto cmdList = package.tCmd;

		cmdList->RegistInitState(
			newBuffer->GetInitState(),
			newBuffer.get());
		cmdList->UpdateResState(
			GPUResourceState_CopyDest,
			newBuffer.get());

		if (instanceStruct) {
			cmdList->RegistInitState(
				instanceStruct->GetInitState(),
				instanceStruct.get());
			cmdList->UpdateResState(
				GPUResourceState_CopySource,
				instanceStruct.get());
			Graphics::CopyBufferRegion(
				cmdList,
				newBuffer.get(),
				newBuffer->GetAddressOffset(0, 0),
				instanceStruct.get(),
				instanceStruct->GetAddressOffset(0, 0),
				instanceStruct->GetStride(0) * instanceStruct->GetElementCount(0));
			Graphics::CopyBufferRegion(
				cmdList,
				newBuffer.get(),
				newBuffer->GetAddressOffset(1, 0),
				instanceStruct.get(),
				instanceStruct->GetAddressOffset(1, 0),
				instanceStruct->GetStride(1) * instanceStruct->GetElementCount(1));
			cmdList->UAVBarrier(
				newBuffer.get());
			instanceStruct->ReleaseAfterFrame(package.frameRes);
		}
		instanceStruct = std::move(newBuffer);
	}
}
void RayTracingManager::BuildRTStruct(
	AllocatedCBufferChunks& allocatedElements,
	Runnable<CBufferChunk(size_t)> const& getCBuffer,
	RenderPackage const& pack) {
	this->allocatedElements = &allocatedElements;
	this->pack = &pack;
	//////// Init
	ID3D12Device5* device = static_cast<ID3D12Device5*>(pack.device->device());
	ID3D12GraphicsCommandList4* cmdList = static_cast<ID3D12GraphicsCommandList4*>(pack.tCmd->GetCmdList());
	for (auto& i : allocatedElements.instanceUploadElements) {
		instanceUploadPool.Return(i);
	}
	allocatedElements.instanceUploadElements.clear();
	for (auto& i : allocatedElements.meshObjUploadElements) {
		meshObjUploadPool.Return(i);
	}
	allocatedElements.meshObjUploadElements.clear();
	for (auto& i : allocatedElements.needClearSBuffers) {
		sbuffers.Delete_Lock(bottomAllocMtx, i);
	}
	allocatedElements.needClearSBuffers.clear();
	sepManager.Execute(
		pack.device,
		lastFrameUpdateFunction,
		addFunction,
		removeFunction,
		updateFunction,
		rendDisposer);
	if (instanceStruct) {
		pack.tCmd->UpdateResState(
			instanceStruct->GetInitState(),
			instanceStruct.get());
	}
}
void RayTracingManager::AddMesh(
	RenderPackage const& pack,
	vstd::vector<StructuredBuffer*>& clearBuffer,
	IMesh const* meshInterface, bool forceUpdateMesh) {

	auto ite = allBottomLevel.Emplace(meshInterface->GetVObjectPtr()->GetInstanceID());
	auto& v = ite.Value();
	if (v.referenceCount == 0 || forceUpdateMesh) {
		//////// Update Mesh
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
		D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& bottomInput = bottomStruct.Inputs;

		bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
		bottomInput.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
		bottomInput.NumDescs = 1;
		bottomInput.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		bottomInput.pGeometryDescs = &geometryDesc;
		GetStaticTriangleGeometryDesc(
			device,
			&geometryDesc,
			meshInterface);
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
		ID3D12Device5* device = static_cast<ID3D12Device5*>(this->device->device());
		device->GetRaytracingAccelerationStructurePrebuildInfo(
			&bottomInput,
			&bottomLevelPrebuildInfo);
		v.bottomBufferChunk = sbuffers.New_Lock(
			bottomAllocMtx,
			this->device,
			std::initializer_list<StructuredBufferElement>{StructuredBufferElement::Get(1, bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes)},
			GPUResourceState_RayTracingStruct,
			DXAllocator::GetBufferAllocator());
		bottomStruct.SourceAccelerationStructureData = 0;
		auto adrs = v.bottomBufferChunk->GetAddress(0, 0);
		bottomStruct.DestAccelerationStructureData = adrs.address;
		//TODO: build bottom

		auto bottomScratchChunk = sbuffers.New_Lock(
			bottomAllocMtx,
			this->device,
			std::initializer_list<StructuredBufferElement>{StructuredBufferElement::Get(1, bottomLevelPrebuildInfo.ScratchDataSizeInBytes)},
			GPUResourceState_UnorderedAccess,
			DXAllocator::GetBufferAllocator());
		clearBuffer.push_back(bottomScratchChunk);
		bottomStruct.ScratchAccelerationStructureData = bottomScratchChunk->GetAddress(0, 0).address;
		ID3D12GraphicsCommandList4* cmdList = static_cast<ID3D12GraphicsCommandList4*>(pack.tCmd->GetCmdList());
		pack.tCmd->ExecuteResBarrier();
		cmdList->BuildRaytracingAccelerationStructure(
			&bottomStruct,
			0,
			nullptr);
		pack.tCmd->UAVBarrier(
			v.bottomBufferChunk);
	}
	v.referenceCount++;
}

void RayTracingManager::RemoveMesh(
	uint64 instanceID,
	vstd::vector<StructuredBuffer*>& clearBuffer) {
	auto ite = allBottomLevel.Find(instanceID);
	if (!ite) return;
	auto& v = ite.Value();
	v.referenceCount--;
	if (v.referenceCount <= 0) {
		clearBuffer.push_back(v.bottomBufferChunk);
	}
	allBottomLevel.Remove(ite);
}
void RayTracingManager::CopyInstanceDescData(RayRendererData* data, uint topLevelIndex) {
	ConstBufferElement instUploadEle = instanceUploadPool.Get(device);
	allocatedElements->instanceUploadElements.push_back(instUploadEle);
	instUploadEle.buffer->CopyData(
		instUploadEle.element,
		&data->instanceDesc);
	static constexpr size_t INST_SIZE = sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
	Graphics::CopyBufferRegion(
		pack->tCmd,
		instanceStruct.get(),
		topLevelIndex * INST_SIZE,
		instUploadEle.buffer,
		instUploadEle.element * INST_SIZE,
		INST_SIZE);
	ConstBufferElement meshUploadEle = meshObjUploadPool.Get(device);
	allocatedElements->meshObjUploadElements.push_back(meshUploadEle);
	meshUploadEle.buffer->CopyData(
		meshUploadEle.element,
		&data->meshObj);
	static constexpr size_t MESH_OBJ_SIZE = sizeof(RayRendererData::MeshObject);
	uint64 meshObjOffset = instanceStruct->GetAddressOffset(1, 0);
	Graphics::CopyBufferRegion(
		pack->tCmd,
		instanceStruct.get(),
		meshObjOffset + topLevelIndex * MESH_OBJ_SIZE,
		meshUploadEle.buffer,
		meshUploadEle.element * MESH_OBJ_SIZE,
		MESH_OBJ_SIZE);
}
void RayTracingManager::CopyInstanceDescData(RayRendererData* data) {
	uint topLevelIndex = data->GetListIndex();
	CopyInstanceDescData(data, topLevelIndex);
}
}// namespace luisa::compute
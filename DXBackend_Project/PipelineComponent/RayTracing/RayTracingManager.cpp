#include "RayTracingManager.h"
#include "../../RenderComponent/RenderComponentInclude.h"
#include "../../LogicComponent/Transform.h"
#include "../../Singleton/ShaderID.h"
#include "../../Singleton/ShaderLoader.h"
#include "../ThreadCommand.h"
RayTracingManager* RayTracingManager::current = nullptr;
namespace RTAccStructUtil {
void GetRayTransform(D3D12_RAYTRACING_INSTANCE_DESC& inst, Transform const* tr) {
	using namespace Math;
	float3 localScale = tr->GetLocalScale();
	float3 right = (Vector3)tr->GetRight() * localScale.x;
	float3 up = (Vector3)tr->GetUp() * localScale.y;
	float3 forward = (Vector3)tr->GetForward() * localScale.z;
	float3 position = tr->GetPosition();
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
	IMesh const* mesh,
	uint subMeshIndex) {
	meshObj.vboDescIndex = mesh->GetVBOSRVDescIndex(device);
	meshObj.iboDescIndex = mesh->GetIBOSRVDescIndex(device);
	if (subMeshIndex == -1) {
		meshObj.vertexOffset = 0;
		meshObj.indexOffset = 0;
	} else {
		auto&& subMesh = mesh->GetSubMesh(subMeshIndex);
		meshObj.vertexOffset = subMesh.vertexOffset;
		meshObj.indexOffset = subMesh.indexOffset;
	}
}
void GetStaticTriangleGeometryDesc(GFXDevice* device, D3D12_RAYTRACING_GEOMETRY_DESC* data, IMesh const* mesh, uint subMeshIndex) {
	auto ibv = mesh->IndexBufferView();
	auto vbv = mesh->VertexBufferViews();
	size_t indexSize;
	if (ibv.Format == GFXFormat_R16_SInt || ibv.Format == GFXFormat_R16_UInt)
		indexSize = 2;
	else
		indexSize = 4;
	D3D12_RAYTRACING_GEOMETRY_DESC& geometryDesc = *data;
	geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
	geometryDesc.Triangles.IndexFormat = ibv.Format;
	geometryDesc.Triangles.Transform3x4 = 0;
	geometryDesc.Triangles.VertexFormat = (DXGI_FORMAT)GFXFormat_R32G32B32_Float;
	geometryDesc.Triangles.VertexBuffer.StrideInBytes = vbv->StrideInBytes;

	if (subMeshIndex == -1) {
		geometryDesc.Triangles.IndexBuffer = ibv.BufferLocation;
		geometryDesc.Triangles.IndexCount = mesh->GetIndexCount();
		geometryDesc.Triangles.VertexBuffer.StartAddress = vbv->BufferLocation;
		geometryDesc.Triangles.VertexCount = mesh->GetVertexCount();

	} else {
		auto&& subMesh = mesh->GetSubMesh(subMeshIndex);
		geometryDesc.Triangles.IndexBuffer = ibv.BufferLocation + indexSize * subMesh.indexOffset;
		geometryDesc.Triangles.IndexCount = subMesh.indexCount;
		geometryDesc.Triangles.VertexBuffer.StartAddress = vbv->BufferLocation + vbv->StrideInBytes * subMesh.vertexOffset;
		geometryDesc.Triangles.VertexCount = mesh->GetVertexCount() - subMesh.vertexOffset;
	}
}

}// namespace RTAccStructUtil
RayRendererData* RayTracingManager::AddRenderer(
	ObjectPtr<IMesh>&& meshPtr,
	uint shaderID,
	uint materialID,
	Transform const* tr,
	uint subMeshIndex) {

	RayRendererData* newRender;
	newRender = rayRenderDataPool.New_Lock(poolMtx, std::move(meshPtr));
	auto&& inst = newRender->instanceDesc;
	RTAccStructUtil::GetRayTransform(inst, tr);
	inst.InstanceID = 0;
	inst.InstanceMask = 1;
	inst.InstanceContributionToHitGroupIndex = 0;
	inst.Flags = 0;
	RayRendererData::MeshObject& meshObj = newRender->meshObj;
	RTAccStructUtil::UpdateMeshObject(
		device,
		meshObj,
		newRender->mesh,
		subMeshIndex);
	meshObj.materialID = materialID;
	meshObj.shaderID = shaderID;
	Command meshBuildCmd(
		Command::CommandType::AddMesh,
		newRender->mesh,
		subMeshIndex);
	commands.Push(meshBuildCmd);
	Command cmd(
		Command::CommandType::Add,
		newRender);
	commands.Push(cmd);
	return newRender;
}

void RayTracingManager::UpdateRenderer(
	ObjectPtr<IMesh>&& mesh,
	uint shaderID,
	uint materialID,
	Transform const* tr,
	RayRendererData* renderer,
	uint subMeshIndex) {

	RayRendererData::MeshObject& meshObj = renderer->meshObj;
	RTAccStructUtil::GetRayTransform(renderer->instanceDesc, tr);

	if (mesh && renderer->mesh != mesh) {
		if (renderer->mesh) {
			Command meshDeleteCmd(
				Command::CommandType::DeleteMesh,
				renderer->mesh,
				subMeshIndex);
			commands.Push(meshDeleteCmd);
		}
		renderer->mesh = std::move(mesh);
		IMesh const* mm = renderer->mesh;
		Command meshBuildCmd(
			Command::CommandType::AddMesh,
			mm,
			subMeshIndex);
		RTAccStructUtil::UpdateMeshObject(
			device,
			meshObj,
			mm,
			subMeshIndex);
		commands.Push(meshBuildCmd);
	}
	if (materialID != -1)
		meshObj.materialID = materialID;
	if (shaderID != -1)
		meshObj.shaderID = shaderID;
	Command cmd(
		Command::CommandType::Add,
		renderer);
	commands.Push(cmd);
}

void RayTracingManager::RemoveRenderer(
	RayRendererData* renderer) {
	Command cmd(
		Command::CommandType::Delete,
		renderer);
	commands.Push(cmd);
	Command meshCmd(
		Command::CommandType::DeleteMesh,
		renderer->mesh,
		-1);
	commands.Push(meshCmd);
}

void RayTracingManager::BuildTopLevelRTStruct(
	RenderPackage const& pack) {
	if (!isTopLevelDirty) return;
	isTopLevelDirty = false;
	topLevelBuildDesc.Inputs.NumDescs = renderersList.size();
	ID3D12GraphicsCommandList4* cmdList = static_cast<ID3D12GraphicsCommandList4*>(pack.tCmd->GetCmdList());
	ID3D12Device5* device = static_cast<ID3D12Device5*>(this->device);
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
		topLevelBuildDesc.Inputs.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(((uint)topLevelBuildDesc.Inputs.Flags)
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

StructuredBuffer const* RayTracingManager::GetRayTracingStruct() const {
	return topLevelAccStruct.get();
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
	if (current) {
		VEngine_Log("Ray Tracing Manager Should be Singleton!\n");
		throw 0;
	}
	_Scene = ShaderID::PropertyToID("_Scene");
	_Meshes = ShaderID::PropertyToID("_Meshes");
	_InstanceBuffer = ShaderID::PropertyToID("_InstanceBuffer");
	_IndexBuffer = ShaderID::PropertyToID("_IndexBuffer");
	_VertexBuffer = ShaderID::PropertyToID("_VertexBuffer");
	current = this;
	rtUtilcs = ShaderLoader::GetComputeShader("RTUtility");
	static constexpr uint64 ACC_ALIGN = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
	ID3D12Device5* device = static_cast<ID3D12Device5*>(originDevice);
	memset(&topLevelBuildDesc, 0, sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC));
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& topLevelInputs = topLevelBuildDesc.Inputs;

	topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
	topLevelInputs.Flags =
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
		| D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
	topLevelInputs.NumDescs = 0;
	topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
}

RayTracingManager::~RayTracingManager() {
	if (current == this) current = nullptr;
}

void RayTracingManager::ReserveStructSize(RenderPackage const& package, uint64 newStrSize, uint64 newScratchSize) {

	/*uint64 instanceSize = sizeof(D3D12_RAYTRACING_INSTANCE_DESC)*/
	bool update;
	RTAccStructUtil::SpreadSize(update, topLevelRayStructSize, newStrSize);
	if (update) {
		if (topLevelAccStruct) {
			//TODO:: Delay Dispose
		}
		topLevelAccStruct = std::unique_ptr<StructuredBuffer>(
			new StructuredBuffer(
				device,
				{StructuredBufferElement::Get(1, topLevelRayStructSize)},
				GFXResourceState_RayTracingStruct,
				nullptr));
	}
	RTAccStructUtil::SpreadSize(update, topLevelScratchSize, newScratchSize);
	if (update) {
		if (scratchStruct) {
			//TODO:: Delay Dispose
		}
		scratchStruct = std::unique_ptr<StructuredBuffer>(
			new StructuredBuffer(
				device,
				{StructuredBufferElement::Get(1, topLevelScratchSize)},
				GFXResourceState_UnorderedAccess,
				nullptr));
	}
}
void RayTracingManager::ReserveInstanceBuffer(RenderPackage const& package, uint64 newObjSize) {
	bool update;
	static constexpr size_t STRIDE = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) + sizeof(RayRendererData::MeshObject);
	uint64 instanceSize = STRIDE * newObjSize;
	RTAccStructUtil::SpreadSize(update, instanceBufferSize, instanceSize);
	if (update) {
		newObjSize = instanceBufferSize / STRIDE;
		instanceBufferSize = STRIDE * newObjSize;
		std::unique_ptr<StructuredBuffer> newBuffer(
			new StructuredBuffer(
				device,
				{StructuredBufferElement::Get(sizeof(D3D12_RAYTRACING_INSTANCE_DESC), newObjSize),
				 StructuredBufferElement::Get(sizeof(RayRendererData::MeshObject), newObjSize)},
				GFXResourceState_NonPixelRead,
				nullptr));
		auto cmdList = package.tCmd;

		cmdList->RegistInitState(
			newBuffer->GetInitState(),
			newBuffer.get());
		cmdList->UpdateResState(
			GFXResourceState_CopyDest,
			newBuffer.get());

		if (instanceStruct) {
			cmdList->RegistInitState(
				instanceStruct->GetInitState(),
				instanceStruct.get());
			cmdList->UpdateResState(
				GFXResourceState_CopySource,
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
			//TODO:: Delay Dispose
		}
		instanceStruct = std::move(newBuffer);
	}
}
void RayTracingManager::SetShaderResources(IShader const* shader, ThreadCommand* cmdList) {
	if (!Avaliable()) return;
	auto heap = Graphics::GetGlobalDescHeap();
	shader->SetResource(
		cmdList,
		_Scene,
		GetRayTracingStruct(),
		0);
	shader->SetBufferByAddress(
		cmdList,
		_Meshes,
		GetMeshObjectAddress());
	shader->SetBufferByAddress(
		cmdList,
		_InstanceBuffer,
		GetInstanceBufferAddress());
	shader->SetResource(
		cmdList,
		_VertexBuffer,
		heap, 0);
	shader->SetResource(
		cmdList,
		_IndexBuffer,
		heap, 0);
}
void RayTracingManager::BuildRTStruct(
	AllocatedCBufferChunks& allocatedElements,
	Runnable<CBufferChunk(size_t)> const& getCBuffer,
	RenderPackage const& pack) {
	//////// Init
	ID3D12Device5* device = static_cast<ID3D12Device5*>(pack.device);
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
	//////// Move the world


	if (instanceStruct) {
		pack.tCmd->RegistInitState(
			instanceStruct->GetInitState(),
			instanceStruct.get());

		if (moveTheWorld && !renderersList.empty()) {
			isTopLevelDirty = true;
			auto cmdList = pack.tCmd;
			auto heap = Graphics::GetGlobalDescHeap();
			rtUtilcs->BindShader(
				cmdList,
				heap);
			struct RTUtilParam {
				float3 _MoveDirection;
				uint _Count;
			};
			CBufferChunk ck = getCBuffer(sizeof(RTUtilParam));
			RTUtilParam param = {
				(float3)moveDir,
				renderersList.size()};
			ck.CopyData(&param);
			rtUtilcs->SetResource(
				cmdList,
				ShaderID::GetParams(),
				ck.GetBuffer(),
				ck.GetOffset());
			rtUtilcs->SetResource(
				cmdList,
				propID._InstanceData,
				instanceStruct.get(),
				0);
			cmdList->UpdateResState(
				GFXResourceState_UnorderedAccess,
				instanceStruct.get());
			
			rtUtilcs->Dispatch(
				cmdList,
				0,
				(renderersList.size() + 63) / 64, 1, 1);
			cmdList->UAVBarrier(instanceStruct.get());
		}
		pack.tCmd->UpdateResState(
			GFXResourceState_CopyDest,
			instanceStruct.get());
	}
	moveTheWorld = false;
	moveDir = {0, 0, 0};
	//////// Execute Commands
	auto CopyInstanceDescData =
		[&](RayRendererData* data) -> void {
		uint topLevelIndex = data->listIndex;
		ConstBufferElement instUploadEle = instanceUploadPool.Get(device);
		allocatedElements.instanceUploadElements.push_back(instUploadEle);
		instUploadEle.buffer->CopyData(
			instUploadEle.element,
			&data->instanceDesc);
		static constexpr size_t INST_SIZE = sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
		Graphics::CopyBufferRegion(
			pack.tCmd,
			instanceStruct.get(),
			topLevelIndex * INST_SIZE,
			instUploadEle.buffer,
			instUploadEle.element * INST_SIZE,
			INST_SIZE);
		ConstBufferElement meshUploadEle = meshObjUploadPool.Get(device);
		allocatedElements.meshObjUploadElements.push_back(meshUploadEle);
		meshUploadEle.buffer->CopyData(
			meshUploadEle.element,
			&data->meshObj);
		static constexpr size_t MESH_OBJ_SIZE = sizeof(RayRendererData::MeshObject);
		uint64 meshObjOffset = instanceStruct->GetAddressOffset(1, 0);
		Graphics::CopyBufferRegion(
			pack.tCmd,
			instanceStruct.get(),
			meshObjOffset + topLevelIndex * MESH_OBJ_SIZE,
			meshUploadEle.buffer,
			meshUploadEle.element * MESH_OBJ_SIZE,
			MESH_OBJ_SIZE);
	};
	Command cmd;
	//////// Build Bottom Level
	while (commands.Pop(&cmd)) {
		isTopLevelDirty = true;
		switch (cmd.type) {
			case Command::CommandType::Add: {

				//Add
				if (cmd.ptr->listIndex == -1) {
					cmd.ptr->listIndex = renderersList.size();
					renderersList.push_back(cmd.ptr);
					ReserveInstanceBuffer(
						pack,
						renderersList.size());
					pack.tCmd->UpdateResState(
						GFXResourceState_CopyDest,
						instanceStruct.get());
				}
				//////// Set Mesh
				auto ite = allBottomLevel.Find(cmd.ptr->mesh->GetVObjectPtr()->GetInstanceID());
				if (!ite) {
					VEngine_Log("Ray Renderer Contains No Mesh!\n");
					throw 0;
				}
				BottomLevelSubMesh* subMesh = nullptr;
				for (auto& i : ite.Value().subMeshes) {
					if (i.subMeshIndex == cmd.ptr->subMeshIndex) {
						subMesh = &i;
					}
				}
				if (!subMesh) {
					VEngine_Log("Ray Renderer Contains No Mesh!\n");
					throw 0;
				}
				cmd.ptr->instanceDesc.AccelerationStructure = subMesh->bottomBufferChunk->GetAddress(0, 0).address;
				CopyInstanceDescData(cmd.ptr);
			} break;
			case Command::CommandType::Delete: {
				//allocatedElements.needClearSBuffers.push_back(cmd.ptr->bottomBufferChunk);
				auto ite = renderersList.end() - 1;
				auto&& i = *ite;
				if (cmd.ptr->listIndex != renderersList.size() - 1) {
					i->listIndex = cmd.ptr->listIndex;
					renderersList[i->listIndex] = i;
					CopyInstanceDescData(i);
				}
				renderersList.erase(ite);
				rayRenderDataPool.Delete_Lock(poolMtx, cmd.ptr);
			} break;
			case Command::CommandType::AddMesh:
				AddMesh(
					pack,
					allocatedElements.needClearSBuffers,
					cmd.mesh,
					cmd.subMeshIndex,
					false);
				cmd.ptr->subMeshIndex = cmd.subMeshIndex;
				break;
			case Command::CommandType::DeleteMesh:
				RemoveMesh(
					cmd.mesh,
					allocatedElements.needClearSBuffers);
				break;
		}
	}
	if (instanceStruct) {
		pack.tCmd->UpdateResState(
			instanceStruct->GetInitState(),
			instanceStruct.get());
	}
}
void RayTracingManager::AddMesh(
	RenderPackage const& pack,
	vengine::vector<StructuredBuffer*>& clearBuffer,
	IMesh const* meshInterface,
	uint subMeshIndex, bool forceUpdateMesh) {

	auto ite = allBottomLevel.Insert(meshInterface->GetVObjectPtr()->GetInstanceID());
	auto& v = ite.Value();
	BottomLevelSubMesh* subMesh = nullptr;
	for (auto& i : v.subMeshes) {
		if (i.subMeshIndex == subMeshIndex) {
			subMesh = &i;
		}
	}
	if (v.referenceCount == 0 || forceUpdateMesh || !subMesh) {
		if (!subMesh) {
			subMesh = &v.subMeshes.emplace_back();
		}
		//////// Update Mesh
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
		D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
		D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& bottomInput = bottomStruct.Inputs;

		bottomInput.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
		bottomInput.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
		bottomInput.NumDescs = 1;
		bottomInput.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
		bottomInput.pGeometryDescs = &geometryDesc;
		RTAccStructUtil::GetStaticTriangleGeometryDesc(
			device,
			&geometryDesc,
			meshInterface, subMeshIndex);
		D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
		ID3D12Device5* device = static_cast<ID3D12Device5*>(this->device);
		device->GetRaytracingAccelerationStructurePrebuildInfo(
			&bottomInput,
			&bottomLevelPrebuildInfo);
		subMesh->bottomBufferChunk = sbuffers.New_Lock(
			bottomAllocMtx,
			device,
			std::initializer_list<StructuredBufferElement>{StructuredBufferElement::Get(1, bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes)},
			GFXResourceState_RayTracingStruct,
			nullptr);//TODO: allocator
		bottomStruct.SourceAccelerationStructureData = 0;
		auto adrs = subMesh->bottomBufferChunk->GetAddress(0, 0);
		bottomStruct.DestAccelerationStructureData = adrs.address;
		//TODO: build bottom

		auto bottomScratchChunk = sbuffers.New_Lock(
			bottomAllocMtx,
			device,
			std::initializer_list<StructuredBufferElement>{StructuredBufferElement::Get(1, bottomLevelPrebuildInfo.ScratchDataSizeInBytes)},
			GFXResourceState_UnorderedAccess,
			nullptr);//TODO: allocator
		clearBuffer.push_back(bottomScratchChunk);
		bottomStruct.ScratchAccelerationStructureData = bottomScratchChunk->GetAddress(0, 0).address;
		ID3D12GraphicsCommandList4* cmdList = static_cast<ID3D12GraphicsCommandList4*>(pack.tCmd->GetCmdList());
		pack.tCmd->ExecuteResBarrier();
		cmdList->BuildRaytracingAccelerationStructure(
			&bottomStruct,
			0,
			nullptr);
		pack.tCmd->UAVBarrier(
			subMesh->bottomBufferChunk);
	}
	v.referenceCount++;
}
RayTracingManager::PropID::PropID() {
	_InstanceData = ShaderID::PropertyToID("_InstanceData");
}

void RayTracingManager::RemoveMesh(
	IMesh const* meshInterface,
	vengine::vector<StructuredBuffer*>& clearBuffer) {
	auto ite = allBottomLevel.Find(meshInterface->GetVObjectPtr()->GetInstanceID());
	if (!ite) return;
	auto& v = ite.Value();
	v.referenceCount--;
	if (v.referenceCount <= 0) {
		for (auto& i : v.subMeshes) {
			clearBuffer.push_back(i.bottomBufferChunk);
		};
	}
	allBottomLevel.Remove(ite);
}

RayTracingManager::AllocatedCBufferChunks::~AllocatedCBufferChunks() {
}

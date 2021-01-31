#pragma once
#include "../../Common/GFXUtil.h"
#include "../../Common/VObject.h"
#include "../../Struct/RenderPackage.h"
#include "../../RenderComponent/CBufferPool.h"
#include "../../RenderComponent/CBufferAllocator.h"
#include "../../Common/LockFreeArrayQueue.h"
#include "../../RenderComponent/RayRendererData.h"
class Transform;
class IShader;
class IMesh;
class StructuredBuffer;
class UploadBuffer;
class ComputeShader;
class ThreadCommand;
class RayTracingManager final {
private:

	//Update And Add Should only called in main job threads, will be unsafe if called in loading thread!

	~RayTracingManager();
	void ReserveStructSize(RenderPackage const& package, uint64 newStrSize, uint64 newScratchSize);
	void AddMesh(RenderPackage const& pack, vengine::vector<StructuredBuffer*>& clearBuffer, IMesh const* meshInterface, uint subMeshIndex, bool forceUpdateMesh);
	void RemoveMesh(IMesh const* meshInterface, vengine::vector<StructuredBuffer*>& clearBuffer);

public:
	class AllocatedCBufferChunks {
		friend class RayTracingManager;
		vengine::vector<ConstBufferElement> instanceUploadElements;
		vengine::vector<ConstBufferElement> meshObjUploadElements;
		vengine::vector<StructuredBuffer*> needClearSBuffers;
		~AllocatedCBufferChunks();
	};
	bool Avaliable() const {
		return !renderersList.empty();
	}
	RayRendererData* AddRenderer(
		ObjectPtr<IMesh>&& meshPtr,
		uint shaderID,
		uint materialID,
		Transform const* renderer,
		uint subMeshIndex);
	void UpdateRenderer(
		ObjectPtr<IMesh>&& mesh,
		uint shaderID,
		uint materialID,
		Transform const* tr,
		RayRendererData* renderer,
		uint subMeshIndex);
	void RemoveRenderer(
		RayRendererData* renderer);
	void BuildTopLevelRTStruct(
		RenderPackage const& pack);
	void MoveTheWorld(Math::Vector3 const& moveDir) {
		//TODO: Disable Move The World,
		//Currently Ray Tracing cannot support Move The World correctly

		this->moveDir += moveDir;
		moveTheWorld = true;
	}
	StructuredBuffer const* GetRayTracingStruct() const;
	GpuAddress GetInstanceBufferAddress() const;
	GpuAddress GetMeshObjectAddress() const;
	static RayTracingManager* GetInstance() { return current; }
	static void DestroyInstance() {
		if (current) delete current;
	}
	RayTracingManager(
		GFXDevice* device);
	void BuildRTStruct(
		AllocatedCBufferChunks& allocatedElements,
		Runnable<CBufferChunk(size_t)> const& getCBuffer,
		RenderPackage const& pack);
	void ReserveInstanceBuffer(
		RenderPackage const& package,
		uint64 newObjSize);
	//A per camera per frame data, to keep upload buffer chunks non-changed before flush
	void SetShaderResources(
		IShader const* shader,
		ThreadCommand* cmdList);
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	GFXDevice* device;
	static RayTracingManager* current;
	struct Command {
		enum class CommandType : uint8_t {
			Add,
			Delete,
			AddMesh,
			DeleteMesh
		};
		CommandType type;
		union {
			RayRendererData* ptr;
			IMesh const* mesh;
		};
		uint subMeshIndex;
		Command() {}
		Command(
			CommandType type,
			RayRendererData* ptr) {
			this->type = type;
			this->ptr = ptr;
		}
		Command(
			CommandType type,
			IMesh const* mesh,
			uint subMeshIndex) {
			this->type = type;
			this->mesh = mesh;
			this->subMeshIndex = subMeshIndex;
		}
	};
	LockFreeArrayQueue<Command> commands;
	CBufferPool instanceUploadPool;
	CBufferPool meshObjUploadPool;
	Pool<RayRendererData, true, true> rayRenderDataPool;
	Pool<StructuredBuffer, true, false> sbuffers;
	vengine::vector<RayRendererData*> renderersList;
	std::unique_ptr<StructuredBuffer> topLevelAccStruct;
	std::unique_ptr<StructuredBuffer> instanceStruct;
	std::unique_ptr<StructuredBuffer> scratchStruct;
	mutable std::mutex poolMtx;
	mutable std::mutex bottomAllocMtx;
	ComputeShader const* rtUtilcs;
	int64 topLevelRayStructSize = -1;
	int64 topLevelScratchSize = -1;
	int64 instanceBufferSize = -1;
	bool isTopLevelDirty = false;
	bool moveTheWorld = false;
	Math::Vector3 moveDir = {0, 0, 0};
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
	struct PropID {
		uint _InstanceData;
		PropID();
	};
	PropID propID;
	struct BottomLevelSubMesh {
		uint subMeshIndex;
		StructuredBuffer* bottomBufferChunk;
	};
	struct BottomLevelBuild {
		int32 referenceCount = 0;
		ArrayList<BottomLevelSubMesh> subMeshes;
	};
	HashMap<uint64, BottomLevelBuild> allBottomLevel;
	uint _Scene;
	uint _Meshes;
	uint _InstanceBuffer;
	uint _VertexBuffer;
	uint _IndexBuffer;
};
#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Struct/RenderPackage.h>
#include <RenderComponent/CBufferPool.h>
#include <RenderComponent/CBufferAllocator.h>
#include <Common/LockFreeArrayQueue.h>
#include <RenderComponent/RayRendererData.h>
#include <RenderComponent/Utility/SeparableRendererManager.h>
class IShader;
class IMesh;
class StructuredBuffer;
class UploadBuffer;
class ComputeShader;
class ThreadCommand;
namespace RTAccStructUtil {
class RemoveMeshFunctor;
}
template<typename T, bool useVEngineMalloc = true, bool isTrivially = std::is_trivially_destructible<T>::value>
using VEnginePool = Pool<T, useVEngineMalloc, isTrivially>;
namespace luisa::compute {
class VENGINE_DLL_RENDERER RayTracingManager final {
	friend class RTAccStructUtil::RemoveMeshFunctor;

private:
	//Update And Add Should only called in main job threads, will be unsafe if called in loading thread!
	~RayTracingManager();
	void ReserveStructSize(RenderPackage const& package, uint64 newStrSize, uint64 newScratchSize);
	void AddMesh(RenderPackage const& pack, vengine::vector<StructuredBuffer*>& clearBuffer, IMesh const* meshInterface, uint subMeshIndex, bool forceUpdateMesh);
	void RemoveMesh(uint64 instanceID, vengine::vector<StructuredBuffer*>& clearBuffer);
	void CopyInstanceDescData(RayRendererData* data, uint index);
	void CopyInstanceDescData(RayRendererData* data);

public:
	class AllocatedCBufferChunks {
		friend class RayTracingManager;
		vengine::vector<ConstBufferElement> instanceUploadElements;
		vengine::vector<ConstBufferElement> meshObjUploadElements;
		vengine::vector<StructuredBuffer*> needClearSBuffers;
	};
	bool Avaliable() const;
	RayRendererData* AddRenderer(
		ObjectPtr<IMesh>&& meshPtr,
		uint shaderID,
		uint materialID,
		float4x4 localToWorldMat,
		uint subMeshIndex);
	void UpdateRenderer(
		ObjectPtr<IMesh>&& mesh,
		uint shaderID,
		uint materialID,
		RayRendererData* renderer,
		uint subMeshIndex);
	void RemoveRenderer(
		RayRendererData* renderer);
	void BuildTopLevelRTStruct(
		RenderPackage const& pack);
	StructuredBuffer const* GetRayTracingStruct() const {
		return topLevelAccStruct.get();
	}
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
	struct Command {
		enum class CommandType : uint8_t {
			AddMesh,
			DeleteMesh
		};
		CommandType type;
		union {
			RayRendererData* ptr;
			IMesh const* mesh;
			uint64 instanceID;
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
		Command(
			CommandType type,
			uint64 instanceID,
			uint subMeshIndex) {
			this->type = type;
			this->instanceID = instanceID;
			this->subMeshIndex = subMeshIndex;
		}
	};
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	enum class UpdateOperator : uint {
		UpdateTrans = 1,
		UpdateMesh = 2
	};
	GFXDevice* device;
	AllocatedCBufferChunks* allocatedElements;
	RenderPackage const* pack;
	static RayTracingManager* current;

	LockFreeArrayQueue<Command> commands;
	CBufferPool instanceUploadPool;
	CBufferPool meshObjUploadPool;
	VEnginePool<RayRendererData, true, true> rayRenderDataPool;
	VEnginePool<StructuredBuffer, true, false> sbuffers;
	std::unique_ptr<StructuredBuffer> topLevelAccStruct;
	std::unique_ptr<StructuredBuffer> instanceStruct;
	std::unique_ptr<StructuredBuffer> scratchStruct;
	mutable std::mutex poolMtx;
	mutable std::mutex bottomAllocMtx;
	int64 topLevelRayStructSize = -1;
	int64 topLevelScratchSize = -1;
	int64 instanceBufferSize = -1;
	bool isTopLevelDirty = false;
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
	struct BottomLevelSubMesh {
		uint subMeshIndex;
		StructuredBuffer* bottomBufferChunk;
	};
	struct BottomLevelBuild {
		int32 referenceCount = 0;
		ArrayList<BottomLevelSubMesh> subMeshes;
	};
	HashMap<uint64, BottomLevelBuild> allBottomLevel;
	SeparableRendererManager sepManager;
	Runnable<void(GFXDevice*, SeparableRenderer*, uint)> lastFrameUpdateFunction;
	Runnable<bool(GFXDevice*, SeparableRenderer*, uint)> addFunction;
	Runnable<void(GFXDevice*, SeparableRenderer*, SeparableRenderer*, uint, bool)> removeFunction;// device, current, last, custom, isLast
	Runnable<bool(GFXDevice*, SeparableRenderer*, uint)> updateFunction;
	Runnable<void(SeparableRenderer*)> rendDisposer;
};
}// namespace luisa::compute
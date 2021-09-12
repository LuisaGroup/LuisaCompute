#pragma once
#include <Common/GFXUtil.h>
#include <vstl/VObject.h>
#include <Struct/RenderPackage.h>
#include <RenderComponent/CBufferPool.h>
#include <RenderComponent/CBufferAllocator.h>
#include <vstl/LockFreeArrayQueue.h>
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
template<typename T, VEngine_AllocType useVEngineMalloc = VEngine_AllocType::VEngine, bool isTrivially = std::is_trivially_destructible<T>::value>
using VEnginePool = Pool<T, useVEngineMalloc, isTrivially>;
namespace luisa::compute {
class VENGINE_DLL_RENDERER RayTracingManager final {
	friend class RTAccStructUtil::RemoveMeshFunctor;

private:
	//Update And Add Should only called in main job threads, will be unsafe if called in loading thread!
	void ReserveStructSize(RenderPackage const& package, uint64 newStrSize, uint64 newScratchSize);
	void AddMesh(RenderPackage const& pack, vstd::vector<StructuredBuffer*>& clearBuffer, IMesh const* meshInterface, bool forceUpdateMesh);
	void RemoveMesh(uint64 instanceID, vstd::vector<StructuredBuffer*>& clearBuffer);
	void CopyInstanceDescData(RayRendererData* data, uint index);
	void CopyInstanceDescData(RayRendererData* data);

public:
	~RayTracingManager();

	class AllocatedCBufferChunks {
		friend class RayTracingManager;
		vstd::vector<ConstBufferElement> instanceUploadElements;
		vstd::vector<ConstBufferElement> meshObjUploadElements;
		vstd::vector<StructuredBuffer*> needClearSBuffers;
	};
	bool Avaliable() const;
	RayRendererData* AddRenderer(
		IMesh* meshPtr,
		uint shaderID,
		uint materialID,
		float4x4 localToWorldMat);
	void UpdateRenderer(
		uint shaderID,
		uint materialID,
		RayRendererData* renderer);
	void RemoveRenderer(
		RayRendererData* renderer);
	void BuildTopLevelRTStruct(
		RenderPackage const& pack);
	StructuredBuffer const* GetRayTracingStruct() const {
		return topLevelAccStruct.get();
	}
	GpuAddress GetInstanceBufferAddress() const;
	GpuAddress GetMeshObjectAddress() const;
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
	
	VSTL_OVERRIDE_OPERATOR_NEW
private:
	GFXDevice* device;
	AllocatedCBufferChunks* allocatedElements;
	RenderPackage const* pack;

	CBufferPool instanceUploadPool;
	CBufferPool meshObjUploadPool;
	VEnginePool<RayRendererData, VEngine_AllocType::VEngine, true> rayRenderDataPool;
	VEnginePool<StructuredBuffer, VEngine_AllocType::VEngine, false> sbuffers;
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
	struct BottomLevelBuild {
		int32 referenceCount = 0;
		StructuredBuffer* bottomBufferChunk;
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

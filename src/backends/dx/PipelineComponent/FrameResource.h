#pragma once
#include <Common/Common.h>
#include <vstl/VObject.h>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/CBufferAllocator.h>
namespace luisa::compute {
class FrameResource final {
public:
	CBufferAllocator* cbAlloc;
	ThreadCommand tCmd;
	uint64 signalIndex = 0;
	vstd::vector<Microsoft::WRL::ComPtr<GFXResource>> deferredDeleteResource;
	vstd::vector<ObjectPtr<VObject>> deferredDeleteObj;
	vstd::vector<Runnable<void()>> afterSyncTask;
	vstd::vector<CBufferChunk> deferredReleaseCBuffer;
	FrameResource(GFXDevice* device, GFXCommandListType type, CBufferAllocator* cbAlloc);
	CBufferChunk AllocateCBuffer(size_t sz);
	void ReleaseTemp();
	~FrameResource();
};
}// namespace luisa::compute

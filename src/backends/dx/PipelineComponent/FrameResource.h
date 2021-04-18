#pragma once
#include <Common/Common.h>
#include <Common/VObject.h>
#include <PipelineComponent/ThreadCommand.h>
#include <RenderComponent/CBufferAllocator.h>
namespace luisa::compute {
class FrameResource final {
public:
	CBufferAllocator* cbAlloc;
	ThreadCommand tCmd;
	uint64 signalIndex;
	vengine::vector<ObjectPtr<VObject>> deferredDeleteObj;
	vengine::vector<Runnable<void()>> afterSyncTask;
	vengine::vector<CBufferChunk> deferredReleaseCBuffer;
	FrameResource(GFXDevice* device, GFXCommandListType type, CBufferAllocator* cbAlloc);
	CBufferChunk AllocateCBuffer(size_t sz);
	void ReleaseTemp();
	~FrameResource();
};
}// namespace luisa::compute

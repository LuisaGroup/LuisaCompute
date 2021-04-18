#include <PipelineComponent/FrameResource.h>
namespace luisa::compute {
FrameResource::FrameResource(GFXDevice* device, GFXCommandListType type, CBufferAllocator* cbAlloc) : tCmd(device, type), signalIndex(0), cbAlloc(cbAlloc) {}
CBufferChunk FrameResource::AllocateCBuffer(size_t sz) {
	auto chunk = cbAlloc->Allocate(sz);
	deferredReleaseCBuffer.push_back(chunk);
	return chunk;
}
void FrameResource::ReleaseTemp() {
	deferredDeleteObj.clear();
	for (auto& i : afterSyncTask) {
		i();
	}
	afterSyncTask.clear();
	for (auto& i : deferredReleaseCBuffer) {
		cbAlloc->Release(i);
	}
	deferredReleaseCBuffer.clear();
}
FrameResource::~FrameResource() {
}
}// namespace luisa::compute
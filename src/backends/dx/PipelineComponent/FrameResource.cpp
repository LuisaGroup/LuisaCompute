#include <PipelineComponent/FrameResource.h>
namespace luisa::compute {
FrameResource::FrameResource(GFXDevice* device, GFXCommandListType type) : tCmd(device, type), signalIndex(0) {}
void FrameResource::ReleaseTemp() {
	deferredDeleteObj.clear();
	for (auto& i : afterSyncTask) {
		i();
	}
	afterSyncTask.clear();
}
FrameResource::~FrameResource() {
}
}// namespace luisa::compute
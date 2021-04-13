#pragma once
#include <Common/Common.h>
#include <Common/VObject.h>
#include <PipelineComponent/ThreadCommand.h>
namespace luisa::compute {
class FrameResource final {
public:
	ThreadCommand tCmd;
	uint64 signalIndex;
	vengine::vector<ObjectPtr<VObject>> deferredDeleteObj;
	vengine::vector<Runnable<void()>> afterSyncTask;
	FrameResource(GFXDevice* device, GFXCommandListType type);
	void ReleaseTemp();
	~FrameResource();
};
}// namespace luisa::compute

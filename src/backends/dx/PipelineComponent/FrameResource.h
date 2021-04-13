#pragma once
#include <Common/Common.h>
#include <Common/VObject.h>
#include <PipelineComponent/ThreadCommand.h>
namespace luisa::compute {
class FrameResource final{
public:
	ThreadCommand tCmd;
	uint64 signalIndex;
	vengine::vector<ObjectPtr<VObject>> deferredDeleteObj;
	FrameResource(GFXDevice* device, GFXCommandListType type);
	void ReleaseTemp();
	~FrameResource();
};
}// namespace luisa::compute

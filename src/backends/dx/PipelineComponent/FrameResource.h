#pragma once
#include <Common/Common.h>
#include <Common/VObject.h>
#include <PipelineComponent/ThreadCommand.h>
namespace luisa::compute {
class FrameResource : public VObject {
public:
	ThreadCommand tCmd;
	uint64 signalIndex;
	FrameResource(GFXDevice* device, GFXCommandListType type) : tCmd(device, type), signalIndex(0) {}
};
}// namespace luisa::compute

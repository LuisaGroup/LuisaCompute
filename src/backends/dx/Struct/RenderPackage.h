#pragma once
#include <Common/GFXUtil.h>
namespace luisa::compute {
class FrameResource;
}
class ThreadCommand;
struct RenderPackage {
	GFXDevice* device;
	ThreadCommand* tCmd;
	luisa::compute::FrameResource* frameRes;
	constexpr RenderPackage(
		GFXDevice* device,
		ThreadCommand* tCmd,
		luisa::compute::FrameResource* frameRes)
		: device(device),
		  frameRes(frameRes),
		  tCmd(tCmd) {}

	constexpr RenderPackage()
		: device(nullptr),
		  tCmd(nullptr){
	}
	constexpr bool operator==(const RenderPackage& p) const {
		return device == p.device && tCmd == p.tCmd;
	}
	constexpr bool operator!=(const RenderPackage& p) const {
		return !operator==(p);
	}
};
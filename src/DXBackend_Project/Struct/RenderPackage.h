#pragma once
#include "../Common/GFXUtil.h"
class ThreadCommand;
struct RenderPackage {
	GFXDevice* device;
	ThreadCommand* tCmd;
	constexpr RenderPackage(
		GFXDevice* device,
		ThreadCommand* tCmd)
		: device(device),
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
#pragma once
#include <Common/GFXUtil.h>
#include <vstl/vstlconfig.h>
class IGPUResourceState {
public:
	virtual GFXResourceState GetGFXResourceState(GPUResourceState gfxState) const {
		return (D3D12_RESOURCE_STATES)gfxState;
	}
};

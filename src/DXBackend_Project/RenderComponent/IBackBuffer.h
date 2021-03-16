#pragma once
#include "../Common/GFXUtil.h"
class GPUResourceBase;
class IBackBuffer {
public:
	virtual D3D12_CPU_DESCRIPTOR_HANDLE GetRTVHandle() const = 0;
	virtual GPUResourceBase const* GetBackBufferGPUResource() const = 0;
};
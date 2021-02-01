#pragma once
#include "GPUResourceBase.h"
class IBuffer : public GPUResourceBase {
public:
	virtual uint64 GetByteSize() const = 0;
	GpuAddress GetBufferAddress() const {
		return {Resource->GetGPUVirtualAddress()};
	}
};
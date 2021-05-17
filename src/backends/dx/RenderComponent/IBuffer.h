#pragma once
#include <RenderComponent/GPUResourceBase.h>
class IBuffer : public GPUResourceBase {
public:
	IBuffer(GFXDevice* device, IGPUAllocator* alloc) : GPUResourceBase(device, GPUResourceType::Buffer, alloc) {}
	virtual uint64 GetByteSize() const = 0;
	virtual ~IBuffer() noexcept = default;
	GpuAddress GetBufferAddress() const {
		return {Resource->GetGPUVirtualAddress()};
	}
};
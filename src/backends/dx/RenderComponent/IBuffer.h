#pragma once
#include <RenderComponent/GPUResourceBase.h>
class IBuffer : public GPUResourceBase {
public:
	IBuffer() : GPUResourceBase(GPUResourceType::Buffer) {}
	virtual uint64 GetByteSize() const = 0;
	virtual ~IBuffer() noexcept = default;
	GpuAddress GetBufferAddress() const {
		return {Resource->GetGPUVirtualAddress()};
	}
};
#pragma once
#include "IGpuResource.h"
namespace lc_rhi {
class ICommandBuffer;
class IBuffer : public IGpuResource {
protected:
	size_t stride;
	size_t elementCount;
	IBuffer(
		size_t stride,
		size_t elementCount)
		: stride(stride),
		  elementCount(elementCount) {
	}

public:
	size_t GetBufferByteLength() const {
		return stride * elementCount;
	}
	size_t GetStride() const {
		return stride;
	}
	size_t GetElementCount() const {
		return elementCount;
	}
	virtual void SetData(ICommandBuffer* commandBuffer, void const* data) const = 0;
	virtual void SetData(ICommandBuffer* commandBuffer, void const* data, size_t startElement, size_t elementCount) const = 0;
	virtual void GetData(ICommandBuffer* commandBuffer, void* data) const = 0;
	virtual void GetData(ICommandBuffer* commandBuffer, void* data, size_t startElement, size_t elementCount) const = 0;
	virtual ~IBuffer() {}
};
}// namespace lc_rhi
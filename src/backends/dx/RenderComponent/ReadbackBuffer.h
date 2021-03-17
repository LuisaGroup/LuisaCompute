#pragma once

#include "IBuffer.h"
class ReadbackBuffer final : public IBuffer {
public:
	ReadbackBuffer(GFXDevice* device, uint elementCount, size_t stride);
	ReadbackBuffer() : mMappedData(0),
					   mStride(0),
					   mElementCount(0) {}
	void Create(GFXDevice* device, uint elementCount, size_t stride);
	virtual ~ReadbackBuffer() {
		if (Resource != nullptr && mMappedData != nullptr)
			Resource->Unmap(0, nullptr);
	}

	inline GpuAddress GetAddress(uint elementCount) const {
		return {Resource->GetGPUVirtualAddress() + elementCount * mStride};
	}
	constexpr size_t GetStride() const { return mStride; }

	inline uint GetElementCount() const {
		return mElementCount;
	}
	inline void* GetMappedPtr(uint index) const {
		size_t sz = (size_t)mMappedData;
		return (void*)(sz + mStride * index);
	}
	ReadbackBuffer(const ReadbackBuffer& another) = delete;
	void Map();
	void UnMap();
	virtual GFXResourceState GetInitState() const {
		return GFXResourceState_CopyDest;
	}
	uint64 GetByteSize() const override {
		return mElementCount * mStride;
	}

private:
	void* mMappedData = nullptr;
	size_t mStride;
	uint mElementCount;
};
#pragma once

#include "IBuffer.h"
class VENGINE_DLL_RENDERER ReadbackBuffer final : public IBuffer {
public:
	ReadbackBuffer(GFXDevice* device, uint64 elementCount, size_t stride);
	ReadbackBuffer() : mMappedData(0),
					   mStride(0),
					   mElementCount(0) {}
	void Create(GFXDevice* device, uint64 elementCount, size_t stride);
	virtual ~ReadbackBuffer() {
		if (Resource != nullptr && mMappedData != nullptr)
			Resource->Unmap(0, nullptr);
	}

	inline GpuAddress GetAddress(uint64 elementCount) const {
		return {Resource->GetGPUVirtualAddress() + elementCount * mStride};
	}
	constexpr size_t GetStride() const { return mStride; }

	inline uint64 GetElementCount() const {
		return mElementCount;
	}
	inline void const* GetMappedPtr(uint64 index) const {
		size_t sz = (size_t)mMappedData;
		return (void*)(sz + mStride * index);
	}
	ReadbackBuffer(const ReadbackBuffer& another) = delete;
	void Map();
	void UnMap();
	virtual GPUResourceState GetInitState() const {
		return GPUResourceState_CopyDest;
	}
	uint64 GetByteSize() const override {
		return mElementCount * mStride;
	}

private:
	void const* mMappedData = nullptr;
	size_t mStride;
	uint64 mElementCount;
};
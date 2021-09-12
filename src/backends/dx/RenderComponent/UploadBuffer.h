#pragma once

#include <RenderComponent/IBuffer.h>
class IBufferAllocator;
class DescriptorHeap;
class VENGINE_DLL_RENDERER UploadBuffer final : public IBuffer {
public:
	UploadBuffer(GFXDevice* device, uint64 elementCount, bool isConstantBuffer, uint64_t stride, IBufferAllocator* allocator = nullptr);
	~UploadBuffer();
	void CopyData(uint64 elementIndex, const void* data) const;
	void CopyData(uint64 elementIndex, const void* data, uint64 byteSize) const;
	void CopyData(uint64 elementIndex, uint64 bufferByteOffset, const void* data, uint64 byteSize) const;
	void CopyDatas(uint64 startElementIndex, uint64 elementCount, const void* data) const;
	GpuAddress GetAddress(uint64 elementCount) const;
	uint64_t GetAddressOffset(uint64 elementCount) const;
	void* GetMappedDataPtr(uint64 element) const;
	uint64_t GetStride() const { return mStride; }
	uint64_t GetAlignedStride() const { return mElementByteSize; }
	uint64 GetByteSize() const override { return mElementByteSize * mElementCount; }
	uint64 GetElementCount() const {
		return mElementCount;
	}
	VSTL_DELETE_COPY_CONSTRUCT(UploadBuffer)
	virtual GPUResourceState GetInitState() const {
		return GPUResourceState_GenericRead;
	}
	void BindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	uint GetSRVDescIndex(GFXDevice* device) const;

private:
	uint64_t mStride;
	void* mMappedData;
	IBufferAllocator* allocator;
	uint64 mElementCount;
	uint64 mElementByteSize;
	mutable luisa::spin_mutex srvDescLock;
	mutable uint srvDescIndex = -1;
	bool mIsConstantBuffer;
	void InitGlobalDesc(GFXDevice* device) const;
	void ReturnGlobalDesc();
};
#pragma once
#include <Resource/Buffer.h>
#include <Resource/AllocHandle.h>
namespace toolhub::directx {
class UploadBuffer final : public Buffer {
private:
	AllocHandle allocHandle;
	uint64 byteSize;

public:
	ID3D12Resource* GetResource() const override { return allocHandle.resource.Get(); }
	D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const override { return allocHandle.resource->GetGPUVirtualAddress(); }
	uint64 GetByteSize() const override { return byteSize; }
	UploadBuffer(
		Device* device,
		uint64 byteSize,
		IGpuAllocator* allocator = nullptr);
	~UploadBuffer();
	void CopyData(uint64 offset, vstd::span<vbyte const> data) const;
	D3D12_RESOURCE_STATES GetInitState() const override {
		return D3D12_RESOURCE_STATE_GENERIC_READ;
	}
	Tag GetTag() const override {
		return Tag::UploadBuffer;
	}
	UploadBuffer(UploadBuffer&&) = default;
	KILL_COPY_CONSTRUCT(UploadBuffer)
	VSTD_SELF_PTR
};

}// namespace toolhub::directx
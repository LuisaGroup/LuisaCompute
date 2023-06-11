#pragma once
#include <Resource/Buffer.h>
#include <Resource/AllocHandle.h>

namespace lc::dx {
class ReadbackBuffer final : public Buffer {
private:
	AllocHandle allocHandle;
	uint64 byteSize;

public:
	ID3D12Resource* GetResource() const override { return allocHandle.resource.Get(); }
	D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const override { return allocHandle.resource->GetGPUVirtualAddress(); }
	uint64 GetByteSize() const override { return byteSize; }
	ReadbackBuffer(
		Device* device,
		uint64 byteSize,
		GpuAllocator* allocator = nullptr);
	~ReadbackBuffer();
	void CopyData(uint64 offset, vstd::span<uint8_t> data) const;
	D3D12_RESOURCE_STATES GetInitState() const override {
		return D3D12_RESOURCE_STATE_COPY_DEST;
	}
	Tag GetTag() const override {
		return Tag::ReadbackBuffer;
	}
	ReadbackBuffer(ReadbackBuffer&&) = default;
	KILL_COPY_CONSTRUCT(ReadbackBuffer)
};
}// namespace lc::dx

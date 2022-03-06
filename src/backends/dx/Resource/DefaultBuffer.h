#pragma once
#include <Resource/Buffer.h>
#include <Resource/AllocHandle.h>
namespace toolhub::directx {
class DefaultBuffer final : public Buffer {
private:
	AllocHandle allocHandle;
	uint64 byteSize;
	D3D12_RESOURCE_STATES initState;

public:
	vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(bool isRaw = false) const override;
	vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(bool isRaw = false) const override;
	vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
	vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
	
	ID3D12Resource* GetResource() const override { return allocHandle.resource.Get(); }
	D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const override { return allocHandle.resource->GetGPUVirtualAddress(); }
	uint64 GetByteSize() const override { return byteSize; }
	DefaultBuffer(
		Device* device,
		uint64 byteSize,
		IGpuAllocator* allocator = nullptr,
		D3D12_RESOURCE_STATES initState = VEngineShaderResourceState);
	~DefaultBuffer();
	D3D12_RESOURCE_STATES GetInitState() const override {
		return initState;
	}
	Tag GetTag() const override {
		return Tag::DefaultBuffer;
	}
	DefaultBuffer(DefaultBuffer&&) = default;
	KILL_COPY_CONSTRUCT(DefaultBuffer)
	VSTD_SELF_PTR
};
}// namespace toolhub::directx
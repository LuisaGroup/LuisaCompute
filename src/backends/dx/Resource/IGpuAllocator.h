#pragma once
#include <DXRuntime/Device.h>
namespace toolhub::directx {
class IGpuAllocator : public vstd::ISelfPtr {
public:
	enum class Tag : vbyte {
		None,
		DefaultAllocator
	};
	static IGpuAllocator* CreateAllocator(
		Device* device,
		Tag tag);
	virtual uint64 AllocateBufferHeap(
		Device* device,
		uint64_t targetSizeInBytes,
		D3D12_HEAP_TYPE heapType,
		ID3D12Heap** heap, uint64_t* offset) = 0;
	virtual uint64 AllocateTextureHeap(
		Device* device,
		GFXFormat format,
		uint32_t width,
		uint32_t height,
		uint32_t depthSlice,
		TextureDimension dimension,
		uint32_t mipCount,
		ID3D12Heap** heap, uint64_t* offset,
		bool isRenderTexture) = 0;
	virtual void Release(uint64 handle) = 0;
	virtual ~IGpuAllocator() = default;
};
}// namespace toolhub::directx
#pragma once
#include <RenderComponent/Utility/ITextureAllocator.h>
namespace D3D12MA {
class Allocator;
class Allocation;
}// namespace D3D12MA
class VENGINE_DLL_RENDERER DefaultTextureAllocator final : public ITextureAllocator {
private:
	HashMap<uint64, D3D12MA::Allocation*> allocatedTexs;
	std::mutex mtx;
	D3D12MA::Allocator* allocator = nullptr;

public:
	~DefaultTextureAllocator();
	void AllocateTextureHeap(
		GFXDevice* device,
		GFXFormat format,
		uint32_t width,
		uint32_t height,
		uint32_t depthSlice,
		TextureDimension dimension,
		uint32_t mipCount,
		ID3D12Heap** heap, uint64_t* offset,
		bool isRenderTexture,
		uint64 instanceID) override;
	void Release(uint64 instanceID) override;
	DefaultTextureAllocator(
		GFXDevice* device);
};
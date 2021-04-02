#pragma once
#include "ITextureAllocator.h"
namespace D3D12MA
{
	class Allocator;
	class Allocation;
}
class VENGINE_DLL_RENDERER DefaultTextureAllocator final : public ITextureAllocator
{
private:
	HashMap<uint64, D3D12MA::Allocation*> allocatedTexs;
	std::mutex mtx;
	D3D12MA::Allocator* allocator = nullptr;
public:
	~DefaultTextureAllocator();
	virtual void AllocateTextureHeap(
		GFXDevice* device,
		GFXFormat format,
		uint32_t width,
		uint32_t height,
		uint32_t depthSlice,
		TextureDimension dimension,
		uint32_t mipCount,
		ID3D12Heap** heap, uint64_t* offset,
		bool isRenderTexture,
		TextureBase* currentPtr);
	virtual void ReturnTexture(TextureBase* tex);
	DefaultTextureAllocator(
		GFXDevice* device,
		IDXGIAdapter* adapter);
};
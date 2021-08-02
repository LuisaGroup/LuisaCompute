#pragma once
#include <Common/GFXUtil.h>
#include <core/vstl/VObject.h>
class VENGINE_DLL_RENDERER TextureHeap
{
private:
	Microsoft::WRL::ComPtr<ID3D12Heap> heap;
	uint64_t chunkSize;
public:
	TextureHeap() : heap(nullptr), chunkSize(0) {}
	TextureHeap(TextureHeap const&) = delete;
	TextureHeap(TextureHeap&&) = delete;
	uint64_t GetChunkSize() const { return chunkSize; }
	TextureHeap(GFXDevice* device, uint64_t chunkSize, bool isRenderTexture);
	void Create(GFXDevice* device, uint64_t chunkSize, bool isRenderTexture);
	ID3D12Heap* GetHeap() const
	{
		return heap.Get();
	}
};
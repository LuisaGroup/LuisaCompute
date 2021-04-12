#pragma once
#include <Common/GFXUtil.h>
#include <Common/MetaLib.h>
#include <RenderComponent/TextureBase.h>
class ITextureAllocator
{
public:
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
		TextureBase* currentPtr) = 0;
	virtual void ReturnTexture(TextureBase* tex) = 0;
	virtual ~ITextureAllocator() {}
};
#pragma once
#include <Common/GFXUtil.h>
#include <vstl/MetaLib.h>
#include <RenderComponent/TextureBase.h>
#include <RenderComponent/Utility/IGPUAllocator.h>
class ITextureAllocator : public IGPUAllocator {
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
		uint64 instanceID) = 0;
	virtual ~ITextureAllocator() {}
};

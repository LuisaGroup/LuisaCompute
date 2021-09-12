#pragma once
#include <util/vstlconfig.h>
#include <RenderComponent/GPUResourceBase.h>
enum class TextureDimension : uint {
	Tex2D = 0,
	Tex3D = 1,
	Cubemap = 2,
	Tex2DArray = 3,
	Num = 4
};
class DescriptorHeap;
class VENGINE_DLL_RENDERER TextureBase : public GPUResourceBase {
protected:
	uint64_t resourceSize = 0;

private:
	uint srvDescID = 0;

protected:
	GFXFormat mFormat;
	uint depthSlice;
	uint mWidth = 0;
	uint mHeight = 0;
	uint mipCount = 1;
	TextureDimension dimension;
	TextureBase(GFXDevice* device, IGPUAllocator* alloc);
#if VENGINE_PLATFORM_DIRECTX_12 == 1
	static constexpr D3D12_RESOURCE_STATES GENERIC_READ_STATE =
		(D3D12_RESOURCE_STATES)(
			(uint)D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
			| (uint)D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
			| (uint)D3D12_RESOURCE_STATE_COPY_SOURCE);
	static constexpr D3D12_RESOURCE_STATES PIXEL_READ_GENERIC_READ_STATE =
		(D3D12_RESOURCE_STATES)(
			(uint)D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
			| (uint)D3D12_RESOURCE_STATE_COPY_SOURCE);
#endif

public:
	GFXResourceState GetGFXResourceState(GPUResourceState gfxState) const override;
	virtual ~TextureBase();
	uint64_t GetResourceSize() const {
		return resourceSize;
	}

	TextureDimension GetDimension() const {
		return dimension;
	}
	GFXFormat GetFormat() const {
		return mFormat;
	}
	uint GetDepthSlice() const {
		return depthSlice;
	}
	uint GetWidth() const {
		return mWidth;
	}
	uint GetHeight() const {
		return mHeight;
	}
	uint GetMipCount() const {
		return mipCount;
	}
	uint GetGlobalDescIndex() const {
		return srvDescID;
	}
	virtual void BindSRVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device) const = 0;
	virtual uint GetGlobalUAVDescIndex(uint mipLevel) const {
		vstl_log("GetGlobalUAVDescIndex Not Implemented!"_sv);
		VSTL_ABORT();
	}
};

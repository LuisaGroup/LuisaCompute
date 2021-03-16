#pragma once
#include "GPUResourceBase.h"
enum class TextureDimension : uint
{
	Tex2D = 0,
	Tex3D = 1,
	Cubemap = 2,
	Tex2DArray = 3,
	Num = 4
};
class DescriptorHeap;
class TextureBase : public GPUResourceBase
{
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
	TextureBase();
public:
	virtual ~TextureBase();
	uint64_t GetResourceSize() const
	{
		return resourceSize;
	}

	TextureDimension GetDimension() const
	{
		return dimension;
	}
	GFXFormat GetFormat() const
	{
		return mFormat;
	}
	uint GetDepthSlice() const
	{
		return depthSlice;
	}
	uint GetWidth() const
	{
		return mWidth;
	}
	uint GetHeight() const
	{
		return mHeight;
	}
	uint GetMipCount() const
	{
		return mipCount;
	}
	uint GetGlobalDescIndex() const
	{
		return srvDescID;
	}
	virtual void BindSRVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device) const = 0;
};
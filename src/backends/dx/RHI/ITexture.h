#pragma once
#include <RHI/IGpuResource.h>
namespace lc_rhi {
enum class TextureDimension : uint {
	Tex2D,
	Tex3D
};
enum class TextureFormat : uint {
	RGBA_Float,
	RG_Float,
	R_Float,
	RGBA_UInt,
	RG_UInt,
	R_UInt,
	RGBA_SInt,
	RG_SInt,
	R_SInt,
	RGBA_UNorm,
	RG_UNorm,
	R_UNorm,
	RGBA_SNorm,
	RG_SNorm,
	R_SNorm
};
class ITexture : public IGpuResource{
protected:
	uint width;
	uint height;
	uint volumeDepth;
	uint mipCount;
	TextureDimension dimension;
	TextureFormat format;
	ITexture(
		uint width,
		uint height,
		uint volumeDepth,
		uint mipCount,
		TextureDimension dimension,
		TextureFormat format)
		: width(width),
		  mipCount(mipCount),
		  height(height),
		  volumeDepth(volumeDepth),
		  dimension(dimension),
		  format(format) {
	}

public:
	virtual ~ITexture() {}
	uint GetWidth() const { return width; }
	uint GetHeight() const { return height; }
	uint GetVolumeDepth() const { return volumeDepth; }
	uint GetMipCount() const { return mipCount; }
	TextureFormat GetFormat() const { return format; }
	TextureDimension GetDimension() const { return dimension; }
};
}// namespace lc_rhi
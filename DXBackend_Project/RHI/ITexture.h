#pragma once
#include <stdint.h>
namespace lc_rhi {
enum class TextureDimension : uint32_t {
	Tex2D,
	Tex3D
};
enum class TextureFormat : uint32_t {
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
class ITexture {
protected:
	uint32_t width;
	uint32_t height;
	uint32_t volumeDepth;
	uint32_t mipCount;
	TextureDimension dimension;
	TextureFormat format;
	ITexture(
		uint32_t width,
		uint32_t height,
		uint32_t volumeDepth,
		uint32_t mipCount,
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
	uint32_t GetWidth() const { return width; }
	uint32_t GetHeight() const { return height; }
	uint32_t GetVolumeDepth() const { return volumeDepth; }
	uint32_t GetMipCount() const { return mipCount; }
	uint32_t GetMipCount() const { return mipCount; }
	TextureFormat GetFormat() const { return format; }
	TextureDimension GetDimension() const { return dimension; }
};
}// namespace lc_rhi
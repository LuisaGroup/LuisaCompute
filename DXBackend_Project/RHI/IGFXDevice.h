#pragma once
#include <string_view>
#include "ITexture.h"
namespace lc_rhi {
class IRayTracingShader;
class IComputeShader;
class IRenderTexture;
class ILoadedTexture;
class IBuffer;
class IGFXDevice {
public:
	virtual IComputeShader const* GetComputeShader(std::string_view name) const = 0;
	virtual IRayTracingShader const* GetRayTracingShader(std::string_view name) const = 0;
	virtual ILoadedTexture const* LoadTexture(std::string_view path) const = 0;
	virtual IRenderTexture const* CreateTexture2D(
		uint32_t width,
		uint32_t height,
		TextureFormat format) const = 0;
	virtual IRenderTexture const* CreateTexture3D(
		uint32_t width,
		uint32_t height,
		uint32_t volumeDepth,
		TextureFormat format) const = 0;
	virtual IBuffer const* CreateBuffer(
		size_t elementCount,
		size_t stride) const = 0;
	virtual ~IGFXDevice() {}
};
}// namespace lc_rhi
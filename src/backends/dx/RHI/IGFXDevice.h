#pragma once
#include <string_view>
#include <RHI/ITexture.h>
#include <RHI/IMesh.h>
namespace lc_rhi {
class IRayTracingShader;
class IComputeShader;
class IRenderTexture;
class ILoadedTexture;
class ICommandBuffer;
class IBuffer;
class IGFXDevice {
public:
	virtual IComputeShader const* GetComputeShader(std::string_view name) const = 0;
	virtual IRayTracingShader const* GetRayTracingShader(std::string_view name) const = 0;
	virtual ILoadedTexture* LoadTexture(
		ICommandBuffer* commandBuffer,
		std::string_view path) const = 0;
	virtual IMesh* LoadMesh(
		ICommandBuffer* commandBuffer,
		std::string_view path) const = 0;
	virtual IMesh* CreateMesh(
		ICommandBuffer* commandBuffer,
		uint vertexCount,
		uint vertexStride,
		uint const* indices,
		uint indexCount,
		SubMesh const* subMeshes,
		uint subMeshCount
	) const = 0;
	virtual IRenderTexture* CreateTexture2D(
		ICommandBuffer* commandBuffer,
		uint width,
		uint height,
		uint mipCount,
		TextureFormat format,
		float4 defaultColor = float4(0, 0, 0, 0)) const = 0;
	virtual IRenderTexture* CreateTexture3D(
		ICommandBuffer* commandBuffer,
		uint width,
		uint height,
		uint volumeDepth,
		uint mipCount,
		TextureFormat format,
		float4 defaultColor = float4(0, 0, 0, 0)) const = 0;
	virtual IBuffer* CreateBuffer(
		ICommandBuffer* commandBuffer,
		size_t elementCount,
		size_t stride) const = 0;
	virtual void SubmitCommand(ICommandBuffer* commandBuffer) = 0;
	virtual ~IGFXDevice() {}
};
}// namespace lc_rhi
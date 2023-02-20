#pragma once
#include <DXRuntime/Device.h>
namespace toolhub::directx {
class Resource : public vstd::ISelfPtr {
public:
	enum class Tag : uint8_t {
		None,
		UploadBuffer,
		ReadbackBuffer,
		DefaultBuffer,
		RenderTexture,
		DescriptorHeap,
		BindlessArray,
		Mesh,
		SwapChain,
		DepthBuffer,
		ExternalBuffer,
		ExternalTexture
	};

protected:
	Device* device;

public:
	static uint64 GetTextureSize(
		Device* device,
		uint width,
		uint height,
		GFXFormat Format,
		TextureDimension type,
		uint depthCount,
		uint mipCount);
	static uint64 GetTexturePixelSize(
		GFXFormat format);
    static bool IsBCtex(GFXFormat format);
	Device* GetDevice() const { return device; }
	Resource(Device* device)
		: device(device) {}
	Resource(Resource&&) = default;
	Resource(Resource const&) = delete;
	virtual Tag GetTag() const = 0;
	virtual ~Resource() = default;
	virtual ID3D12Resource* GetResource() const { return nullptr; }
	virtual D3D12_RESOURCE_STATES GetInitState() const { return D3D12_RESOURCE_STATE_COMMON; }
};
}// namespace toolhub::directx
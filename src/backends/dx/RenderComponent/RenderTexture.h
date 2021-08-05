#pragma once
#include <util/VObject.h>
#include <Common/GFXUtil.h>
#include <RenderComponent/TextureBase.h>
#include <RenderComponent/DescriptorHeap.h>
#include <RenderComponent/IBackBuffer.h>
class TextureHeap;
class ITextureAllocator;
class ThreadCommand;
enum CubeMapFace {
	CubeMapFace_PositiveX = 0,
	CubeMapFace_NegativeX = 1,
	CubeMapFace_PositiveY = 2,
	CubeMapFace_NegativeY = 3,
	CubeMapFace_PositiveZ = 4,
	CubeMapFace_NegativeZ = 5
};
enum RenderTextureDepthSettings {
	RenderTextureDepthSettings_None,
	RenderTextureDepthSettings_Depth16,
	RenderTextureDepthSettings_Depth32,
	RenderTextureDepthSettings_DepthStencil
};

enum class RenderTextureUsage : bool {
	ColorBuffer = false,
	DepthBuffer = true
};

enum class RenderTextureState : UCHAR {
	Render_Target = 0,
	Unordered_Access = 1,
	Generic_Read = 2,
	Common = 3,
	Default = 4,
	Non_Pixel_SRV = 5
};

struct RenderTextureFormat {
	RenderTextureUsage usage;
	union {
		GFXFormat colorFormat;
		RenderTextureDepthSettings depthFormat;
	};
	static RenderTextureFormat GetColorFormat(GFXFormat format) {
		RenderTextureFormat f;
		f.usage = RenderTextureUsage::ColorBuffer;
		f.colorFormat = format;
		return f;
	}

	static RenderTextureFormat GetDepthFormat(RenderTextureDepthSettings depthFormat) {
		RenderTextureFormat f;
		f.usage = RenderTextureUsage::DepthBuffer;
		f.depthFormat = depthFormat;
		return f;
	}
};

struct RenderTextureDescriptor {
	uint width;
	uint height;
	uint depthSlice;
	TextureDimension type = TextureDimension::Tex2D;
	RenderTextureFormat rtFormat;
	RenderTextureState initState = RenderTextureState::Default;
	constexpr bool operator==(const RenderTextureDescriptor& other) const {
		bool value = width == other.width && height == other.height && depthSlice == other.depthSlice && type == other.type && rtFormat.usage == other.rtFormat.usage && initState == other.initState;
		if (value) {
			if (rtFormat.usage == RenderTextureUsage::ColorBuffer) {
				return rtFormat.colorFormat == other.rtFormat.colorFormat;
			} else {
				return rtFormat.depthFormat == other.rtFormat.depthFormat;
			}
		}
		return false;
	}

	constexpr bool operator!=(const RenderTextureDescriptor& other) const {
		return !operator==(other);
	}
};
namespace vstd {
template<>
struct hash<RenderTextureDescriptor> {
public:
	size_t operator()(const RenderTextureDescriptor& o) const {
		hash<uint> hashFunc;
		return hashFunc(
			(uint)o.initState ^ (uint)o.type ^ o.width ^ o.height ^ o.depthSlice ^ (o.rtFormat.usage == RenderTextureUsage::ColorBuffer ? (uint)o.rtFormat.colorFormat : (uint)o.rtFormat.depthFormat));
	}
};
}// namespace vstd

class VENGINE_DLL_RENDERER RenderTexture final
	: public TextureBase,
	  public IBackBuffer {
private:
	RenderTextureUsage usage;
	GPUResourceState initState;
	DescriptorHeap rtvHeap;
	ArrayList<uint> uavDescIndices;
	float clearColor;
	void GetColorViewDesc(D3D12_SHADER_RESOURCE_VIEW_DESC& srvDesc) const;
	void GetColorUAVDesc(D3D12_UNORDERED_ACCESS_VIEW_DESC& uavDesc, uint targetMipLevel) const;
	GFXResourceState GetGFXResourceState(GPUResourceState gfxState) const override;

public:
	GPUResourceBase const* GetBackBufferGPUResource() const override {
		return this;
	};

	KILL_COPY_CONSTRUCT(RenderTexture)
	uint GetGlobalUAVDescIndex(uint mipLevel) const override;
	static uint64_t GetSizeFromProperty(
		GFXDevice* device,
		uint width,
		uint height,
		RenderTextureFormat rtFormat,
		TextureDimension type,
		uint depthCount,
		uint mipCount,
		RenderTextureState initState);
	RenderTexture(
		GFXDevice* device,
		uint width,
		uint height,
		RenderTextureFormat rtFormat,
		TextureDimension type,
		uint depthCount,
		uint mipCount,
		RenderTextureState initState = RenderTextureState::Default,
		TextureHeap* targetHeap = nullptr,
		uint64_t placedOffset = 0,
		float clearColor = 0);
	RenderTexture(
		GFXDevice* device,
		ITextureAllocator* allocator,
		uint width,
		uint height,
		RenderTextureFormat rtFormat,
		TextureDimension type,
		uint depthCount,
		uint mipCount,
		RenderTextureState initState = RenderTextureState::Default,
		float clearColor = 0);
	~RenderTexture();
	virtual GPUResourceState GetInitState() const {
		return initState;
	}
	RenderTextureUsage GetUsage() const { return usage; }
	void BindRTVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device, uint slice, uint mip) const;
	void SetViewport(ThreadCommand* commandList, uint mipCount = 0) const;
	D3D12_CPU_DESCRIPTOR_HANDLE GetColorDescriptor(uint slice, uint mip) const;
	virtual void BindSRVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device) const;
	void BindUAVToHeap(DescriptorHeap* targetHeap, uint index, GFXDevice* device, uint targetMipLevel) const;
	void ClearRenderTarget(ThreadCommand* commandList, uint slice, uint mip) const;
	GFXFormat GetBackBufferFormat() const {
		return GetFormat();
	}
	D3D12_CPU_DESCRIPTOR_HANDLE GetRTVHandle() const override {
		return GetColorDescriptor(0, 0);
	}
};

#pragma once
#include <Resource/Resource.h>
#include <Resource/GpuAllocator.h>
#include <runtime/rhi/pixel.h>
#include <core/logging.h>
using namespace luisa::compute;
namespace toolhub::directx {
class TextureBase : public Resource {
protected:
    uint width;
    uint height;
    GFXFormat format;
    TextureDimension dimension;
    uint depth;
    uint mip;
    //	vstd::unique_ptr<std::atomic<D3D12_BARRIER_LAYOUT>> layouts;
protected:
    D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDescBase(uint targetMipLevel) const;
    D3D12_RENDER_TARGET_VIEW_DESC GetRenderTargetDescBase(uint mipOffset) const;
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDescBase(uint mipOffset) const;

public:
    //	vstd::span<std::atomic<D3D12_BARRIER_LAYOUT>> Layouts() const;
    static GFXFormat ToGFXFormat(PixelFormat format);
    uint Width() const { return width; }
    uint Height() const { return height; }
    GFXFormat Format() const { return format; }
    TextureDimension Dimension() const { return dimension; }
    uint Depth() const { return depth; }
    uint Mip() const { return mip; }
    virtual uint GetGlobalSRVIndex(uint mipOffset = 0) const = 0;
    virtual uint GetGlobalUAVIndex(uint mipLevel) const = 0;
    virtual D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc(uint mipOffset = 0) const = 0;
    virtual D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDesc(uint targetMipLevel) const {
        LUISA_ERROR("Texture type not support random write!");
        return {};
    }
    virtual D3D12_DEPTH_STENCIL_VIEW_DESC GetDepthDesc() const {
        LUISA_ERROR("Texture type not support depth!");
        return {};
    }
    virtual D3D12_RENDER_TARGET_VIEW_DESC GetRenderTargetDesc(uint mipOffset = 0) const {
        LUISA_ERROR("Texture type not support render target!");
        return {};
    }
    TextureBase(
        Device *device,
        uint width,
        uint height,
        GFXFormat format,
        TextureDimension dimension,
        uint depth,
        uint mip);
    virtual ~TextureBase();
    TextureBase(TextureBase &&) = default;
    KILL_COPY_CONSTRUCT(TextureBase)
};
}// namespace toolhub::directx
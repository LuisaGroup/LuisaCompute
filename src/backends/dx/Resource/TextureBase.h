#pragma once
#include <Resource/Resource.h>
#include <Resource/GpuAllocator.h>
#include <luisa/runtime/rhi/pixel.h>
namespace lc::dx {
using namespace luisa::compute;
class TextureBase : public Resource {
protected:
    uint width;
    uint height;
    GFXFormat format;
    TextureDimension dimension;
    uint depth;
    uint mip;
    mutable std::atomic<D3D12_RESOURCE_STATES> initState;
    //	vstd::unique_ptr<std::atomic<D3D12_BARRIER_LAYOUT>> layouts;
    D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDescBase(uint targetMipLevel) const;
    D3D12_RENDER_TARGET_VIEW_DESC GetRenderTargetDescBase(uint mipOffset) const;
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDescBase(uint mipOffset) const;
    uint GetGlobalSRVIndexBase(uint mipOffset, std::mutex &allocMtx, vstd::unordered_map<uint, uint> &srvIdcs) const;
    uint GetGlobalUAVIndexBase(uint mipLevel, std::mutex &allocMtx, vstd::unordered_map<uint, uint> &uavIdcs) const;
    D3D12_RESOURCE_DESC GetResourceDescBase(bool allowUav, bool allowSimul, bool allowRaster, bool reserved) const;
    D3D12_RESOURCE_DESC GetResourceDescBase(uint3 size, uint mip, bool allowUav, bool allowSimul, bool allowRaster, bool reserved) const;

public:
    //	vstd::span<std::atomic<D3D12_BARRIER_LAYOUT>> Layouts() const;
    static GFXFormat ToGFXFormat(PixelFormat format);
    static PixelFormat ToPixelFormat(GFXFormat format);
    uint Width() const { return width; }
    uint Height() const { return height; }
    GFXFormat Format() const { return format; }
    TextureDimension Dimension() const { return dimension; }
    uint Depth() const { return depth; }
    uint Mip() const { return mip; }
    virtual uint GetGlobalSRVIndex(uint mipOffset = 0) const;
    virtual uint GetGlobalUAVIndex(uint mipLevel) const;
    virtual D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc(uint mipOffset = 0) const;
    virtual D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDesc(uint targetMipLevel) const;
    virtual D3D12_DEPTH_STENCIL_VIEW_DESC GetDepthDesc() const;
    virtual D3D12_RENDER_TARGET_VIEW_DESC GetRenderTargetDesc(uint mipOffset = 0) const;
    TextureBase(
        Device *device,
        uint width,
        uint height,
        GFXFormat format,
        TextureDimension dimension,
        uint depth,
        uint mip,
        D3D12_RESOURCE_STATES initState);
    virtual ~TextureBase() override;
    TextureBase(TextureBase &&) = delete;
    KILL_COPY_CONSTRUCT(TextureBase)
};
}// namespace lc::dx

#pragma once
#include <Resource/Resource.h>
#include <Resource/IGpuAllocator.h>
#include <runtime/pixel.h>
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

public:
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
    virtual D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDesc(uint targetMipLevel) const VENGINE_PURE_VIRTUAL_RET;
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
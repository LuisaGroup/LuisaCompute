
#include <Resource/TextureBase.h>
namespace toolhub::directx {
TextureBase::TextureBase(
    Device *device,
    uint width,
    uint height,
    GFXFormat format,
    TextureDimension dimension,
    uint depth,
    uint mip)
    : Resource(device),
      width(width),
      height(height),
      format(format),
      dimension(dimension),
      depth(depth),
      mip(mip) {
    this->depth = std::max<uint>(this->depth, 1);
    this->mip = std::max<uint>(this->mip, 1);
    switch (dimension) {
        case TextureDimension::Tex1D:
            this->depth = 1;
            this->height = 1;
            break;
        case TextureDimension::Tex2D:
            this->depth = 1;
            break;
        case TextureDimension::Cubemap:
            this->depth = 6;
            break;
        default: assert(false); break;
    }
    //layouts = vstd::create_unique(vengine_new_array<std::atomic<D3D12_BARRIER_LAYOUT>>(mip, D3D12_BARRIER_LAYOUT_COMMON));
}
D3D12_SHADER_RESOURCE_VIEW_DESC TextureBase::GetColorSrvDescBase(uint mipOffset, ID3D12Resource* resource) const {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    auto format = resource->GetDesc();
    srvDesc.Format = format.Format;
    auto mipSize = std::max<int>(0, (int32)format.MipLevels - (int32)mipOffset);
    switch (dimension) {
        case TextureDimension::Cubemap:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srvDesc.TextureCube.MostDetailedMip = mipOffset;
            srvDesc.TextureCube.MipLevels = mipSize;
            srvDesc.TextureCube.ResourceMinLODClamp = 0.0f;
            break;
        case TextureDimension::Tex2D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MostDetailedMip = mipOffset;
            srvDesc.Texture2D.MipLevels = mipSize;
            srvDesc.Texture2D.PlaneSlice = 0;
            srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
            break;
        case TextureDimension::Tex1D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            srvDesc.Texture1D.MipLevels = mipSize;
            srvDesc.Texture1D.MostDetailedMip = mipOffset;
            srvDesc.Texture1D.ResourceMinLODClamp = 0.0f;
            break;
        case TextureDimension::Tex2DArray:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srvDesc.Texture2DArray.MostDetailedMip = mipOffset;
            srvDesc.Texture2DArray.MipLevels = mipSize;
            srvDesc.Texture2DArray.PlaneSlice = 0;
            srvDesc.Texture2DArray.ResourceMinLODClamp = 0.0f;
            srvDesc.Texture2DArray.ArraySize = depth;
            srvDesc.Texture2DArray.FirstArraySlice = 0;
            break;
        case TextureDimension::Tex3D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srvDesc.Texture3D.MipLevels = mipSize;
            srvDesc.Texture3D.MostDetailedMip = mipOffset;
            srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;
            break;
        default: assert(false); break;
    }
    return srvDesc;
}
D3D12_RENDER_TARGET_VIEW_DESC TextureBase::GetRenderTargetDescBase(uint mipOffset) const {
    D3D12_RENDER_TARGET_VIEW_DESC rtv;
    rtv.Format = static_cast<DXGI_FORMAT>(format);
    assert(dimension == TextureDimension::Tex2D);
    rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    rtv.Texture2D.MipSlice = mipOffset;
    rtv.Texture2D.PlaneSlice = 0;
    return rtv;
}
// vstd::span<std::atomic<D3D12_BARRIER_LAYOUT>> TextureBase::Layouts() const {
// 	return {layouts.get(), size_t(mip)};
// }
D3D12_UNORDERED_ACCESS_VIEW_DESC TextureBase::GetColorUavDescBase(uint targetMipLevel, ID3D12Resource *resource) const {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
    auto desc = resource->GetDesc();
    uint maxLevel = desc.MipLevels - 1;
    targetMipLevel = std::min(targetMipLevel, maxLevel);
    uavDesc.Format = desc.Format;
    switch (dimension) {
        case TextureDimension::Tex2D:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uavDesc.Texture2D.MipSlice = targetMipLevel;
            uavDesc.Texture2D.PlaneSlice = 0;
            break;
        case TextureDimension::Tex1D:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
            uavDesc.Texture1D.MipSlice = targetMipLevel;
            break;
        case TextureDimension::Tex3D:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uavDesc.Texture3D.FirstWSlice = 0;
            uavDesc.Texture3D.MipSlice = targetMipLevel;
            uavDesc.Texture3D.WSize = depth >> targetMipLevel;
            break;
        default:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uavDesc.Texture2DArray.ArraySize = depth;
            uavDesc.Texture2DArray.FirstArraySlice = 0;
            uavDesc.Texture2DArray.MipSlice = targetMipLevel;
            uavDesc.Texture2DArray.PlaneSlice = 0;
            break;
    }
    return uavDesc;
}
TextureBase::~TextureBase() {
}
}// namespace toolhub::directx
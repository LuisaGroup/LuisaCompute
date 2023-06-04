
#include <Resource/TextureBase.h>
#include <Resource/DescriptorHeap.h>
namespace lc::dx {
TextureBase::TextureBase(
    Device *device,
    uint width,
    uint height,
    GFXFormat format,
    TextureDimension dimension,
    uint depth,
    uint mip,
    D3D12_RESOURCE_STATES initState)
    : Resource(device),
      width(width),
      height(height),
      format(format),
      dimension(dimension),
      depth(depth),
      initState(initState),
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
        default:
            assert(dimension == TextureDimension::Tex3D);
            break;
    }
    //layouts = vstd::create_unique(vengine_new_array<std::atomic<D3D12_BARRIER_LAYOUT>>(mip, D3D12_BARRIER_LAYOUT_COMMON));
}
D3D12_SHADER_RESOURCE_VIEW_DESC TextureBase::GetColorSrvDescBase(uint mipOffset) const {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = static_cast<DXGI_FORMAT>(format);
    auto mipSize = std::max<int>(0, (int32)mip - (int32)mipOffset);
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
D3D12_UNORDERED_ACCESS_VIEW_DESC TextureBase::GetColorUavDescBase(uint targetMipLevel) const {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
    uint maxLevel = mip - 1;
    targetMipLevel = std::min(targetMipLevel, maxLevel);
    uavDesc.Format = static_cast<DXGI_FORMAT>(format);
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
uint TextureBase::GetGlobalSRVIndexBase(uint mipOffset, std::mutex &allocMtx, vstd::unordered_map<uint, uint> &srvIdcs) const {
    std::lock_guard lck(allocMtx);
    auto ite = srvIdcs.try_emplace(
        mipOffset,
        vstd::lazy_eval([&]() -> uint {
            auto v = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateSRV(
                GetResource(),
                GetColorSrvDesc(mipOffset),
                v);
            return v;
        }));
    return ite.first->second;
}
uint TextureBase::GetGlobalUAVIndexBase(uint mipLevel, std::mutex &allocMtx, vstd::unordered_map<uint, uint> &uavIdcs) const {
    mipLevel = std::min<uint>(mipLevel, mip - 1);
    std::lock_guard lck(allocMtx);
    auto ite = uavIdcs.try_emplace(
        mipLevel,
        vstd::lazy_eval([&]() -> uint {
            auto v = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateUAV(
                GetResource(),
                GetColorUavDesc(mipLevel),
                v);
            return v;
        }));
    return ite.first->second;
}
TextureBase::~TextureBase() {
}
GFXFormat TextureBase::ToGFXFormat(PixelFormat format) {
    switch (format) {
        case PixelFormat::R8SInt:
            return GFXFormat_R8_SInt;
        case PixelFormat::R8UInt:
            return GFXFormat_R8_UInt;
        case PixelFormat::R8UNorm:
            return GFXFormat_R8_UNorm;
        case PixelFormat::RG8SInt:
            return GFXFormat_R8G8_SInt;
        case PixelFormat::RG8UInt:
            return GFXFormat_R8G8_UInt;
        case PixelFormat::RG8UNorm:
            return GFXFormat_R8G8_UNorm;
        case PixelFormat::RGBA8SInt:
            return GFXFormat_R8G8B8A8_SInt;
        case PixelFormat::RGBA8UInt:
            return GFXFormat_R8G8B8A8_UInt;
        case PixelFormat::RGBA8UNorm:
            return GFXFormat_R8G8B8A8_UNorm;
        case PixelFormat::R16SInt:
            return GFXFormat_R16_SInt;
        case PixelFormat::R16UInt:
            return GFXFormat_R16_UInt;
        case PixelFormat::R16UNorm:
            return GFXFormat_R16_UNorm;
        case PixelFormat::RG16SInt:
            return GFXFormat_R16G16_SInt;
        case PixelFormat::RG16UInt:
            return GFXFormat_R16G16_UInt;
        case PixelFormat::RG16UNorm:
            return GFXFormat_R16G16_UNorm;
        case PixelFormat::RGBA16SInt:
            return GFXFormat_R16G16B16A16_SInt;
        case PixelFormat::RGBA16UInt:
            return GFXFormat_R16G16B16A16_UInt;
        case PixelFormat::RGBA16UNorm:
            return GFXFormat_R16G16B16A16_UNorm;
        case PixelFormat::R32SInt:
            return GFXFormat_R32_SInt;
        case PixelFormat::R32UInt:
            return GFXFormat_R32_UInt;
        case PixelFormat::RG32SInt:
            return GFXFormat_R32G32_SInt;
        case PixelFormat::RG32UInt:
            return GFXFormat_R32G32_UInt;
        case PixelFormat::RGBA32SInt:
            return GFXFormat_R32G32B32A32_SInt;
        case PixelFormat::RGBA32UInt:
            return GFXFormat_R32G32B32A32_UInt;
        case PixelFormat::R16F:
            return GFXFormat_R16_Float;
        case PixelFormat::RG16F:
            return GFXFormat_R16G16_Float;
        case PixelFormat::RGBA16F:
            return GFXFormat_R16G16B16A16_Float;
        case PixelFormat::R32F:
            return GFXFormat_R32_Float;
        case PixelFormat::RG32F:
            return GFXFormat_R32G32_Float;
        case PixelFormat::RGBA32F:
            return GFXFormat_R32G32B32A32_Float;
        case PixelFormat::BC6HUF16:
            return GFXFormat_BC6H_UF16;
        case PixelFormat::BC7UNorm:
            return GFXFormat_BC7_UNorm;
        case PixelFormat::BC5UNorm:
            return GFXFormat_BC5_UNorm;
        case PixelFormat::BC4UNorm:
            return GFXFormat_BC4_UNorm;
    }
    LUISA_ERROR_WITH_LOCATION("Unreachable.");
}

}// namespace lc::dx
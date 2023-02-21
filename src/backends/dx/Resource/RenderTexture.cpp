
#include <Resource/RenderTexture.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/BufferView.h>
namespace toolhub::directx {
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
}

RenderTexture::RenderTexture(
    Device *device,
    uint width,
    uint height,
    GFXFormat format,
    TextureDimension dimension,
    uint depth,
    uint mip,
    bool allowUav,
    GpuAllocator *allocator)
    : TextureBase(device, width, height, format, dimension, depth, mip),
      allocHandle(allocator),
      allowUav(allowUav) {
    D3D12_RESOURCE_DESC texDesc{};
    switch (dimension) {
        case TextureDimension::Cubemap:
        case TextureDimension::Tex2DArray:
        case TextureDimension::Tex2D:
            texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            break;
        case TextureDimension::Tex3D:
            texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            break;
        case TextureDimension::Tex1D:
            texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
            break;
        default: assert(false); break;
    }
    texDesc.Alignment = 0;
    texDesc.Width = this->width;
    texDesc.Height = this->height;
    texDesc.DepthOrArraySize = this->depth;
    texDesc.MipLevels = mip;
    texDesc.Format = (DXGI_FORMAT)format;
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = allowUav ? (D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) : D3D12_RESOURCE_FLAG_NONE;

    if (!allocator) {
        auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_HEAP_PROPERTIES const *propPtr = &prop;
        ThrowIfFailed(device->device->CreateCommittedResource(
            propPtr,
            D3D12_HEAP_FLAG_NONE,
            &texDesc,
            GetInitState(),
            nullptr,
            IID_PPV_ARGS(&allocHandle.resource)));
    } else {
        ID3D12Heap *heap;
        uint64 offset;
        auto allocateInfo = device->device->GetResourceAllocationInfo(
            0, 1, &texDesc);
        auto byteSize = allocateInfo.SizeInBytes;
        allocHandle.allocateHandle = allocator->AllocateTextureHeap(
            device,
            byteSize,
            &heap,
            &offset,
            true);
        ThrowIfFailed(device->device->CreatePlacedResource(
            heap,
            offset,
            &texDesc,
            GetInitState(),
            nullptr,
            IID_PPV_ARGS(&allocHandle.resource)));
    }
    //Setup Desc
}

D3D12_SHADER_RESOURCE_VIEW_DESC RenderTexture::GetColorSrvDesc(uint mipOffset) const {
    return GetColorSrvDescBase(mipOffset);
}
D3D12_UNORDERED_ACCESS_VIEW_DESC RenderTexture::GetColorUavDesc(uint targetMipLevel) const {
    return GetColorUavDescBase(targetMipLevel);
}
D3D12_RENDER_TARGET_VIEW_DESC RenderTexture::GetRenderTargetDesc(uint mipOffset) const {
    return GetRenderTargetDescBase(mipOffset);
}
uint RenderTexture::GetGlobalSRVIndex(uint mipOffset) const {
    std::lock_guard lck(allocMtx);
    auto ite = srvIdcs.try_emplace(
        mipOffset,
        vstd::LazyEval([&]() -> uint {
            auto v = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateSRV(
                GetResource(),
                GetColorSrvDesc(mipOffset),
                v);
            return v;
        }));
    return ite.first->second;
}
uint RenderTexture::GetGlobalUAVIndex(uint mipLevel) const {
    assert(allowUav);
    mipLevel = std::min<uint>(mipLevel, mip - 1);
    std::lock_guard lck(allocMtx);
    auto ite = uavIdcs.try_emplace(
        mipLevel,
        vstd::LazyEval([&]() -> uint {
            auto v = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateUAV(
                GetResource(),
                GetColorUavDesc(mipLevel),
                v);
            return v;
        }));
    return ite.first->second;
}
RenderTexture::~RenderTexture() {
    auto &globalHeap = *device->globalHeap.get();
    for (auto &&i : uavIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
    for (auto &&i : srvIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
}
TexView::TexView(
    TextureBase const *tex,
    uint64 mipStart,
    uint64 mipCount)
    : tex(tex),
      mipStart(mipStart),
      mipCount(mipCount) {
}
TexView::TexView(
    TextureBase const *tex,
    uint64 mipStart)
    : tex(tex),
      mipStart(mipStart) {
    mipCount = tex->Mip() - mipStart;
}
}// namespace toolhub::directx
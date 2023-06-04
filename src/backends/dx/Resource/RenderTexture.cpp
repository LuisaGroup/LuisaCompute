
#include <Resource/RenderTexture.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/BufferView.h>
namespace lc::dx {
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
    : TextureBase(device, width, height, format, dimension, depth, mip, GetInitState()),
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
    assert(allowUav);
    return GetColorUavDescBase(targetMipLevel);
}
D3D12_RENDER_TARGET_VIEW_DESC RenderTexture::GetRenderTargetDesc(uint mipOffset) const {
    return GetRenderTargetDescBase(mipOffset);
}
uint RenderTexture::GetGlobalSRVIndex(uint mipOffset) const {
    return GetGlobalSRVIndexBase(mipOffset, allocMtx, srvIdcs);
}
uint RenderTexture::GetGlobalUAVIndex(uint mipLevel) const {
    return GetGlobalUAVIndexBase(mipLevel, allocMtx, uavIdcs);
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
}// namespace lc::dx
#include <Resource/DepthBuffer.h>
#include <Resource/DescriptorHeap.h>
namespace toolhub::directx {
GFXFormat DepthBuffer::GetDepthFormat(DepthFormat f) {
    using namespace luisa::compute;
    switch (f) {
        case DepthFormat::D16:
            return GFXFormat_D16_UNorm;
        case DepthFormat::D24S8:
            return GFXFormat_D24_UNorm_S8_UInt;
        case DepthFormat::D32:
            return GFXFormat_D32_Float;
        case DepthFormat::D32S8A24:
            return GFXFormat_D32_Float_S8X24_UInt;
        default:
            return GFXFormat_Unknown;
    }
}
DepthBuffer::DepthBuffer(
    Device *device,
    uint width,
    uint height,
    luisa::compute::DepthFormat format,
    GpuAllocator *alloc)
    : TextureBase(
          device, width, height,
          GetDepthFormat(format),
          TextureDimension::Tex2D, 1, 1),
      allocHandle(alloc) {
    D3D12_RESOURCE_DESC texDesc{};
    texDesc.Alignment = 0;
    texDesc.Width = this->width;
    texDesc.Height = this->height;
    texDesc.DepthOrArraySize = depth;
    texDesc.MipLevels = mip;
    texDesc.Format = (DXGI_FORMAT)this->format;
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if (!alloc) {
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
        allocHandle.allocateHandle = alloc->AllocateTextureHeap(
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
}
DepthBuffer::~DepthBuffer() {
    if (srvIdx != ~0u) device->globalHeap->ReturnIndex(srvIdx);
}
D3D12_SHADER_RESOURCE_VIEW_DESC DepthBuffer::GetColorSrvDesc(uint mipOffset) const {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    switch (format) {
        case GFXFormat_D16_UNorm:
            srvDesc.Format = DXGI_FORMAT_R16_UNORM;
            break;
        case GFXFormat_D24_UNorm_S8_UInt:
            srvDesc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            break;
        case GFXFormat_D32_Float:
            srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
            break;
        case GFXFormat_D32_Float_S8X24_UInt:
            srvDesc.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            break;
        default: assert(false); break;
    }
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.PlaneSlice = 0;
    srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
    return srvDesc;
}
uint DepthBuffer::GetGlobalSRVIndex(uint mipOffset) const {
    std::lock_guard lck(allocMtx);
    if (srvIdx != ~0u) return srvIdx;
    srvIdx = device->globalHeap->AllocateIndex();
    device->globalHeap->CreateSRV(
        GetResource(),
        GetColorSrvDesc(),
        srvIdx);
    return srvIdx;
}
D3D12_DEPTH_STENCIL_VIEW_DESC DepthBuffer::GetDepthDesc() const {
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv;
    dsv.Format = static_cast<DXGI_FORMAT>(format);
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dsv.Flags = D3D12_DSV_FLAG_NONE;
    dsv.Texture2D.MipSlice = 0;
    return dsv;
}
}// namespace toolhub::directx
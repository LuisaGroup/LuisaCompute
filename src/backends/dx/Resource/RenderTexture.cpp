#include <Resource/RenderTexture.h>
#include <Resource/DescriptorHeap.h>
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
    bool allowSimul,
    GpuAllocator *allocator)
    : TextureBase(device, width, height, format, dimension, depth, mip, GetInitState()),
      allocHandle(allocator) {
    auto texDesc = GetResourceDescBase(allowUav, allowSimul);
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
            allowUav);
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
}// namespace lc::dx

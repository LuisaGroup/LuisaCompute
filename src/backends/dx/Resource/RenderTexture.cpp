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
    bool allowRaster,
    GpuAllocator *allocator,
    bool shared_adaptor)
    : TextureBase(device, width, height, format, dimension, depth, mip, GetInitState(), allowUav),
      allocHandle(allocator),
      allowSimul(allowSimul) {
    if (format == GFXFormat_BC7_UNorm_SRGB || format == GFXFormat_R8G8B8A8_UNorm_SRGB) {
        allowUav = false;
    }
    auto texDesc = GetResourceDescBase(allowUav, allowSimul, allowRaster, false);

    D3D12_CLEAR_VALUE optimizedClear{};
    D3D12_CLEAR_VALUE *clearPtr = nullptr;
    if (allowRaster) {
        optimizedClear.Format = static_cast<DXGI_FORMAT>(format);
        optimizedClear.Color[0] = 0.f;
        optimizedClear.Color[1] = 0.f;
        optimizedClear.Color[2] = 0.f;
        optimizedClear.Color[3] = 0.f;
        clearPtr = &optimizedClear;
    }

    if (!allocator) {
        auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_HEAP_PROPERTIES const *propPtr = &prop;
        ThrowIfFailed(device->device->CreateCommittedResource(
            propPtr,
            shared_adaptor ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
            &texDesc,
            GetInitState(),
            clearPtr,
            IID_PPV_ARGS(&allocHandle.resource)));
    } else {
        ID3D12Heap *heap;
        uint64 offset;
        auto allocateInfo = device->device->GetResourceAllocationInfo(
            0, 1, &texDesc);
        auto byteSize = allocateInfo.SizeInBytes;
        allocHandle.allocateHandle = allocator->AllocateTextureHeap(
            device,
            "render texture",
            byteSize,
            &heap,
            &offset,
            allowRaster,
            shared_adaptor ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE);
        ThrowIfFailed(device->device->CreatePlacedResource(
            heap,
            offset,
            &texDesc,
            GetInitState(),
            clearPtr,
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
    auto &global_heap = *device->global_heap.get();
    for (auto &&i : uavIdcs) {
        global_heap.ReturnIndex(i.second);
    }
    for (auto &&i : srvIdcs) {
        global_heap.ReturnIndex(i.second);
    }
}
}// namespace lc::dx

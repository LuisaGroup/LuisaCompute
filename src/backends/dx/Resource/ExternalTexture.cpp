#include <Resource/ExternalTexture.h>
#include <Resource/DescriptorHeap.h>
namespace toolhub::directx {
ExternalTexture::ExternalTexture(
    Device *device,
    ID3D12Resource *resource,
    D3D12_RESOURCE_STATES initState,
    uint width,
    uint height,
    GFXFormat format,
    TextureDimension dimension,
    uint depth,
    uint mip,
    bool allowUav)
    : TextureBase{
          device,
          width,
          height,
          format,
          dimension,
          depth,
          mip},
      resource{resource}, initState{initState}, allowUav{allowUav} {}
ExternalTexture::~ExternalTexture() {}

D3D12_SHADER_RESOURCE_VIEW_DESC ExternalTexture::GetColorSrvDesc(uint mipOffset) const {
    return GetColorSrvDescBase(mipOffset, resource);
}
D3D12_UNORDERED_ACCESS_VIEW_DESC ExternalTexture::GetColorUavDesc(uint targetMipLevel) const {
    return GetColorUavDescBase(targetMipLevel, resource);
}
D3D12_RENDER_TARGET_VIEW_DESC ExternalTexture::GetRenderTargetDesc(uint mipOffset) const {
    return GetRenderTargetDescBase(mipOffset);
}
uint ExternalTexture::GetGlobalSRVIndex(uint mipOffset) const {
    std::lock_guard lck(allocMtx);
    srvIdcs.New();
    auto ite = srvIdcs->Emplace(
        mipOffset,
        vstd::LazyEval([&]() -> uint {
            auto v = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateSRV(
                GetResource(),
                GetColorSrvDesc(mipOffset),
                v);
            return v;
        }));
    return ite.Value();
}
uint ExternalTexture::GetGlobalUAVIndex(uint mipLevel) const {
    if (!allowUav) return std::numeric_limits<uint>::max();
    mipLevel = std::min<uint>(mipLevel, mip - 1);
    std::lock_guard lck(allocMtx);
    uavIdcs.New();
    auto ite = uavIdcs->Emplace(
        mipLevel,
        vstd::LazyEval([&]() -> uint {
            auto v = device->globalHeap->AllocateIndex();
            device->globalHeap->CreateUAV(
                GetResource(),
                GetColorUavDesc(mipLevel),
                v);
            return v;
        }));
    return ite.Value();
}
}// namespace toolhub::directx
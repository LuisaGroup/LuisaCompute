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
          mip,
          initState},
      resource{resource}, allowUav{allowUav} {
}
ExternalTexture::~ExternalTexture() {
    auto &globalHeap = *device->globalHeap.get();
    for (auto &&i : uavIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
    for (auto &&i : srvIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
}

D3D12_SHADER_RESOURCE_VIEW_DESC ExternalTexture::GetColorSrvDesc(uint mipOffset) const {
    return GetColorSrvDescBase(mipOffset);
}
D3D12_UNORDERED_ACCESS_VIEW_DESC ExternalTexture::GetColorUavDesc(uint targetMipLevel) const {
    return GetColorUavDescBase(targetMipLevel);
}
D3D12_RENDER_TARGET_VIEW_DESC ExternalTexture::GetRenderTargetDesc(uint mipOffset) const {
    return GetRenderTargetDescBase(mipOffset);
}
uint ExternalTexture::GetGlobalSRVIndex(uint mipOffset) const {
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
uint ExternalTexture::GetGlobalUAVIndex(uint mipLevel) const {
    assert(allowUav);
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
}// namespace toolhub::directx
#pragma once
#include <Resource/TextureBase.h>
#include <Resource/AllocHandle.h>
namespace toolhub::directx {
class ExternalTexture final : public TextureBase {
private:
    mutable vstd::optional<vstd::HashMap<uint, uint>> uavIdcs;
    mutable vstd::optional<vstd::HashMap<uint, uint>> srvIdcs;
    mutable std::mutex allocMtx;
    ID3D12Resource *resource;
    D3D12_RESOURCE_STATES initState;
    bool allowUav;

public:
    ExternalTexture(
        Device *device,
        ID3D12Resource *resource,
        D3D12_RESOURCE_STATES initState,
        uint width,
        uint height,
        GFXFormat format,
        TextureDimension dimension,
        uint depth,
        uint mip,
        bool allowUav);
    ~ExternalTexture();
    D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDesc(uint targetMipLevel) const override;
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc(uint mipOffset = 0) const override;
    D3D12_RENDER_TARGET_VIEW_DESC GetRenderTargetDesc(uint mipOffset) const override;
    uint GetGlobalSRVIndex(uint mipOffset = 0) const override;
    uint GetGlobalUAVIndex(uint mipLevel) const override;
    ID3D12Resource *GetResource() const override { return resource; }
    D3D12_RESOURCE_STATES GetInitState() const override {
        return initState;
    }
    Tag GetTag() const override { return Tag::ExternalTexture; }
    VSTD_SELF_PTR
};
}// namespace toolhub::directx
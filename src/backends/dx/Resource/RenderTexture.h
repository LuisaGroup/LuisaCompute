#pragma once
#include <Resource/TextureBase.h>
#include <Resource/AllocHandle.h>
namespace toolhub::directx {
class RenderTexture final : public TextureBase {
private:
    bool allowUav;
    AllocHandle allocHandle;

public:
    RenderTexture(
        Device *device,
        uint width,
        uint height,
        GFXFormat format,
        TextureDimension dimension,
        uint depth,
        uint mip,
        bool allowUav,
        IGpuAllocator *allocator = nullptr);
    ~RenderTexture();
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc() const override;
    ID3D12Resource *GetResource() const override {
        return allocHandle.resource.Get();
    }
    D3D12_RESOURCE_STATES GetInitState() const {
        return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDesc(uint targetMipLevel) const override;
    Tag GetTag() const override { return Tag::RenderTexture; }
    VSTD_SELF_PTR
};
}// namespace toolhub::directx
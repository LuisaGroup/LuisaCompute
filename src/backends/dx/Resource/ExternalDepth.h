#pragma once
#include <Resource/TextureBase.h>
#include <runtime/depth_format.h>
namespace toolhub::directx {
class ExternalDepth final : public TextureBase {
private:
    ID3D12Resource *resource;
    mutable uint srvIdx{~0u};
    mutable std::mutex allocMtx;

public:
    ID3D12Resource *GetResource() const override {
        return resource;
    }
    D3D12_RESOURCE_STATES GetInitState() const override {
        return initState;
    }
    uint GetGlobalUAVIndex(uint mipLevel) const override {
        return ~0u;
    }
    Tag GetTag() const override { return Tag::ExternalDepth; }
    D3D12_DEPTH_STENCIL_VIEW_DESC GetDepthDesc() const override;
    ExternalDepth(
        ID3D12Resource *res,
        Device *device,
        uint width,
        uint height,
        luisa::compute::DepthFormat format,
        D3D12_RESOURCE_STATES initState);
    ~ExternalDepth();
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc(uint mipOffset = 0) const override;
    uint GetGlobalSRVIndex(uint mipOffset = 0) const override;
    VSTD_SELF_PTR
};
}// namespace toolhub::directx
#pragma once
#include <Resource/TextureBase.h>
#include <luisa/runtime/depth_format.h>
#include <Resource/AllocHandle.h>
namespace lc::dx {
class DepthBuffer final : public TextureBase {
private:
    AllocHandle allocHandle;
    mutable uint srvIdx{~0u};
    mutable std::mutex allocMtx;

public:
    static DepthFormat GFXFormatToDepth(GFXFormat f);
    static GFXFormat GetDepthFormat(DepthFormat f);
    ID3D12Resource *GetResource() const override {
        return allocHandle.resource.Get();
    }
    D3D12_RESOURCE_STATES GetInitState() const override {
        return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    Tag GetTag() const override { return Tag::DepthBuffer; }
    D3D12_DEPTH_STENCIL_VIEW_DESC GetDepthDesc() const override;
    DepthBuffer(
        Device *device,
        uint width,
        uint height,
        luisa::compute::DepthFormat format,
        GpuAllocator *alloc = nullptr);
    ~DepthBuffer();
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc(uint mipOffset = 0) const override;
    uint GetGlobalSRVIndex(uint mipOffset = 0) const override;
};
}// namespace lc::dx

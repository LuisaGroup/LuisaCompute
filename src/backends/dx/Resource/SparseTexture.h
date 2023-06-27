#pragma once
#include <Resource/TextureBase.h>
namespace lc::dx {
class SparseTexture final : public TextureBase {
private:
    ComPtr<ID3D12Resource> resource;
    mutable vstd::unordered_map<uint, uint> uavIdcs;
    mutable vstd::unordered_map<uint, uint> srvIdcs;
    mutable std::mutex allocMtx;
    uint3 tileSize;

public:
    uint3 TilingSize() const;
    SparseTexture(
        Device *device,
        uint width,
        uint height,
        GFXFormat format,
        TextureDimension dimension,
        uint depth,
        uint mip,
        bool allowUav,
        bool allowSimul);
    ~SparseTexture();
    ID3D12Resource *GetResource() const override {
        return resource.Get();
    }
    D3D12_RESOURCE_STATES GetInitState() const override {
        return D3D12_RESOURCE_STATE_COMMON;
    }
    Tag GetTag() const override { return Tag::SparseTexture; }
    D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDesc(uint targetMipLevel) const override;
    D3D12_RENDER_TARGET_VIEW_DESC GetRenderTargetDesc(uint mipOffset) const override;
    uint GetGlobalSRVIndex(uint mipOffset = 0) const override;
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDesc(uint mipOffset = 0) const override;
    uint GetGlobalUAVIndex(uint mipLevel) const override;
    void AllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint3 size, uint mipLevel, uint64 alloc) const;
    void DeAllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint3 size, uint mipLevel) const;
};
}// namespace lc::dx

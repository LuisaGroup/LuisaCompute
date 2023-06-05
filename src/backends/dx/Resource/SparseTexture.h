#pragma once
#include <Resource/TextureBase.h>
#include <Resource/AllocHandle.h>
namespace lc::dx {
class SparseTexture final : public TextureBase {
public:
    struct Tile {
        uint coords[3];
        uint mipLevel;
    };
    struct TileHash {
        size_t operator()(Tile const &t) const {
            return luisa::hash64(&t, sizeof(Tile), luisa::hash64_default_seed);
        }
    };
    struct TileLess {
        bool operator()(Tile const &a, Tile const &b) const {
            return memcmp(&a, &b, sizeof(Tile)) < 0;
        }
    };
    struct TileInfo {
        uint size[3];
        uint allocatorHandle;
    };

private:
    GpuAllocator *allocator;
    uint64 allocatorPool;
    ComPtr<ID3D12Resource> resource;
    mutable vstd::unordered_map<uint, uint> uavIdcs;
    mutable vstd::unordered_map<uint, uint> srvIdcs;
    mutable vstd::unordered_map<Tile, TileInfo, TileHash, TileLess> allocatedTiles;
    mutable std::mutex allocMtx;
    bool allowUav;

public:
    SparseTexture(
        Device *device,
        uint width,
        uint height,
        GFXFormat format,
        TextureDimension dimension,
        uint depth,
        uint mip,
        bool allowUav,
        GpuAllocator &allocator);
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
    void AllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint3 size, uint mipLevel) const;
    void DeAllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint mipLevel, vstd::vector<std::pair<GpuAllocator*, uint64>>& deallocatedHandle)const;
};
}// namespace lc::dx
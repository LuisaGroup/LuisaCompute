#pragma once
#include <Resource/Buffer.h>
namespace lc::dx {
class SparseBuffer final : public Buffer {
private:
    GpuAllocator *allocator;
    uint64 allocatorPool;
    ComPtr<ID3D12Resource> resource;
    uint64 byteSize;
    D3D12_RESOURCE_STATES initState;
    struct TileInfo {
        uint size;
        uint64 allocatorHandle;
    };
    mutable vstd::unordered_map<uint, TileInfo> allocatedTiles;
    mutable std::mutex allocMtx;

public:
    vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(bool isRaw = false) const override;
    vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(bool isRaw = false) const override;
    vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
    vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;

    ID3D12Resource *GetResource() const override { return resource.Get(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const override { return resource->GetGPUVirtualAddress(); }
    uint64 GetByteSize() const override { return byteSize; }
    void AllocateTile(ID3D12CommandQueue *queue, uint coord, uint size) const;
    void DeAllocateTile(ID3D12CommandQueue *queue, uint coord) const;
    void FreeTileMemory(ID3D12CommandQueue *queue, uint coord) const;
    SparseBuffer(
        Device *device,
        uint64 byteSize,
        GpuAllocator &allocator,
        D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_COMMON);
    ~SparseBuffer();
    D3D12_RESOURCE_STATES GetInitState() const override {
        return initState;
    }
    Tag GetTag() const override {
        return Tag::SparseBuffer;
    }
    SparseBuffer(SparseBuffer &&) = default;
    void ClearTile(ID3D12CommandQueue *queue) const;
    KILL_COPY_CONSTRUCT(SparseBuffer)
};
}// namespace lc::dx
#include "SparseBuffer.h"
// #include <Resource/GpuAllocator.h>
#include <core/logging.h>
namespace lc::dx {
SparseBuffer::SparseBuffer(
    Device *device,
    uint64 byteSize,
    D3D12_RESOURCE_STATES initState)
    : Buffer(device),
      sparseAllocator(device, false),
      byteSize(byteSize),
      initState(initState) {
    auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    // allocatorPool = allocator.CreatePool(D3D12_HEAP_TYPE_DEFAULT);

    ThrowIfFailed(device->device->CreateReservedResource(
        &buffer,
        GetInitState(),
        nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())));
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> SparseBuffer::GetColorSrvDesc(bool isRaw) const {
    return GetColorSrvDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> SparseBuffer::GetColorUavDesc(bool isRaw) const {
    return GetColorUavDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> SparseBuffer::GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
    return GetColorSrvDescBase(offset, byteSize, isRaw);
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> SparseBuffer::GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
    return GetColorUavDescBase(offset, byteSize, isRaw);
}
SparseBuffer::~SparseBuffer() {
}
void SparseBuffer::FreeTileMemory(ID3D12CommandQueue *queue, uint coord) const {
    std::lock_guard lck{allocMtx};
    auto iter = allocatedTiles.find(coord);
    if (iter == allocatedTiles.end()) [[unlikely]] {
        return;
    }
    auto &tileInfo = iter->second;
    for (auto &i : tileInfo.heaps) {
        sparseAllocator.Deallocate(i.heap, {tileInfo.offsets.data() + i.offset, i.size});
    }
    allocatedTiles.erase(iter);
}
void SparseBuffer::AllocateTile(ID3D12CommandQueue *queue, uint coord, uint size) const {
    std::lock_guard lck{allocMtx};
    auto iter = allocatedTiles.try_emplace(
        coord,
        vstd::lazy_eval([&] {
            TileInfo tileInfo;
            tileInfo.size = size;
            tileInfo.offsets.push_back_uninitialized(size);
            return tileInfo;
        }));
    auto &tileInfo = iter.first->second;
    uint tileIndex = 0;
    auto allocateFunc = [&](ID3D12Heap *heap, vstd::span<const uint> offsets) {
        D3D12_TILED_RESOURCE_COORDINATE tileCoord{
            .X = tileIndex + coord,
            .Subresource = 0};
        D3D12_TILE_REGION_SIZE tileSize{
            .NumTiles = static_cast<uint>(offsets.size()),
            .UseBox = true,
            .Width = static_cast<uint>(offsets.size()),
            .Height = 1,
            .Depth = 1};
        bytes.clear();
        bytes.push_back_uninitialized(
            sizeof(D3D12_TILE_RANGE_FLAGS) * offsets.size() +
            sizeof(uint) * offsets.size());
        auto flags = reinterpret_cast<D3D12_TILE_RANGE_FLAGS *>(bytes.data());
        auto rangeTiles = reinterpret_cast<uint *>(flags + offsets.size());
        tileInfo.heaps.emplace_back(AllocateOffsets{
            .heap = heap,
            .offset = tileIndex,
            .size = static_cast<uint>(offsets.size())});
        for (auto i : vstd::range(offsets.size())) {
            tileInfo.offsets[tileIndex + i] = offsets[i];
            flags[i] = D3D12_TILE_RANGE_FLAG_NONE;
            rangeTiles[i] = 1;
        }
        queue->UpdateTileMappings(
            resource.Get(), 1,
            &tileCoord,
            &tileSize,
            heap, offsets.size(),
            flags,
            offsets.data(),
            rangeTiles,
            D3D12_TILE_MAPPING_FLAG_NONE);
        tileIndex += offsets.size();
    };
    sparseAllocator.AllocateTiles(size, allocateFunc);
}
void SparseBuffer::ClearTile(ID3D12CommandQueue *queue) const {
    {
        std::lock_guard lck{allocMtx};
        sparseAllocator.ClearAllocate();
        allocatedTiles.clear();
    }
    D3D12_TILE_RANGE_FLAGS RangeFlags = D3D12_TILE_RANGE_FLAG_NULL;
    queue->UpdateTileMappings(resource.Get(), 1, nullptr, nullptr, nullptr, 1, &RangeFlags, nullptr, nullptr, D3D12_TILE_MAPPING_FLAG_NONE);
}
}// namespace lc::dx
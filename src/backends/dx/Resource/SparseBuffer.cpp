#include "SparseBuffer.h"
#include <Resource/GpuAllocator.h>
#include <core/logging.h>
namespace lc::dx {
SparseBuffer::SparseBuffer(
    Device *device,
    uint64 byteSize,
    GpuAllocator &allocator,
    D3D12_RESOURCE_STATES initState)
    : Buffer(device),
      allocator(&allocator),
      byteSize(byteSize),
      initState(initState) {
    if (byteSize < D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) [[unlikely]] {
        LUISA_ERROR("Currently do not support packed tile.");
    }
    auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    allocatorPool = allocator.CreatePool(D3D12_HEAP_TYPE_DEFAULT);

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
    for (auto &&i : allocatedTiles) {
        allocator->Release(i.second.allocatorHandle);
    }
    allocator->DestroyPool(allocatorPool);
}
void SparseBuffer::FreeTileMemory(ID3D12CommandQueue *queue, uint coord) const {
    TileInfo tileInfo;
    {
        std::lock_guard lck{allocMtx};
        auto iter = allocatedTiles.find(coord);
        if (iter == allocatedTiles.end()) [[unlikely]] {
            return;
        }
        tileInfo = iter->second;
        allocatedTiles.erase(iter);
    }
    allocator->Release(tileInfo.allocatorHandle);
}
void SparseBuffer::AllocateTile(ID3D12CommandQueue *queue, uint coord, uint size) const {
    TileInfo tileInfo;
    tileInfo.size = size;
    ID3D12Heap *heap;
    uint64 offset;
    uint offsetTile;
    tileInfo.allocatorHandle =
        allocator->AllocateBufferHeap(device, size * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset, allocatorPool);
    {
        std::lock_guard lck{allocMtx};
        auto iter = allocatedTiles.try_emplace(coord, tileInfo);
    }
    D3D12_TILED_RESOURCE_COORDINATE tileCoord{
        .X = coord,
        .Subresource = 0};
    D3D12_TILE_REGION_SIZE tileSize{
        .NumTiles = size,
        .UseBox = true,
        .Width = size,
        .Height = 1,
        .Depth = 1};
    uint rangeTileCount = tileSize.NumTiles;
    offsetTile = offset / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    queue->UpdateTileMappings(
        resource.Get(), 1,
        &tileCoord,
        &tileSize,
        heap, 1,
        vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
        &offsetTile,
        &rangeTileCount,
        D3D12_TILE_MAPPING_FLAG_NONE);
}
void SparseBuffer::DeAllocateTile(ID3D12CommandQueue *queue, uint coord) const {
    TileInfo tileInfo;
    {
        std::lock_guard lck{allocMtx};
        auto iter = allocatedTiles.find(coord);
        if (iter == allocatedTiles.end()) [[unlikely]] {
            return;
        }
        tileInfo = iter->second;
        allocatedTiles.erase(iter);
    }
    D3D12_TILED_RESOURCE_COORDINATE tileCoord{
        .X = coord,
        .Subresource = 0};
    D3D12_TILE_REGION_SIZE tileSize{
        .NumTiles = tileInfo.size,
        .UseBox = true,
        .Width = tileInfo.size,
        .Height = 1,
        .Depth = 1};
    queue->UpdateTileMappings(
        resource.Get(), 1,
        &tileCoord,
        &tileSize,
        nullptr, 1,
        vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NULL),
        nullptr,
        nullptr,
        D3D12_TILE_MAPPING_FLAG_NONE);
}
}// namespace lc::dx
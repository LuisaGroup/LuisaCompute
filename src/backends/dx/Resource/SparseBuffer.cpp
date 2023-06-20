#include "SparseBuffer.h"
#include <Resource/GpuAllocator.h>
#include <luisa/core/logging.h>
namespace lc::dx {
SparseBuffer::SparseBuffer(
    Device *device,
    uint64 byteSize,
    D3D12_RESOURCE_STATES initState)
    : Buffer(device),
      byteSize(byteSize),
      initState(initState) {
    auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
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
    resource.Reset();
    auto alloc = device->defaultAllocator.get();
    for (auto &&i : allocatedTiles) {
        alloc->Release(i.second.allocation);
    }
}
void SparseBuffer::DeAllocateTile(ID3D12CommandQueue *queue, uint coord, vstd::vector<uint64> &destroyList) const {
    std::lock_guard lck{allocMtx};
    auto iter = allocatedTiles.find(coord);
    if (iter == allocatedTiles.end()) [[unlikely]] {
        return;
    }
    auto &tileInfo = iter->second;
    destroyList.emplace_back(tileInfo.allocation);
    D3D12_TILED_RESOURCE_COORDINATE tileCoord{
        .X = coord,
        .Subresource = 0};
    D3D12_TILE_REGION_SIZE tileSize{
        .NumTiles = static_cast<uint>(tileInfo.tileCount),
        .UseBox = true,
        .Width = static_cast<uint>(tileInfo.tileCount),
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
    allocatedTiles.erase(iter);
}
void SparseBuffer::AllocateTile(ID3D12CommandQueue *queue, uint coord, uint size) const {
    std::lock_guard lck{allocMtx};
    auto iter = allocatedTiles.try_emplace(
        coord,
        vstd::lazy_eval([&] {
            TileInfo tileInfo;
            tileInfo.tileCount = size;
            ID3D12Heap *heap;
            uint64 offset;
            tileInfo.allocation = device->defaultAllocator->AllocateBufferHeap(
                device,
                size * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES,
                D3D12_HEAP_TYPE_DEFAULT,
                &heap, &offset);
            D3D12_TILED_RESOURCE_COORDINATE tileCoord{
                .X = coord,
                .Subresource = 0};
            D3D12_TILE_REGION_SIZE tileSize{
                .NumTiles = size,
                .UseBox = true,
                .Width = size,
                .Height = 1,
                .Depth = 1};
            uint tileOffset = offset / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
            queue->UpdateTileMappings(
                resource.Get(), 1,
                &tileCoord,
                &tileSize,
                heap,
                1,
                vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
                &tileOffset,
                &size,
                D3D12_TILE_MAPPING_FLAG_NONE);
            return tileInfo;
        }));
}
}// namespace lc::dx

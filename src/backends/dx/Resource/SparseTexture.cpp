#include <Resource/SparseTexture.h>
#include <Resource/DescriptorHeap.h>
#include <core/logging.h>
namespace lc::dx {
SparseTexture::SparseTexture(
    Device *device,
    uint width,
    uint height,
    GFXFormat format,
    TextureDimension dimension,
    uint depth,
    uint mip,
    bool allowUav,
    GpuAllocator &allocator)
    : TextureBase(device, width, height, format, dimension, depth, mip, GetInitState()),
      allocator(&allocator),
      allowUav(allowUav) {
    allocatorPool = allocator.CreatePool(D3D12_HEAP_TYPE_DEFAULT);
    auto texDesc = GetResourceDescBase(allowUav, true);
    ThrowIfFailed(device->device->CreateReservedResource(
        &texDesc,
        GetInitState(),
        nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())));
    D3D12_SUBRESOURCE_TILING tilingInfo;
    uint subresourceCount = 1;
    device->device->GetResourceTiling(
        resource.Get(),
        nullptr,
        nullptr,
        nullptr,
        &subresourceCount,
        0,
        &tilingInfo);
    auto lastMipSize = uint3(width, height, depth) >> (mip - 1);
    // TODO: may need packed mip in the future?
    if (lastMipSize.x < tilingInfo.WidthInTiles || lastMipSize.y < tilingInfo.HeightInTiles || lastMipSize.z < tilingInfo.DepthInTiles) [[unlikely]] {
        LUISA_ERROR("Currently do not support packed tile.");
    }
    tilingSize = uint3(
        width / tilingInfo.WidthInTiles,
        height / tilingInfo.HeightInTiles,
        depth / tilingInfo.DepthInTiles);
}
SparseTexture::~SparseTexture() {
    auto &globalHeap = *device->globalHeap.get();
    for (auto &&i : uavIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
    for (auto &&i : srvIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
    for (auto &&i : allocatedTiles) {
        allocator->Release(i.second.allocatorHandle);
    }
    allocator->DestroyPool(allocatorPool);
}
D3D12_SHADER_RESOURCE_VIEW_DESC SparseTexture::GetColorSrvDesc(uint mipOffset) const {
    return GetColorSrvDescBase(mipOffset);
}
D3D12_UNORDERED_ACCESS_VIEW_DESC SparseTexture::GetColorUavDesc(uint targetMipLevel) const {
    assert(allowUav);
    return GetColorUavDescBase(targetMipLevel);
}
D3D12_RENDER_TARGET_VIEW_DESC SparseTexture::GetRenderTargetDesc(uint mipOffset) const {
    return GetRenderTargetDescBase(mipOffset);
}
uint SparseTexture::GetGlobalSRVIndex(uint mipOffset) const {
    return GetGlobalSRVIndexBase(mipOffset, allocMtx, srvIdcs);
}
uint SparseTexture::GetGlobalUAVIndex(uint mipLevel) const {
    return GetGlobalUAVIndexBase(mipLevel, allocMtx, uavIdcs);
}
void SparseTexture::AllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint3 size, uint mipLevel) const {
    Tile tile;
    TileInfo tileInfo;
    tile.mipLevel = mipLevel;
    for (auto i : vstd::range(3)) {
        tile.coords[i] = coord[i];
        tileInfo.size[i] = size[i];
    }
    ID3D12Heap *heap;
    uint64 offset;
    auto allocateInfo = device->device->GetResourceAllocationInfo(
        0, 1, vstd::get_rval_ptr(GetResourceDescBase(size, 1, allowUav, true)));

    tileInfo.allocatorHandle = allocator->AllocateTextureHeap(device, allocateInfo.SizeInBytes, &heap, &offset, true, allocatorPool);
    {
        std::lock_guard lck{allocMtx};
        auto iter = allocatedTiles.try_emplace(tile, tileInfo);
        if (!iter.second) [[unlikely]] {
            LUISA_ERROR("Tile already exists");
        }
    }
    D3D12_TILED_RESOURCE_COORDINATE tileCoord{
        .X = coord.x,
        .Y = coord.y,
        .Z = coord.z,
        .Subresource = mipLevel};
    auto sizeInTiles = size / tilingSize;
    D3D12_TILE_REGION_SIZE tileSize{
        .NumTiles = sizeInTiles.x * sizeInTiles.y * sizeInTiles.z,
        .UseBox = true,
        .Width = sizeInTiles.x,
        .Height = static_cast<uint16_t>(sizeInTiles.y),
        .Depth = static_cast<uint16_t>(sizeInTiles.z)};
    uint rangeTileCount = tileSize.NumTiles;
    queue->UpdateTileMappings(
        resource.Get(), 1,
        &tileCoord,
        &tileSize,
        heap, 1,
        vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
        vstd::get_rval_ptr(static_cast<uint>(offset)),
        &rangeTileCount,
        D3D12_TILE_MAPPING_FLAG_NONE);
}
void SparseTexture::DeAllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint mipLevel, vstd::vector<std::pair<GpuAllocator *, uint64>> &deallocatedHandle) const {
    Tile tile;
    tile.mipLevel = mipLevel;
    for (auto i : vstd::range(3)) {
        tile.coords[i] = coord[i];
    }
    TileInfo tileInfo;
    {
        std::lock_guard lck{allocMtx};
        auto iter = allocatedTiles.find(tile);
        if (iter == allocatedTiles.end()) [[unlikely]] {
            LUISA_ERROR("Tile not exists");
        }
        tileInfo = iter->second;
        allocatedTiles.erase(iter);
    }
    D3D12_TILED_RESOURCE_COORDINATE tileCoord{
        .X = coord.x,
        .Y = coord.y,
        .Z = coord.z,
        .Subresource = mip};
    D3D12_TILE_REGION_SIZE tileSize{
        .NumTiles = tileInfo.size[0] * tileInfo.size[1] * tileInfo.size[2],
        .UseBox = true,
        .Width = tileInfo.size[0],
        .Height = static_cast<uint16_t>(tileInfo.size[1]),
        .Depth = static_cast<uint16_t>(tileInfo.size[2])};
    queue->UpdateTileMappings(
        resource.Get(), 1,
        &tileCoord,
        &tileSize,
        nullptr, 0,
        vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
        nullptr,
        nullptr,
        D3D12_TILE_MAPPING_FLAG_NONE);
    deallocatedHandle.emplace_back(allocator, tileInfo.allocatorHandle);
}
}// namespace lc::dx
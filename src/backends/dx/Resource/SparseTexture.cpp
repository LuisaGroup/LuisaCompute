#include <Resource/SparseTexture.h>
#include <Resource/DescriptorHeap.h>
#include <luisa/core/logging.h>
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
    bool allowSimul)
    : TextureBase(device, width, height, format, dimension, depth, mip, GetInitState()),
      sparseAllocator(device, true),
      allowUav(allowUav) {
    auto texDesc = GetResourceDescBase(allowUav, allowSimul, true);
    texDesc.Flags &= ~D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ThrowIfFailed(device->device->CreateReservedResource(
        &texDesc,
        GetInitState(),
        nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())));
    D3D12_SUBRESOURCE_TILING tilingInfo;
    uint subresourceCount = 1;
    uint numTiles;
    device->device->GetResourceTiling(
        resource.Get(),
        &numTiles,
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
    tileSize = uint3(tilingInfo.WidthInTiles, tilingInfo.HeightInTiles, tilingInfo.DepthInTiles);
}
uint3 SparseTexture::TilingSize() const {
    return uint3(width / tileSize.x, height / tileSize.y, depth / tileSize.z);
}
SparseTexture::~SparseTexture() {
    auto &globalHeap = *device->globalHeap.get();
    for (auto &&i : uavIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
    for (auto &&i : srvIdcs) {
        globalHeap.ReturnIndex(i.second);
    }
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
    uint tileCount = size.x * size.y * size.z;
    Tile tile;
    tile.mipLevel = mipLevel;
    tile.coords[0] = coord[0];
    tile.coords[1] = coord[1];
    tile.coords[2] = coord[2];
    std::lock_guard lck{allocMtx};
    auto iter = allocatedTiles.try_emplace(
        tile,
        vstd::lazy_eval([&] {
            TileInfo tileInfo;
            tileInfo.size[0] = size[0];
            tileInfo.size[1] = size[1];
            tileInfo.size[2] = size[2];
            tileInfo.offsets.push_back_uninitialized(tileCount);
            return tileInfo;
        }));
    auto &tileInfo = iter.first->second;
    auto endTile = coord + size;
    uint tileIndex = 0;
    D3D12_TILE_REGION_SIZE region{
        .NumTiles = 1,
        .UseBox = true,
        .Width = 1,
        .Height = 1,
        .Depth = 1};
    auto toDim = [&](uint d) -> D3D12_TILED_RESOURCE_COORDINATE {
        D3D12_TILED_RESOURCE_COORDINATE c;
        c.Subresource = mipLevel;
        auto sizexy = (size.y * size.x);
        c.Z = d / sizexy;
        c.Y = (d - (c.Z * sizexy)) / size.x;
        c.X = d % size.x;
        c.X += coord.x;
        c.Y += coord.y;
        c.Z += coord.z;
        return c;
    };
    auto allocateFunc = [&](ID3D12Heap *heap, vstd::span<const uint> offsets) {
        bytes.clear();
        bytes.push_back_uninitialized(
            sizeof(D3D12_TILED_RESOURCE_COORDINATE) * offsets.size() +
            sizeof(D3D12_TILE_REGION_SIZE) * offsets.size() +
            sizeof(D3D12_TILE_RANGE_FLAGS) * offsets.size() +
            sizeof(uint) * offsets.size());
        auto coords = reinterpret_cast<D3D12_TILED_RESOURCE_COORDINATE *>(bytes.data());
        auto flags = reinterpret_cast<D3D12_TILE_RANGE_FLAGS *>(coords + offsets.size());
        auto regions = reinterpret_cast<D3D12_TILE_REGION_SIZE *>(flags + offsets.size());
        auto rangeTiles = reinterpret_cast<uint *>(regions + offsets.size());
        tileInfo.heaps.emplace_back(AllocateOffsets{
            .heap = heap,
            .offset = tileIndex,
            .size = static_cast<uint>(offsets.size())});
        for (auto i : vstd::range(offsets.size())) {
            flags[i] = D3D12_TILE_RANGE_FLAG_NONE;
            rangeTiles[i] = 1;
            regions[i] = region;
            auto &c = coords[i];
            c = toDim(tileIndex + i);
            tileInfo.offsets[tileIndex + i] = offsets[i];
        }
        queue->UpdateTileMappings(
            resource.Get(), offsets.size(),
            coords,
            regions,
            heap, offsets.size(),
            flags,
            offsets.data(),
            rangeTiles,
            D3D12_TILE_MAPPING_FLAG_NONE);
        tileIndex += offsets.size();
    };
    sparseAllocator.AllocateTiles(tileCount, allocateFunc);
    // D3D12_TILED_RESOURCE_COORDINATE tileCoord{
    //     .X = coord.x,
    //     .Y = coord.y,
    //     .Z = coord.z,
    //     .Subresource = mipLevel};
    // D3D12_TILE_REGION_SIZE tileSize{
    //     .NumTiles = size.x * size.y * size.z,
    //     .UseBox = true,
    //     .Width = size.x,
    //     .Height = static_cast<uint16_t>(size.y),
    //     .Depth = static_cast<uint16_t>(size.z)};
    // uint rangeTileCount = tileSize.NumTiles;
    // offsetTile = offset / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    // queue->UpdateTileMappings(
    //     resource.Get(), 1,
    //     &tileCoord,
    //     &tileSize,
    //     heap, 1,
    //     vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
    //     &offsetTile,
    //     &rangeTileCount,
    //     D3D12_TILE_MAPPING_FLAG_NONE);
}
void SparseTexture::FreeTileMemory(ID3D12CommandQueue *queue, uint3 coord, uint mipLevel) const {
    Tile tile;
    tile.mipLevel = mipLevel;
    for (auto i : vstd::range(3)) {
        tile.coords[i] = coord[i];
    }
    std::lock_guard lck{allocMtx};
    auto iter = allocatedTiles.find(tile);
    if (iter == allocatedTiles.end()) [[unlikely]] {
        return;
    }
    auto &tileInfo = iter->second;
    for (auto &i : tileInfo.heaps) {
        sparseAllocator.Deallocate(i.heap, {tileInfo.offsets.data() + i.offset, i.size});
    }
    allocatedTiles.erase(iter);
}
void SparseTexture::ClearTile(ID3D12CommandQueue *queue) const {
    {
        std::lock_guard lck{allocMtx};
        allocatedTiles.clear();
        sparseAllocator.ClearAllocate();
    }
    D3D12_TILE_RANGE_FLAGS RangeFlags = D3D12_TILE_RANGE_FLAG_NULL;
    queue->UpdateTileMappings(resource.Get(), 1, nullptr, nullptr, nullptr, 1, &RangeFlags, nullptr, nullptr, D3D12_TILE_MAPPING_FLAG_NONE);
}
}// namespace lc::dx

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
    if (lastMipSize.x < tilingInfo.WidthInTiles || lastMipSize.y < tilingInfo.HeightInTiles || (dimension == TextureDimension::Tex3D && lastMipSize.z < tilingInfo.DepthInTiles)) [[unlikely]] {
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
    resource->Release();
    auto alloc = device->defaultAllocator.get();
    for (auto &&i : allocatedTiles) {
        alloc->Release(i.second.allocation);
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
            ID3D12Heap *heap;
            uint64 offset;
            D3D12_TILED_RESOURCE_COORDINATE tileCoord{
                .X = coord.x,
                .Y = coord.y,
                .Z = coord.z,
                .Subresource = mipLevel};
            D3D12_TILE_REGION_SIZE tileSize{
                .NumTiles = size.x * size.y * size.z,
                .UseBox = true,
                .Width = size.x,
                .Height = static_cast<uint16_t>(size.y),
                .Depth = static_cast<uint16_t>(size.z)};
            tileInfo.allocation = device->defaultAllocator->AllocateTextureHeap(device, tileSize.NumTiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, &heap, &offset, true);
            uint rangeTileCount = tileSize.NumTiles;
            uint offsetTile = offset / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
            queue->UpdateTileMappings(
                resource.Get(), 1,
                &tileCoord,
                &tileSize,
                heap, 1,
                vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
                &offsetTile,
                &rangeTileCount,
                D3D12_TILE_MAPPING_FLAG_NONE);
            return tileInfo;
        }));
}
void SparseTexture::DeAllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint mipLevel, vstd::vector<uint64> &destroyList) const {
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
    destroyList.emplace_back(tileInfo.allocation);
    auto size = uint3(tileInfo.size[0], tileInfo.size[1], tileInfo.size[2]);
    D3D12_TILED_RESOURCE_COORDINATE tileCoord{
        .X = coord.x,
        .Y = coord.y,
        .Z = coord.z,
        .Subresource = mipLevel};
    D3D12_TILE_REGION_SIZE tileSize{
        .NumTiles = size.x * size.y * size.z,
        .UseBox = true,
        .Width = size.x,
        .Height = static_cast<uint16_t>(size.y),
        .Depth = static_cast<uint16_t>(size.z)};
    queue->UpdateTileMappings(
        resource.Get(), 1, &tileCoord,
        &tileSize,
        nullptr, 1,
        vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NULL),
        nullptr,
        nullptr,
        D3D12_TILE_MAPPING_FLAG_NONE);
    allocatedTiles.erase(iter);
}
}// namespace lc::dx

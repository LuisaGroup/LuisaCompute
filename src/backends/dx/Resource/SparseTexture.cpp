#include <Resource/SparseTexture.h>
#include <Resource/DescriptorHeap.h>
#include <luisa/core/logging.h>
#include "SparseHeap.h"
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
    : TextureBase(device, width, height, format, dimension, depth, mip, GetInitState()) {
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
}
D3D12_SHADER_RESOURCE_VIEW_DESC SparseTexture::GetColorSrvDesc(uint mipOffset) const {
    return GetColorSrvDescBase(mipOffset);
}
D3D12_UNORDERED_ACCESS_VIEW_DESC SparseTexture::GetColorUavDesc(uint targetMipLevel) const {
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
void SparseTexture::AllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint3 size, uint mipLevel, uint64 alloc) const {
    auto heap = reinterpret_cast<SparseHeap const *>(alloc);
    if (heap->size_bytes < (size.x * size.y * size.z) * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) [[unlikely]] {
        LUISA_ERROR("Map size out of range. Required size: {}, heap size: {}", (size.x * size.y * size.z) * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, heap->size_bytes);
    }
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
    uint rangeTileCount = tileSize.NumTiles;
    uint offsetTile = heap->offset / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    queue->UpdateTileMappings(
        resource.Get(), 1,
        &tileCoord,
        &tileSize,
        heap->heap, 1,
        vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NONE),
        &offsetTile,
        &rangeTileCount,
        D3D12_TILE_MAPPING_FLAG_NONE);
}
void SparseTexture::DeAllocateTile(ID3D12CommandQueue *queue, uint3 coord, uint3 size, uint mipLevel) const {
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
}
}// namespace lc::dx

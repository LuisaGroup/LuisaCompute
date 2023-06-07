#include <runtime/sparse_texture.h>
#include <runtime/rhi/device_interface.h>

namespace luisa::compute {

void SparseTexture::UpdateTiles::operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept {
    device->update_sparse_texture(stream_handle, handle, std::move(operations));
}

SparseTexture::SparseTexture(DeviceInterface *device, const SparseTextureCreationInfo &info) noexcept
    : Resource{device, Tag::SPARSE_TEXTURE, info},
      _tile_size_bytes{info.tile_size_bytes},
      _tile_size{info.tile_size} {
}

SparseTexture::UpdateTiles SparseTexture::update() noexcept {
    return {handle(), std::move(_operations)};
}

SparseTexture::~SparseTexture() noexcept {
    if (*this) { device()->destroy_sparse_texture(handle()); }
}

}// namespace luisa::compute

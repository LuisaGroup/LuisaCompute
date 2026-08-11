#include "simd_texture.h"

#include <cstring>

namespace luisa::compute::simd {

SIMDTexture::SIMDTexture(
    PixelStorage storage, uint dimension, uint3 size,
    uint mip_levels) noexcept
    : _texture{storage, dimension, size, mip_levels},
      _dimension{dimension} {}

SIMDTexture::SIMDTexture(
    PixelStorage storage, uint dimension, uint3 size,
    uint mip_levels, std::byte *external_memory) noexcept
    : _texture{storage, dimension, size, mip_levels,
               external_memory},
      _dimension{dimension} {}

void SIMDTexture::_read_float(
    void *texture, uint32_t level,
    uint32_t x, uint32_t y, uint32_t z, void *value) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    auto view = self->view(level);
    auto pixel = self->dimension() == 2u ?
        view.read2d<float>(make_uint2(x, y)) :
        view.read3d<float>(make_uint3(x, y, z));
    std::memcpy(value, &pixel, sizeof(pixel));
}

void SIMDTexture::_read_uint(
    void *texture, uint32_t level,
    uint32_t x, uint32_t y, uint32_t z, void *value) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    auto view = self->view(level);
    auto pixel = self->dimension() == 2u ?
        view.read2d<uint>(make_uint2(x, y)) :
        view.read3d<uint>(make_uint3(x, y, z));
    std::memcpy(value, &pixel, sizeof(pixel));
}

void SIMDTexture::_write_float(
    void *texture, uint32_t level,
    uint32_t x, uint32_t y, uint32_t z,
    const void *value) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    auto view = self->view(level);
    float4 pixel{};
    std::memcpy(&pixel, value, sizeof(pixel));
    if (self->dimension() == 2u) {
        view.write2d(make_uint2(x, y), pixel);
    } else {
        view.write3d(make_uint3(x, y, z), pixel);
    }
}

void SIMDTexture::_write_uint(
    void *texture, uint32_t level,
    uint32_t x, uint32_t y, uint32_t z,
    const void *value) noexcept {
    auto *self = static_cast<SIMDTexture *>(texture);
    auto view = self->view(level);
    uint4 pixel{};
    std::memcpy(&pixel, value, sizeof(pixel));
    if (self->dimension() == 2u) {
        view.write2d(make_uint2(x, y), pixel);
    } else {
        view.write3d(make_uint3(x, y, z), pixel);
    }
}

uint32_t SIMDTexture::_size(
    void *texture, uint32_t level, uint32_t axis) noexcept {
    auto size = static_cast<SIMDTexture *>(texture)->view(level).size3d();
    return axis < 3u ? size[axis] : 0u;
}

SIMDHostTextureView SIMDTexture::host_view(uint level) noexcept {
    return {
        .texture = this,
        .read_float = _read_float,
        .read_uint = _read_uint,
        .write_float = _write_float,
        .write_uint = _write_uint,
        .size = _size,
        .level = level,
        .dimension = _dimension,
    };
}

}// namespace luisa::compute::simd

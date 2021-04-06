//
// Created by Mike Smith on 2021/3/29.
//

#include <runtime/device.h>
#include <dsl/texture.h>

namespace luisa::compute::dsl {

namespace detail {

[[nodiscard]] auto valid_mipmap_levels(uint width, uint height, uint requested_levels) noexcept {
    auto rounded_size = next_pow2(std::min(width, height));
    auto max_levels = static_cast<uint>(std::log2(rounded_size));
    return requested_levels == 0u
               ? max_levels
               : std::min(requested_levels, max_levels);
}

}// namespace detail

Texture::Texture(Device &device, PixelFormat format, uint width, uint height, uint mipmap_levels) noexcept
    : _device{&device},
      _handle{device.create_texture(
          format, 2, width, height, 1,
          detail::valid_mipmap_levels(width, height, mipmap_levels),
          false)},
      _format{format},
      _width{width},
      _height{height},
      _mipmap_levels{detail::valid_mipmap_levels(width, height, mipmap_levels)} {}

Texture::~Texture() noexcept {
    if (_device != nullptr) {
        _device->dispose_texture(_handle);
    }
}

Texture::Texture(Texture &&another) noexcept
    : _device{another._device},
      _handle{another._handle},
      _format{another._format},
      _width{another._width},
      _height{another._height},
      _mipmap_levels{another._mipmap_levels} { another._device = nullptr; }

Texture &Texture::operator=(Texture &&rhs) noexcept {
    if (&rhs != this) {
        _device->dispose_texture(_handle);
        _device = rhs._device;
        _handle = rhs._handle;
        _format = rhs._format;
        _width = rhs._width;
        _height = rhs._height;
        _mipmap_levels = rhs._mipmap_levels;
        rhs._device = nullptr;
    }
    return *this;
}

}// namespace luisa::compute::dsl

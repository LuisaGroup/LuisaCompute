//
// Created by Mike Smith on 2021/3/29.
//

#include <runtime/texture.h>
#include <runtime/device.h>

namespace luisa::compute {

Texture2D::Texture2D(Device *device, PixelFormat format, uint width, uint height, uint mipmap_levels) noexcept
    : _device{device},
      _handle{device->_create_texture(format, 2, width, height, 1, mipmap_levels)},
      _format{format},
      _width{width},
      _height{height},
      _mipmap_levels{mipmap_levels} {}

Texture2D::~Texture2D() noexcept {
    if (_device != nullptr) {
        _device->_dispose_texture(_handle);
    }
}

Texture2D::Texture2D(Texture2D &&another) noexcept
    : _device{another._device},
      _handle{another._handle},
      _format{another._format},
      _width{another._width},
      _height{another._height},
      _mipmap_levels{another._mipmap_levels} { another._device = nullptr; }

Texture2D &Texture2D::operator=(Texture2D &&rhs) noexcept {
    if (&rhs != this) {
        _device->_dispose_texture(_handle);
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

}// namespace luisa::compute

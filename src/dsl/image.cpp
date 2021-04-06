//
// Created by Mike Smith on 2021/3/29.
//

#include "ast/function_builder.h"
#include <dsl/image.h>

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

Image::Image(Device &device, PixelFormat format, uint2 size) noexcept
    : _device{&device},
      _handle{device.create_texture(
          format, 2u,
          size.x, size.y, 1u,
          1u, false)},
      _size{size},
      _format{format} {}

Image::~Image() noexcept {
    if (_device != nullptr) {
        _device->dispose_texture(_handle);
    }
}

Image::Image(Image &&another) noexcept
    : _device{another._device},
      _handle{another._handle},
      _size{another._size},
      _format{another._format} { another._device = nullptr; }

Image &Image::operator=(Image &&rhs) noexcept {
    if (&rhs != this) {
        _device->dispose_texture(_handle);
        _device = rhs._device;
        _handle = rhs._handle;
        _size = rhs._size;
        _format = rhs._format;
        rhs._device = nullptr;
    }
    return *this;
}

ImageView Image::view() const noexcept {
    return ImageView{_handle, _format, _size};
}

detail::ImageAccess ImageView::operator[](detail::Expr<uint2> uv) const noexcept {
    auto self = _expression ? _expression : FunctionBuilder::current()->image_binding(_handle);
    return {self, uv.expression()};
}

}// namespace luisa::compute::dsl

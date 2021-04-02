//
// Created by Mike Smith on 2020/12/2.
//

#include <core/mathematics.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/texture.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return Stream{this, _create_stream()};
}

const Context &Device::context() const noexcept { return _ctx; }

namespace detail {

[[nodiscard]] auto max_mipmap_levels(uint width, uint height) noexcept {
    auto rounded_size = next_pow2(std::min(width, height));
    return static_cast<uint>(std::log2(rounded_size));
}

}// namespace detail

Texture Device::create_texture(
    PixelFormat format,
    uint width, uint height,
    uint mipmap_levels) noexcept {

    auto max_mipmap_level = detail::max_mipmap_levels(width, height);
    auto valid_mipmap_levels = mipmap_levels == 0u
                                   ? max_mipmap_level
                                   : std::min(mipmap_levels, max_mipmap_level);
    return {this, format, width, height, valid_mipmap_levels};
}

}// namespace luisa::compute

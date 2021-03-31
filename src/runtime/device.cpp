//
// Created by Mike Smith on 2020/12/2.
//

#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/texture.h>

namespace luisa::compute {

Stream Device::create_stream() noexcept {
    return Stream{this, _create_stream()};
}

const Context &Device::context() const noexcept { return _ctx; }

Texture Device::create_texture(
    PixelFormat format,
    uint width, uint height,
    uint mipmap_levels) noexcept {
    
    return {this, format, width, height, mipmap_levels};
}

}// namespace luisa::compute

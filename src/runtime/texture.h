//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/concepts.h>
#include <runtime/pixel_format.h>

namespace luisa::compute {

class Device;

class Texture : concepts::Noncopyable {

private:
    Device *_device;
    uint64_t _handle;
    PixelFormat _format;
    uint _width;
    uint _height;
    uint _mipmap_levels;

private:
    friend class Device;
    Texture(Device *device, PixelFormat format, uint width, uint height, uint mipmap_levels) noexcept;

public:
    Texture(Texture &&another) noexcept;
    Texture &operator=(Texture &&rhs) noexcept;
    ~Texture() noexcept;
};

}// namespace luisa::compute

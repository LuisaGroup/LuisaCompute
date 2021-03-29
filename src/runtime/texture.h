//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/concepts.h>
#include <runtime/pixel_format.h>

namespace luisa::compute {

class Device;

class Texture2D : concepts::Noncopyable {

public:

private:
    Device *_device;
    uint64_t _handle;
    PixelFormat _format;
    uint _width;
    uint _height;
    uint _mipmap_levels;

private:
    friend class Device;
    Texture2D(Device *device, PixelFormat format, uint width, uint height, uint mipmap_levels) noexcept;

public:
    Texture2D(Texture2D &&another) noexcept;
    Texture2D &operator=(Texture2D &&rhs) noexcept;
    ~Texture2D() noexcept;
};

}// namespace luisa::compute

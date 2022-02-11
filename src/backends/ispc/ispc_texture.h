//
// Created by Mike Smith on 2022/2/11.
//

#pragma once

#include <runtime/pixel.h>

namespace luisa::compute::ispc {

// TODO
class ISPCTexture {

public:
    struct Handle {
    };

public:
    ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept;
    [[nodiscard]] Handle handle() const noexcept;
};

}// namespace luisa::compute::ispc

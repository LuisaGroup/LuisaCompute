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
        const void *ptr;// TODO
    };

public:
    ISPCTexture(PixelFormat format, uint dim, uint3 size, uint mip_levels) noexcept;
    [[nodiscard]] Handle handle() const noexcept;
};

}// namespace luisa::compute::ispc

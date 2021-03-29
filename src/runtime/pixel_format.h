//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <cstdint>

namespace luisa::compute {

enum struct PixelFormat : uint32_t {
    R8U,
    R8U_SRGB,
    RG8U,
    RG8U_SRGB,
    RGBA8U,
    RGBA8U_SRGB,
    R32F,
    RG32F,
    RGBA32F,
};

}

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

/*
enum struct PixelFormat : uint32_t {
    RGBA8U_SRGB,

    R32F,
    RG32F,
    RGBA32F,

    R32UInt,
    RG32UInt,
    RGBA32UInt,

    R32Int,
    RG32Int,
    RGBA32Int,

    R16F,
    RG16F,
    RGBA16F,

    R16UInt,
    RG16UInt,
    RGBA16UInt,

    R16UNorm,
    RG16UNorm,
    RGBA16UNorm,

    R16SNorm,
    RG16SNorm,
    RGBA16SNorm,
    R16Int,
    RG16Int,
    RGBA16Int,

    R8UInt,
    RG8UInt,
    RGBA8UInt,

    R8SNorm,
    RG8SNorm,
    RGBA8SNorm,

    R8Int,
    RG8Int,
    RGBA8Int,
};
*/

constexpr auto uchar_linear = PixelFormat::R8U;
constexpr auto uchar_sRGB = PixelFormat::R8U_SRGB;
constexpr auto uchar2_linear = PixelFormat::RG8U;
constexpr auto uchar2_sRGB = PixelFormat::RG8U_SRGB;
constexpr auto uchar4_linear = PixelFormat::RGBA8U;
constexpr auto uchar4_sRGB = PixelFormat::RGBA8U_SRGB;
constexpr auto float_linear = PixelFormat::R32F;
constexpr auto float2_linear = PixelFormat::RG32F;
constexpr auto float4_linear = PixelFormat::RGBA32F;

[[nodiscard]] constexpr auto pixel_format_size(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8U:
        case PixelFormat::R8U_SRGB:
            return sizeof(uint8_t);
        case PixelFormat::RG8U:
        case PixelFormat::RG8U_SRGB:
            return sizeof(uint8_t) * 2u;
        case PixelFormat::RGBA8U:
        case PixelFormat::RGBA8U_SRGB:
            return sizeof(uint8_t) * 4u;
        case PixelFormat::R32F:
            return sizeof(float);
        case PixelFormat::RG32F:
            return sizeof(float) * 2u;
        case PixelFormat::RGBA32F:
            return sizeof(float) * 4u;
    }
    return static_cast<size_t>(0u);
}

[[nodiscard]] constexpr auto pixel_format_is_srgb(PixelFormat format) noexcept {
    return format == PixelFormat::R8U_SRGB
           || format == PixelFormat::RG8U_SRGB
           || format == PixelFormat::RGBA8U_SRGB;
}

}// namespace luisa::compute

//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/basic_types.h>

namespace luisa::compute {

enum struct PixelStorage : uint32_t {

    BYTE1,
    BYTE2,
    BYTE4,

    SHORT1,
    SHORT2,
    SHORT4,

    INT1,
    INT2,
    INT4,

    HALF1,
    HALF2,
    HALF4,

    FLOAT1,
    FLOAT2,
    FLOAT4,

    NUM
};

enum struct PixelFormat : uint32_t {

    R8SInt,
    R8UInt,
    R8UNorm,

    RG8SInt,
    RG8UInt,
    RG8UNorm,

    RGBA8SInt,
    RGBA8UInt,
    RGBA8UNorm,

    R16SInt,
    R16UInt,
    R16UNorm,

    RG16SInt,
    RG16UInt,
    RG16UNorm,

    RGBA16SInt,
    RGBA16UInt,
    RGBA16UNorm,

    R32SInt,
    R32UInt,

    RG32SInt,
    RG32UInt,

    RGBA32SInt,
    RGBA32UInt,

    R16F,
    RG16F,
    RGBA16F,

    R32F,
    RG32F,
    RGBA32F,

    NUM
};

[[nodiscard]] constexpr auto pixel_format_to_storage(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt:
        case PixelFormat::R8UInt:
        case PixelFormat::R8UNorm:
            return PixelStorage::BYTE1;
        case PixelFormat::RG8SInt:
        case PixelFormat::RG8UInt:
        case PixelFormat::RG8UNorm:
            return PixelStorage::BYTE2;
        case PixelFormat::RGBA8SInt:
        case PixelFormat::RGBA8UInt:
        case PixelFormat::RGBA8UNorm:
            return PixelStorage::BYTE4;
        case PixelFormat::R16SInt:
        case PixelFormat::R16UInt:
        case PixelFormat::R16UNorm:
            return PixelStorage::SHORT1;
        case PixelFormat::RG16SInt:
        case PixelFormat::RG16UInt:
        case PixelFormat::RG16UNorm:
            return PixelStorage::SHORT2;
        case PixelFormat::RGBA16SInt:
        case PixelFormat::RGBA16UInt:
        case PixelFormat::RGBA16UNorm:
            return PixelStorage::SHORT4;
        case PixelFormat::R32SInt:
        case PixelFormat::R32UInt:
            return PixelStorage::INT1;
        case PixelFormat::RG32SInt:
        case PixelFormat::RG32UInt:
            return PixelStorage::INT2;
        case PixelFormat::RGBA32SInt:
        case PixelFormat::RGBA32UInt:
            return PixelStorage::INT4;
        case PixelFormat::R16F:
            return PixelStorage::HALF1;
        case PixelFormat::RG16F:
            return PixelStorage::HALF2;
        case PixelFormat::RGBA16F:
            return PixelStorage::HALF4;
        case PixelFormat::R32F:
            return PixelStorage::FLOAT1;
        case PixelFormat::RG32F:
            return PixelStorage::FLOAT2;
        case PixelFormat::RGBA32F:
            return PixelStorage::FLOAT4;
    }
    return PixelStorage{};
}

[[nodiscard]] constexpr auto pixel_storage_size(PixelStorage storage) noexcept {
    switch (storage) {
        case PixelStorage::BYTE1: return sizeof(std::byte) * 1u;
        case PixelStorage::BYTE2: return sizeof(std::byte) * 2u;
        case PixelStorage::BYTE4: return sizeof(std::byte) * 4u;
        case PixelStorage::SHORT1: return sizeof(short) * 1u;
        case PixelStorage::SHORT2: return sizeof(short) * 2u;
        case PixelStorage::SHORT4: return sizeof(short) * 4u;
        case PixelStorage::INT1: return sizeof(int) * 1u;
        case PixelStorage::INT2: return sizeof(int) * 2u;
        case PixelStorage::INT4: return sizeof(int) * 4u;
        case PixelStorage::HALF1: return sizeof(short) * 1u;
        case PixelStorage::HALF2: return sizeof(short) * 2u;
        case PixelStorage::HALF4: return sizeof(short) * 4u;
        case PixelStorage::FLOAT1: return sizeof(float) * 1u;
        case PixelStorage::FLOAT2: return sizeof(float) * 2u;
        case PixelStorage::FLOAT4: return sizeof(float) * 4u;
    }
    return static_cast<size_t>(0u);
}

template<typename T>
[[nodiscard]] constexpr auto pixel_storage_to_format(PixelStorage storage) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        switch (storage) {
            case PixelStorage::BYTE1: return PixelFormat::R8UNorm;
            case PixelStorage::BYTE2: return PixelFormat::RG8UNorm;
            case PixelStorage::BYTE4: return PixelFormat::RGBA8UNorm;
            case PixelStorage::SHORT1: return PixelFormat::R16UNorm;
            case PixelStorage::SHORT2: return PixelFormat::RG16UNorm;
            case PixelStorage::SHORT4: return PixelFormat::RGBA16UNorm;
            case PixelStorage::HALF1: return PixelFormat::R16F;
            case PixelStorage::HALF2: return PixelFormat::RG16F;
            case PixelStorage::HALF4: return PixelFormat::RGBA16F;
            case PixelStorage::FLOAT1: return PixelFormat::R32F;
            case PixelStorage::FLOAT2: return PixelFormat::RG32F;
            case PixelStorage::FLOAT4: return PixelFormat::RGBA32F;
            default: LUISA_ERROR_WITH_LOCATION("Invalid pixel storage for float format.");
        }
    } else if constexpr (std::is_same_v<T, int>) {
        switch (storage) {
            case PixelStorage::BYTE1: return PixelFormat::R8SInt;
            case PixelStorage::BYTE2: return PixelFormat::RG8SInt;
            case PixelStorage::BYTE4: return PixelFormat::RGBA8SInt;
            case PixelStorage::SHORT1: return PixelFormat::R16SInt;
            case PixelStorage::SHORT2: return PixelFormat::RG16SInt;
            case PixelStorage::SHORT4: return PixelFormat::RGBA16SInt;
            case PixelStorage::INT1: return PixelFormat::R32SInt;
            case PixelStorage::INT2: return PixelFormat::RG32SInt;
            case PixelStorage::INT4: return PixelFormat::RGBA32SInt;
            default: LUISA_ERROR_WITH_LOCATION("Invalid pixel storage for int format.");
        }
    } else if constexpr (std::is_same_v<T, uint>) {
        switch (storage) {
            case PixelStorage::BYTE1: return PixelFormat::R8UInt;
            case PixelStorage::BYTE2: return PixelFormat::RG8UInt;
            case PixelStorage::BYTE4: return PixelFormat::RGBA8UInt;
            case PixelStorage::SHORT1: return PixelFormat::R16UInt;
            case PixelStorage::SHORT2: return PixelFormat::RG16UInt;
            case PixelStorage::SHORT4: return PixelFormat::RGBA16UInt;
            case PixelStorage::INT1: return PixelFormat::R32UInt;
            case PixelStorage::INT2: return PixelFormat::RG32UInt;
            case PixelStorage::INT4: return PixelFormat::RGBA32UInt;
            default: LUISA_ERROR_WITH_LOCATION("Invalid pixel storage for uint format.");
        }
    } else {
        static_assert(always_false_v<T>);
    }
    return PixelFormat{};
}

[[nodiscard]] constexpr auto pixel_format_size(PixelFormat format) noexcept {
    return pixel_storage_size(pixel_format_to_storage(format));
}

}// namespace luisa::compute

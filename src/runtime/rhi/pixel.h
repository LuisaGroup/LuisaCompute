//
// Created by Mike Smith on 2021/3/29.
//

#pragma once

#include <core/dll_export.h>
#include <core/basic_types.h>

namespace luisa::compute {

namespace detail {
[[noreturn]] LC_RUNTIME_API void error_pixel_invalid_format(const char *name) noexcept;
}

enum struct PixelStorage : uint8_t {

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

    BC4,
    BC5,
    BC6,
    BC7,
    //TODO: ASTC
};

enum struct PixelFormat : uint8_t {

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

    BC4UNorm,
    BC5UNorm,
    BC6HUF16,
    BC7UNorm,

    //TODO: ASTC
};

constexpr auto pixel_storage_count = to_underlying(PixelStorage::BC7) + 1u;
constexpr auto pixel_format_count = to_underlying(PixelFormat::BC7UNorm) + 1u;

[[nodiscard]] constexpr auto is_block_compressed(PixelStorage s) noexcept {
    return s == PixelStorage::BC4 || s == PixelStorage::BC5 ||
           s == PixelStorage::BC6 || s == PixelStorage::BC7;
}

[[nodiscard]] constexpr auto is_block_compressed(PixelFormat f) noexcept {
    return f == PixelFormat::BC4UNorm || f == PixelFormat::BC5UNorm ||
           f == PixelFormat::BC6HUF16 || f == PixelFormat::BC7UNorm;
}

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
        case PixelFormat::BC6HUF16:
            return PixelStorage::BC6;
        case PixelFormat::BC7UNorm:
            return PixelStorage::BC7;
        case PixelFormat::BC5UNorm:
            return PixelStorage::BC5;
        case PixelFormat::BC4UNorm:
            return PixelStorage::BC4;
        default:
            break;
    }
    return PixelStorage{};
}

[[nodiscard]] constexpr size_t pixel_storage_size(PixelStorage storage, uint3 size) noexcept {
    if (is_block_compressed(storage)) {
        auto block_width = (size.x + 3u) / 4u;
        auto block_height = (size.y + 3u) / 4u;
        auto block_count = block_width * block_height * std::max(size.z, 1u);
        switch (storage) {
            case PixelStorage::BC4: return block_count * 8u;
            case PixelStorage::BC5: return block_count * 16u;
            case PixelStorage::BC6: return block_count * 16u;
            case PixelStorage::BC7: return block_count * 16u;
            default: break;
        }
        detail::error_pixel_invalid_format("unknown.");
    }
    auto pixel_count = size.x * size.y * size.z;
    switch (storage) {
        case PixelStorage::BYTE1: return pixel_count * sizeof(std::byte) * 1u;
        case PixelStorage::BYTE2: return pixel_count * sizeof(std::byte) * 2u;
        case PixelStorage::BYTE4: return pixel_count * sizeof(std::byte) * 4u;
        case PixelStorage::SHORT1: return pixel_count * sizeof(short) * 1u;
        case PixelStorage::SHORT2: return pixel_count * sizeof(short) * 2u;
        case PixelStorage::SHORT4: return pixel_count * sizeof(short) * 4u;
        case PixelStorage::INT1: return pixel_count * sizeof(int) * 1u;
        case PixelStorage::INT2: return pixel_count * sizeof(int) * 2u;
        case PixelStorage::INT4: return pixel_count * sizeof(int) * 4u;
        case PixelStorage::HALF1: return pixel_count * sizeof(short) * 1u;
        case PixelStorage::HALF2: return pixel_count * sizeof(short) * 2u;
        case PixelStorage::HALF4: return pixel_count * sizeof(short) * 4u;
        case PixelStorage::FLOAT1: return pixel_count * sizeof(float) * 1u;
        case PixelStorage::FLOAT2: return pixel_count * sizeof(float) * 2u;
        case PixelStorage::FLOAT4: return pixel_count * sizeof(float) * 4u;
        default: break;
    }
    detail::error_pixel_invalid_format("unknown");
}

[[nodiscard]] constexpr auto pixel_storage_channel_count(PixelStorage storage) noexcept {
    switch (storage) {
        case PixelStorage::BYTE1: return 1u;
        case PixelStorage::BYTE2: return 2u;
        case PixelStorage::BYTE4: return 4u;
        case PixelStorage::SHORT1: return 1u;
        case PixelStorage::SHORT2: return 2u;
        case PixelStorage::SHORT4: return 4u;
        case PixelStorage::INT1: return 1u;
        case PixelStorage::INT2: return 2u;
        case PixelStorage::INT4: return 4u;
        case PixelStorage::HALF1: return 1u;
        case PixelStorage::HALF2: return 2u;
        case PixelStorage::HALF4: return 4u;
        case PixelStorage::FLOAT1: return 1u;
        case PixelStorage::FLOAT2: return 2u;
        case PixelStorage::FLOAT4: return 4u;
        case PixelStorage::BC4: return 1u;
        case PixelStorage::BC5: return 2u;
        case PixelStorage::BC6: return 3u;
        case PixelStorage::BC7: return 4u;
        default: break;
    }
    return 0u;
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
            case PixelStorage::BC4: return PixelFormat ::BC4UNorm;
            case PixelStorage::BC5: return PixelFormat ::BC5UNorm;
            case PixelStorage::BC6: return PixelFormat ::BC6HUF16;
            case PixelStorage::BC7: return PixelFormat ::BC7UNorm;
            default: detail::error_pixel_invalid_format("float");
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
            default: detail::error_pixel_invalid_format("int");
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
            default: detail::error_pixel_invalid_format("uint");
        }
    } else {
        static_assert(always_false_v<T>);
    }
    // unreachable
    return PixelFormat{};
}

[[nodiscard]] constexpr auto pixel_format_size(PixelFormat format, uint3 size) noexcept {
    return pixel_storage_size(pixel_format_to_storage(format), size);
}

[[nodiscard]] constexpr auto pixel_format_channel_count(PixelFormat format) noexcept {
    return pixel_storage_channel_count(pixel_format_to_storage(format));
}

}// namespace luisa::compute

#pragma once

#include <algorithm>
#include <limits>

#include <luisa/core/dll_export.h>
#include <luisa/core/basic_types.h>

// WORKAROUND: Windows SDK corecrt_math.h defines OVERFLOW as a macro
// (expanded to the constant _OVERFLOW == 3), which conflicts with
// PixelStorageSizeStatus::OVERFLOW. Undefine it before the enum.
#ifdef OVERFLOW
#undef OVERFLOW
#endif

namespace luisa::compute {

namespace detail {
[[noreturn]] LUISA_RUNTIME_API void error_pixel_invalid_format(const char *name) noexcept;
[[noreturn]] LUISA_RUNTIME_API void error_pixel_size_overflow(
    uint32_t storage, uint3 size) noexcept;
}// namespace detail

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

    R10G10B10A2,
    R11G11B10,

    BC1,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6,
    BC7,
    BC7_SRGB,
    BYTE4_SRGB,
    // ASTC block formats (all blocks are 16 bytes; block dimensions differ).
    // ASTC is mandatory on Android GPUs, which generally do not support BC.
    ASTC_4x4,
    ASTC_5x5,
    ASTC_6x6,
    ASTC_8x8,
    ASTC_10x10,
    ASTC_12x12,
    ASTC_4x4_SRGB,
    ASTC_5x5_SRGB,
    ASTC_6x6_SRGB,
    ASTC_8x8_SRGB,
    ASTC_10x10_SRGB,
    ASTC_12x12_SRGB,
};

enum struct PixelStorageSizeStatus : uint8_t {
    SUCCESS,
    INVALID_STORAGE,
    OVERFLOW
};

struct PixelStorageSizeResult {
    PixelStorageSizeStatus status;
    size_t size;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == PixelStorageSizeStatus::SUCCESS;
    }
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

    R10G10B10A2UInt,
    R10G10B10A2UNorm,
    R11G11B10F,

    BC1UNorm,
    BC2UNorm,
    BC3UNorm,
    BC4UNorm,
    BC5UNorm,
    BC6HUF16,
    BC7UNorm,
    BC7SRGB,
    RGBA8SRGB,

    // ASTC block formats, mandatory on Android GPUs. Both UNORM and SRGB
    // variants are exposed so ASTC textures can be requested through the DSL.
    ASTC_4x4,
    ASTC_5x5,
    ASTC_6x6,
    ASTC_8x8,
    ASTC_10x10,
    ASTC_12x12,
    ASTC_4x4_SRGB,
    ASTC_5x5_SRGB,
    ASTC_6x6_SRGB,
    ASTC_8x8_SRGB,
    ASTC_10x10_SRGB,
    ASTC_12x12_SRGB,
};

constexpr auto pixel_storage_count = to_underlying(PixelStorage::ASTC_12x12_SRGB) + 1u;
constexpr auto pixel_format_count = to_underlying(PixelFormat::ASTC_12x12_SRGB) + 1u;

[[nodiscard]] constexpr auto is_block_compressed(PixelStorage s) noexcept {
    return luisa::to_underlying(s) >= luisa::to_underlying(PixelStorage::BC1) &&
           luisa::to_underlying(s) <= luisa::to_underlying(PixelStorage::ASTC_12x12_SRGB);
}

[[nodiscard]] constexpr auto is_block_compressed(PixelFormat f) noexcept {
    return luisa::to_underlying(f) >= luisa::to_underlying(PixelFormat::BC1UNorm) &&
           luisa::to_underlying(f) <= luisa::to_underlying(PixelFormat::ASTC_12x12_SRGB);
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
        case PixelFormat::RGBA8SRGB:
            return PixelStorage::BYTE4_SRGB;
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
        case PixelFormat::BC7SRGB:
            return PixelStorage::BC7_SRGB;
        case PixelFormat::BC5UNorm:
            return PixelStorage::BC5;
        case PixelFormat::BC4UNorm:
            return PixelStorage::BC4;
        case PixelFormat::BC3UNorm:
            return PixelStorage::BC3;
        case PixelFormat::BC2UNorm:
            return PixelStorage::BC2;
        case PixelFormat::BC1UNorm:
            return PixelStorage::BC1;
        case PixelFormat::ASTC_4x4:
        case PixelFormat::ASTC_4x4_SRGB:
            return PixelStorage::ASTC_4x4;
        case PixelFormat::ASTC_5x5:
        case PixelFormat::ASTC_5x5_SRGB:
            return PixelStorage::ASTC_5x5;
        case PixelFormat::ASTC_6x6:
        case PixelFormat::ASTC_6x6_SRGB:
            return PixelStorage::ASTC_6x6;
        case PixelFormat::ASTC_8x8:
        case PixelFormat::ASTC_8x8_SRGB:
            return PixelStorage::ASTC_8x8;
        case PixelFormat::ASTC_10x10:
        case PixelFormat::ASTC_10x10_SRGB:
            return PixelStorage::ASTC_10x10;
        case PixelFormat::ASTC_12x12:
        case PixelFormat::ASTC_12x12_SRGB:
            return PixelStorage::ASTC_12x12;
        case PixelFormat::R10G10B10A2UNorm:
        case PixelFormat::R10G10B10A2UInt:
            return PixelStorage::R10G10B10A2;
        case PixelFormat::R11G11B10F:
            return PixelStorage::R11G11B10;
        default:
            break;
    }
    return PixelStorage{};
}

[[nodiscard]] constexpr bool is_srgb(PixelStorage storage) noexcept {
    return storage == PixelStorage::BC7_SRGB || storage == PixelStorage::BYTE4_SRGB ||
           storage == PixelStorage::ASTC_4x4_SRGB ||
           storage == PixelStorage::ASTC_5x5_SRGB ||
           storage == PixelStorage::ASTC_6x6_SRGB ||
           storage == PixelStorage::ASTC_8x8_SRGB ||
           storage == PixelStorage::ASTC_10x10_SRGB ||
           storage == PixelStorage::ASTC_12x12_SRGB;
}

[[nodiscard]] constexpr bool is_srgb(PixelFormat format) noexcept {
    return format == PixelFormat::BC7SRGB || format == PixelFormat::RGBA8SRGB ||
           format == PixelFormat::ASTC_4x4_SRGB ||
           format == PixelFormat::ASTC_5x5_SRGB ||
           format == PixelFormat::ASTC_6x6_SRGB ||
           format == PixelFormat::ASTC_8x8_SRGB ||
           format == PixelFormat::ASTC_10x10_SRGB ||
           format == PixelFormat::ASTC_12x12_SRGB;
}

[[nodiscard]] constexpr PixelStorage decay_srgb(PixelStorage storage) noexcept {
    switch (storage) {
        case PixelStorage::BC7_SRGB:
            return PixelStorage::BC7;
        case PixelStorage::BYTE4_SRGB:
            return PixelStorage::BYTE4;
        case PixelStorage::ASTC_4x4_SRGB:
            return PixelStorage::ASTC_4x4;
        case PixelStorage::ASTC_5x5_SRGB:
            return PixelStorage::ASTC_5x5;
        case PixelStorage::ASTC_6x6_SRGB:
            return PixelStorage::ASTC_6x6;
        case PixelStorage::ASTC_8x8_SRGB:
            return PixelStorage::ASTC_8x8;
        case PixelStorage::ASTC_10x10_SRGB:
            return PixelStorage::ASTC_10x10;
        case PixelStorage::ASTC_12x12_SRGB:
            return PixelStorage::ASTC_12x12;
        default:
            return storage;
    }
}

[[nodiscard]] constexpr PixelFormat decay_srgb(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::BC7SRGB:
            return PixelFormat::BC7UNorm;
        case PixelFormat::RGBA8SRGB:
            return PixelFormat::RGBA8UNorm;
        case PixelFormat::ASTC_4x4_SRGB:
            return PixelFormat::ASTC_4x4;
        case PixelFormat::ASTC_5x5_SRGB:
            return PixelFormat::ASTC_5x5;
        case PixelFormat::ASTC_6x6_SRGB:
            return PixelFormat::ASTC_6x6;
        case PixelFormat::ASTC_8x8_SRGB:
            return PixelFormat::ASTC_8x8;
        case PixelFormat::ASTC_10x10_SRGB:
            return PixelFormat::ASTC_10x10;
        case PixelFormat::ASTC_12x12_SRGB:
            return PixelFormat::ASTC_12x12;
        default:
            return format;
    }
}

[[nodiscard]] constexpr size_t pixel_storage_align(PixelStorage storage) noexcept {
    if (is_block_compressed(storage)) {
        switch (storage) {
            case PixelStorage::BC1:
            case PixelStorage::BC4: return 8u;
            case PixelStorage::BC2:
            case PixelStorage::BC3:
            case PixelStorage::BC5:
            case PixelStorage::BC6:
            case PixelStorage::BC7:
            case PixelStorage::BC7_SRGB:
            case PixelStorage::ASTC_4x4:
            case PixelStorage::ASTC_5x5:
            case PixelStorage::ASTC_6x6:
            case PixelStorage::ASTC_8x8:
            case PixelStorage::ASTC_10x10:
            case PixelStorage::ASTC_12x12:
            case PixelStorage::ASTC_4x4_SRGB:
            case PixelStorage::ASTC_5x5_SRGB:
            case PixelStorage::ASTC_6x6_SRGB:
            case PixelStorage::ASTC_8x8_SRGB:
            case PixelStorage::ASTC_10x10_SRGB:
            case PixelStorage::ASTC_12x12_SRGB:
                return 16u;
            default: break;
        }
        detail::error_pixel_invalid_format("unknown.");
    }
    switch (storage) {
        case PixelStorage::BYTE1: return alignof(std::byte) * 1u;
        case PixelStorage::BYTE2: return alignof(std::byte) * 2u;
        case PixelStorage::BYTE4:
        case PixelStorage::BYTE4_SRGB: return alignof(std::byte) * 4u;
        case PixelStorage::SHORT1: return alignof(short) * 1u;
        case PixelStorage::SHORT2: return alignof(short) * 2u;
        case PixelStorage::SHORT4: return alignof(short) * 4u;
        case PixelStorage::INT1: return alignof(int) * 1u;
        case PixelStorage::INT2: return alignof(int) * 2u;
        case PixelStorage::INT4: return alignof(int) * 4u;
        case PixelStorage::HALF1: return alignof(short) * 1u;
        case PixelStorage::HALF2: return alignof(short) * 2u;
        case PixelStorage::HALF4: return alignof(short) * 4u;
        case PixelStorage::FLOAT1: return alignof(float) * 1u;
        case PixelStorage::FLOAT2: return alignof(float) * 2u;
        case PixelStorage::FLOAT4: return alignof(float) * 4u;
        case PixelStorage::R10G10B10A2:
        case PixelStorage::R11G11B10:
            return 4;
        default: break;
    }
    detail::error_pixel_invalid_format("unknown");
}

[[nodiscard]] constexpr PixelStorageSizeResult
checked_pixel_storage_size(PixelStorage storage, uint3 size) noexcept {
    auto checked_multiply = [](size_t lhs, size_t rhs,
                               size_t &result) constexpr noexcept {
        if (rhs != 0u &&
            lhs > std::numeric_limits<size_t>::max() / rhs) {
            return false;
        }
        result = lhs * rhs;
        return true;
    };

    size_t unit_size = 0u;
    auto block_compressed = false;
    // ASTC block dimensions differ per format; BC is always 4x4.
    auto block_width = 4u;
    auto block_height = 4u;
    switch (storage) {
        case PixelStorage::BYTE1:
            unit_size = sizeof(std::byte);
            break;
        case PixelStorage::BYTE2:
            unit_size = sizeof(std::byte) * 2u;
            break;
        case PixelStorage::BYTE4:
        case PixelStorage::BYTE4_SRGB:
            unit_size = sizeof(std::byte) * 4u;
            break;
        case PixelStorage::SHORT1:
        case PixelStorage::HALF1:
            unit_size = sizeof(short);
            break;
        case PixelStorage::SHORT2:
        case PixelStorage::HALF2:
            unit_size = sizeof(short) * 2u;
            break;
        case PixelStorage::SHORT4:
        case PixelStorage::HALF4:
            unit_size = sizeof(short) * 4u;
            break;
        case PixelStorage::INT1:
            unit_size = sizeof(int);
            break;
        case PixelStorage::INT2:
            unit_size = sizeof(int) * 2u;
            break;
        case PixelStorage::INT4:
            unit_size = sizeof(int) * 4u;
            break;
        case PixelStorage::FLOAT1:
            unit_size = sizeof(float);
            break;
        case PixelStorage::FLOAT2:
            unit_size = sizeof(float) * 2u;
            break;
        case PixelStorage::FLOAT4:
            unit_size = sizeof(float) * 4u;
            break;
        case PixelStorage::R10G10B10A2:
        case PixelStorage::R11G11B10:
            unit_size = 4u;
            break;
        case PixelStorage::BC1:
        case PixelStorage::BC4:
            unit_size = 8u;
            block_compressed = true;
            break;
        case PixelStorage::BC2:
        case PixelStorage::BC3:
        case PixelStorage::BC5:
        case PixelStorage::BC6:
        case PixelStorage::BC7:
        case PixelStorage::BC7_SRGB:
            unit_size = 16u;
            block_compressed = true;
            break;
        case PixelStorage::ASTC_4x4:
        case PixelStorage::ASTC_4x4_SRGB:
            unit_size = 16u;
            block_compressed = true;
            block_width = 4u;
            block_height = 4u;
            break;
        case PixelStorage::ASTC_5x5:
        case PixelStorage::ASTC_5x5_SRGB:
            unit_size = 16u;
            block_compressed = true;
            block_width = 5u;
            block_height = 5u;
            break;
        case PixelStorage::ASTC_6x6:
        case PixelStorage::ASTC_6x6_SRGB:
            unit_size = 16u;
            block_compressed = true;
            block_width = 6u;
            block_height = 6u;
            break;
        case PixelStorage::ASTC_8x8:
        case PixelStorage::ASTC_8x8_SRGB:
            unit_size = 16u;
            block_compressed = true;
            block_width = 8u;
            block_height = 8u;
            break;
        case PixelStorage::ASTC_10x10:
        case PixelStorage::ASTC_10x10_SRGB:
            unit_size = 16u;
            block_compressed = true;
            block_width = 10u;
            block_height = 10u;
            break;
        case PixelStorage::ASTC_12x12:
        case PixelStorage::ASTC_12x12_SRGB:
            unit_size = 16u;
            block_compressed = true;
            block_width = 12u;
            block_height = 12u;
            break;
        default:
            return {PixelStorageSizeStatus::INVALID_STORAGE, 0u};
    }

    auto width = static_cast<size_t>(size.x);
    auto height = static_cast<size_t>(size.y);
    auto depth = static_cast<size_t>(size.z);
    if (block_compressed) {
        // Avoid `(extent + 3) / 4`: the addition itself wraps for UINT32_MAX.
        width = width / block_width +
                static_cast<size_t>(width % block_width != 0u);
        height = height / block_height +
                 static_cast<size_t>(height % block_height != 0u);
        depth = std::max<size_t>(depth, 1u);
    }
    size_t element_count = 0u;
    size_t byte_size = 0u;
    if (!checked_multiply(width, height, element_count) ||
        !checked_multiply(element_count, depth, element_count) ||
        !checked_multiply(element_count, unit_size, byte_size)) {
        return {PixelStorageSizeStatus::OVERFLOW, 0u};
    }
    return {PixelStorageSizeStatus::SUCCESS, byte_size};
}

[[nodiscard]] constexpr size_t pixel_storage_size(
    PixelStorage storage, uint3 size) noexcept {
    auto result = checked_pixel_storage_size(storage, size);
    if (result.status == PixelStorageSizeStatus::INVALID_STORAGE) {
        detail::error_pixel_invalid_format("unknown");
    }
    if (result.status == PixelStorageSizeStatus::OVERFLOW) {
        detail::error_pixel_size_overflow(
            luisa::to_underlying(storage), size);
    }
    return result.size;
}

[[nodiscard]] constexpr size_t pixel_format_align(PixelFormat format) noexcept {
    return pixel_storage_align(pixel_format_to_storage(format));
}

[[nodiscard]] constexpr auto pixel_storage_channel_count(PixelStorage storage) noexcept {
    switch (storage) {
        case PixelStorage::BYTE1: return 1u;
        case PixelStorage::BYTE2: return 2u;
        case PixelStorage::BYTE4_SRGB:
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

        case PixelStorage::R11G11B10: return 3u;
        case PixelStorage::R10G10B10A2: return 4u;

        case PixelStorage::BC4: return 1u;
        case PixelStorage::BC5: return 2u;
        case PixelStorage::BC1:
        case PixelStorage::BC6: return 3u;
        case PixelStorage::BC2:
        case PixelStorage::BC3:
        case PixelStorage::BC7:
        case PixelStorage::BC7_SRGB:
        case PixelStorage::ASTC_4x4:
        case PixelStorage::ASTC_5x5:
        case PixelStorage::ASTC_6x6:
        case PixelStorage::ASTC_8x8:
        case PixelStorage::ASTC_10x10:
        case PixelStorage::ASTC_12x12:
        case PixelStorage::ASTC_4x4_SRGB:
        case PixelStorage::ASTC_5x5_SRGB:
        case PixelStorage::ASTC_6x6_SRGB:
        case PixelStorage::ASTC_8x8_SRGB:
        case PixelStorage::ASTC_10x10_SRGB:
        case PixelStorage::ASTC_12x12_SRGB:
            return 4u;
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
            case PixelStorage::BYTE4_SRGB: return PixelFormat::RGBA8SRGB;
            case PixelStorage::SHORT1: return PixelFormat::R16UNorm;
            case PixelStorage::SHORT2: return PixelFormat::RG16UNorm;
            case PixelStorage::SHORT4: return PixelFormat::RGBA16UNorm;
            case PixelStorage::HALF1: return PixelFormat::R16F;
            case PixelStorage::HALF2: return PixelFormat::RG16F;
            case PixelStorage::HALF4: return PixelFormat::RGBA16F;
            case PixelStorage::FLOAT1: return PixelFormat::R32F;
            case PixelStorage::FLOAT2: return PixelFormat::RG32F;
            case PixelStorage::FLOAT4: return PixelFormat::RGBA32F;
            case PixelStorage::R11G11B10: return PixelFormat::R11G11B10F;
            case PixelStorage::R10G10B10A2: return PixelFormat::R10G10B10A2UNorm;
            case PixelStorage::BC1: return PixelFormat ::BC1UNorm;
            case PixelStorage::BC2: return PixelFormat ::BC2UNorm;
            case PixelStorage::BC3: return PixelFormat ::BC3UNorm;
            case PixelStorage::BC4: return PixelFormat ::BC4UNorm;
            case PixelStorage::BC5: return PixelFormat ::BC5UNorm;
            case PixelStorage::BC6: return PixelFormat ::BC6HUF16;
            case PixelStorage::BC7: return PixelFormat ::BC7UNorm;
            case PixelStorage::BC7_SRGB: return PixelFormat ::BC7SRGB;
            case PixelStorage::ASTC_4x4: return PixelFormat ::ASTC_4x4;
            case PixelStorage::ASTC_5x5: return PixelFormat ::ASTC_5x5;
            case PixelStorage::ASTC_6x6: return PixelFormat ::ASTC_6x6;
            case PixelStorage::ASTC_8x8: return PixelFormat ::ASTC_8x8;
            case PixelStorage::ASTC_10x10: return PixelFormat ::ASTC_10x10;
            case PixelStorage::ASTC_12x12: return PixelFormat ::ASTC_12x12;
            case PixelStorage::ASTC_4x4_SRGB: return PixelFormat ::ASTC_4x4_SRGB;
            case PixelStorage::ASTC_5x5_SRGB: return PixelFormat ::ASTC_5x5_SRGB;
            case PixelStorage::ASTC_6x6_SRGB: return PixelFormat ::ASTC_6x6_SRGB;
            case PixelStorage::ASTC_8x8_SRGB: return PixelFormat ::ASTC_8x8_SRGB;
            case PixelStorage::ASTC_10x10_SRGB: return PixelFormat ::ASTC_10x10_SRGB;
            case PixelStorage::ASTC_12x12_SRGB: return PixelFormat ::ASTC_12x12_SRGB;
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
            case PixelStorage::R10G10B10A2: return PixelFormat::R10G10B10A2UInt;
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

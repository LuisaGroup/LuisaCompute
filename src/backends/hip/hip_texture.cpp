//
// Created by mike on 1/10/26.
//

#include "hip_check.h"
#include "hip_texture.h"
#include <luisa/core/stl/memory.h>

namespace luisa::compute::hip {

HIPTexture::HIPTexture() noexcept = default;

HIPTexture::~HIPTexture() noexcept {
    if (_direct_descriptor) {
        LUISA_CHECK_HIP(hipFree(_direct_descriptor));
    }
    for (auto i = 0u; i < _levels; i++) {
        if (_mip_surfaces[i]) { LUISA_CHECK_HIP(hipDestroySurfaceObject(_mip_surfaces[i])); }
        LUISA_CHECK_HIP(hipArrayDestroy(_mip_arrays[i]));
    }
}

hipArray_t HIPTexture::level(uint32_t i) const noexcept {
    LUISA_ASSERT(i < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 i, _levels);
    return _mip_arrays[i];
}

HIPSurface HIPTexture::surface(uint32_t level) const noexcept {
    LUISA_ASSERT(level < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 level, _levels);
    LUISA_ASSERT(!is_block_compressed(format()),
                 "Block compressed textures cannot be used as HIP surfaces.");
    return binding(level);
}

HIPSurface HIPTexture::binding(uint32_t level) const noexcept {
    LUISA_ASSERT(level < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 level, _levels);
    const auto is_bc = is_block_compressed(format());
    LUISA_ASSERT(!is_bc ||
                     (format() == PixelFormat::BC1UNorm ||
                      format() == PixelFormat::BC2UNorm ||
                      format() == PixelFormat::BC3UNorm ||
                      format() == PixelFormat::BC4UNorm ||
                      format() == PixelFormat::BC5UNorm ||
                      format() == PixelFormat::BC6HUF16 ||
                      format() == PixelFormat::BC7UNorm ||
                      format() == PixelFormat::BC7SRGB),
                 "HIP block-compressed format {} cannot be used for texture sampling.",
                 luisa::to_underlying(format()));
    LUISA_ASSERT(_direct_descriptor != nullptr,
                 "HIP direct texture descriptor is not initialized.");
    auto descriptor = reinterpret_cast<uint64_t>(_direct_descriptor);
    LUISA_ASSERT((descriptor & direct_descriptor_mip_tag_mask) == 0u,
                 "HIP direct texture descriptor address 0x{:016x} is not {}-byte aligned.",
                 descriptor, 1u << direct_descriptor_mip_tag_bits);
    descriptor |= static_cast<uint64_t>(level);
    auto surface_handle = _mip_surfaces[level];
    if (is_bc) {
        surface_handle = reinterpret_cast<hipSurfaceObject_t>(
            reinterpret_cast<std::byte *>(_direct_descriptor) +
            offsetof(HIPDirectTextureDescriptor, images) +
            static_cast<size_t>(level) * sizeof(HIPImageDescriptor));
    }
    return HIPSurface{surface_handle, descriptor};
}

namespace {

[[nodiscard]] auto hip_array_format(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt: [[fallthrough]];
        case PixelFormat::RG8SInt: [[fallthrough]];
        case PixelFormat::RGBA8SInt: return HIP_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::R8UInt: [[fallthrough]];
        case PixelFormat::R8UNorm: [[fallthrough]];
        case PixelFormat::RG8UInt: [[fallthrough]];
        case PixelFormat::RG8UNorm: [[fallthrough]];
        case PixelFormat::RGBA8UInt: [[fallthrough]];
        case PixelFormat::RGBA8UNorm: return HIP_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::R16SInt: [[fallthrough]];
        case PixelFormat::RG16SInt: [[fallthrough]];
        case PixelFormat::RGBA16SInt: return HIP_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::R16UInt: [[fallthrough]];
        case PixelFormat::R16UNorm: [[fallthrough]];
        case PixelFormat::RG16UInt: [[fallthrough]];
        case PixelFormat::RG16UNorm: [[fallthrough]];
        case PixelFormat::RGBA16UInt: [[fallthrough]];
        case PixelFormat::RGBA16UNorm: return HIP_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::R32SInt: [[fallthrough]];
        case PixelFormat::RGBA32SInt: return HIP_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::R32UInt: [[fallthrough]];
        case PixelFormat::RG32SInt: [[fallthrough]];
        case PixelFormat::RG32UInt: [[fallthrough]];
        case PixelFormat::RGBA32UInt: return HIP_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::R16F: [[fallthrough]];
        case PixelFormat::RG16F: [[fallthrough]];
        case PixelFormat::RGBA16F: return HIP_AD_FORMAT_HALF;
        case PixelFormat::R32F: [[fallthrough]];
        case PixelFormat::RG32F: [[fallthrough]];
        case PixelFormat::RGBA32F: return HIP_AD_FORMAT_FLOAT;
        // HIP does not expose a native 10/10/10/2 array descriptor. Keep the
        // resource packed as one 32-bit channel; image read/write and sampling
        // unpack the word in the LLVM backend.
        case PixelFormat::R10G10B10A2UInt: [[fallthrough]];
        case PixelFormat::R10G10B10A2UNorm: return HIP_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::BC1UNorm: [[fallthrough]];
        case PixelFormat::BC4UNorm: [[fallthrough]];
        case PixelFormat::BC2UNorm: [[fallthrough]];
        case PixelFormat::BC3UNorm: [[fallthrough]];
        case PixelFormat::BC5UNorm: [[fallthrough]];
        case PixelFormat::BC6HUF16: [[fallthrough]];
        case PixelFormat::BC7UNorm: [[fallthrough]];
        case PixelFormat::BC7SRGB: return HIP_AD_FORMAT_UNSIGNED_INT32;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported pixel format 0x{:02x}.",
                              luisa::to_underlying(format));
}

[[nodiscard]] auto hip_array_channel_count(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::BC1UNorm: [[fallthrough]];
        case PixelFormat::BC4UNorm: return 2u;
        case PixelFormat::BC2UNorm: [[fallthrough]];
        case PixelFormat::BC3UNorm: [[fallthrough]];
        case PixelFormat::BC5UNorm: [[fallthrough]];
        case PixelFormat::BC6HUF16: [[fallthrough]];
        case PixelFormat::BC7UNorm: [[fallthrough]];
        case PixelFormat::BC7SRGB: return 4u;
        case PixelFormat::R10G10B10A2UInt: [[fallthrough]];
        case PixelFormat::R10G10B10A2UNorm: return 1u;
        case PixelFormat::R11G11B10F: [[fallthrough]];
        case PixelFormat::RGBA8SRGB: LUISA_ERROR_WITH_LOCATION(
            "HIPTexture does not support special formats "
            "R11G11B10F and sRGB RGBA8 as array formats.");
        default: break;
    }
    return pixel_format_channel_count(format);
}

[[nodiscard]] auto hip_resource_view_format(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt: return HIP_RES_VIEW_FORMAT_SINT_1X8;
        case PixelFormat::R8UInt: return HIP_RES_VIEW_FORMAT_UINT_1X8;
        case PixelFormat::R8UNorm: return HIP_RES_VIEW_FORMAT_UINT_1X8;
        case PixelFormat::RG8SInt: return HIP_RES_VIEW_FORMAT_SINT_2X8;
        case PixelFormat::RG8UInt: return HIP_RES_VIEW_FORMAT_UINT_2X8;
        case PixelFormat::RG8UNorm: return HIP_RES_VIEW_FORMAT_UINT_2X8;
        case PixelFormat::RGBA8SInt: return HIP_RES_VIEW_FORMAT_SINT_4X8;
        case PixelFormat::RGBA8UInt: return HIP_RES_VIEW_FORMAT_UINT_4X8;
        case PixelFormat::RGBA8UNorm: return HIP_RES_VIEW_FORMAT_UINT_4X8;
        case PixelFormat::R16SInt: return HIP_RES_VIEW_FORMAT_SINT_1X16;
        case PixelFormat::R16UInt: return HIP_RES_VIEW_FORMAT_UINT_1X16;
        case PixelFormat::R16UNorm: return HIP_RES_VIEW_FORMAT_UINT_1X16;
        case PixelFormat::RG16SInt: return HIP_RES_VIEW_FORMAT_SINT_2X16;
        case PixelFormat::RG16UInt: return HIP_RES_VIEW_FORMAT_UINT_2X16;
        case PixelFormat::RG16UNorm: return HIP_RES_VIEW_FORMAT_UINT_2X16;
        case PixelFormat::RGBA16SInt: return HIP_RES_VIEW_FORMAT_SINT_4X16;
        case PixelFormat::RGBA16UInt: return HIP_RES_VIEW_FORMAT_UINT_4X16;
        case PixelFormat::RGBA16UNorm: return HIP_RES_VIEW_FORMAT_UINT_4X16;
        case PixelFormat::R32SInt: return HIP_RES_VIEW_FORMAT_SINT_1X32;
        case PixelFormat::R32UInt: return HIP_RES_VIEW_FORMAT_UINT_1X32;
        case PixelFormat::RG32SInt: return HIP_RES_VIEW_FORMAT_SINT_2X32;
        case PixelFormat::RG32UInt: return HIP_RES_VIEW_FORMAT_UINT_2X32;
        case PixelFormat::RGBA32SInt: return HIP_RES_VIEW_FORMAT_SINT_4X32;
        case PixelFormat::RGBA32UInt: return HIP_RES_VIEW_FORMAT_UINT_4X32;
        case PixelFormat::R16F: return HIP_RES_VIEW_FORMAT_FLOAT_1X16;
        case PixelFormat::RG16F: return HIP_RES_VIEW_FORMAT_FLOAT_2X16;
        case PixelFormat::RGBA16F: return HIP_RES_VIEW_FORMAT_FLOAT_4X16;
        case PixelFormat::R32F: return HIP_RES_VIEW_FORMAT_FLOAT_1X32;
        case PixelFormat::RG32F: return HIP_RES_VIEW_FORMAT_FLOAT_2X32;
        case PixelFormat::RGBA32F: return HIP_RES_VIEW_FORMAT_FLOAT_4X32;
        case PixelFormat::BC1UNorm: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC1;
        case PixelFormat::BC2UNorm: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC2;
        case PixelFormat::BC3UNorm: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC3;
        case PixelFormat::BC4UNorm: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC4;
        case PixelFormat::BC5UNorm: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC5;
        case PixelFormat::BC6HUF16: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H;
        case PixelFormat::BC7UNorm: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC7;
        case PixelFormat::BC7SRGB: return HIP_RES_VIEW_FORMAT_UNSIGNED_BC7;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported pixel format 0x{:02x} for resource view.",
                              luisa::to_underlying(format));
}

[[nodiscard]] auto hip_texture_address_mode(Sampler::Address mode) noexcept {
    switch (mode) {
        case Sampler::Address::EDGE: return HIP_TR_ADDRESS_MODE_CLAMP;
        case Sampler::Address::REPEAT: return HIP_TR_ADDRESS_MODE_WRAP;
        case Sampler::Address::MIRROR: return HIP_TR_ADDRESS_MODE_MIRROR;
        case Sampler::Address::ZERO: return HIP_TR_ADDRESS_MODE_BORDER;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported sampler address mode {}.",
                              luisa::to_underlying(mode));
}

[[nodiscard]] auto hip_texture_filter_mode(Sampler::Filter filter) noexcept {
    switch (filter) {
        case Sampler::Filter::POINT: return HIP_TR_FILTER_MODE_POINT;
        case Sampler::Filter::LINEAR_POINT: return HIP_TR_FILTER_MODE_LINEAR;
        case Sampler::Filter::LINEAR_LINEAR: return HIP_TR_FILTER_MODE_LINEAR;
        case Sampler::Filter::ANISOTROPIC: return HIP_TR_FILTER_MODE_LINEAR;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported sampler filter mode {}.",
                              luisa::to_underlying(filter));
}

[[nodiscard]] auto hip_texture_mipmap_filter_mode(Sampler::Filter filter, bool is_mipmapped) noexcept {
    switch (filter) {
        case Sampler::Filter::POINT: return HIP_TR_FILTER_MODE_POINT;
        case Sampler::Filter::LINEAR_POINT: return HIP_TR_FILTER_MODE_POINT;
        case Sampler::Filter::LINEAR_LINEAR: return is_mipmapped ? HIP_TR_FILTER_MODE_LINEAR : HIP_TR_FILTER_MODE_POINT;
        case Sampler::Filter::ANISOTROPIC: return is_mipmapped ? HIP_TR_FILTER_MODE_LINEAR : HIP_TR_FILTER_MODE_POINT;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported sampler filter mode {}.",
                              luisa::to_underlying(filter));
}

[[nodiscard]] auto hip_texture_max_anisotropy(Sampler::Filter filter, bool is_mipmapped) noexcept {
    switch (filter) {
        case Sampler::Filter::POINT: return 0u;
        case Sampler::Filter::LINEAR_POINT: return 0u;
        case Sampler::Filter::LINEAR_LINEAR: return 0u;
        case Sampler::Filter::ANISOTROPIC: return is_mipmapped ? 16u : 0u;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported sampler filter mode {}.",
                              luisa::to_underlying(filter));
}

[[nodiscard]] auto hip_texture_mip_level_clamp(Sampler::Filter filter, bool is_mipmapped) noexcept {
    switch (filter) {
        case Sampler::Filter::POINT: return 0.0f;
        case Sampler::Filter::LINEAR_POINT: return 0.0f;
        case Sampler::Filter::LINEAR_LINEAR: return is_mipmapped ? 999.0f : 0.0f;
        case Sampler::Filter::ANISOTROPIC: return is_mipmapped ? 999.0f : 0.0f;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported sampler filter mode {}.",
                              luisa::to_underlying(filter));
}

[[nodiscard]] bool hip_texture_is_bc_format(PixelFormat format) noexcept {
    return format == PixelFormat::BC1UNorm ||
           format == PixelFormat::BC2UNorm ||
           format == PixelFormat::BC3UNorm ||
           format == PixelFormat::BC4UNorm ||
           format == PixelFormat::BC5UNorm ||
           format == PixelFormat::BC6HUF16 ||
           format == PixelFormat::BC7UNorm ||
           format == PixelFormat::BC7SRGB;
}

[[nodiscard]] uint8_t parse_hip_gfx_gen(luisa::string_view arch) noexcept {
    if (arch.starts_with("gfx12")) { return 12u; }
    if (arch.starts_with("gfx11")) { return 11u; }
    if (arch.starts_with("gfx10")) { return 10u; }
    if (arch.starts_with("gfx9")) { return 9u; }
    return 0u;
}

[[nodiscard]] uint32_t hip_bc_hw_format(PixelFormat format, uint8_t gfx_gen) noexcept {
    // Values verified from AMD PAL gfx12_merged_enum.h and gfx9_plus_merged_enum.h.
    if (gfx_gen == 10u) {
        switch (format) {
            case PixelFormat::BC1UNorm: return 0xa9u;
            case PixelFormat::BC2UNorm: return 0xabu;
            case PixelFormat::BC3UNorm: return 0xadu;
            case PixelFormat::BC4UNorm: return 0xafu;
            case PixelFormat::BC5UNorm: return 0xb1u;
            case PixelFormat::BC6HUF16: return 0xb3u;
            case PixelFormat::BC7UNorm: return 0xb5u;
            case PixelFormat::BC7SRGB: return 0xb6u;
            default: break;
        }
    } else {
        // GFX11 and GFX12 share IMG_FMT_BC* numeric identifiers.
        switch (format) {
            case PixelFormat::BC1UNorm: return 0x6du;
            case PixelFormat::BC2UNorm: return 0x6fu;
            case PixelFormat::BC3UNorm: return 0x71u;
            case PixelFormat::BC4UNorm: return 0x73u;
            case PixelFormat::BC5UNorm: return 0x75u;
            case PixelFormat::BC6HUF16: return 0x77u;
            case PixelFormat::BC7UNorm: return 0x79u;
            case PixelFormat::BC7SRGB: return 0x7au;
            default: break;
        }
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported HIP BC pixel format 0x{:02x}.",
                              luisa::to_underlying(format));
}

void patch_bc_image_descriptor(
    HIPImageDescriptor &desc,
    PixelFormat format,
    uint3 level_size,
    uint8_t gfx_gen) noexcept {
    LUISA_ASSERT(gfx_gen == 10u || gfx_gen == 11u || gfx_gen == 12u,
                 "HIP hardware BC texture sampling requires GFX10/GFX11/GFX12 (got gfx{}).",
                 gfx_gen);
    LUISA_ASSERT((desc.words[6u] & (1u << 21u)) == 0u,
                 "Unexpected hardware DCC compression_en on raw block HIPImageDescriptor.");

    const uint32_t blocks_x = (level_size.x + 3u) / 4u;
    const uint32_t blocks_y = (level_size.y + 3u) / 4u;
    const uint32_t w_m1 = level_size.x - 1u;
    const uint32_t h_m1 = level_size.y - 1u;
    const bool is_64bpp = (format == PixelFormat::BC1UNorm || format == PixelFormat::BC4UNorm);
    const uint32_t sw_mode = (desc.words[3u] >> 20u) & 0x1fu;
    const uint32_t bc_fmt = hip_bc_hw_format(format, gfx_gen);

    if (gfx_gen == 12u) {
        const uint32_t expected_raw_fmt = is_64bpp ? 0x30u : 0x3du;
        const uint32_t raw_fmt = (desc.words[1u] >> 17u) & 0xffu;
        const uint32_t raw_w_m1 = ((desc.words[1u] >> 30u) & 0x3u) | ((desc.words[2u] & 0x3fffu) << 2u);
        const uint32_t raw_h_m1 = (desc.words[2u] >> 14u) & 0xffffu;
        LUISA_ASSERT(raw_fmt == expected_raw_fmt && raw_w_m1 == blocks_x - 1u && raw_h_m1 == blocks_y - 1u,
                     "Unexpected GFX12 raw T# layout for BC patching: fmt=0x{:x} (exp 0x{:x}), w_m1={} (exp {}), h_m1={} (exp {}).",
                     raw_fmt, expected_raw_fmt, raw_w_m1, blocks_x - 1u, raw_h_m1, blocks_y - 1u);
        desc.words[1u] = (desc.words[1u] & ~(0xffu << 17u) & ~(0x3u << 30u)) |
                         ((bc_fmt & 0xffu) << 17u) | ((w_m1 & 0x3u) << 30u);
        desc.words[2u] = (desc.words[2u] & ~0x3fffu & ~(0xffffu << 14u)) |
                         ((w_m1 >> 2u) & 0x3fffu) | ((h_m1 & 0xffffu) << 14u);
        if (sw_mode == 0u) {
            const uint32_t pitch_m1 = desc.words[4u] & 0xffffu;
            const uint32_t pitch_blocks = pitch_m1 != 0u ? (pitch_m1 + 1u) : blocks_x;
            const uint32_t pitch_pixels = pitch_blocks * 4u;
            if (pitch_pixels > level_size.x) {
                desc.words[4u] = (desc.words[4u] & ~0xffffu) | ((pitch_pixels - 1u) & 0xffffu);
            }
        }
    } else if (gfx_gen == 11u) {
        const uint32_t expected_raw_fmt = is_64bpp ? 0x30u : 0x3du;
        const uint32_t raw_fmt = (desc.words[1u] >> 20u) & 0xffu;
        const uint32_t raw_w_m1 = ((desc.words[1u] >> 30u) & 0x3u) | ((desc.words[2u] & 0x0fffu) << 2u);
        const uint32_t raw_h_m1 = (desc.words[2u] >> 14u) & 0x3fffu;
        LUISA_ASSERT(raw_fmt == expected_raw_fmt && raw_w_m1 == blocks_x - 1u && raw_h_m1 == blocks_y - 1u,
                     "Unexpected GFX11 raw T# layout for BC patching: fmt=0x{:x} (exp 0x{:x}), w_m1={} (exp {}), h_m1={} (exp {}).",
                     raw_fmt, expected_raw_fmt, raw_w_m1, blocks_x - 1u, raw_h_m1, blocks_y - 1u);
        desc.words[1u] = (desc.words[1u] & ~(0xffu << 20u) & ~(0x3u << 30u)) |
                         ((bc_fmt & 0xffu) << 20u) | ((w_m1 & 0x3u) << 30u);
        desc.words[2u] = (desc.words[2u] & ~0x0fffu & ~(0x3fffu << 14u)) |
                         ((w_m1 >> 2u) & 0x0fffu) | ((h_m1 & 0x3fffu) << 14u);
        if (sw_mode == 0u) {
            const uint32_t pitch_m1 = desc.words[4u] & 0x3fffu;
            const uint32_t pitch_blocks = pitch_m1 != 0u ? (pitch_m1 + 1u) : blocks_x;
            const uint32_t pitch_pixels = pitch_blocks * 4u;
            if (pitch_pixels > level_size.x) {
                desc.words[4u] = (desc.words[4u] & ~0x3fffu) | ((pitch_pixels - 1u) & 0x3fffu);
            }
        }
    } else {
        // GFX10
        const uint32_t expected_raw_fmt = is_64bpp ? 0x3eu : 0x4bu;
        const uint32_t raw_fmt = (desc.words[1u] >> 20u) & 0x1ffu;
        const uint32_t raw_w_m1 = ((desc.words[1u] >> 30u) & 0x3u) | ((desc.words[2u] & 0x3fffu) << 2u);
        const uint32_t raw_h_m1 = (desc.words[2u] >> 14u) & 0xffffu;
        LUISA_ASSERT(raw_fmt == expected_raw_fmt && raw_w_m1 == blocks_x - 1u && raw_h_m1 == blocks_y - 1u,
                     "Unexpected GFX10 raw T# layout for BC patching: fmt=0x{:x} (exp 0x{:x}), w_m1={} (exp {}), h_m1={} (exp {}).",
                     raw_fmt, expected_raw_fmt, raw_w_m1, blocks_x - 1u, raw_h_m1, blocks_y - 1u);
        desc.words[1u] = (desc.words[1u] & ~(0x1ffu << 20u) & ~(0x3u << 30u)) |
                         ((bc_fmt & 0x1ffu) << 20u) | ((w_m1 & 0x3u) << 30u);
        desc.words[2u] = (desc.words[2u] & ~0x3fffu & ~(0xffffu << 14u)) |
                         ((w_m1 >> 2u) & 0x3fffu) | ((h_m1 & 0xffffu) << 14u);
        if (sw_mode == 0u) {
            const uint32_t pitch_m1 = desc.words[4u] & 0xffffu;
            const uint32_t pitch_blocks = pitch_m1 != 0u ? (pitch_m1 + 1u) : blocks_x;
            const uint32_t pitch_pixels = pitch_blocks * 4u;
            if (pitch_pixels > level_size.x) {
                desc.words[4u] = (desc.words[4u] & ~0xffffu) | ((pitch_pixels - 1u) & 0xffffu);
            }
        }
    }

    // Channel swizzle in SQ_IMG_RSRC_WORD3 [11:0] (SQ_SEL_0=0, SQ_SEL_1=1, X=4, Y=5, Z=6, W=7).
    uint32_t dst_sel = 4u | (5u << 3u) | (6u << 6u) | (7u << 9u);
    if (format == PixelFormat::BC4UNorm) {
        dst_sel = 4u | (0u << 3u) | (0u << 6u) | (1u << 9u);
    } else if (format == PixelFormat::BC5UNorm) {
        dst_sel = 4u | (5u << 3u) | (0u << 6u) | (1u << 9u);
    } else if (format == PixelFormat::BC6HUF16) {
        dst_sel = 4u | (5u << 3u) | (6u << 6u) | (1u << 9u);
    }
    desc.words[3u] = (desc.words[3u] & ~0xfffu) | dst_sel;
}

[[nodiscard]] auto hip_texture_is_samplable(PixelFormat format) noexcept {
    return format == PixelFormat::R8UNorm ||
           format == PixelFormat::RG8UNorm ||
           format == PixelFormat::RGBA8UNorm ||
           format == PixelFormat::R16UNorm ||
           format == PixelFormat::RG16UNorm ||
           format == PixelFormat::RGBA16UNorm ||
           format == PixelFormat::R32F ||
           format == PixelFormat::RG32F ||
           format == PixelFormat::RGBA32F ||
           format == PixelFormat::R16F ||
           format == PixelFormat::RG16F ||
           format == PixelFormat::RGBA16F ||
           format == PixelFormat::R10G10B10A2UNorm ||
           hip_texture_is_bc_format(format);
}

[[nodiscard]] auto mip_size(uint3 size, uint32_t level) noexcept {
    auto extent = size >> level;
    return make_uint3(
        std::max(extent.x, 1u),
        std::max(extent.y, 1u),
        std::max(extent.z, 1u));
}

}// namespace

namespace {

[[nodiscard]] hipTextureObject_t create_texture_object(
    hipArray_t array, PixelFormat format, uint dimension,
    uint3 level_size, Sampler sampler) noexcept {
    HIP_RESOURCE_DESC res_desc{};
    res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = array;
    HIP_TEXTURE_DESC tex_desc{};
    auto address_mode = hip_texture_address_mode(sampler.address());
    tex_desc.addressMode[0] = address_mode;
    tex_desc.addressMode[1] = address_mode;
    tex_desc.addressMode[2] = address_mode;
    tex_desc.filterMode = hip_texture_filter_mode(sampler.filter());
    // Luisa's HIP backend stores every mip as an independent array. Mip
    // selection/filtering is therefore implemented in LLVM codegen; the
    // native sampler is only responsible for spatial filtering.
    tex_desc.mipmapFilterMode = HIP_TR_FILTER_MODE_POINT;
    tex_desc.maxAnisotropy = 0u;
    tex_desc.maxMipmapLevelClamp = 0.0f;
    tex_desc.flags = HIP_TRSF_NORMALIZED_COORDINATES;
    if (is_srgb(format) && !hip_texture_is_bc_format(format)) { tex_desc.flags |= HIP_TRSF_SRGB; }
    hipTextureObject_t object{};
    if (is_block_compressed(format) && !hip_texture_is_bc_format(format)) {
        HIP_RESOURCE_VIEW_DESC view_desc{};
        view_desc.format = hip_resource_view_format(format);
        view_desc.width = level_size.x;
        view_desc.height = level_size.y;
        view_desc.depth = dimension == 2u ? 0u : level_size.z;
        view_desc.firstMipmapLevel = 0u;
        view_desc.lastMipmapLevel = 0u;
        view_desc.firstLayer = 0u;
        view_desc.lastLayer = 0u;
        LUISA_CHECK_HIP(hipTexObjectCreate(&object, &res_desc, &tex_desc, &view_desc));
    } else {
        LUISA_CHECK_HIP(hipTexObjectCreate(&object, &res_desc, &tex_desc, nullptr));
    }
    return object;
}

}// namespace

void HIPTexture::create_texture_objects(luisa::span<hipTextureObject_t> objects, Sampler s) const noexcept {
    LUISA_ASSERT(hip_texture_is_samplable(format()) && !is_block_compressed(format()),
                 "Pixel format {} cannot be used for raw hipTextureObject_t sampling.",
                 luisa::to_underlying(format()));
    LUISA_ASSERT(objects.size() >= _levels,
                 "Texture object span size {} is smaller than texture level count {}.",
                 objects.size(), _levels);
    auto base_size = size();
    for (auto level = 0u; level < _levels; level++) {
        objects[level] = create_texture_object(
            _mip_arrays[level], format(), _dimension,
            mip_size(base_size, level), s);
    }
}

void HIPTexture::copy_image_descriptors(luisa::span<HIPImageDescriptor> descriptors) const noexcept {
    LUISA_ASSERT(hip_texture_is_samplable(format()),
                 "Pixel format {} cannot be used for texture sampling.",
                 luisa::to_underlying(format()));
    LUISA_ASSERT(descriptors.size() >= _levels,
                 "Image descriptor span size {} is smaller than texture level count {}.",
                 descriptors.size(), _levels);
    auto base_size = size();
    const bool patch_bc = hip_texture_is_bc_format(format());
    for (auto level = 0u; level < _levels; level++) {
        const auto level_ext = mip_size(base_size, level);
        auto object = create_texture_object(
            _mip_arrays[level], format(), _dimension,
            level_ext, Sampler::point_edge());
        auto address = reinterpret_cast<hipDeviceptr_t>(object);
        LUISA_CHECK_HIP(hipMemcpyDtoH(&descriptors[level], address,
                                      sizeof(HIPImageDescriptor)));
        LUISA_CHECK_HIP(hipTexObjectDestroy(object));
        if (patch_bc) {
            patch_bc_image_descriptor(descriptors[level], format(), level_ext, _gfx_gen);
        }
    }
}

void HIPTexture::copy_sampler_descriptors(luisa::span<HIPSamplerDescriptor> descriptors) const noexcept {
    static constexpr auto sampler_count = 16u;
    LUISA_ASSERT(hip_texture_is_samplable(format()),
                 "Pixel format {} cannot be used for texture sampling.",
                 luisa::to_underlying(format()));
    LUISA_ASSERT(descriptors.size() >= sampler_count,
                 "Sampler descriptor span size {} is smaller than {}.",
                 descriptors.size(), sampler_count);
    for (auto code = 0u; code < sampler_count; code++) {
        auto object = create_texture_object(
            _mip_arrays[0], format(), _dimension,
            size(), Sampler::decode(code));
        auto address = reinterpret_cast<hipDeviceptr_t>(
            reinterpret_cast<std::byte *>(object) +
            HIP_SAMPLER_OBJECT_OFFSET_DWORD * sizeof(uint32_t));
        LUISA_CHECK_HIP(hipMemcpyDtoH(&descriptors[code], address,
                                      sizeof(HIPSamplerDescriptor)));
        LUISA_CHECK_HIP(hipTexObjectDestroy(object));
    }
}

void HIPTexture::_initialize_direct_descriptor() noexcept {
    LUISA_ASSERT(_direct_descriptor == nullptr,
                 "HIP direct texture descriptor is already initialized.");
    HIPDirectTextureDescriptor host_descriptor{};
    host_descriptor.level_count = _levels;
    host_descriptor.storage = to_underlying(storage());
    auto texture_size = size();
    host_descriptor.size_xy = static_cast<uint64_t>(texture_size.x) |
                              (static_cast<uint64_t>(texture_size.y) << 32u);
    host_descriptor.size_z = texture_size.z;
    if (hip_texture_is_samplable(format())) {
        copy_image_descriptors(
            luisa::span{host_descriptor.images, static_cast<size_t>(_levels)});
        copy_sampler_descriptors(luisa::span{host_descriptor.samplers});
    }
    LUISA_CHECK_HIP(hipMalloc(&_direct_descriptor, sizeof(host_descriptor)));
    auto descriptor = reinterpret_cast<uint64_t>(_direct_descriptor);
    LUISA_ASSERT((descriptor & direct_descriptor_mip_tag_mask) == 0u,
                 "hipMalloc returned a direct texture descriptor address "
                 "0x{:016x} that cannot carry the {}-bit mip tag.",
                 descriptor, direct_descriptor_mip_tag_bits);
    LUISA_CHECK_HIP(hipMemcpyHtoD(
        _direct_descriptor, &host_descriptor, sizeof(host_descriptor)));
}

HIPTexture *HIPTexture::create_device_texture(PixelFormat format, uint dim, uint3 size, uint32_t mip_levels, luisa::string_view amdgpu_arch) noexcept {
    LUISA_ASSERT(dim == 2u || dim == 3u,
                 "HIPTexture::create_device_texture() only supports 2D and 3D textures.");
    LUISA_ASSERT(mip_levels >= 1u && mip_levels <= max_level_count,
                 "HIPTexture::create_device_texture() mip levels {} out of range [1, {}].",
                 mip_levels, max_level_count);
    auto is_bc = is_block_compressed(format);
    if (is_bc) {
        LUISA_ASSERT(dim == 2u && hip_texture_is_bc_format(format),
                     "Unsupported HIP block-compressed texture format {} (dim={}).",
                     luisa::to_underlying(format), dim);
    }
    auto t = luisa::new_with_allocator<HIPTexture>();
    t->_size[0] = static_cast<uint16_t>(size.x);
    t->_size[1] = static_cast<uint16_t>(size.y);
    t->_size[2] = static_cast<uint16_t>(size.z);
    t->_format = static_cast<uint8_t>(format);
    t->_levels = static_cast<uint8_t>(mip_levels);
    t->_dimension = static_cast<uint8_t>(dim);
    t->_gfx_gen = parse_hip_gfx_gen(amdgpu_arch);
    for (auto i = 0u; i < mip_levels; i++) {
        auto level_size = mip_size(size, i);
        HIP_ARRAY3D_DESCRIPTOR array_desc{};
        array_desc.Width = is_bc ? (level_size.x + 3u) / 4u : level_size.x;
        array_desc.Height = is_bc ? (level_size.y + 3u) / 4u : level_size.y;
        array_desc.Depth = dim == 2u ? 0u : level_size.z;
        array_desc.Format = hip_array_format(format);
        array_desc.NumChannels = hip_array_channel_count(format);
        hipArray_t array_handle{nullptr};
        LUISA_CHECK_HIP(hipArray3DCreate(&array_handle, &array_desc));
        t->_mip_arrays[i] = array_handle;
    }
    t->_base_array = t->_mip_arrays[0];
    if (!is_bc) {
        for (auto i = 0u; i < mip_levels; i++) {
            hipResourceDesc res_desc{};
            res_desc.resType = hipResourceTypeArray;
            res_desc.res.array.array = t->_mip_arrays[i];
            LUISA_CHECK_HIP(hipCreateSurfaceObject(&t->_mip_surfaces[i], &res_desc));
        }
    }
    t->_initialize_direct_descriptor();
    return t;
}

HIPTexture *HIPTexture::import_external_texture(uint64_t external_array, PixelFormat format,
                                                uint dim, uint3 size, uint32_t mip_levels) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip

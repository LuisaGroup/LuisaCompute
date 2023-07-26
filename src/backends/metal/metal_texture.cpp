#include <luisa/core/logging.h>
#include "metal_texture.h"

namespace luisa::compute::metal {

MetalTexture::MetalTexture(MTL::Device *device, PixelFormat format, uint dimension,
                           uint width, uint height, uint depth, uint mipmap_levels,
                           bool allow_simultaneous_access) noexcept
    : _format{format} {

    if (is_block_compressed(format)) {
        auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
        LUISA_ASSERT(device->supportsBCTextureCompression(),
                     "Metal device '{}' does not support "
                     "block-compressed texture compression.",
                     device->name()->cString(NS::UTF8StringEncoding));
        autorelease_pool->release();
    }

    auto desc = MTL::TextureDescriptor::alloc()->init();
    switch (dimension) {
        case 2u: desc->setTextureType(MTL::TextureType2D); break;
        case 3u: desc->setTextureType(MTL::TextureType3D); break;
        default: LUISA_ERROR_WITH_LOCATION("Invalid MetalTexture dimension {}.", dimension);
    }
    switch (format) {
        case PixelFormat::R8SInt: desc->setPixelFormat(MTL::PixelFormatR8Sint); break;
        case PixelFormat::R8UInt: desc->setPixelFormat(MTL::PixelFormatR8Uint); break;
        case PixelFormat::R8UNorm: desc->setPixelFormat(MTL::PixelFormatR8Unorm); break;
        case PixelFormat::RG8SInt: desc->setPixelFormat(MTL::PixelFormatRG8Sint); break;
        case PixelFormat::RG8UInt: desc->setPixelFormat(MTL::PixelFormatRG8Uint); break;
        case PixelFormat::RG8UNorm: desc->setPixelFormat(MTL::PixelFormatRG8Unorm); break;
        case PixelFormat::RGBA8SInt: desc->setPixelFormat(MTL::PixelFormatRGBA8Sint); break;
        case PixelFormat::RGBA8UInt: desc->setPixelFormat(MTL::PixelFormatRGBA8Uint); break;
        case PixelFormat::RGBA8UNorm: desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm); break;
        case PixelFormat::R16SInt: desc->setPixelFormat(MTL::PixelFormatR16Sint); break;
        case PixelFormat::R16UInt: desc->setPixelFormat(MTL::PixelFormatR16Uint); break;
        case PixelFormat::R16UNorm: desc->setPixelFormat(MTL::PixelFormatR16Unorm); break;
        case PixelFormat::RG16SInt: desc->setPixelFormat(MTL::PixelFormatRG16Sint); break;
        case PixelFormat::RG16UInt: desc->setPixelFormat(MTL::PixelFormatRG16Uint); break;
        case PixelFormat::RG16UNorm: desc->setPixelFormat(MTL::PixelFormatRG16Unorm); break;
        case PixelFormat::RGBA16SInt: desc->setPixelFormat(MTL::PixelFormatRGBA16Sint); break;
        case PixelFormat::RGBA16UInt: desc->setPixelFormat(MTL::PixelFormatRGBA16Uint); break;
        case PixelFormat::RGBA16UNorm: desc->setPixelFormat(MTL::PixelFormatRGBA16Unorm); break;
        case PixelFormat::R32SInt: desc->setPixelFormat(MTL::PixelFormatR32Sint); break;
        case PixelFormat::R32UInt: desc->setPixelFormat(MTL::PixelFormatR32Uint); break;
        case PixelFormat::RG32SInt: desc->setPixelFormat(MTL::PixelFormatRG32Sint); break;
        case PixelFormat::RG32UInt: desc->setPixelFormat(MTL::PixelFormatRG32Uint); break;
        case PixelFormat::RGBA32SInt: desc->setPixelFormat(MTL::PixelFormatRGBA32Sint); break;
        case PixelFormat::RGBA32UInt: desc->setPixelFormat(MTL::PixelFormatRGBA32Uint); break;
        case PixelFormat::R16F: desc->setPixelFormat(MTL::PixelFormatR16Float); break;
        case PixelFormat::RG16F: desc->setPixelFormat(MTL::PixelFormatRG16Float); break;
        case PixelFormat::RGBA16F: desc->setPixelFormat(MTL::PixelFormatRGBA16Float); break;
        case PixelFormat::R32F: desc->setPixelFormat(MTL::PixelFormatR32Float); break;
        case PixelFormat::RG32F: desc->setPixelFormat(MTL::PixelFormatRG32Float); break;
        case PixelFormat::RGBA32F: desc->setPixelFormat(MTL::PixelFormatRGBA32Float); break;
        case PixelFormat::BC4UNorm: desc->setPixelFormat(MTL::PixelFormatBC4_RUnorm); break;
        case PixelFormat::BC5UNorm: desc->setPixelFormat(MTL::PixelFormatBC5_RGUnorm); break;
        case PixelFormat::BC6HUF16: desc->setPixelFormat(MTL::PixelFormatBC6H_RGBUfloat); break;
        case PixelFormat::BC7UNorm: desc->setPixelFormat(MTL::PixelFormatBC7_RGBAUnorm); break;
    }
    desc->setWidth(std::max(width, 1u));
    desc->setHeight(std::max(height, 1u));
    desc->setDepth(std::max(depth, 1u));
    desc->setMipmapLevelCount(std::clamp(mipmap_levels, 1u, max_level_count));
    desc->setAllowGPUOptimizedContents(true);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setHazardTrackingMode(MTL::HazardTrackingModeTracked);
    desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    desc->setAllowGPUOptimizedContents(allow_simultaneous_access);
    _maps[0u] = device->newTexture(desc);
    desc->release();
    auto n = _maps[0u]->mipmapLevelCount();
    auto pixel_format = _maps[0u]->pixelFormat();
    auto texture_type = _maps[0u]->textureType();
    for (auto i = 1u; i < n; i++) {
        _maps[i] = _maps[0u]->newTextureView(
            pixel_format, texture_type,
            NS::Range{i, 1u}, NS::Range{0u, 1u});
    }
}

MetalTexture::~MetalTexture() noexcept {
    auto n = _maps[0u]->mipmapLevelCount();
    for (auto i = 0u; i < n; i++) { _maps[i]->release(); }
}

MTL::Texture *MetalTexture::handle(uint level) const noexcept {
#ifndef NDEBUG
    LUISA_ASSERT(level < _maps[0u]->mipmapLevelCount(),
                 "Invalid mipmap level {} for "
                 "MetalTexture with {} level(s).",
                 level, _maps[0u]->mipmapLevelCount());
#endif
    return _maps[level];
}

MetalTexture::Binding MetalTexture::binding(uint level) const noexcept {
    return {handle(level)->gpuResourceID()};
}

void MetalTexture::set_name(luisa::string_view name) noexcept {
    auto n = _maps[0u]->mipmapLevelCount();
    if (name.empty()) {
        for (auto i = 0u; i < n; i++) {
            _maps[i]->setLabel(nullptr);
        }
    } else {
        for (auto i = 0u; i < n; i++) {
            auto level_name = luisa::format("{} (level {})", name, i);
            auto mtl_name = NS::String::alloc()->init(
                level_name.data(), level_name.size(),
                NS::UTF8StringEncoding, false);
            _maps[i]->setLabel(mtl_name);
            mtl_name->release();
        }
    }
}

}// namespace luisa::compute::metal

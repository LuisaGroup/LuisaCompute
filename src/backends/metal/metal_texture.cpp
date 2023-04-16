//
// Created by Mike Smith on 2023/4/15.
//

#include <core/logging.h>
#include <backends/metal/metal_texture.h>

namespace luisa::compute::metal {

MetalTexture::MetalTexture(MTL::Device *device, PixelFormat format, uint dimension,
                           uint width, uint height, uint depth, uint mipmap_levels) noexcept {

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
    _maps[0u] = device->newTexture(desc);
    for (auto i = 1u; i < desc->mipmapLevelCount(); i++) {
        _maps[i] = _maps[0u]->newTextureView(
            _maps[0u]->pixelFormat(), _maps[0u]->textureType(),
            NS::Range{i, 1u}, NS::Range{0u, 1u});
    }
    desc->release();
}

MetalTexture::~MetalTexture() noexcept {
    for (auto i = 0u; i < _maps.front()->mipmapLevelCount(); i++) {
        _maps[i]->release();
    }
}

MTL::Texture *MetalTexture::level(uint level) const noexcept {
#ifndef NDEBUG
    LUISA_ASSERT(level < _maps.front()->mipmapLevelCount(),
                 "Invalid mipmap level {} for "
                 "MetalTexture with {} level(s).",
                 level, _maps.front()->mipmapLevelCount());
#endif
    return _maps[level];
}

void MetalTexture::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        for (auto i = 0u; i < _maps.front()->mipmapLevelCount(); i++) {
            _maps[i]->setLabel(nullptr);
        }
    } else {
        auto autorelease_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        for (auto i = 0u; i < _maps.front()->mipmapLevelCount(); i++) {
            auto level_name = luisa::format("{} (level {})", name, i);
            _maps[i]->setLabel(NS::String::string(level_name.c_str(), NS::UTF8StringEncoding));
        }
    }
}

}// namespace luisa::compute::metal

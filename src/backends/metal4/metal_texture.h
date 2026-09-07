#pragma once

#include <array>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/pixel.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalTextureBase {

public:
    enum class Kind : uint8_t {
        TEXTURE,
        DEPTH,
    };

    struct Binding {
        MTL::ResourceID handle;
    };

public:
    MetalTextureBase() noexcept = default;
    virtual ~MetalTextureBase() noexcept = default;
    MetalTextureBase(MetalTextureBase &&) noexcept = delete;
    MetalTextureBase(const MetalTextureBase &) noexcept = delete;
    MetalTextureBase &operator=(MetalTextureBase &&) noexcept = delete;
    MetalTextureBase &operator=(const MetalTextureBase &) noexcept = delete;
    [[nodiscard]] virtual Kind kind() const noexcept = 0;
    [[nodiscard]] virtual MTL::Texture *handle(uint level = 0u) const noexcept = 0;
    [[nodiscard]] Binding binding(uint level = 0u) const noexcept {
        return {handle(level)->gpuResourceID()};
    }
    virtual void set_name(luisa::string_view name) noexcept = 0;
};

class MetalTexture final : public MetalTextureBase {

public:
    static constexpr auto max_level_count = 15u;

private:
    std::array<MTL::Texture *, max_level_count> _maps{};
    PixelFormat _format{};
    bool _raster_target{};

public:
    MetalTexture(MTL::Device *device, PixelFormat format, uint dimension,
                 uint width, uint height, uint depth, uint mipmap_levels,
                 bool allow_simultaneous_access, bool allow_raster_target) noexcept;
    ~MetalTexture() noexcept override;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::TEXTURE; }
    [[nodiscard]] MTL::Texture *handle(uint level = 0u) const noexcept override;
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] auto storage() const noexcept { return pixel_format_to_storage(_format); }
    [[nodiscard]] auto pixel_format() const noexcept { return _maps[0u]->pixelFormat(); }
    [[nodiscard]] auto is_raster_target() const noexcept { return _raster_target; }
    void set_name(luisa::string_view name) noexcept override;
};

}// namespace luisa::compute::metal

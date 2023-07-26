#pragma once

#include <array>

#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/pixel.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalTexture {

public:
    static constexpr auto max_level_count = 15u;

    struct Binding {
        MTL::ResourceID handle;
    };

private:
    std::array<MTL::Texture *, max_level_count> _maps{};
    PixelFormat _format{};

public:
    MetalTexture(MTL::Device *device, PixelFormat format, uint dimension,
                 uint width, uint height, uint depth, uint mipmap_levels,
                 bool allow_simultaneous_access) noexcept;
    ~MetalTexture() noexcept;
    [[nodiscard]] MTL::Texture *handle(uint level = 0u) const noexcept;
    [[nodiscard]] Binding binding(uint level = 0u) const noexcept;
    [[nodiscard]] auto format() const noexcept { return _format; }
    [[nodiscard]] auto storage() const noexcept { return pixel_format_to_storage(_format); }
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal


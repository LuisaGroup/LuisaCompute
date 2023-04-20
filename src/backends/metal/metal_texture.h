//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <array>

#include <core/stl/string.h>
#include <runtime/rhi/pixel.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalTexture {

public:
    static constexpr auto max_level_count = 16u;

private:
    std::array<MTL::Texture *, max_level_count> _maps{};

public:
    MetalTexture(MTL::Device *device,
                 PixelFormat format, uint dimension,
                 uint width, uint height, uint depth,
                 uint mipmap_levels) noexcept;
    ~MetalTexture() noexcept;
    [[nodiscard]] MTL::Texture *level(uint level) const noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal

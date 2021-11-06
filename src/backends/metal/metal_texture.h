//
// Created by Mike Smith on 2021/11/7.
//

#pragma once

#import <array>
#import <Metal/Metal.h>

#import <core/spin_mutex.h>

namespace luisa::compute::metal {

class MetalMipmapTexture {

public:
    static constexpr auto max_level_count = 15u;

private:
    id<MTLTexture> _handle;
    std::array<id<MTLTexture>, max_level_count> _mipmaps{};

public:
    explicit MetalMipmapTexture(id<MTLTexture> handle) noexcept
        : _handle{handle} {}

};

static_assert(sizeof(MetalMipmapTexture) == 128u);

}

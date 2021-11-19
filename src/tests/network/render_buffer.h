//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <vector>
#include <span>

#include <network/render_tile.h>

namespace luisa::compute {

class RenderBuffer {

public:
    using pixel_type = float4;

private:
    uint2 _frame_size;
    uint2 _tile_size;
    size_t _total_tiles;
    size_t _accum_tiles;
    std::vector<float4> _framebuffer;
    std::vector<bool> _tile_marks;

public:
    RenderBuffer(uint2 frame_size, uint2 tile_size) noexcept;
    [[nodiscard]] bool accumulate(RenderTile tile, std::span<const std::byte> tile_buffer) noexcept;
    [[nodiscard]] auto framebuffer() noexcept { return std::span{_framebuffer}; }
    [[nodiscard]] auto framebuffer() const noexcept { return std::span{_framebuffer}; }
    [[nodiscard]] auto total_tile_count() const noexcept { return _total_tiles; }
    [[nodiscard]] auto done() const noexcept { return _accum_tiles >= _total_tiles; }
    [[nodiscard]] auto accumulated_tile_count() const noexcept { return _accum_tiles; }
};

}// namespace luisa::compute

//
// Created by Mike Smith on 2021/9/24.
//

#include <network/render_buffer.h>

namespace luisa::compute {

RenderBuffer::RenderBuffer(uint2 frame_size, uint2 tile_size) noexcept
    : _frame_size{frame_size},
      _tile_size{tile_size},
      _total_tiles{},
      _accum_tiles{} {
    auto tile_count = (_frame_size + tile_size - 1u) / tile_size;
    _total_tiles = tile_count.x * tile_count.y;
}

bool RenderBuffer::accumulate(RenderTile tile, std::span<const std::byte> tile_buffer) noexcept {
    if (!all(tile.offset() < _frame_size)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid tile offset ({}, {}) for frame size ({}, {}).",
            tile.offset().x, tile.offset().y, _frame_size.x, _frame_size.y);
        return false;
    }
    if (auto pixel_count = _frame_size.x * _frame_size.y; _framebuffer.size() != pixel_count) {
        _framebuffer.resize(pixel_count);
    }
    for (auto y_tile = 0u; y_tile < _tile_size.y && y_tile + tile.offset().y < _frame_size.y; y_tile++) {
        auto p_frame = _framebuffer.data() + (y_tile + tile.offset().y) * _frame_size.x + tile.offset().x;
        auto p_tile = tile_buffer.data() + y_tile * _tile_size.x * sizeof(pixel_type);
        auto valid_w = std::min(tile.offset().x + _tile_size.x, _frame_size.x) - tile.offset().x;
        std::memcpy(p_frame, p_tile, valid_w * sizeof(pixel_type));
    }
    _accum_tiles++;
    return true;
}

}// namespace luisa::compute

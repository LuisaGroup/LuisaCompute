//
// Created by Mike Smith on 2021/9/24.
//

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <network/render_buffer.h>

namespace luisa::compute {

RenderBuffer::RenderBuffer(uint2 frame_size, uint2 tile_size) noexcept
    : _frame_size{frame_size},
      _tile_size{tile_size},
      _total_tiles{},
      _accum_tiles{} {
    auto tile_count = (_frame_size + tile_size - 1u) / tile_size;
    _total_tiles = tile_count.x * tile_count.y;
    _tile_marks.resize(_total_tiles, false);
    _framebuffer.resize(_frame_size.x * _frame_size.y);
}

bool RenderBuffer::accumulate(RenderTile tile, std::span<const std::byte> tile_buffer) noexcept {
    if (!all(tile.offset() < _frame_size) ||
        !all(tile.offset() % _tile_size == 0u)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid tile offset ({}, {}) for frame size ({}, {}).",
            tile.offset().x, tile.offset().y, _frame_size.x, _frame_size.y);
        return false;
    }
    auto tile_count_x = (_frame_size.x + _tile_size.x - 1u) / _tile_size.x;
    auto tile_id = tile.offset() / _tile_size;
    auto tile_index = tile_id.y * tile_count_x + tile_id.x;
    if (_tile_marks[tile_index]) {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring duplicate tile: render_id = {}, frame_id = {}, offset = ({}, {}).",
            tile.render_id(), tile.frame_id(), tile.offset().x, tile.offset().y);
        return false;
    }

    // convert from rgbe to float4
    auto width = 0;
    auto height = 0;
    auto channels = 0;
    std::unique_ptr<float4, void (*)(void *)> pixels{
        reinterpret_cast<float4 *>(
            stbi_loadf_from_memory(
                reinterpret_cast<const uint8_t *>(tile_buffer.data()),
                static_cast<int>(tile_buffer.size_bytes()),
                &width, &height, &channels, 4)),
        stbi_image_free};
    if (width != _tile_size.x || height != _tile_size.y) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid tile: width = {}, height = {}, channels = {}.",
            width, height, channels);
        return false;
    }
    for (auto y_tile = 0u; y_tile < _tile_size.y && y_tile + tile.offset().y < _frame_size.y; y_tile++) {
        auto p_frame = _framebuffer.data() + (y_tile + tile.offset().y) * _frame_size.x + tile.offset().x;
        auto p_tile = pixels.get() + y_tile * _tile_size.x;
        auto valid_w = std::min(tile.offset().x + _tile_size.x, _frame_size.x) - tile.offset().x;
        std::memcpy(p_frame, p_tile, valid_w * sizeof(pixel_type));
    }
    _tile_marks[tile_index] = true;
    _accum_tiles++;
    return true;
}

}// namespace luisa::compute

//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <span>

#include <core/basic_types.h>
#include <core/allocator.h>
#include <network/binary_buffer.h>

namespace luisa::compute {

class RenderTile {

private:
    uint _render_index;
    uint _frame_index;
    uint2 _frame_size;
    uint2 _offset;
    uint2 _size;

public:
    constexpr RenderTile(uint render_id, uint frame_id, uint2 frame_size, uint2 offset, uint2 size) noexcept
        : _render_index{render_id},
          _frame_index{frame_id},
          _frame_size{frame_size},
          _offset{offset},
          _size{size} {}
    [[nodiscard]] constexpr auto render_index() const noexcept { return _render_index; }
    [[nodiscard]] constexpr auto frame_index() const noexcept { return _frame_index; }
    [[nodiscard]] constexpr auto frame_size() const noexcept { return _frame_size; }
    [[nodiscard]] constexpr auto offset() const noexcept { return _offset; }
    [[nodiscard]] constexpr auto size() const noexcept { return _size; }
};

}// namespace luisa::compute

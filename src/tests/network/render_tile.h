//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <span>
#include <utility>

#include <core/basic_types.h>
#include <core/allocator.h>
#include <network/binary_buffer.h>

namespace luisa::compute {

class alignas(16u) RenderTile {

private:
    uint _render_id{};
    uint _frame_id{};
    uint2 _offset{};

public:
    constexpr RenderTile() noexcept = default;
    constexpr RenderTile(uint render_id, uint frame_id, uint2 offset) noexcept
        : _render_id{render_id}, _frame_id{frame_id}, _offset{offset} {}
    [[nodiscard]] constexpr auto render_id() const noexcept { return _render_id; }
    [[nodiscard]] constexpr auto frame_id() const noexcept { return _frame_id; }
    [[nodiscard]] constexpr auto offset() const noexcept { return _offset; }
    [[nodiscard]] constexpr auto operator==(RenderTile rhs) const noexcept {
        return _render_id == rhs._render_id &&
               _frame_id == rhs._frame_id &&
               _offset.x == rhs._offset.x &&
               _offset.y == rhs._offset.y;
    }
};

}// namespace luisa::compute

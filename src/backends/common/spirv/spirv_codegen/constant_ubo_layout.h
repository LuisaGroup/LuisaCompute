#pragma once

#include <cstddef>
#include <limits>

namespace lc::spirv {

// Vulkan guarantees maxUniformBufferRange >= 16 KiB. Native SPIR-V codegen
// cannot inspect the selected physical device, so keeping the generated
// constant block within this portable limit makes the result valid on every
// conformant Vulkan device. The runtime checks the actual limit again.
inline constexpr size_t portable_constant_ubo_max_range = 16u * 1024u;

enum class ConstantUBOLayoutStatus {
    SUCCESS,
    INVALID_LAYOUT,
    ARITHMETIC_OVERFLOW,
    RANGE_EXCEEDED,
};

struct ConstantUBOMemberLayout {
    ConstantUBOLayoutStatus status{ConstantUBOLayoutStatus::INVALID_LAYOUT};
    size_t member_offset{0u};
    size_t array_stride{0u};
    size_t end_offset{0u};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == ConstantUBOLayoutStatus::SUCCESS;
    }
};

namespace detail {

[[nodiscard]] constexpr bool checked_constant_ubo_align_up(
    size_t value, size_t alignment, size_t &result) noexcept {
    if (alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u) {
        return false;
    }
    auto remainder = value & (alignment - 1u);
    auto padding = remainder == 0u ? 0u : alignment - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        return false;
    }
    result = value + padding;
    return true;
}

[[nodiscard]] constexpr bool checked_constant_ubo_product(
    size_t lhs, size_t rhs, size_t &result) noexcept {
    if (rhs != 0u && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

}// namespace detail

// Plans one std140 array member after an already planned prefix. Arrays round
// their element alignment up to 16 bytes; their stride is the occupied element
// size rounded up to that alignment. No partial result is returned on failure.
[[nodiscard]] constexpr ConstantUBOMemberLayout plan_constant_ubo_member(
    size_t current_size, size_t element_base_alignment,
    size_t element_occupied_size, size_t element_count,
    size_t max_range = portable_constant_ubo_max_range) noexcept {
    if (element_base_alignment == 0u || element_occupied_size == 0u ||
        element_count == 0u || max_range == 0u ||
        (element_base_alignment & (element_base_alignment - 1u)) != 0u) {
        return {.status = ConstantUBOLayoutStatus::INVALID_LAYOUT};
    }
    size_t member_alignment;
    size_t array_stride;
    size_t member_offset;
    size_t member_size;
    if (!detail::checked_constant_ubo_align_up(
            element_base_alignment, 16u, member_alignment) ||
        !detail::checked_constant_ubo_align_up(
            element_occupied_size, member_alignment, array_stride) ||
        !detail::checked_constant_ubo_align_up(
            current_size, member_alignment, member_offset) ||
        !detail::checked_constant_ubo_product(
            array_stride, element_count, member_size) ||
        member_offset > std::numeric_limits<size_t>::max() - member_size) {
        return {.status = ConstantUBOLayoutStatus::ARITHMETIC_OVERFLOW};
    }
    auto end_offset = member_offset + member_size;
    if (end_offset > max_range) {
        return {.status = ConstantUBOLayoutStatus::RANGE_EXCEEDED};
    }
    return {
        .status = ConstantUBOLayoutStatus::SUCCESS,
        .member_offset = member_offset,
        .array_stride = array_stride,
        .end_offset = end_offset};
}

// Stateful greedy planner used by codegen. A failed append leaves the accepted
// prefix unchanged, which lets the caller fall back for one constant and keep
// considering subsequent constants.
class ConstantUBOLayoutPlanner {
private:
    size_t _size_bytes{0u};
    size_t _max_range;

public:
    constexpr explicit ConstantUBOLayoutPlanner(
        size_t max_range = portable_constant_ubo_max_range) noexcept
        : _max_range{max_range} {}

    [[nodiscard]] constexpr ConstantUBOMemberLayout try_append(
        size_t element_base_alignment,
        size_t element_occupied_size,
        size_t element_count) noexcept {
        auto layout = plan_constant_ubo_member(
            _size_bytes, element_base_alignment,
            element_occupied_size, element_count, _max_range);
        if (layout) { _size_bytes = layout.end_offset; }
        return layout;
    }

    [[nodiscard]] constexpr size_t size_bytes() const noexcept {
        return _size_bytes;
    }
};

}// namespace lc::spirv

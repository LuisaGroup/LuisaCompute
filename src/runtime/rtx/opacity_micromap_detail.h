#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rtx/opacity_micromap.h>

// Internal helpers shared by the frontend implementation and its unit test.
// Everything lives in `opacity_micromap_detail` (a file-unique namespace, as the
// unity/jumbo build requires) and is header-only so the test can include it
// without linking a private symbol.
namespace luisa::compute::opacity_micromap_detail {

using Format = OpacityMicromap::Format;

// A micromap is a bit vector; these are the two unit sizes it packs.
inline constexpr uint32_t bits_per_micro_triangle_two_state = 1u;
inline constexpr uint32_t bits_per_micro_triangle_four_state = 2u;

// Two-state values, VkMicromapTriangleEXT-relative: 0 = transparent (the micro
// triangle is cut out, the intersection is ignored), 1 = opaque.
inline constexpr uint32_t two_state_transparent = 0u;
inline constexpr uint32_t two_state_opaque = 1u;

// Four-state values.
inline constexpr uint32_t four_state_transparent = 0u;
inline constexpr uint32_t four_state_opaque = 1u;
inline constexpr uint32_t four_state_unknown_transparent = 2u;
inline constexpr uint32_t four_state_unknown_opaque = 3u;

[[nodiscard]] inline constexpr uint32_t bits_per_micro_triangle(Format format) noexcept {
    return format == Format::FourState ? bits_per_micro_triangle_four_state :
                                         bits_per_micro_triangle_two_state;
}

// How many micro triangles one 32-bit output word holds.
[[nodiscard]] inline constexpr uint32_t micro_triangles_per_word(Format format) noexcept {
    return 32u / bits_per_micro_triangle(format);
}

// Number of 32-bit words `triangle_count` triangles occupy (the baker's dispatch
// size, always rounded up so the last word is fully written).
[[nodiscard]] inline constexpr uint64_t word_count(
    Format format, uint32_t level, uint64_t triangle_count) noexcept {
    auto micro = OpacityMicromap::micro_triangle_count(level) * triangle_count;
    auto bits = micro * bits_per_micro_triangle(format);
    return (bits + 31u) / 32u;
}

// Packed-bit accessors. `words` must already be large enough; the index is the
// micro-triangle index *within the whole micromap* (triangle-major, then the
// space-filling-curve index inside the triangle).
inline void set_two_state(luisa::span<uint> words, uint64_t index, bool opaque) noexcept {
    auto bit = index;
    auto word = static_cast<size_t>(bit / 32u);
    auto shift = static_cast<uint32_t>(bit % 32u);
    LUISA_ASSERT(word < words.size(), "micromap bit {} is out of range ({} words).", bit, words.size());
    auto mask = 1u << shift;
    words[word] = opaque ? (words[word] | mask) : (words[word] & ~mask);
}

[[nodiscard]] inline bool get_two_state(luisa::span<const uint> words, uint64_t index) noexcept {
    auto bit = index;
    auto word = static_cast<size_t>(bit / 32u);
    auto shift = static_cast<uint32_t>(bit % 32u);
    LUISA_ASSERT(word < words.size(), "micromap bit {} is out of range ({} words).", bit, words.size());
    return (words[word] >> shift) & 1u;
}

inline void set_four_state(luisa::span<uint> words, uint64_t index, uint32_t value) noexcept {
    LUISA_ASSERT(value <= four_state_unknown_opaque, "four-state value {} is out of range.", value);
    auto bit = index * bits_per_micro_triangle_four_state;
    auto word = static_cast<size_t>(bit / 32u);
    auto shift = static_cast<uint32_t>(bit % 32u);
    LUISA_ASSERT(word < words.size(), "micromap bits {} are out of range ({} words).", bit, words.size());
    auto mask = 3u << shift;
    words[word] = (words[word] & ~mask) | (value << shift);
}

[[nodiscard]] inline uint32_t get_four_state(luisa::span<const uint> words, uint64_t index) noexcept {
    auto bit = index * bits_per_micro_triangle_four_state;
    auto word = static_cast<size_t>(bit / 32u);
    auto shift = static_cast<uint32_t>(bit % 32u);
    LUISA_ASSERT(word < words.size(), "micromap bits {} are out of range ({} words).", bit, words.size());
    return (words[word] >> shift) & 3u;
}

}// namespace luisa::compute::opacity_micromap_detail

#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::metal {

struct MetalLibVersion {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;

    [[nodiscard]] constexpr bool operator==(const MetalLibVersion &) const noexcept = default;
};

struct MetalLibTarget {
    MetalLibVersion file_format;
    MetalLibVersion platform;
    MetalLibVersion air;
    MetalLibVersion metal;

    [[nodiscard]] constexpr bool operator==(const MetalLibTarget &) const noexcept = default;
};

enum class MetalLibProgramType : uint8_t {
    VERTEX = 0u,
    FRAGMENT = 1u,
    KERNEL = 2u,
    UNQUALIFIED = 3u,
    VISIBLE = 4u,
    EXTERN = 5u,
    INTERSECTION = 6u
};

struct MetalLibFunction {
    luisa::string_view name;
    luisa::span<const std::byte> air_module;
    MetalLibProgramType type{MetalLibProgramType::KERNEL};
};

[[nodiscard]] MetalLibTarget metallib_target_for_macos(
    uint16_t major, uint16_t minor = 0u, uint16_t patch = 0u) noexcept;

[[nodiscard]] luisa::vector<std::byte> make_metallib(
    const MetalLibTarget &target,
    luisa::span<const MetalLibFunction> functions) noexcept;

[[nodiscard]] bool validate_metallib(
    luisa::span<const std::byte> data,
    luisa::span<const luisa::string_view> expected_entry_points = {},
    luisa::span<const MetalLibProgramType> expected_program_types = {}) noexcept;

}// namespace luisa::compute::metal

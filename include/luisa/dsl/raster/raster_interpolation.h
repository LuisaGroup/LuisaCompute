#pragma once

#include <array>
#include <cstdint>

#include <luisa/ast/attribute.h>
#include <luisa/core/macro.h>

namespace luisa::compute {

/// Interpolation mode for a user vertex-to-fragment varying.
enum class RasterInterpolation : uint8_t {
    DEFAULT,
    CENTER_PERSPECTIVE,
    CENTER_NO_PERSPECTIVE,
    CENTROID_PERSPECTIVE,
    CENTROID_NO_PERSPECTIVE,
    SAMPLE_PERSPECTIVE,
    SAMPLE_NO_PERSPECTIVE,
    FLAT,
};

inline constexpr auto raster_interpolation_attribute_key =
    luisa::string_view{"interpolation"};

[[nodiscard]] constexpr luisa::string_view raster_interpolation_name(
    RasterInterpolation interpolation) noexcept {
    switch (interpolation) {
        case RasterInterpolation::DEFAULT: return "default";
        case RasterInterpolation::CENTER_PERSPECTIVE:
            return "center_perspective";
        case RasterInterpolation::CENTER_NO_PERSPECTIVE:
            return "center_no_perspective";
        case RasterInterpolation::CENTROID_PERSPECTIVE:
            return "centroid_perspective";
        case RasterInterpolation::CENTROID_NO_PERSPECTIVE:
            return "centroid_no_perspective";
        case RasterInterpolation::SAMPLE_PERSPECTIVE:
            return "sample_perspective";
        case RasterInterpolation::SAMPLE_NO_PERSPECTIVE:
            return "sample_no_perspective";
        case RasterInterpolation::FLAT: return "flat";
    }
    return {};
}

[[nodiscard]] inline Attribute raster_interpolation_attribute(
    RasterInterpolation interpolation) noexcept {
    return Attribute{
        luisa::string{raster_interpolation_attribute_key},
        luisa::string{raster_interpolation_name(interpolation)}};
}

[[nodiscard]] inline Attribute raster_position_attribute() noexcept {
    return Attribute{luisa::string{"position"}, {}};
}

[[nodiscard]] constexpr bool parse_raster_interpolation(
    const Attribute &attribute,
    RasterInterpolation &interpolation) noexcept {
    if (attribute.key != raster_interpolation_attribute_key) {
        return false;
    }
    auto value = luisa::string_view{attribute.value};
    if (value == "default") {
        interpolation = RasterInterpolation::DEFAULT;
    } else if (value == "center_perspective") {
        interpolation = RasterInterpolation::CENTER_PERSPECTIVE;
    } else if (value == "center_no_perspective") {
        interpolation = RasterInterpolation::CENTER_NO_PERSPECTIVE;
    } else if (value == "centroid_perspective") {
        interpolation = RasterInterpolation::CENTROID_PERSPECTIVE;
    } else if (value == "centroid_no_perspective") {
        interpolation = RasterInterpolation::CENTROID_NO_PERSPECTIVE;
    } else if (value == "sample_perspective") {
        interpolation = RasterInterpolation::SAMPLE_PERSPECTIVE;
    } else if (value == "sample_no_perspective") {
        interpolation = RasterInterpolation::SAMPLE_NO_PERSPECTIVE;
    } else if (value == "flat") {
        interpolation = RasterInterpolation::FLAT;
    } else {
        return false;
    }
    return true;
}

}// namespace luisa::compute

#define LUISA_RASTER_VARYING_INTERPOLATION_ATTRIBUTE(mode) \
    ::luisa::compute::raster_interpolation_attribute(      \
        ::luisa::compute::RasterInterpolation::mode)

/// Declares interpolation for every varying after the mandatory first
/// float4 position member. Place this declaration inside the reflected host
/// structure and list one RasterInterpolation enumerator per remaining member.
#define LUISA_RASTER_VARYING_INTERPOLATION(...)                                         \
    [[nodiscard]] static auto luisa_compute_member_attributes() noexcept {              \
        return std::array{                                                              \
            ::luisa::compute::raster_position_attribute(),                              \
            LUISA_MAP_LIST(LUISA_RASTER_VARYING_INTERPOLATION_ATTRIBUTE, __VA_ARGS__)}; \
    }

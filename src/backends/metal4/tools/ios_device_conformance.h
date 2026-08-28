#pragma once

#include <array>
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>

namespace luisa::compute::metal {

struct IOSMetal4ConformanceResult {
    bool success{};
    luisa::string failed_stage;
    luisa::string error;
    luisa::string printer_message;
    luisa::string acceleration_structure_path;
    luisa::string motion_blur;
    luisa::string component_motion;
    uint32_t bindless_value{};
    uint32_t indirect_checksum{};
    uint32_t raster_colored_pixels{};
    std::array<uint8_t, 4u> raster_center{};
    uint32_t path_trace_nonblack_pixels{};
    uint8_t path_trace_max_channel{};
    double path_trace_mean_luma{};
    double printer_ms{};
    double bindless_indirect_ms{};
    double raster_compile_ms{};
    double raster_dispatch_readback_ms{};
    double acceleration_build_ms{};
    double path_trace_compile_ms{};
    double path_trace_dispatch_readback_ms{};
    luisa::vector<std::array<uint8_t, 4u>> pixels;
};

[[nodiscard]] IOSMetal4ConformanceResult
run_ios_metal4_conformance(
    Device &device,
    uint32_t width,
    uint32_t height,
    uint32_t samples_per_pixel) noexcept;

}// namespace luisa::compute::metal

#pragma once

#include <array>
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>

namespace luisa::compute::metal {

struct Metal4DeviceConformanceResult {
    bool success{};
    luisa::string failed_stage;
    luisa::string error;
    luisa::string printer_message;
    luisa::string device_name;
    luisa::string gpu_family;
    luisa::string metal4_runtime;
    luisa::string ray_tracing;
    luisa::string acceleration_structure_path;
    luisa::string motion_blur;
    luisa::string component_motion;
    uint32_t abi_layout_checksum{};
    uint32_t atomic_value{};
    std::array<float, 4u> texture_read{};
    uint32_t native_include_checksum{};
    uint64_t timeline_value{};
    uint32_t matrix_motion_hit_count{};
    uint32_t component_motion_hit_count{};
    double matrix_motion_centroid_delta{};
    double component_motion_centroid_delta{};
    double motion_instance_ms{};
    bool matrix_motion_valid{};
    bool component_motion_exercised{};
    bool component_motion_valid{};
    uint32_t bindless_value{};
    uint32_t indirect_checksum{};
    uint32_t raster_colored_pixels{};
    uint32_t raster_stencil_colored_pixels{};
    std::array<uint8_t, 4u> raster_center{};
    uint32_t path_trace_nonblack_pixels{};
    uint8_t path_trace_max_channel{};
    double path_trace_mean_luma{};
    double printer_ms{};
    double compute_abi_ms{};
    double native_include_ms{};
    double timeline_event_ms{};
    double bindless_indirect_ms{};
    double raster_compile_ms{};
    double raster_dispatch_readback_ms{};
    double acceleration_build_ms{};
    double path_trace_compile_ms{};
    double path_trace_dispatch_readback_ms{};
    luisa::vector<std::array<uint8_t, 4u>> pixels;
};

[[nodiscard]] Metal4DeviceConformanceResult
run_metal4_device_conformance(
    Device &device,
    uint32_t width,
    uint32_t height,
    uint32_t samples_per_pixel) noexcept;

}// namespace luisa::compute::metal

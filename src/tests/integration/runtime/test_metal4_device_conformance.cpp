#include "ut/ut.hpp"
#include "test_device.h"

#include <stb/stb_image_write.h>

#include "metal4_device_conformance.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);

    constexpr auto width = 256u;
    constexpr auto height = 256u;
    constexpr auto samples_per_pixel = 4u;
    auto result = metal::run_metal4_device_conformance(
        device, width, height, samples_per_pixel);
    if (!result.success) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal4 device conformance failed at '{}': {}",
            result.failed_stage, result.error);
        return 1;
    }

    constexpr auto output_path = "metal4_device_conformance.png";
    if (stbi_write_png(
            output_path, static_cast<int>(width), static_cast<int>(height),
            4, result.pixels.data(), static_cast<int>(width * 4u)) == 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to save Metal4 conformance image to '{}'.", output_path);
        return 2;
    }

    LUISA_INFO(
        "Metal4 device conformance passed: device='{}' family='{}', "
        "runtime='{}', ray_tracing='{}', AS='{}', motion='{}', "
        "component_motion='{}', printer='{}' ({:.2f} ms), "
        "ABI checksum={} atomic={} texture=({:.6f}, {:.6f}, {:.6f}, {:.6f}) "
        "({:.2f} ms), native_include={} ({:.2f} ms), "
        "timeline=0x{:016x} ({:.2f} ms), "
        "motion=matrix({} hits, delta={:.2f}) component({}: {} hits, "
        "delta={:.2f}) ({:.2f} ms), "
        "bindless=0x{:08x}, indirect_checksum={}, bindless/indirect={:.2f} ms, "
        "raster={} pixels stencil={} pixels center=({}, {}, {}, {}) "
        "compile={:.2f} ms draw={:.2f} ms, "
        "AS build={:.2f} ms, RTX compile={:.2f} ms, RTX dispatch={:.2f} ms, "
        "nonblack={}, max={}, mean_luma={:.6f}, output='{}'.",
        result.device_name,
        result.gpu_family,
        result.metal4_runtime,
        result.ray_tracing,
        result.acceleration_structure_path,
        result.motion_blur,
        result.component_motion,
        result.printer_message,
        result.printer_ms,
        result.abi_layout_checksum,
        result.atomic_value,
        result.texture_read[0u],
        result.texture_read[1u],
        result.texture_read[2u],
        result.texture_read[3u],
        result.compute_abi_ms,
        result.native_include_checksum,
        result.native_include_ms,
        result.timeline_value,
        result.timeline_event_ms,
        result.matrix_motion_hit_count,
        result.matrix_motion_centroid_delta,
        result.component_motion_exercised,
        result.component_motion_hit_count,
        result.component_motion_centroid_delta,
        result.motion_instance_ms,
        result.bindless_value,
        result.indirect_checksum,
        result.bindless_indirect_ms,
        result.raster_colored_pixels,
        result.raster_stencil_colored_pixels,
        result.raster_center[0u],
        result.raster_center[1u],
        result.raster_center[2u],
        result.raster_center[3u],
        result.raster_compile_ms,
        result.raster_dispatch_readback_ms,
        result.acceleration_build_ms,
        result.path_trace_compile_ms,
        result.path_trace_dispatch_readback_ms,
        result.path_trace_nonblack_pixels,
        result.path_trace_max_channel,
        result.path_trace_mean_luma,
        output_path);
    return 0;
}

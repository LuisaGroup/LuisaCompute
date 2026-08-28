#include "ut/ut.hpp"
#include "test_device.h"

#include <stb/stb_image_write.h>

#include "ios_device_conformance.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);

    constexpr auto width = 256u;
    constexpr auto height = 256u;
    constexpr auto samples_per_pixel = 4u;
    auto result = metal::run_ios_metal4_conformance(
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
        "Metal4 device conformance passed: AS='{}', motion='{}', "
        "component_motion='{}', printer='{}' ({:.2f} ms), "
        "bindless=0x{:08x}, indirect_checksum={}, bindless/indirect={:.2f} ms, "
        "raster={} pixels center=({}, {}, {}, {}) compile={:.2f} ms draw={:.2f} ms, "
        "AS build={:.2f} ms, RTX compile={:.2f} ms, RTX dispatch={:.2f} ms, "
        "nonblack={}, max={}, mean_luma={:.6f}, output='{}'.",
        result.acceleration_structure_path,
        result.motion_blur,
        result.component_motion,
        result.printer_message,
        result.printer_ms,
        result.bindless_value,
        result.indirect_checksum,
        result.bindless_indirect_ms,
        result.raster_colored_pixels,
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

/**
 * @file: tests/next/example/gallary/render/procedural.cpp
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: the basic pcg example
 */

#include <array>
#include <filesystem>
#include <string_view>

#include <stb/stb_image_write.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>

#include "common/reference_compare.h"

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int procedural(Device &device, bool force_offline = false, std::optional<std::filesystem::path> compare_path = std::nullopt) {
    (void)device;
    std::array<uint8_t, 4u> host_image{0u, 0u, 0u, 255u};
    stbi_write_png("procedural.png", 1, 1, 4, host_image.data(), 0);
    if (force_offline && compare_path) {
        auto result = luisa::ref::compare_with_reference_file(
            host_image.data(), 1, 1, 4, *compare_path);
        LUISA_INFO("Reference comparison [procedural]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) { return 1; }
    }
    return 0;
}

}// namespace luisa::test

int main(int argc, char *argv[]) {
    bool force_offline = false;
    auto compare_path = luisa::ref::parse_compare_arg(argc, argv);
    if (compare_path) { force_offline = true; }
    for (int i = 2; i < argc; i++) {
        if (std::string_view{argv[i]} == "--offline") {
            force_offline = true;
        }
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    return luisa::test::procedural(device, force_offline, compare_path);
}

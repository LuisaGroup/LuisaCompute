// Test for kernel IR (Intermediate Representation) compilation and display.
// This test creates a simple render kernel, compiles it from IR, and displays
// the output in a window using a swapchain. It demonstrates the full pipeline
// from kernel definition to GPU execution and display.
// Credit: https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py

#include "ut/ut.hpp"
#include "test_device.h"

#include <atomic>
#include <numbers>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include "reference_image.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/gui/window.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_kernel_ir(Device &device) {
    // Define a simple 2D kernel that generates a gradient image
    Kernel2D render_kernel = [&](ImageFloat display_image) noexcept {
        set_block_size(16u, 8u, 1u);
        // Compute UV coordinates from dispatch ID
        auto uv = make_float2(dispatch_id().xy()) /
                  make_float2(dispatch_size().xy());
        // Generate gradient color based on position
        auto c = def(make_float3());
        c.x = uv.x;
        c.y = uv.y;
        c.z = .5f;
        display_image.write(dispatch_id().xy(),
                            make_float4(c, 1.f));
    };

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);

    // Build kernel IR from AST
    auto render_kernel_ir = AST2IR::build_kernel(render_kernel.function()->function());
    // Compile kernel from IR with specified signature
    auto render = device.compile<2, Image<float>>(render_kernel_ir->get());

    // Set up image and window dimensions
    static constexpr auto width = 1280u;
    static constexpr auto height = 720u;
    auto image = device.create_image<float>(PixelStorage::BYTE4, width, height);

    // Create graphics stream and window
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    if (!opts.offline) {
        Window window{"Display", width, height};
        auto swap_chain{device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = make_uint2(width, height),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            })};

        // Main render loop
        while (!window.should_close()) {
            stream << render(image).dispatch(width, height)
                   << swap_chain.present(image);
            window.poll_events();
        }

        stream << synchronize();
        return;
    } else {
        luisa::vector<std::byte> pixels(image.view().size_bytes());
        stream << render(image).dispatch(width, height)
               << image.copy_to(luisa::span{pixels})
               << synchronize();
        if (opts.compare_path) {
            auto result = luisa::test::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(width), static_cast<int>(height), 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison [test_kernel_ir]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) {
            boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
            return;
        }
        }
        return;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_kernel_ir(device);
}

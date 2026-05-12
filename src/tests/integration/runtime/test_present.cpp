// Swapchain Presentation Timing Test
// Measures the timing of swapchain present operations.
// Useful for analyzing display latency and VSync behavior.
//
// Features demonstrated:
// - Swapchain present timing measurement
// - Statistical analysis of frame times
// - Empty frame presentation

#include "ut/ut.hpp"
#include "test_device.h"

#include <iostream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include "../../reference_image.h"
#include <luisa/gui/window.h>
#include <luisa/ast/ast2json.h>
using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_present(Device &device) {

    log_level_verbose();

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    (void)opts;
    // Note: test_present directly tests swapchain presentation timing, offline mode not applicable
    static constexpr uint2 resolution = make_uint2(1024u);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    // Create window and swapchain
    Window window{"path tracing", resolution};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 3,
        });
    Image<float> ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
    double last_time = 0.0;
    uint frame_count = 0u;
    Clock clock;

    // Statistics for timing analysis
    double mean_dt = 0;
    double mean_dt2 = 0;
    size_t cnt = 0;
    while (!window.should_close()) {
        const auto tic = clock.toc();
        stream << swap_chain.present(ldr_image) << synchronize();
        window.poll_events();
        const auto toc = clock.toc();

        const auto dt = toc - tic;
        mean_dt += dt;
        mean_dt2 += dt * dt;
        cnt++;
        LUISA_INFO("mean: {}, var:{}", mean_dt / cnt, -(mean_dt * mean_dt / cnt - mean_dt2));
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_present(device);
}

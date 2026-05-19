// Test for runtime features including:
// - Multi-stream execution (compute and graphics streams)
// - Event synchronization between streams
// - Timeline events for frame pacing
// - Triple-buffering implementation
// - Swapchain presentation
// - HDR and VSync configuration
// - Real-time rendering loop
// - Stream statistics and profiling

#include "ut/ut.hpp"
#include "test_device.h"

#include <numeric>

#include "reference_image.h"

#include <filesystem>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/dynamic_module.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/stats_ext.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_runtime(Device &device_from_ut) {
    luisa::log_level_verbose();

    auto argv = boost::ut::detail::cfg::largv;
    (void)device_from_ut;
    Buffer<float> buffer;
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);

    // Configure device with explicit settings
    Context context{argv[0]};
    DeviceConfig device_config{
        .device_index = 0,
        .inqueue_buffer_limit = false};
    // To avoid memory overflows, the backend automatically waits 2 - 3 frames before committing,
    // set .inqueue_buffer_limit to false when multi-stream interactions are involved
    Device device = context.create_device(argv[1], &device_config, true /*use validation layer for debug*/);

    // Get statistics extension for profiling
    auto stats = device.extension<StatsExt>();

    // Create graphics stream for presentation
    Stream graphics_stream = device.create_stream(StreamTag::GRAPHICS);
    // Create compute stream for kernel execution
    Stream compute_stream = device.create_stream(StreamTag::COMPUTE);

    // Event to let graphics stream wait for compute stream
    Event compute_event = device.create_event();

    // Triple-buffer implementation
    static constexpr uint32_t framebuffer_count = 3;
    // Event to let host wait for kernel before 3 frames
    TimelineEvent graphics_event = device.create_timeline_event();

    static constexpr uint2 resolution = make_uint2(1024u);
    if (!opts.offline) {
        Window window{"test runtime", resolution.x, resolution.x};

        // Create swapchain with configuration
        Swapchain swap_chain{device.create_swapchain(
            graphics_stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = resolution,
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = framebuffer_count - 1u})};

        // Create image for rendering
        Image<float> ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
        ldr_image.set_name("present");
        compute_stream.set_name("my compute");
        graphics_stream.set_name("my present");

        // Kernel that renders a time-varying pattern
        Kernel2D kernel = [&](Float time, UInt2 offset, Float z_axis) {
            UInt2 coord = dispatch_id().xy();
            Float2 uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
            ldr_image->write(coord + offset, make_float4(uv, sin(time + z_axis) * 0.5f + 0.5f, 1.0f));
        };
        auto shader = device.compile(kernel);

        Clock clk;
        clk.tic();
        // Fence index is a self-incremental integer
        uint64_t frame_index = 0;

        // Main rendering loop
        while (!window.should_close()) {
            // current frame's index
            uint64_t this_frame = frame_index;
            // next frame's index
            frame_index += 1;

            // Wait for last cycle
            if (this_frame >= framebuffer_count) {
                graphics_event.synchronize(this_frame - (framebuffer_count - 1));
            }

            // Try this: without synchronize, texture will be used by multiple streams simultaneously, this is illegal.
            // If you REALLY want to access one resource with multiple streams simultaneously, you should mark
            // simultaneously_access = true in create_image<T> and create_volume<T>, this may cause performance loss in some backends.
            // Buffer is always simultaneously_accessible

            // #define NO_SYNC_ERROR
// compute stream must wait last frame's graphics stream
#ifndef NO_SYNC_ERROR
            if (this_frame > 0) {
                compute_stream << graphics_event.wait(this_frame);
            }
#endif
            // Begin statistics collection
            if (frame_index == 5) {
                stats->begin_stats();
            }

            // Dispatch compute task
            stats->set_next_dispatch_name("Compute Task");
            compute_stream
                << shader(clk.toc() / 200.0f, uint2(0), pi).dispatch(resolution.x, resolution.y / 2)
                // make a signal after compute_stream's tasks
                << compute_event.signal();

            // Update frame - graphics stream waits for compute
            stats->set_next_dispatch_name("Graphics Task");
            graphics_stream
                // wait compute_stream's tasks
                << compute_event.wait()
                << shader(clk.toc() / 200.0f, uint2(0, resolution.y / 2), 0.0f).dispatch(resolution.x, resolution.y / 2)
                << swap_chain.present(ldr_image)
                // let host wait here
                << graphics_event.signal(frame_index);
            window.poll_events();

            // End statistics collection and print results
            if (frame_index == 5) {
                auto &streams = stats->end_stats();
                for (auto &i : streams) {
                    for (auto &j : i.second.stream_scopes) {
                        LUISA_INFO("Stream {} task {} from time {} to time {}", luisa::to_string(i.second.stream_tag), j->name, j->start_time, j->finished_time);
                    }
                }
            }
        }
        // Final synchronization
        compute_stream << synchronize();
        graphics_stream << synchronize();
        return;
    } else {
        Image<float> ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
        ldr_image.set_name("present");
        compute_stream.set_name("my compute");
        graphics_stream.set_name("my present");

        Kernel2D kernel = [&](Float time, UInt2 offset, Float z_axis) {
            UInt2 coord = dispatch_id().xy();
            Float2 uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
            ldr_image->write(coord + offset, make_float4(uv, sin(time + z_axis) * 0.5f + 0.5f, 1.0f));
        };
        auto shader = device.compile(kernel);
        luisa::vector<std::byte> pixels(ldr_image.view().size_bytes());

        stats->set_next_dispatch_name("Compute Task");
        compute_stream
            << shader(0.0f, uint2(0), pi).dispatch(resolution.x, resolution.y / 2)
            << compute_event.signal();

        stats->set_next_dispatch_name("Graphics Task");
        graphics_stream
            << compute_event.wait()
            << shader(0.0f, uint2(0, resolution.y / 2), 0.0f).dispatch(resolution.x, resolution.y / 2)
            << ldr_image.copy_to(luisa::span{pixels})
            << synchronize();

        if (opts.compare_path) {
            auto result = luisa::test::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison [test_runtime]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
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
    test_runtime(device);
}

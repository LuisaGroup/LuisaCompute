//
// Created by Mike Smith on 2021/2/27.
//

#include <numeric>

#include <core/clock.h>
#include <core/logging.h>
#include <core/dynamic_module.h>
#include <runtime/device.h>
#include <runtime/context.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <runtime/bindless_array.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <gui/window.h>
using namespace luisa;
using namespace luisa::compute;

struct Base {
    float a;
};

struct Derived : Base {
    float b;
    constexpr Derived(float a, float b) noexcept : Base{a}, b{b} {}
};
int main(int argc, char *argv[]) {
    luisa::log_level_verbose();

    Context context{argv[0]};

    Buffer<float> buffer;
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    DeviceConfig device_config{
        .device_index = 0,
        // To avoid memory overflows, the backend automatically waits 2 - 3 frames before committing, set .inqueue_buffer_limit to false when multi-stream interactions are involved
        .inqueue_buffer_limit = false};
    auto device = context.create_device(argv[1]);
    // graphics stream for present
    auto graphics_stream = device.create_stream(StreamTag::GRAPHICS);
    // compute stream for kernel
    auto compute_stream = device.create_stream(StreamTag::COMPUTE);
    // Event to let graphics stream wait compute stream
    auto compute_event = device.create_event();
    // Do triple-buffer implementation here
    // Event to let host wait kernel before 3 frame
    static constexpr uint32_t framebuffer_count = 3;
    std::array<Event, framebuffer_count> graphics_events;
    uint64_t frame{};
    for (auto &i : graphics_events) {
        i = device.create_event();
    }
    static constexpr auto resolution = make_uint2(1024u);
    auto ldr_image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    Kernel2D kernel = [&](Float time) {
        auto coord = dispatch_id().xy();
        auto uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
        ldr_image->write(coord, make_float4(uv, sin(time) * 0.5f + 0.5f, 1.f));
    };
    auto shader = device.compile(kernel);
    Window window{"test runtime", resolution.x, resolution.x, false};
    auto swap_chain{device.create_swapchain(
        window.native_handle(),
        graphics_stream,
        resolution,
        true, false, framebuffer_count - 1)};
    Clock clk;
    clk.tic();
    while (!window.should_close()) {
        auto resource_frame = frame % framebuffer_count;
        auto &grpahics_event = graphics_events[resource_frame];
        grpahics_event.synchronize();
        // Use Commandlist to store commands
        CommandList cmd_list;
        cmd_list << shader(clk.toc() / 200.0f).dispatch(resolution);
        // compute stream must wait last frame's graphics stream
        if (frame > 0) [[likely]] {
            auto &last_event = graphics_events[(frame - 1) % framebuffer_count];
            compute_stream << last_event.wait();
        }
        compute_stream
            << cmd_list.commit()
            // make a signal after compute_stream's tasks
            << compute_event.signal();
        graphics_stream
            // wait compute_stream's tasks
            << compute_event.wait()
            << swap_chain.present(ldr_image)
            // let host wait here
            << grpahics_event.signal();
        window.pool_event();
        frame++;
    }
}

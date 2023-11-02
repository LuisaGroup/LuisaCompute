#include <numeric>

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

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();

    Context context{argv[0]};

    Buffer<float> buffer;
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    DeviceConfig device_config{
        .device_index = 0,
        // To avoid memory overflows, the backend automatically waits 2 - 3 frames before committing, set .inqueue_buffer_limit to false when multi-stream interactions are involved
        .inqueue_buffer_limit = false};
    Device device = context.create_device(argv[1], &device_config, true /*use validation layer for debug*/);
    // graphics stream for present
    Stream graphics_stream = device.create_stream(StreamTag::GRAPHICS);
    // compute stream for kernel
    Stream compute_stream = device.create_stream(StreamTag::COMPUTE);
    // Event to let graphics stream wait compute stream
    Event compute_event = device.create_event();
    // Do triple-buffer implementation here
    // Event to let host wait kernel before 3 frame
    static constexpr uint32_t framebuffer_count = 3;
    TimelineEvent graphics_event = device.create_timeline_event();
    static constexpr uint2 resolution = make_uint2(1024u);
    Window window{"test runtime", resolution.x, resolution.x};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        graphics_stream,
        resolution,
        false, false, framebuffer_count - 1)};
    Image<float> ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
    ldr_image.set_name("present");
    compute_stream.set_name("my compute");
    graphics_stream.set_name("my present");
    Kernel2D kernel = [&](Float time) {
        UInt2 coord = dispatch_id().xy();
        Float2 uv = (make_float2(coord) + 0.5f) / make_float2(dispatch_size().xy());
        ldr_image->write(coord, make_float4(uv, sin(time) * 0.5f + 0.5f, 1.f));
    };
    Shader2D<float> shader = device.compile(kernel);

    Clock clk;
    clk.tic();
    // Fence index is a self-incremental integer
    uint64_t frame_index = 0;
    while (!window.should_close()) {
        // current frame's index
        uint64_t this_frame = frame_index;
        // next frame's index
        frame_index += 1;
        // Wait for last cycle
        if (this_frame >= framebuffer_count) {
            graphics_event.synchronize(this_frame - (framebuffer_count - 1));
        }
        CommandList cmd_list;
        cmd_list << shader(clk.toc() / 200.0f).dispatch(resolution);

        // Try this: without synchronize, texture will be used by multiple streams simultaneously, this is illegal.
        // If you REALLY want to access one resource with multiple streams simultaneously, you should mark simultaneously_access = true in create_image<T> and create_volume<T>, this may cause performance loss in some backends.
        // Buffer is always simultaneously_accessible

        // #define NO_SYNC_ERROR
// compute stream must wait last frame's graphics stream
#ifndef NO_SYNC_ERROR
        if (this_frame > 0) {
            compute_stream << graphics_event.wait(this_frame);
        }
#endif
        compute_stream
            << cmd_list.commit()
            // make a signal after compute_stream's tasks
            << compute_event.signal();
        // update frame
        graphics_stream
            // wait compute_stream's tasks
            << compute_event.wait()
            << swap_chain.present(ldr_image)
            // let host wait here
            << graphics_event.signal(frame_index);
        window.poll_events();
    }
    compute_stream << synchronize();
    graphics_stream << synchronize();
}

#include <iostream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>
#include <luisa/gui/window.h>
#include <luisa/ast/ast2json.h>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    static constexpr uint2 resolution = make_uint2(1024u);
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

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
        // double dt = clock.toc() - last_time;
        // last_time = clock.toc();
        // LUISA_INFO("spp: {}, time: {} ms, spp/s: {}",
        //            frame_count, dt, spp_per_dispatch / dt * 1000);
        LUISA_INFO("mean: {}, var:{}", mean_dt / cnt, -(mean_dt * mean_dt / cnt - mean_dt2));
    }
}

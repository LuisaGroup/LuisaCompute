//
// Created by Mike on 3/31/2023.
//

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <runtime/swapchain.h>
#include <dsl/syntax.h>
#include <gui/window.h>
#include <gui/framerate.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    static constexpr auto width = 2048u;
    static constexpr auto height = 1024u;
    static constexpr auto resolution = make_uint2(width, height);

    auto draw = device.compile<2>([](ImageFloat image, Float time) noexcept {
        auto p = dispatch_id().xy();
        auto uv = make_float2(p) / make_float2(resolution) * 2.0f - 1.0f;
        auto color = def(make_float4());
        Constant<float> scales{pi, luisa::exp(1.f), luisa::sqrt(2.f)};
        for (auto i = 0u; i < 3u; i++) {
            color[i] = cos(time * scales[i] + uv.y * 11.f +
                           sin(-time * scales[2u - i] + uv.x * 7.f) * 4.f) *
                           .5f +
                       .5f;
        }
        color[3] = 1.0f;
        image.write(p, color);
    });

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    auto image = device.create_image<float>(PixelStorage::BYTE4, resolution);

    struct PackagedWindow {
        Window window;
        Swapchain swapchain;
    };

    static constexpr auto window_count = 4u;
    luisa::vector<PackagedWindow> windows;
    windows.reserve(window_count);
    for (auto i = 0u; i < window_count; i++) {
        Window window{luisa::format("Window #{}", i), resolution >> i};
        auto swpachain = device.create_swapchain(
            window.native_handle(), stream,
            resolution, false, false, 3);
        windows.emplace_back(PackagedWindow{
            std::move(window),
            std::move(swpachain)});
    }

    Clock clk;
    Framerate framerate;
    while (std::all_of(windows.cbegin(), windows.cend(), [](auto &&w) noexcept {
        return !w.window.should_close();
    })) {
        stream << draw(image, static_cast<float>(clk.toc() * 1e-3))
                      .dispatch(resolution);
        for (auto &&w : windows) { stream << w.swapchain.present(image); }
        framerate.record(1u);
        LUISA_INFO("FPS: {}", framerate.report());
        for (auto &&w : windows) { w.window.poll_events(); }
    }
}

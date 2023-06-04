//
// Created by Mike on 3/31/2023.
//

#include <stb/stb_image.h>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <gui/window.h>
#include <gui/framerate.h>
#include <runtime/swapchain.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    auto width = 0;
    auto height = 0;
    auto channels = 0;
    auto pixels = stbi_load("logo.png", &width, &height, &channels, 4);
    auto resolution = make_uint2(width, height);
    auto image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    stream << image.copy_from(pixels) << synchronize();
    stbi_image_free(pixels);

    Window window{"Display", resolution};
    auto swapchain = device.create_swapchain(
        window.native_handle(), stream,
        resolution, false, true, 8);

    Clock clk;
    Framerate framerate;
    while (!window.should_close()) {
        stream << swapchain.present(image);
        framerate.record(1u);
        LUISA_INFO("FPS: {}", framerate.report());
        window.poll_events();
    }
}

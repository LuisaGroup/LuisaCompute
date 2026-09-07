// Static Image Swapchain Test
// Demonstrates displaying a static image using the swapchain.
// Loads an image from file and presents it continuously.
//
// Features demonstrated:
// - Loading images with stb_image
// - Creating swapchains for image display
// - Framerate measurement
// - Static content presentation

#include <stb/stb_image.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/gui/window.h>
#include <luisa/gui/framerate.h>
#include <luisa/runtime/swapchain.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, metal, vk, hip, fallback", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // Load image from file using stb_image
    auto width = 0;
    auto height = 0;
    auto channels = 0;
    auto pixels = stbi_load("src/tests/logo.png", &width, &height, &channels, 4);
    auto resolution = make_uint2(width, height);
    auto image = device.create_image<float>(PixelStorage::BYTE4, resolution);
    stream << image.copy_from(luisa::span{pixels, static_cast<size_t>(width * height * 4)}) << synchronize();
    stbi_image_free(pixels);

    // Create window and swapchain
    Window window{"Display", resolution};
    SwapchainOption sc_options{
        .display = window.native_display(),
        .window = window.native_handle(),
        .size = resolution,
        .wants_hdr = false,
        .wants_vsync = true,
        .back_buffer_count = 8,
    };
    auto swapchain = device.create_swapchain(stream, sc_options);

    // Display loop with FPS measurement
    Clock clk;
    Framerate framerate;
    while (!window.should_close()) {
        stream << swapchain.present(image);
        framerate.record(1u);
        LUISA_INFO("FPS: {}", framerate.report());
        window.poll_events();
    }
}

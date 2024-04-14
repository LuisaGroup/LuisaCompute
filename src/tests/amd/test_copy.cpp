#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/gui/window.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    static constexpr uint32_t width = 1024;
    static constexpr uint32_t height = 1024;
    Context context{argv[0]};
    Device device = context.create_device("dx");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Stream copy_stream = device.create_stream(StreamTag::COPY);
    Image<float> image = device.create_image<float>(PixelStorage::BYTE4, width, height);
    Buffer<uint> buffer = device.create_buffer<uint>(width * height);
    luisa::vector<uint8_t> pixels(width * height * 4);
    for (size_t x = 0; x < width; ++x)
        for (size_t y = 0; y < height; ++y) {
            size_t pixel_pos = x + y * width;
            float2 uv = make_float2(x, y) / make_float2(width, height);
            pixels[pixel_pos * 4] = static_cast<uint8_t>(uv.x * 255);
            pixels[pixel_pos * 4 + 1] = static_cast<uint8_t>(uv.y * 255);
            pixels[pixel_pos * 4 + 2] = 127;
            pixels[pixel_pos * 4 + 3] = 255;
        }
    copy_stream << buffer.copy_from(pixels.data())
                // Sub range copy failed in AMD Radeon RX 7900 XTX
                << luisa::make_unique<BufferToTextureCopyCommand>(
                       buffer.handle(),
                       0,
                       image.handle(),
                       image.storage(),
                       0,
                       uint3(width, height / 2, 1),// half range
                       uint3(0))
                // Full range copy is fine
                // << luisa::make_unique<BufferToTextureCopyCommand>(
                //        buffer.handle(),
                //        0,
                //        image.handle(),
                //        image.storage(),
                //        0,
                //        uint3(width, height, 1),
                //        uint3(0))

                << synchronize();
    Window window{"test copy", uint2(width, height)};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = make_uint2(width, height),
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });
    while (!window.should_close()) {
        window.poll_events();
        stream << swap_chain.present(image);
    }
}
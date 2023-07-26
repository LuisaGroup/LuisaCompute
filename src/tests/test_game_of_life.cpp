#include <iostream>
#include <random>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/sugar.h>
#include <luisa/gui/window.h>
#include <luisa/runtime/swapchain.h>

using namespace luisa;
using namespace luisa::compute;

struct ImagePair {
    Image<uint> prev;
    Image<uint> curr;
    ImagePair(Device &device, PixelStorage storage, uint width, uint height) noexcept
        : prev{device.create_image<uint>(storage, width, height)},
          curr{device.create_image<uint>(storage, width, height)} {}
    void swap() noexcept { std::swap(prev, curr); }
};

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Keys: SPACE - Run/Pause, R - Reset, ESC - Quit");

    Callable read_state = [](ImageUInt prev, UInt2 uv) noexcept {
        return prev.read(uv).x == 255u;
    };

    Kernel2D kernel = [&](ImageUInt prev, ImageUInt curr) noexcept {
        set_block_size(16, 16, 1);
        UInt count = def(0u);
        UInt2 uv = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        Bool state = read_state(prev, uv);
        Int2 p = make_int2(uv);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    Int2 q = p + make_int2(dx, dy) + make_int2(size);
                    Bool neighbor = read_state(prev, make_uint2(q) % size);
                    count += ite(neighbor, 1, 0);
                }
            }
        }
        Bool c0 = count == 2u;
        Bool c1 = count == 3u;
        curr.write(uv, make_uint4(make_uint3(ite((state & c0) | c1, 255u, 0u)), 255u));
    };
    Shader2D<Image<uint>, Image<uint>> shader = device.compile(kernel);
    Kernel2D display_kernel = [&](ImageUInt in_tex, ImageFloat out_tex) noexcept {
        set_block_size(16, 16, 1);
        UInt2 uv = dispatch_id().xy();
        UInt2 coord = uv / 4u;
        UInt4 value = in_tex.read(coord);
        out_tex.write(uv, make_float4(value) / 255.0f);
    };
    Shader2D<Image<uint>, Image<float>> display_shader = device.compile(display_kernel);
    static constexpr uint width = 128u;
    static constexpr uint height = 128u;
    ImagePair image_pair{device, PixelStorage::BYTE4, width, height};

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    std::mt19937 rng{std::random_device{}()};
    Window window{"Game of Life", width * 4u, height * 4u};
    Swapchain swap_chain = device.create_swapchain(window.native_handle(), stream, window.size());
    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), window.size());

    // reset
    luisa::vector<uint> host_image;
    host_image.push_back_uninitialized(width * height);
    for (uint v : host_image) {
        uint x = (rng() % 4u == 0u) * 255u;
        v = x * 0x00010101u | 0xff000000u;
    }
    stream << image_pair.prev.copy_from(host_image.data()) << synchronize();
    while (!window.should_close()) {
        stream << shader(image_pair.prev, image_pair.curr).dispatch(width, height)
               << display_shader(image_pair.curr, display).dispatch(width * 4u, height * 4u)
               << swap_chain.present(display);
        image_pair.swap();
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
        window.poll_events();
    }
    stream << synchronize();
}


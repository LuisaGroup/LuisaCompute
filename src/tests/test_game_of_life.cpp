//
// Created by Mike Smith on 2021/6/25.
//

#include <iostream>
#include <random>

#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/sugar.h>
#include <gui/window.h>
#include <runtime/swap_chain.h>

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
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    LUISA_INFO("Keys: SPACE - Run/Pause, R - Reset, ESC - Quit");

    Callable read_state = [](ImageUInt prev, UInt2 uv) noexcept {
        return prev.read(uv).x == 255u;
    };

    Kernel2D kernel = [&](ImageUInt prev, ImageUInt curr) noexcept {
        set_block_size(16, 16, 1);
        auto count = def(0u);
        auto uv = dispatch_id().xy();
        auto size = dispatch_size().xy();
        auto state = read_state(prev, uv);
        auto p = make_int2(uv);
        for (auto dy = -1; dy <= 1; dy++) {
            for (auto dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    auto q = p + make_int2(dx, dy) + make_int2(size);
                    auto neighbor = read_state(prev, make_uint2(q) % size);
                    count += neighbor;
                }
            }
        }
        auto c0 = count == 2u;
        auto c1 = count == 3u;
        curr.write(uv, make_uint4(make_uint3(ite((state & c0) | c1, 255u, 0u)), 255u));
    };
    auto shader = device.compile(kernel);
    Kernel2D display_kernel = [&](ImageUInt in_tex, ImageFloat out_tex) noexcept {
        set_block_size(16, 16, 1);
        auto uv = dispatch_id().xy();
        auto coord = uv / 4u;
        auto value = in_tex.read(coord);
        out_tex.write(uv, make_float4(value) / 255.0f);
    };
    auto display_shader = device.compile(display_kernel);
    static constexpr auto width = 128u;
    static constexpr auto height = 128u;
    ImagePair image_pair{device, PixelStorage::BYTE4, width, height};

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    std::mt19937 rng{std::random_device{}()};
    Window window{"Game of Life", width * 4u, height * 4u, false};
    auto swap_chain = device.create_swapchain(window.native_handle(), stream, window.size());
    auto display = device.create_image<float>(PixelStorage::BYTE4, window.size());

    auto reset = [&] {
        luisa::vector<uint> host_image;
        host_image.push_back_uninitialized(width * height);
        for (auto &v : host_image) {
            auto x = (rng() % 4u == 0u) * 255u;
            v = x * 0x00010101u | 0xff000000u;
        }
        stream << image_pair.prev.copy_from(host_image.data()) << synchronize();
    };
    reset();
    while (!window.should_close()) {
        stream << shader(image_pair.prev, image_pair.curr).dispatch(width, height)
               << display_shader(image_pair.curr, display).dispatch(width * 4u, height * 4u)
               << swap_chain.present(display);
        image_pair.swap();
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
        window.pool_event();
    }
    stream << synchronize();
}

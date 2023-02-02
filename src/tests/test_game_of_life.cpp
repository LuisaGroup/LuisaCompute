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
#include <gui/backup/window.h>

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

LUISA_BINDING_GROUP(ImagePair, prev, curr)

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    LUISA_INFO("Keys: SPACE - Run/Pause, R - Reset, ESC - Quit");

    Callable read_state = [](Var<ImagePair> pair, UInt2 uv) noexcept {
        return pair.prev.read(uv).x == 255u;
    };

    Kernel2D kernel = [&](Var<ImagePair> pair) noexcept {
        auto count = def(0u);
        auto uv = dispatch_id().xy();
        auto size = dispatch_size().xy();
        auto state = read_state(pair, uv);
        auto p = make_int2(uv);
        for (auto dy = -1; dy <= 1; dy++) {
            for (auto dx = -1; dx <= 1; dx++) {
                if (dx != 0 || dy != 0) {
                    auto q = p + make_int2(dx, dy) + make_int2(size);
                    auto neighbor = read_state(pair, make_uint2(q) % size);
                    count += neighbor;
                }
            }
        }
        auto c0 = count == 2u;
        auto c1 = count == 3u;
        pair.curr.write(uv, make_uint4(make_uint3(ite((state & c0) | c1, 255u, 0u)), 255u));
    };
    auto shader = device.compile(kernel);

    static constexpr auto width = 128u;
    static constexpr auto height = 128u;
    ImagePair image_pair{device, PixelStorage::BYTE4, width, height};
    auto stream = device.create_stream();

    auto should_start = false;
    luisa::vector<uint> host_image(width * height);
    std::mt19937 rng{std::random_device{}()};
    auto reset = [&] {
        for (auto &v : host_image) {
            auto x = (rng() % 4u == 0u) * 255u;
            v = x * 0x00010101u | 0xff000000u;
        }
        stream << image_pair.prev.copy_from(host_image.data());
        should_start = false;
    };

    Window window{"Game of Life", make_uint2(width, height) * 4u};
    window.set_key_callback([&](int key, int action) noexcept {
        if (action == GLFW_PRESS && (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)) {
            window.set_should_close();
        }
        if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
            should_start = !should_start;
        }
        if (action == GLFW_PRESS && key == GLFW_KEY_R) {
            reset();
        }
    });

    reset();
    while (!window.should_close()) {
        if (should_start) {
            stream << shader(image_pair).dispatch(width, height)
                   << image_pair.curr.copy_to(host_image.data())
                   << synchronize();
            image_pair.swap();
        }
        window.run_one_frame([&] {
            using Pixel = std::array<uint8_t, 4>;
            auto p = reinterpret_cast<const Pixel *>(host_image.data());
            window.set_background(p, make_uint2(width, height), false);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }
}

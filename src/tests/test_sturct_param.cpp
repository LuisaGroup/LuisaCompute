//
// Created by Mike on 2023/2/5.
//

#include <luisa-compute.h>
#include <dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct foo {
    bool b1;
    bool b2;
    luisa::float3 test1;
    luisa::int2 a;
    luisa::float3 test2[2];
    bool b3;
};
LUISA_STRUCT(foo, b1, b2, test1, a, test2, b3){};

int main(int args, char *argv[]) {
    std::vector<std::array<uint8_t, 4u>> download_image(1280 * 720);
    {
        Context context{argv[0]};
        auto device = context.create_device("dx");
        auto stream = device.create_stream();
        auto device_image = device.create_image<float>(PixelStorage::BYTE4, 1280, 720, 0u);

        Kernel2D fill_image_kernel = [&](Var<foo> bar) noexcept {
            Var coord = dispatch_id().xy();
            device_image.write(coord, make_float4(bar.test2[1], 1.f));
        };
        auto fill_image = device.compile(fill_image_kernel);

        foo b;
        b.test2[1] = {0, 1, 0};

        stream
            << fill_image(b).dispatch(1280, 720)
            << device_image.copy_to(download_image.data())
            << synchronize();// Step 5: Synchronize the stream
    }

    Window window{"Display", uint2{1280, 720}};
    window.set_key_callback([&](int key, int action) noexcept {
        if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
            window.set_should_close();
        }
    });
    auto frame_count = 0u;
    window.run([&] {
        window.set_background(download_image.data(), uint2{1280, 720});
    });
}

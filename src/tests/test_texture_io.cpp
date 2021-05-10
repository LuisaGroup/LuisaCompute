//
// Created by Mike Smith on 2021/4/6.
//

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tests/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

#if defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    Kernel2D clear_image = [](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        image.write(coord, float4{float3{}, 1.0f});
    };

    Kernel2D fill_image = [](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(launch_size().xy());
        image.write(coord, make_float4(rg, 1.0f, 1.0f));
    };

    device.compile(clear_image, fill_image);
    /*auto device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u);
    std::vector<uint8_t> host_image(1024u * 1024u * 4u);

    auto event = device.create_event();
    auto stream = device.create_stream();
    auto copy_stream = device.create_stream();

    stream << clear_image(device_image).launch(1024u, 1024u)
           << fill_image(device_image.view(256u, 512u)).launch(512u, 512u)
           << event.signal();

    copy_stream << event.wait()
                << device_image.view().copy_to(host_image.data())
                << event.signal();

    event.synchronize();
    stbi_write_png("result.png", 1024u, 1024u, 4u, host_image.data(), 0u);

    auto volume = device.create_volume<float>(PixelStorage::FLOAT4, 64u, 64u, 64u);*/
}

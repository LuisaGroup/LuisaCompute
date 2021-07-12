//
// Created by Mike Smith on 2021/4/6.
//

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tests/stb_image.h>
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

    Kernel2D clear_image_kernel = [](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, make_float4(make_float2(0.3f, 0.4f), 0.5f, 1.0f));
    };

    Kernel2D fill_image_kernel = [](TextureHeapVar heap, ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var uv = make_float2(coord) / make_float2(dispatch_size().xy()) * 3.0f - 1.0f;
        image.write(coord, heap.sample(0u, uv));
    };

    auto clear_image = device.compile(clear_image_kernel);
    auto fill_image = device.compile(fill_image_kernel);

    auto texture_heap = device.create_texture_heap(512_mb);
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("src/tests/logo.png", &image_width, &image_height, &image_channels, 4);
    auto texture = texture_heap.create(0u, PixelStorage::BYTE4, uint2(image_width, image_height), TextureSampler::bilinear_edge(), 1u);
    auto device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u);
    std::vector<uint8_t> host_image(1024u * 1024u * 4u);

    auto event = device.create_event();
    auto stream = device.create_stream();
    auto upload_stream = device.create_stream();

    upload_stream << texture.load(image_pixels)
                  << event.signal();

    stream << clear_image(device_image).dispatch(1024u, 1024u)
           << event.wait()
           << fill_image(texture_heap,
                         device_image.view(make_uint2(128u), make_uint2(1024u - 256u)))
                  .dispatch(make_uint2(1024u - 256u))
           << device_image.view().copy_to(host_image.data())
           << event.signal();

    event.synchronize();
    stbi_write_png("result.png", 1024u, 1024u, 4u, host_image.data(), 0u);
}

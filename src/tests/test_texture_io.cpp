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

#if defined(LUISA_BACKEND_CUDA_ENABLED)
    auto device = context.create_device("cuda");
#elif defined(LUISA_BACKEND_METAL_ENABLED)
    auto device = context.create_device("metal");
#elif defined(LUISA_BACKEND_DX_ENABLED)
    auto device = context.create_device("dx");
#else
    auto device = FakeDevice::create(context);
#endif

    Callable linear_to_srgb = [](Float4 linear) noexcept {
        auto x = linear.xyz();
        return make_float4(
            select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                   12.92f * x,
                   x <= 0.00031308f),
            linear.w);
    };

    Kernel2D clear_image_kernel = [](ImageFloat image) noexcept {
        auto coord = dispatch_id().xy();
        auto rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, make_float4(make_float2(0.3f, 0.4f), 0.5f, 1.0f));
    };

    Kernel2D fill_image_kernel = [&linear_to_srgb](ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, linear_to_srgb(make_float4(rg, 1.0f, 1.0f)));
    };

    Kernel2D fill_buffer_kernel = [](BufferFloat4 image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image[coord.x + coord.y * dispatch_size_x()] = make_float4(rg, 1.0f, 1.0f);
    };

    Kernel2D copy_texture_kernel = [](BufferFloat4 buffer, ImageFloat image) noexcept {
        Var coord = dispatch_id().xy();
        buffer[coord.x + coord.y * dispatch_size_x()] = image.read(coord);
    };

    auto clear_image = device.compile(clear_image_kernel);
    auto fill_image = device.compile(fill_image_kernel);
    auto fill_buffer = device.compile(fill_buffer_kernel);
    auto copy = device.compile(copy_texture_kernel);

    //    for (auto i = 0u; i < 2u; i++) {
    //        clear_image = device.compile(clear_image_kernel);
    //        fill_image = device.compile(fill_image_kernel);
    //    }

    auto device_image = device.create_image<float>(PixelStorage::FLOAT4, 1024u, 1024u, 1u);
    std::vector<float> host_image(1024u * 1024u * 4u, 1.0f);
    std::vector<float> download_image(1024u * 1024u * 4u);
    auto device_buffer = device.create_buffer<float4>(1024 * 1024);

    auto event = device.create_event();
    auto stream = device.create_stream();

    stream << clear_image(device_image.view(0)).dispatch(1024u, 1024u)
           << fill_image(device_image.view(0).region(make_uint2(256u), make_uint2(512u))).dispatch(512u, 512u)
           //           << fill_buffer(device_buffer).dispatch(1024, 1024)
           //           << device_buffer.copy_from(host_image.data())
           //           << device_image.view(0).copy_to(device_buffer.view())
           << copy(device_buffer, device_image).dispatch(1024, 1024)
           << device_buffer.copy_to(download_image.data())
           << synchronize();

    //    event.synchronize();
    stbi_write_hdr("result.hdr", 1024, 1024, 4, download_image.data());
    //    stbi_write_png("result.png", 1024u, 1024u, 4u, host_image.data(), 0u);

    auto volume = device.create_volume<float>(PixelStorage::FLOAT4, 64u, 64u, 64u);
}

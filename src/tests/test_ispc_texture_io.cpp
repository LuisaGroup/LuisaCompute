//
// Created by Mike Smith on 2021/4/6.
//

#include <stb/stb_image_write.h>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <tests/fake_device.h>
#include <backends/ispc/ISPCTest/Types.h>

using namespace luisa;
using namespace luisa::compute;

#include <cstdio>

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};

    auto device = context.create_device("ispc");

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
        image.write(coord.x + coord.y * dispatch_size_x(), make_float4(rg, 1.0f, 1.0f));
    };

    Kernel2D copy_texture_kernel = [](BufferFloat4 buffer, ImageFloat image) noexcept {
        Var coord = dispatch_id().xy();
        buffer.write(coord.x + coord.y * dispatch_size_x(), image.read(coord));
    };

    auto clear_image = device.compile(clear_image_kernel);
    auto fill_image = device.compile(fill_image_kernel);
    auto fill_buffer = device.compile(fill_buffer_kernel);
    auto copy = device.compile(copy_texture_kernel);

    //    for (auto i = 0u; i < 2u; i++) {
    //        clear_image = device.compile(clear_image_kernel);
    //        fill_image = device.compile(fill_image_kernel);
    //    }

    LUISA_INFO("kernel compiled");
    auto device_image = device.create_image<float>(PixelStorage::FLOAT4, 1024u, 1024u, 1u);
    LUISA_WARNING("texture dim {} {}", reinterpret_cast<Texture2D*>(device_image.handle())->width, reinterpret_cast<Texture2D*>(device_image.handle())->height);
    std::vector<float> download_image(1024u * 1024u * 4u);
    auto device_buffer = device.create_buffer<float4>(1024 * 1024);

    auto event = device.create_event();
    auto stream = device.create_stream();

    LUISA_INFO("a: {}", 1); 
    stream << clear_image(device_image.view(0)).dispatch(1024u, 1024u) << synchronize();
    LUISA_INFO("a: {}", 2);
    stream << fill_image(device_image.view(0).region(make_uint2(256u), make_uint2(512u))).dispatch(512u, 512u) << synchronize();
    LUISA_INFO("a: {}", 3);
    stream << fill_buffer(device_buffer).dispatch(1024, 1024) << synchronize();
    LUISA_INFO("a: {}", 4);
    stream << device_image.view(0).copy_to(download_image.data()) << synchronize();
    LUISA_INFO("a: {}", 5);
        stbi_write_hdr("result.hdr", 1024u, 1024u, 4u, download_image.data());

    auto volume = device.create_volume<float>(PixelStorage::FLOAT4, 64u, 64u, 64u);
}

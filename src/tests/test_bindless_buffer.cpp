//
// Created by Mike Smith on 2021/4/6.
//

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize.h>

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    auto device = context.create_device("metal");

    Kernel2D clear_image_kernel = [](BufferVar<uint> image) noexcept {
        Var coord = dispatch_id().xy();
        Var index = coord.y * dispatch_size_x() + coord.x;
        image.write(index, 0u);
    };

    // Callable sample = [](BindlessVar heap, Float2 uv, Float mip) noexcept {
    //     return heap.tex2d(0u).sample(uv, make_float2(), make_float2());
    // };

    Callable readBuffer = [](BindlessVar heap, UInt index) noexcept {
        UInt x = 0;
        return heap.buffer<uint>(0u).read(index);
    };
    // auto useless_shader = device.compile(useless_kernel);

    Kernel2D fill_image_kernel = [&](BindlessVar heap, BufferUInt image) noexcept {
        // Var coord = dispatch_id().xy();
        // Var index = coord.x;
        auto index = dispatch_y() * dispatch_size_x() + dispatch_x();
        // Var uv = make_float2(coord) / make_float2(dispatch_size().xy());
        // Var r = length(uv - 0.5f);
        // Var t = log(sin(sqrt(r) * 100.0f - constants::pi_over_two) + 2.0f);
        // image.write(index, 0u);
        image.write(index, readBuffer(heap, index));
    };

    auto clear_image = device.compile(clear_image_kernel);
    auto fill_image = device.compile(fill_image_kernel);

    auto heap = device.create_bindless_array();
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("src/tests/logo.png", &image_width, &image_height, &image_channels, 4);
    auto texture = device.create_buffer<uint>(image_width * image_height);
    auto device_image = device.create_buffer<uint>(1024u * 1024u);
    std::vector<uint8_t> host_image(1024u * 1024u * 4u);

    // auto event = device.create_event();
    auto stream = device.create_stream();
    // auto upload_stream = device.create_stream();

    // std::vector<uint8_t> mipmaps(image_width * image_height * 4u);
    // auto in_pixels = image_pixels;
    // auto out_pixels = mipmaps.data();

    // generate mip-maps
    auto cmd = stream.command_buffer();
    cmd << heap.emplace(0, texture).update()
        << texture.copy_from(image_pixels);
    // for (auto i = 1u; i < texture.mip_levels(); i++) {
    //     auto half_w = std::max(image_width / 2, 1);
    //     auto half_h = std::max(image_height / 2, 1);
    //     stbir_resize_uint8_srgb_edgemode(
    //         in_pixels,
    //         image_width, image_height, 0,
    //         out_pixels,
    //         half_w, half_h, 0,
    //         4, STBIR_ALPHA_CHANNEL_NONE, 0,
    //         STBIR_EDGE_REFLECT);
    //     image_width = half_w;
    //     image_height = half_h;
    //     stbi_write_png(fmt::format("level-{}.png", i).c_str(), image_width, image_height, 4, out_pixels, 0);
    //     cmd << texture.view(i).copy_from(out_pixels);
    //     in_pixels = out_pixels;
    //     out_pixels += image_width * image_height * 4u;
    // }
    cmd //<< event.signal()
        << commit();
    
    stream << clear_image(device_image).dispatch(1024u, 1024u)
        //    << event.wait()
           << fill_image(heap, device_image).dispatch(1024u, 1024u)
           << device_image.copy_to(host_image.data())
        //    << event.signal()
           << synchronize();

    stbi_write_png("result.png", 1024u, 1024u, 4u, host_image.data(), 0u);
}

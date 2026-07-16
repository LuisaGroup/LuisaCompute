#include "ut/ut.hpp"
#include "test_device.h"
#include <stb/stb_image_write.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_texture_io(Device &device) {

    log_level_verbose();

    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u, 1u);

    Callable linear_to_srgb = [](Float4 linear) noexcept {
        auto x = linear.xyz();
        return make_float4(
            select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                   12.92f * x,
                   x <= 0.00031308f),
            linear.w);
    };

    Kernel2D fill_image_kernel = [&]() noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        device_image->write(coord, linear_to_srgb(make_float4(rg, 1.0f, 1.0f)));
    };

    Kernel2D change_color_kernel = [&]() noexcept {
        Var coord = dispatch_id().xy();
        auto c = device_image->read(coord);
        device_image->write(coord, make_float4(lerp(c.xyz(), 1.f, 0.2f), 1.0f));
    };

    auto fill_image = device.compile(fill_image_kernel);
    auto change_color = device.compile(change_color_kernel);
    std::vector<std::byte> download_image(1024u * 1024u * 4u);

    Stream stream = device.create_stream();
    stream << fill_image().dispatch(1024u, 1024u)
           << change_color().dispatch(512u, 512u)
           << device_image.copy_to(luisa::span{download_image})
           << synchronize();
    expect(true) << "texture io test completed";
    stbi_write_png("result.png", 1024u, 1024u, 4u, download_image.data(), 0u);

    Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, 64u, 64u, 64u);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_texture_io(device);
}

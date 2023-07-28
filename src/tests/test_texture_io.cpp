#include <stb/stb_image_write.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    Callable linear_to_srgb = [](Float4 linear) noexcept {
        auto x = linear.xyz();
        return make_float4(
            select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                   12.92f * x,
                   x <= 0.00031308f),
            linear.w);
    };

    Kernel2D fill_image_kernel = [&linear_to_srgb](ImageFloat image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, linear_to_srgb(make_float4(rg, 1.0f, 1.0f)));
    };

    Kernel2D change_color_kernel = [](ImageFloat image) noexcept {
        Var coord = dispatch_id().xy();
        auto c = image.read(coord);
        image.write(coord, make_float4(lerp(c.xyz(), 1.f, 0.2f), 1.0f));
    };

    Shader2D<Image<float>> fill_image = device.compile(fill_image_kernel);
    Shader2D<Image<float>> change_color = device.compile(change_color_kernel);
    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u, 0u);
    std::vector<std::byte> download_image(1024u * 1024u * 4u);

    Stream stream = device.create_stream();
    stream << fill_image(device_image.view(0)).dispatch(1024u, 1024u)
           << change_color(device_image.view(0)).dispatch(512u, 512u)
           << device_image.copy_to(download_image.data())
           << synchronize();
    stbi_write_png("result.png", 1024u, 1024u, 4u, download_image.data(), 0u);

    Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, 64u, 64u, 64u);
}


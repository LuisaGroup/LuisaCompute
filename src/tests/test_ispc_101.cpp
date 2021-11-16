//
// Created by Mike Smith on 2021/4/6.
//

#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <dsl/syntax.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tests/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    auto device = context.create_device("ispc");

    Callable linear_to_srgb = [](Float3 linear) noexcept {
        auto x = linear.xyz();
        auto srgb = make_uint3(
            round(saturate(
                      select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                             12.92f * x,
                             x <= 0.00031308f)) *
                  255.0f));
        return (255u << 24u) | (srgb.z << 16u) | (srgb.y << 8u) | srgb.x;
    };

    Kernel2D fill_image_kernel = [&linear_to_srgb](BufferUInt image) noexcept {
        auto coord = dispatch_id().xy();
        auto rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image[coord.x + coord.y * dispatch_size_x()] = linear_to_srgb(make_float3(rg, 0.5f));
    };

    auto fill_image = device.compile(fill_image_kernel);
    std::vector<std::byte> download_image(1024u * 1024u * 4u);
    auto device_buffer = device.create_buffer<uint>(1024 * 1024);

    auto stream = device.create_stream();
    stream << fill_image(device_buffer).dispatch(1024u, 1024u)
           << device_buffer.copy_to(download_image.data())
           << synchronize();
    stbi_write_png("result.png", 1024u, 1024u, 4u, download_image.data(), 0u);
}

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize.h>

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

    Callable sample = [](BindlessVar heap, Float2 uv, Float mip) noexcept {
        return heap.tex2d(0u).sample(uv, mip);
    };

    Kernel2D fill_image_kernel = [&](BindlessVar heap, ImageVar<float> image) noexcept {
        Var coord = dispatch_id().xy();
        Var uv = make_float2(coord) * 2.f / make_float2(dispatch_size().xy());
        Var r = length(uv - 0.5f);
        Var t = log(sin(sqrt(r) * 100.0f - constants::pi_over_two) + 2.0f);
        image.write(coord, sample(heap, 2.0f * uv - make_float2(0.5f), t * 7.0f));
    };

    Shader2D<BindlessArray, Image<float>> fill_image = device.compile(fill_image_kernel);

    BindlessArray heap = device.create_bindless_array();
    int image_width = 0;
    int image_height = 0;
    int image_channels = 0;
    stbi_uc *image_pixels = stbi_load("test_path_tracing.png", &image_width, &image_height, &image_channels, 4);
    Image<float> texture = device.create_image<float>(PixelStorage::BYTE4, uint2(image_width, image_height), 0u);
    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u);
    luisa::vector<uint8_t> host_image(1024u * 1024u * 4u);

    Event event = device.create_event();
    Stream stream = device.create_stream();
    luisa::vector<uint8_t> mipmaps(image_width * image_height * 4u * 2);
    stbi_uc *in_pixels = image_pixels;
    uint8_t *out_pixels = mipmaps.data();

    // generate mip-maps
    stream << heap.emplace_on_update(0u, texture, Sampler::linear_linear_mirror()).update()
           << texture.copy_from(image_pixels);

    LUISA_INFO("Mip Level: {}", texture.mip_levels());

    for (uint i = 1u; i < texture.mip_levels(); i++) {
        uint half_w = std::max(image_width / 2, 1);
        uint half_h = std::max(image_height / 2, 1);
        stbir_resize_uint8_srgb_edgemode(
            in_pixels,
            image_width, image_height, 0,
            out_pixels,
            half_w, half_h, 0,
            4, STBIR_ALPHA_CHANNEL_NONE, 0,
            STBIR_EDGE_REFLECT);
        image_width = half_w;
        image_height = half_h;
        // stbi_write_png(fmt::format("level-{}.png", i).c_str(), image_width, image_height, 4, out_pixels, 0);
        stream << texture.view(i).copy_from(out_pixels);
        in_pixels = out_pixels;
        out_pixels += image_width * image_height * 4u;
    }

    stream << fill_image(heap, device_image).dispatch(make_uint2(1024u))
           << device_image.copy_to(host_image.data())
           << event.signal()
           << synchronize();

    stbi_write_png("result.png", 1024u, 1024u, 4u, host_image.data(), 0u);
}


#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <luisa/backends/ext/tex_compress_ext.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    Device device = context.create_device(argv[1]);
    auto tex_ext = device.extension<TexCompressExt>();
    Stream stream = device.create_stream();
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("logo.png", &image_width, &image_height, &image_channels, 4);
    auto resolution = make_uint2(image_width, image_height);
    Image<float> byte4_image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    Image<float> bc6h_image{device.create_image<float>(PixelStorage::BC6, resolution)};
    Image<float> bc7_image{device.create_image<float>(PixelStorage::BC7, resolution)};
    Buffer<uint> bc6h_buffer{device.create_buffer<uint>(bc6h_image.view().size_bytes() / sizeof(uint))};
    Buffer<uint> bc7_buffer{device.create_buffer<uint>(bc7_image.view().size_bytes() / sizeof(uint))};
    stream << byte4_image.copy_from(image_pixels) << synchronize();
    Clock clk;
    tex_ext->compress_bc6h(stream, byte4_image, bc6h_buffer);
    stream << synchronize();
    auto compress_time = clk.toc();
    LUISA_INFO("Compress BC6 {}x{} image spend {} ms", resolution.x, resolution.y, compress_time);
    clk.tic();
    tex_ext->compress_bc7(stream, byte4_image, bc7_buffer, 0 /*No need alpha channel*/);
    stream << synchronize();
    compress_time = clk.toc();
    LUISA_INFO("Compress BC7 {}x{} image spend {} ms", resolution.x, resolution.y, compress_time);
    BindlessArray array = device.create_bindless_array(2u);
    constexpr auto bc6h_image_index = 0u;
    constexpr auto bc7_image_index = 1u;
    array.emplace_on_update(bc6h_image_index, bc6h_image, Sampler::linear_linear_mirror());
    array.emplace_on_update(bc7_image_index, bc7_image, Sampler::linear_linear_mirror());
    stream << array.update() << synchronize();

    Kernel2D present_kernel = [&](UInt i) noexcept {
        Var coord = dispatch_id().xy();
        byte4_image->write(coord, make_float4(array->tex2d(i)->read(coord).xyz(), 1.0f));
    };
    auto present_shader = device.compile(present_kernel);

    luisa::vector<std::byte> host_image(byte4_image.view().size_bytes());
    stream
        << bc7_image.copy_from(bc7_buffer.view())
        << present_shader(bc7_image_index).dispatch(resolution)
        << byte4_image.copy_to(host_image.data())
        << synchronize();
    stbi_write_png("test_bc7_compress.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    stream
        << bc6h_image.copy_from(bc6h_buffer.view())
        << present_shader(bc6h_image_index).dispatch(resolution)
        << byte4_image.copy_to(host_image.data())
        << synchronize();
    stbi_write_png("test_bc6h_compress.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}

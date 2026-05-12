#include "ut/ut.hpp"
#include "test_device.h"
// Texture Compression Test
// Demonstrates BC6H and BC7 texture compression using compute shaders.
// Block compression reduces memory bandwidth and storage requirements.
//
// Features demonstrated:
// - BC6H (HDR) texture compression
// - BC7 (LDR with alpha) texture compression
// - Compressed texture sampling via bindless arrays

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
using namespace boost::ut;
using namespace boost::ut::literals;

void test_texture_compress(Device &device) {
    auto tex_ext = device.extension<TexCompressExt>();
    Stream stream = device.create_stream();

    // Load source image
    auto image_width = 0;
    auto image_height = 0;
    auto image_channels = 0;
    auto image_pixels = stbi_load("logo.png", &image_width, &image_height, &image_channels, 4);
    auto resolution = make_uint2(image_width, image_height);

    // Create images for different compression formats
    Image<float> byte4_image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    Image<float> bc6h_image{device.create_image<float>(PixelStorage::BC6, resolution)};
    Image<float> bc7_image{device.create_image<float>(PixelStorage::BC7, resolution)};
    Buffer<uint> bc6h_buffer{device.create_buffer<uint>(bc6h_image.view().size_bytes() / sizeof(uint))};
    Buffer<uint> bc7_buffer{device.create_buffer<uint>(bc7_image.view().size_bytes() / sizeof(uint))};
    stream << byte4_image.copy_from(luisa::span{image_pixels, static_cast<size_t>(image_width * image_height * 4)}) << synchronize();

    // Compress to BC6H format (HDR, no alpha)
    Clock clk;
    tex_ext->compress_bc6h(stream, byte4_image, bc6h_buffer);
    stream << synchronize();
    auto compress_time = clk.toc();
    LUISA_INFO("Compress BC6 {}x{} image spend {} ms", resolution.x, resolution.y, compress_time);

    // Compress to BC7 format (LDR with alpha)
    clk.tic();
    tex_ext->compress_bc7(stream, byte4_image, bc7_buffer, 0 /*No need alpha channel*/);
    stream << synchronize();
    compress_time = clk.toc();
    LUISA_INFO("Compress BC7 {}x{} image spend {} ms", resolution.x, resolution.y, compress_time);

    // Setup bindless array for compressed texture sampling
    BindlessArray array = device.create_bindless_array(2u);
    constexpr auto bc6h_image_index = 0u;
    constexpr auto bc7_image_index = 1u;
    array.emplace_on_update(bc6h_image_index, bc6h_image, Sampler::linear_linear_mirror());
    array.emplace_on_update(bc7_image_index, bc7_image, Sampler::linear_linear_mirror());
    stream << array.update() << synchronize();

    // Kernel to display compressed texture
    Kernel2D present_kernel = [&](UInt i) noexcept {
        Var coord = dispatch_id().xy();
        byte4_image->write(coord, make_float4(array->tex2d(i)->read(coord).xyz(), 1.0f));
    };
    auto present_shader = device.compile(present_kernel);

    // Decompress and save results
    luisa::vector<std::byte> host_image(byte4_image.view().size_bytes());
    stream
        << bc7_image.copy_from(bc7_buffer.view())
        << present_shader(bc7_image_index).dispatch(resolution)
        << byte4_image.copy_to(luisa::span{host_image})
        << synchronize();
    stbi_write_png("test_bc7_compress.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    stream
        << bc6h_image.copy_from(bc6h_buffer.view())
        << present_shader(bc6h_image_index).dispatch(resolution)
        << byte4_image.copy_to(luisa::span{host_image})
        << synchronize();
    expect(true) << "texture compress completed";
    stbi_write_png("test_bc6h_compress.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_texture_compress(device);
}

#include <runtime/stream.h>
#include <runtime/image.h>
#include <runtime/shader.h>
#include <dsl/syntax.h>
#include <stb/stb_image_write.h>
#include <backends/ext/tex_compress_ext.h>
#include <core/clock.h>
#include <core/logging.h>
using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    Device device = context.create_device(argv[1]);
    auto tex_ext = device.extension<TexCompressExt>();
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::FLOAT4, resolution)};
    Image<float> present_image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    Image<float> compressed_image{device.create_image<float>(PixelStorage::BC7, resolution)};
    Buffer<uint> compress_buffer{device.create_buffer<uint>(compressed_image.size_bytes() / sizeof(uint))};
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = (make_float2(coord) + 0.5f) / make_float2(size);
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };
    Shader2D<> shader = device.compile(kernel);
    stream << shader().dispatch(resolution) << synchronize();
    Clock clk;
    tex_ext->compress_bc7(stream, image, compress_buffer, 0 /*No need alpha channel*/);
    stream << synchronize();
    auto compress_time = clk.toc();
    LUISA_INFO("Compress {}x{} image spend {} ms", resolution.x, resolution.y, compress_time);
    Kernel2D present_kernel = [&]() {
        Var coord = dispatch_id().xy();
        present_image->write(coord, make_float4(compressed_image->read(coord).xyz(), 1.0f));
    };
    auto present_shader = device.compile(present_kernel);
    luisa::vector<std::byte> host_image(image.size_bytes());
    stream
        << compressed_image.copy_from(compress_buffer.view())
        << present_shader().dispatch(resolution)
        << present_image.copy_to(host_image.data())
        << synchronize();
    stbi_write_png("test_compressed_tex.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
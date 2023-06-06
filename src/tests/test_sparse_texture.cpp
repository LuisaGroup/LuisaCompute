#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/buffer.h>
#include <runtime/sparse_image.h>
#include <core/logging.h>
#include <dsl/syntax.h>
#include <stb/stb_image_write.h>
#include <core/clock.h>

using namespace luisa;
using namespace luisa::compute;
int main(int argc, char *argv[]) {

    Context context{argv[0]};

    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1], nullptr, true);
    auto stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(4096, 4096);
    auto sparse_image = device.create_sparse_image<float>(PixelStorage::BYTE4, resolution.x * 2, resolution.y * 2);
    sparse_image.map_tile(make_uint2(0), resolution, 0);
    Kernel2D kernel = [&](ImageVar<float> img, ImageVar<float> out) {
        Var coord = dispatch_id().xy();
        out.write(coord, img.read(coord));
    };
    Kernel2D write_kernel = [&](ImageVar<float> img) {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = (make_float2(coord) + 0.5f) / make_float2(size);
        img.write(coord, make_float4(uv, 1.f, 1.0f));
    };
    auto shader = device.compile(kernel);
    auto write_shader = device.compile(write_kernel);
    auto buffer = device.create_buffer<uint>(resolution.x * resolution.y);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> result(image.size_bytes());
    stream
        << sparse_image.update()
        << write_shader(image.view()).dispatch(resolution)
        << image.copy_to(result.data())
        << synchronize();
    stbi_write_png("test_sparse.png", resolution.x, resolution.y, 4, result.data(), 0);
}
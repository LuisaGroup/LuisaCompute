#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    luisa::string device_name = argv[1];
    Device device = context.create_device(device_name);
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());
    ExternalCallable<float2(float2, float2)> get_uv{"get_uv"};
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = get_uv(make_float2(coord), make_float2(size));
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };
    ShaderOption option;
    if (device_name == "dx") {
        // native HLSL code
        option.native_include = R"(
float2 get_uv(float2 coord, float2 size){
    return (coord + 0.5) / size;
}
    )";
    } else if (device_name == "cuda") {
        // native CUDA code
        option.native_include = R"(
[[nodiscard]] __device__ inline auto get_uv(lc_float2 coord, lc_float2 size) noexcept {
    return (coord + .5f) / size;
}
    )";
    } else if (device_name == "metal") {
        option.native_include = R"(
[[nodiscard]] inline auto get_uv(float2 coord, float2 size) {
    return (coord + .5f) / size;
}
    )";
    }
    Shader2D<> shader = device.compile(kernel, option);
    stream << shader().dispatch(resolution)
           << image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_native_code.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}

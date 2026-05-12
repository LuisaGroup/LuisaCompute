// Native include test demonstrating how to embed backend-specific
// code (HLSL, CUDA, Metal) directly in kernels.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include "../../reference_image.h"

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_native_include(Device &device) {
    auto argv = boost::ut::detail::cfg::largv;
    luisa::string device_name = argv[1];
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    Stream stream = device.create_stream();

    // Set image resolution
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());

    // Define external callable for UV calculation
    ExternalCallable<float2(float2, float2)> get_uv{"get_uv"};

    // Kernel that writes UV coordinates to image
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = get_uv(make_float2(coord), make_float2(size));
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };

    // Set native include code based on backend
    ShaderOption option;
    if (device_name == "dx" || device_name == "vk") {
        // Native HLSL code
        option.native_include = R"(
float2 get_uv(float2 coord, float2 size){
    return (coord + 0.5) / size;
}
    )";
    } else if (device_name == "cuda") {
        // Native CUDA code
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

    // Compile and execute
    auto shader = device.compile(kernel, option);
    stream << shader().dispatch(resolution)
           << image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("test_native_code.png", resolution.x, resolution.y, 4, host_image.data(), 0);
    auto ref_dir = luisa::test::find_reference_dir(std::filesystem::path{argv[0]}.parent_path());
    auto result = luisa::test::save_and_compare(
        reinterpret_cast<const uint8_t *>(host_image.data()), static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
        "test_native_code", opts.output_dir, ref_dir, opts.update_reference);
    LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
    if (!result.passed) {
        LUISA_ERROR("Reference comparison failed for test_native_code: {}", result.message);
        boost::ut::expect(false) << result.message;
        return;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_native_include(device);
}

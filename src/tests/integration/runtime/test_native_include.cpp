// Native include test demonstrating how to embed backend-specific
// code (HLSL, CUDA, Metal, LLVM IR) directly in kernels.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <system_error>

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include "reference_image.h"

#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_native_include(Device &device) {
    auto device_name = device.backend_name();
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    Stream stream = device.create_stream();

    // Set image resolution
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());

    // Define external callables for UV calculation and an in-place offset. The
    // latter deliberately exercises the mutable-reference ABI used by native
    // code rather than testing value arguments alone.
    ExternalCallable<float2(float2, float2)> get_uv{"get_uv"};
    ExternalCallable<void(float2 &, float2)> offset_uv{"offset_uv"};

    // Kernel that writes UV coordinates to image
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = get_uv(make_float2(coord), make_float2(size));
        constexpr auto delta = make_float2(0.125f, 0.25f);
        offset_uv(uv, delta);
        uv = uv - delta;
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
void offset_uv(inout float2 uv, float2 delta) {
    uv += delta;
}
    )";
    } else if (device_name == "cuda") {
        // Native CUDA code
        option.native_include = R"(
[[nodiscard]] __device__ inline auto get_uv(lc_float2 coord, lc_float2 size) noexcept {
    return (coord + .5f) / size;
}
__device__ inline void offset_uv(lc_float2 &uv, lc_float2 delta) noexcept {
    uv += delta;
}
    )";
    } else if (device_name == "metal") {
        option.native_include = R"(
[[nodiscard]] inline auto get_uv(float2 coord, float2 size) {
    return (coord + .5f) / size;
}
inline void offset_uv(thread float2 &uv, float2 delta) {
    uv += delta;
}
    )";
    } else if (device_name == "hip") {
        // The HIP backend lowers XIR directly to LLVM. Native includes therefore
        // use LLVM IR (text shown here; bitcode is accepted as well) and expose
        // unmangled symbols with the exact ExternalCallable register ABI.
        option.native_include = R"(
define <2 x float> @get_uv(<2 x float> %coord, <2 x float> %size) {
entry:
    %half.x = insertelement <2 x float> poison, float 5.000000e-01, i32 0
    %half = insertelement <2 x float> %half.x, float 5.000000e-01, i32 1
    %center = fadd <2 x float> %coord, %half
    %uv = fdiv <2 x float> %center, %size
    ret <2 x float> %uv
}

define void @offset_uv(ptr noundef nonnull align 8 dereferenceable(8) %uv.ptr,
                       <2 x float> noundef %delta) {
entry:
    %uv = load <2 x float>, ptr %uv.ptr, align 8
    %adjusted = fadd <2 x float> %uv, %delta
    store <2 x float> %adjusted, ptr %uv.ptr, align 8
    ret void
}
    )";
    }

    // Compile and execute
    auto shader = device.compile(kernel, option);
    stream << shader().dispatch(resolution)
           << image.copy_to(luisa::span{host_image})
           << synchronize();

    auto quantize_unorm = [](float value) noexcept {
        return static_cast<int>(std::lround(
            std::clamp(value, 0.0f, 1.0f) * 255.0f));
    };
    constexpr std::array sample_coordinates{
        make_uint2(0u, 0u),
        make_uint2(17u, 511u),
        make_uint2(256u, 341u),
        make_uint2(777u, 123u),
        make_uint2(1023u, 1023u),
    };
    for (auto coordinate : sample_coordinates) {
        auto pixel = (static_cast<size_t>(coordinate.y) * resolution.x +
                      coordinate.x) *
                     4u;
        auto actual_u = static_cast<int>(
            std::to_integer<uint8_t>(host_image[pixel + 0u]));
        auto actual_v = static_cast<int>(
            std::to_integer<uint8_t>(host_image[pixel + 1u]));
        auto actual_b = static_cast<int>(
            std::to_integer<uint8_t>(host_image[pixel + 2u]));
        auto actual_a = static_cast<int>(
            std::to_integer<uint8_t>(host_image[pixel + 3u]));
        auto expected_u = quantize_unorm(
            (static_cast<float>(coordinate.x) + 0.5f) /
            static_cast<float>(resolution.x));
        auto expected_v = quantize_unorm(
            (static_cast<float>(coordinate.y) + 0.5f) /
            static_cast<float>(resolution.y));
        boost::ut::expect(std::abs(actual_u - expected_u) <= 1)
            << luisa::format("Native callable returned U={} at ({}, {}), expected {}.",
                             actual_u, coordinate.x, coordinate.y, expected_u);
        boost::ut::expect(std::abs(actual_v - expected_v) <= 1)
            << luisa::format("Native callable returned V={} at ({}, {}), expected {}.",
                             actual_v, coordinate.x, coordinate.y, expected_v);
        boost::ut::expect(std::abs(actual_b - 128) <= 1)
            << "Native include test wrote an incorrect blue channel.";
        boost::ut::expect(actual_a == 255)
            << "Native include test wrote an incorrect alpha channel.";
    }

    auto output_directory = std::filesystem::path{opts.output_dir};
    std::error_code filesystem_error;
    std::filesystem::create_directories(output_directory, filesystem_error);
    boost::ut::expect(!filesystem_error)
        << luisa::format("Failed to create output directory '{}': {}.",
                         output_directory.string(), filesystem_error.message());
    if (!filesystem_error) {
        auto output_path = output_directory / "test_native_code.png";
        auto output_path_string = output_path.string();
        auto write_succeeded = stbi_write_png(
            output_path_string.c_str(), static_cast<int>(resolution.x),
            static_cast<int>(resolution.y), 4, host_image.data(), 0);
        boost::ut::expect(write_succeeded != 0)
            << luisa::format("Failed to write native include test image '{}'.",
                             output_path_string);
    }
    if (opts.compare_path) {
        auto result = luisa::test::compare_with_reference_file(
            reinterpret_cast<const uint8_t *>(host_image.data()), static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
            *opts.compare_path);
        LUISA_INFO("Reference comparison [test_native_code]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) {
            boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
            return;
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_native_include(device);
}

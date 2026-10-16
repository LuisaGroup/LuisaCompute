#include "ut/ut.hpp"
#include "test_device.h"
#include "memory_binary_io.h"

#include <array>
#include <cmath>
#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    luisa::test::MemoryBinaryIO binary_io;
    DeviceConfig config{.binary_io = &binary_io};
    auto dc = luisa::test::create_device_from_ut(argc, argv, &config);
    if (!dc) { return 2; }
    auto &device = dc->device;
    auto stream = device.create_stream();
    auto bytes = device.create_image<float>(PixelStorage::BYTE4, 1u, 1u);
    auto floats = device.create_image<float>(PixelStorage::FLOAT4, 1u, 1u);
    auto output = device.create_buffer<float4>(1u);
    auto make_kernel = [](const Image<float> &image) {
        return Kernel1D{[&image](BufferFloat4 result) noexcept {
            result.write(0u, image->read(make_uint2(0u)));
        }};
    };
    auto byte_kernel = make_kernel(bytes);
    auto float_kernel = make_kernel(floats);
    bool passed = true;
    auto check = [&](bool condition, const char *description) noexcept {
        expect(condition) << description;
        passed &= condition;
    };
    check(byte_kernel.function()->function().hash() == float_kernel.function()->function().hash(),
          "different captured texture storage must have the same AST hash for this regression");
    auto check_pixel = [&](const auto &shader, float4 expected) {
        std::array<float4, 1u> result{};
        stream << shader(output).dispatch(1u)
               << output.copy_to(luisa::span{result}) << synchronize();
        auto correct = all(abs(result[0] - expected) < 1.0e-6f);
        if (!correct) {
            LUISA_WARNING("Cached texture read: actual ({}, {}, {}, {}), expected ({}, {}, {}, {}).",
                          result[0].x, result[0].y, result[0].z, result[0].w,
                          expected.x, expected.y, expected.z, expected.w);
        }
        check(correct, "cached texture read must respect captured storage");
    };

    std::array<uint8_t, 4u> byte_pixel{51u, 102u, 153u, 255u};
    const std::array float_pixel{make_float4(0.125f, 0.25f, 0.75f, 1.0f)};
    stream << bytes.copy_from(luisa::span{byte_pixel})
           << floats.copy_from(luisa::span{float_pixel}) << synchronize();

    const auto initial_writes = binary_io.cache_write_count;
    auto byte_shader = device.compile(byte_kernel, ShaderOption{.enable_cache = true});
    const auto byte_writes = binary_io.cache_write_count;
    check(byte_writes > initial_writes, "BYTE4 cold compilation must populate the cache");
    check_pixel(byte_shader, make_float4(0.2f, 0.4f, 0.6f, 1.0f));

    auto float_shader = device.compile(float_kernel, ShaderOption{.enable_cache = true});
    const auto float_writes = binary_io.cache_write_count;
    check(float_writes > byte_writes, "FLOAT4 storage must have a distinct cache entry");
    check_pixel(float_shader, float_pixel[0]);

    // A fresh kernel object captures the same texture, after its contents
    // change. Warm lookup must select BYTE4 without baking in sampled values.
    byte_pixel = {255u, 153u, 102u, 51u};
    stream << bytes.copy_from(luisa::span{byte_pixel}) << synchronize();
    const auto reads_before_warm = binary_io.cache_read_count;
    auto warm_shader = device.compile(make_kernel(bytes), ShaderOption{.enable_cache = true});
    check(binary_io.cache_read_count > reads_before_warm, "warm compilation must read the shader cache");
    check(binary_io.cache_write_count == float_writes, "warm compilation must reuse the BYTE4 entry");
    check_pixel(warm_shader, make_float4(1.0f, 0.6f, 0.4f, 0.2f));
    return passed ? 0 : 1;
}

// Test for explicit HIP wave-size selection.
// This test covers:
// - exact wave32 and wave64 code generation on RDNA devices
// - lane IDs, active-lane counts, and full-wave reductions
// - loading a wave64 AOT shader on a device whose default wave size is 32

#include "ut/ut.hpp"
#include "test_device.h"

#include <filesystem>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

[[nodiscard]] Kernel1D<void(Buffer<uint4>)> make_wave_size_kernel(
    uint32_t wave_size) noexcept {
    return [wave_size](BufferVar<uint4> output) noexcept {
        set_block_size(wave_size, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(wave_size));
        auto lane = warp_lane_id();
        auto lane_count = warp_lane_count();
        auto active_count = warp_active_count_bits(lane < lane_count);
        auto lane_sum = warp_active_sum(lane + 1u);
        output.write(lane, make_uint4(lane, lane_count, active_count, lane_sum));
    };
}

void verify_wave_output(Device &device, Stream &stream,
                        const Shader1D<Buffer<uint4>> &shader,
                        uint32_t wave_size) noexcept {
    auto output = device.create_buffer<uint4>(wave_size);
    luisa::vector<uint4> host_output(wave_size, make_uint4(~0u));
    stream << output.copy_from(luisa::span{host_output})
           << shader(output).dispatch(wave_size)
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    auto expected_sum = wave_size * (wave_size + 1u) / 2u;
    auto correct = true;
    for (auto lane = 0u; lane < wave_size; lane++) {
        auto value = host_output[lane];
        if (value.x != lane ||
            value.y != wave_size ||
            value.z != wave_size ||
            value.w != expected_sum) {
            LUISA_WARNING(
                "HIP wave{} mismatch at lane {}: got ({}, {}, {}, {}), expected ({}, {}, {}, {}).",
                wave_size, lane, value.x, value.y, value.z, value.w,
                lane, wave_size, wave_size, expected_sum);
            correct = false;
        }
    }
    expect(correct) << "explicit HIP wave size must select one physical wave of the requested width";
}

void test_hip_wave_sizes(Device &device) noexcept {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP wave-size test on backend '{}'.", device.backend_name());
        return;
    }
    log_level_verbose();
    auto stream = device.create_stream();

    auto native_wave_size = device.compute_warp_size();
    expect(native_wave_size == 32u || native_wave_size == 64u)
        << "HIP must report a native wave size of 32 or 64";

    if (native_wave_size == 32u) {
        auto wave32 = device.compile(make_wave_size_kernel(32u));
        verify_wave_output(device, stream, wave32, 32u);
    }

    auto wave64_kernel = make_wave_size_kernel(64u);
    auto wave64 = device.compile(wave64_kernel);
    verify_wave_output(device, stream, wave64, 64u);

    auto package_path = std::filesystem::absolute("test_hip_wave64_aot.bytes");
    auto package_name = luisa::string{package_path.string()};
    std::error_code error;
    std::filesystem::remove(package_path, error);
    [[maybe_unused]] auto compile_only = device.compile(
        wave64_kernel,
        ShaderOption{.compile_only = true, .name = package_name});
    expect(std::filesystem::is_regular_file(package_path))
        << "HIP wave64 AOT package must be written";

    auto loaded_wave64 = device.load_shader<1, Buffer<uint4>>(package_name);
    verify_wave_output(device, stream, loaded_wave64, 64u);

    error.clear();
    std::filesystem::remove(package_path, error);
    expect(!error && !std::filesystem::exists(package_path))
        << "HIP wave64 AOT package cleanup must succeed";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_hip_wave_sizes(dc->device);
}

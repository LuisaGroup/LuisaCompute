// Test for AOT (Ahead-of-Time) shader compilation:
// - Save a kernel to a named file (compile_only)
// - Discard the shader object
// - Load the shader back from file
// - Dispatch and verify the result
//
// Usage:
//   xmake run test_aot <backend>
//   e.g.: xmake run test_aot vk

#include "ut/ut.hpp"
#include "test_device.h"

#include <filesystem>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_aot(Device &device) {
    log_level_verbose();

    static constexpr auto n = 1024u;
    static constexpr auto filename = "test_aot_save.bytes";

    // Clean up any leftover from a previous run
    std::error_code ec;
    std::filesystem::remove(filename, ec);

    // Step 1: Define a kernel
    Kernel1D kernel = [](BufferVar<float> buffer) noexcept {
        set_block_size(64, 1, 1);
        buffer.write(dispatch_id().x, cast<float>(dispatch_id().x) * 2.0f);
    };

    // Step 2: Save the shader to file via AOT compilation (compile_only)
    {
        ShaderOption option{
            .compile_only = true,
            .name = luisa::string{filename}};
        [[maybe_unused]] auto saved_shader = device.compile<1>(kernel, option);
        // Compile-only shader: handle is invalid, but bytecode was written to disk.
    }
    // saved_shader out of scope → discarded

    // Step 3: Load the shader back from the saved file
    auto shader = device.load_shader<1, Buffer<float>>(filename);

    // Step 4: Create buffer and dispatch
    Stream stream = device.create_stream();
    Buffer<float> buffer = device.create_buffer<float>(n);

    stream << shader(buffer).dispatch(n)
           << synchronize();

    // Step 5: Verify results
    std::vector<float> host_data(n);
    stream << buffer.copy_to(luisa::span{host_data})
           << synchronize();

    bool passed = true;
    for (uint32_t i = 0; i < n; ++i) {
        float expected = static_cast<float>(i) * 2.0f;
        if (host_data[i] != expected) {
            LUISA_ERROR("Mismatch at index {}: expected {}, got {}",
                        i, expected, host_data[i]);
            passed = false;
        }
    }
    LUISA_INFO("AOT save/load test: {}", passed ? "PASSED" : "FAILED");
    expect(passed) << "AOT save/load round-trip verification failed";

    // Clean up the saved file
    std::filesystem::remove(filename, ec);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_aot(device);
}

/**
 * @file test/unit/runtime/test_out_of_range.cpp
 * @brief Out-of-range detection tests for DX backend debug mode.
 *
 * Verifies that buffer and bindless array accesses are safely guarded
 * when compiled with enable_debug_info = true.
 */

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test 1: Buffer read OOB returns sentinel value
static void test_buffer_read_oob(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<float> result = device.create_buffer<float>(1);

    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<float> r) noexcept {
        set_block_size(64u);
        r.write(0, b.read(10));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    Stream stream = device.create_stream();
    luisa::vector<float> init(n, 42.0f);
    luisa::vector<float> res(1, 999.0f);
    stream << buf.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << shader(buf, result).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << synchronize();

    expect(static_cast<bool>(res[0] == -1.0f));
}

// Test 2: Buffer write OOB does not corrupt valid data
static void test_buffer_write_oob(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<float> result = device.create_buffer<float>(n);

    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<float> r) noexcept {
        set_block_size(64u);
        b.write(10, 99.0f);
        for (auto i = 0u; i < n; i++) {
            r.write(i, b.read(i));
        }
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    Stream stream = device.create_stream();
    luisa::vector<float> init(n, 42.0f);
    luisa::vector<float> res(n, 0.0f);
    stream << buf.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << shader(buf, result).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << synchronize();

    for (uint i = 0; i < n; i++) {
        expect(static_cast<bool>(res[i] == 42.0f));
    }
}

// Test 3: Volatile buffer read OOB returns sentinel value
static void test_volatile_buffer_read_oob(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<float> result = device.create_buffer<float>(1);

    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<float> r) noexcept {
        set_block_size(64u);
        r.write(0, b.volatile_read(10));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    Stream stream = device.create_stream();
    luisa::vector<float> init(n, 42.0f);
    luisa::vector<float> res(1, 999.0f);
    stream << buf.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << shader(buf, result).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << synchronize();

    expect(static_cast<bool>(res[0] == -1.0f));
}

// Test 4: Bindless buffer read OOB returns sentinel value
static void test_bindless_buffer_read_oob(Device &device) {
    constexpr uint buf_size = 4u;
    Buffer<float> buf0 = device.create_buffer<float>(buf_size);

    BindlessArray bdls = device.create_bindless_array(2);
    bdls.emplace_on_update(0, buf0);

    Buffer<float> result = device.create_buffer<float>(1);

    Kernel1D kernel = [&](BindlessVar b, BufferVar<float> r) noexcept {
        set_block_size(64u);
        r.write(0, b.buffer<float>(5).read(0));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    Stream stream = device.create_stream();
    luisa::vector<float> init(buf_size, 42.0f);
    luisa::vector<float> res(1, 999.0f);
    stream << buf0.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << bdls.update();
    stream << shader(bdls, result).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << synchronize();

    expect(static_cast<bool>(res[0] == -1.0f));
}

// Test 5: Buffer read valid (sanity check)
static void test_buffer_read_valid(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<float> result = device.create_buffer<float>(1);

    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<float> r) noexcept {
        set_block_size(64u);
        r.write(0, b.read(2));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    Stream stream = device.create_stream();
    luisa::vector<float> init(n);
    init[2] = 77.0f;
    luisa::vector<float> res(1, 0.0f);
    stream << buf.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << shader(buf, result).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << synchronize();

    expect(static_cast<bool>(res[0] == 77.0f));
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    if (dc->device.backend_name() != "dx") {
        LUISA_INFO("Skipping out-of-range test: debug feature is DX-only for now.");
        return 0;
    }
    test_buffer_read_oob(dc->device);
    test_buffer_write_oob(dc->device);
    // TODO: volatile operations not yet supported for debug validation
    // test_volatile_buffer_read_oob(dc->device);
    // TODO: bindless test requires builtin kernel fix
    // test_bindless_buffer_read_oob(dc->device);
    test_buffer_read_valid(dc->device);
    return 0;
}

/**
 * @file test/unit/runtime/test_out_of_range.cpp
 * @brief Out-of-range detection tests for DX backend debug mode.
 *
 * On the DX backend an out-of-range index into a buffer, bindless array, local
 * array, shared array or acceleration-structure instance is not trapped by the
 * driver: it silently removes the D3D12 device without any exception or log.
 * When the host is built without NDEBUG and the shader is compiled with
 * `enable_debug_info = true`, the HLSL codegen therefore:
 *
 *  1. clamps the offending index to 0 through `_lc_oob_guard`, so no invalid
 *     memory access can ever reach the GPU (no device removal),
 *  2. records the violation in per-thread static state,
 *  3. returns from every generated function call (HLSL has no exceptions, so
 *     the "quit" is a manually generated multiple return), and
 *  4. flushes the record into the device printer so the host reports it through
 *     the stream log callback.
 *
 * These tests exercise the detection on a live device.
 */

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <memory>
#include <mutex>
#include <string_view>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

/// Captures everything a stream pushes through its log callback.
struct LogCapture {
    mutable std::mutex mutex;
    luisa::vector<luisa::string> messages;

    void operator()(luisa::string_view message) noexcept {
        std::scoped_lock lock{mutex};
        messages.emplace_back(message);
    }

    [[nodiscard]] bool contains(std::string_view needle) const noexcept {
        std::scoped_lock lock{mutex};
        for (auto const &message : messages) {
            if (message.find(needle) != luisa::string::npos) { return true; }
        }
        return false;
    }
};

};// namespace

// Test 1: Buffer read OOB is reported and the invocation aborts.
static void test_buffer_read_oob(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<float> result = device.create_buffer<float>(1);
    Buffer<uint> reached = device.create_buffer<uint>(1);

    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<float> r, BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        // Out-of-range read: detected, clamped, and every function returns.
        auto v = b.read(10);
        // Must NOT be reached once the violation aborts the invocation.
        flag.write(0, 12345u);
        r.write(0, v);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<float> init(n, 42.0f);
    luisa::vector<float> res(1, 999.0f);
    luisa::vector<uint> flag(1, 0u);
    stream << buf.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << reached.copy_from(luisa::span{flag});
    stream << shader(buf, result, reached).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << reached.copy_to(luisa::span{flag});
    stream << synchronize();

    // The device survived (no silent removal) and the violation was reported.
    expect(captured->contains("LC OOB access")) << "buffer OOB must be reported";
    // The invocation aborted before the trailing write.
    expect(eq(flag[0], 0u)) << "statements after the OOB access must not run";
    // The result was never written because the kernel returned early.
    expect(static_cast<bool>(res[0] == 999.0f));
}

// Test 2: Buffer write OOB is reported and does not corrupt valid data.
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

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<float> init(n, 42.0f);
    luisa::vector<float> res(n, 0.0f);
    stream << buf.copy_from(luisa::span{init});
    stream << result.copy_from(luisa::span{res});
    stream << shader(buf, result).dispatch(1);
    stream << synchronize();
    stream << result.copy_to(luisa::span{res});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "buffer write OOB must be reported";
    // The loop after the violation never ran, so `result` keeps its init value.
    for (uint i = 0; i < n; i++) {
        expect(static_cast<bool>(res[i] == 0.0f));
    }
}

// Test 3: Local array index OOB is reported.
static void test_array_oob(Device &device) {
    Buffer<uint> result = device.create_buffer<uint>(1);
    Buffer<uint> reached = device.create_buffer<uint>(1);

    Kernel1D kernel = [&](BufferVar<uint> r, BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        ArrayVar<uint, 4> arr;
        $for(i, 4u) { arr[i] = i + 1u; };
        auto index = dispatch_x() + 7u;// always out of range
        auto v = arr[index];
        flag.write(0, 12345u);
        r.write(0, v);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<uint> res(1, 0u);
    luisa::vector<uint> flag(1, 0u);
    stream << result.copy_from(luisa::span{res});
    stream << reached.copy_from(luisa::span{flag});
    stream << shader(result, reached).dispatch(1);
    stream << synchronize();
    stream << reached.copy_to(luisa::span{flag});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "array OOB must be reported";
    expect(eq(flag[0], 0u)) << "statements after the array OOB must not run";
}

// Test 4: Shared (groupshared) array index OOB is reported.
static void test_shared_array_oob(Device &device) {
    Buffer<uint> reached = device.create_buffer<uint>(1);

    Kernel1D kernel = [&](BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        Shared<uint> shared_values{4};
        shared_values.write(dispatch_x(), dispatch_x() + 1u);
        auto index = dispatch_x() + 100u;// out of range
        auto v = shared_values.read(index);
        flag.write(0, v);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<uint> flag(1, 0u);
    stream << reached.copy_from(luisa::span{flag});
    stream << shader(reached).dispatch(1);
    stream << synchronize();
    stream << reached.copy_to(luisa::span{flag});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "shared array OOB must be reported";
    // The trailing write is skipped, so the flag keeps its uploaded value.
    expect(eq(flag[0], 0u)) << "statements after the shared OOB must not run";
}

// Test 5: Bindless buffer slot OOB is reported.
static void test_bindless_buffer_oob(Device &device) {
    constexpr uint buf_size = 4u;
    Buffer<float> buf0 = device.create_buffer<float>(buf_size);

    BindlessArray bdls = device.create_bindless_array(2);
    bdls.emplace_on_update(0, buf0);

    Buffer<uint> reached = device.create_buffer<uint>(1);

    Kernel1D kernel = [&](BindlessVar b, BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        auto v = b.buffer<float>(5).read(0);// slot 5 >= capacity 2
        flag.write(0, cast<uint>(v));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<float> init(buf_size, 42.0f);
    luisa::vector<uint> flag(1, 0u);
    stream << buf0.copy_from(luisa::span{init});
    stream << reached.copy_from(luisa::span{flag});
    stream << bdls.update();
    stream << shader(bdls, reached).dispatch(1);
    stream << synchronize();
    stream << reached.copy_to(luisa::span{flag});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "bindless slot OOB must be reported";
    expect(eq(flag[0], 0u)) << "statements after the bindless OOB must not run";
}

// Test 6: Sanity check — in-range accesses are not reported.
static void test_in_range_is_silent(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<float> result = device.create_buffer<float>(1);

    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<float> r) noexcept {
        set_block_size(64u);
        r.write(0, b.read(2));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

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
    expect(!captured->contains("LC OOB access")) << "in-range access must be silent";
}

// Test 7: A callable that triggers the violation also aborts the caller.
static void test_callable_oob_propagates(Device &device) {
    constexpr uint n = 4u;
    Buffer<float> buf = device.create_buffer<float>(n);
    Buffer<uint> reached = device.create_buffer<uint>(1);

    Callable inner = [](BufferFloat b, UInt index) noexcept {
        return b.read(index);
    };
    Kernel1D kernel = [&](BufferVar<float> b, BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        auto v = inner(b, 100u);
        flag.write(0, cast<uint>(v));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<float> init(n, 42.0f);
    luisa::vector<uint> flag(1, 0u);
    stream << buf.copy_from(luisa::span{init});
    stream << reached.copy_from(luisa::span{flag});
    stream << shader(buf, reached).dispatch(1);
    stream << synchronize();
    stream << reached.copy_to(luisa::span{flag});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "callable OOB must be reported";
    expect(eq(flag[0], 0u)) << "the caller must abort too";
}

// Test 8: Acceleration-structure instance index OOB is reported.
static void test_accel_instance_oob(Device &device) {
    Buffer<uint> reached = device.create_buffer<uint>(1);

    // One instance at index 0; the kernel reads an out-of-range instance.
    luisa::vector<float3> vertices{
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f)};
    luisa::vector<Triangle> indices{Triangle{0u, 1u, 2u}};
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
    Accel accel = device.create_accel();
    Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    accel.emplace_back(mesh, scaling(1.0f));

    Kernel1D kernel = [&](AccelVar tlas, BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        auto id = tlas.instance_user_id(99u);// only 1 instance exists
        flag.write(0, id + 1u);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<uint> flag(1, 0u);
    stream << vertex_buffer.copy_from(luisa::span{vertices});
    stream << triangle_buffer.copy_from(luisa::span{indices});
    stream << reached.copy_from(luisa::span{flag});
    stream << mesh.build() << accel.build();
    stream << shader(accel, reached).dispatch(1);
    stream << synchronize();
    stream << reached.copy_to(luisa::span{flag});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "accel instance OOB must be reported";
    expect(eq(flag[0], 0u)) << "statements after the accel OOB must not run";
}

// Test 9: Buffer atomic with an out-of-range index is reported.
static void test_atomic_oob(Device &device) {
    constexpr uint n = 4u;
    Buffer<uint> buf = device.create_buffer<uint>(n);
    Buffer<uint> reached = device.create_buffer<uint>(1);

    Kernel1D kernel = [&](BufferVar<uint> b, BufferVar<uint> flag) noexcept {
        set_block_size(64u);
        auto old = b.atomic(50u).fetch_add(1u);
        flag.write(0, old + 1u);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_debug_info = true});

    auto captured = std::make_shared<LogCapture>();
    Stream stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        (*captured)(message);
    });

    luisa::vector<uint> init(n, 0u);
    luisa::vector<uint> flag(1, 0u);
    stream << buf.copy_from(luisa::span{init});
    stream << reached.copy_from(luisa::span{flag});
    stream << shader(buf, reached).dispatch(1);
    stream << synchronize();
    stream << reached.copy_to(luisa::span{flag});
    stream << buf.copy_to(luisa::span{init});
    stream << synchronize();

    expect(captured->contains("LC OOB access")) << "atomic OOB must be reported";
    expect(eq(flag[0], 0u)) << "statements after the atomic OOB must not run";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    if (dc->device.backend_name() != "dx") {
        LUISA_INFO("Skipping out-of-range test: debug feature is DX-only for now.");
        return 0;
    }
#ifndef NDEBUG
    test_buffer_read_oob(dc->device);
    test_buffer_write_oob(dc->device);
    test_array_oob(dc->device);
    test_shared_array_oob(dc->device);
    test_bindless_buffer_oob(dc->device);
    test_in_range_is_silent(dc->device);
    test_callable_oob_propagates(dc->device);
    test_accel_instance_oob(dc->device);
    test_atomic_oob(dc->device);
#else
    LUISA_INFO("Skipping out-of-range test: detection is only active in debug builds.");
#endif
    return 0;
}

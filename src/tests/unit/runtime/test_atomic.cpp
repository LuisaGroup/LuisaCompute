// Test for atomic operations on buffers and shared memory.
//
// This test verifies various atomic operation types:
// - fetch_add: Atomically add and return old value
// - fetch_sub: Atomically subtract and return old value
// - fetch_max: Atomically compute max and return old value
// - compare_exchange: Atomic compare-and-swap
//
// Atomic operations ensure thread-safe concurrent access to memory
// locations from multiple threads.

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Custom struct for testing atomic operations on complex types
struct Something {
    uint x;
    float3 v;
};

LUISA_STRUCT(Something, x, v) {};

void test_atomic(Device &device) {

    // Enable verbose logging
    log_level_verbose();

    // Create buffer for atomic counter test
    Buffer<uint> buffer = device.create_buffer<uint>(4u);

    // Create a buffer to hold the constant value (1u)
    Buffer<uint> constant_buffer = device.create_buffer<uint>(1);
    uint host_value = 1u;
    Stream stream = device.create_stream();
    stream << constant_buffer.copy_from(luisa::span{&host_value, 1}) << synchronize();

    // Kernel demonstrating atomic fetch_add and conditional write
    // This pattern can be used for counting unique events
    Kernel1D count_kernel = [&](BufferUInt counter_buffer) noexcept {
        // Atomically add 1 to buffer[3], returns old value
        Var x = buffer->atomic(3u).fetch_add(counter_buffer.read(0));

        // Only the first thread to increment writes 1 to buffer[0]
        // This demonstrates atomic counting with flag setting
        if_(x == 0u, [&] {
            buffer->write(0u, 1u);
        });
    };
    auto count = device.compile(count_kernel);

    // Initialize host buffer to zeros
    uint4 host_buffer = make_uint4(0u);

    // Performance test for atomic operations
    Clock clock;
    clock.tic();
    stream << buffer.copy_from(luisa::span{&host_buffer, 1})
           << count(constant_buffer).dispatch(102400u)// Launch many threads
           << buffer.copy_to(luisa::span{&host_buffer, 1})
           << synchronize();
    double time = clock.toc();

    // Validate results:
    // - buffer[0] should be 1 (set by first thread)
    // - buffer[3] should be 102400 (total atomic increments)
    LUISA_INFO("Count: {} {}, Time: {} ms", host_buffer.x, host_buffer.w, time);
    boost::ut::expect(static_cast<bool>(host_buffer.x == 1u && host_buffer.w == 102400u))
        << "Atomic operation failed.";

    // Test atomic operations on float buffers
    Buffer<float> atomic_float_buffer = device.create_buffer<float>(1u);

    // Kernel with atomic subtraction (via negative add)
    Kernel1D add_kernel = [&](BufferFloat buffer) noexcept {
        buffer.atomic(0u).fetch_sub(-1.f);// fetch_sub with negative = addition
    };
    auto add_shader = device.compile(add_kernel);

    // Test atomic operations on vector components
    Kernel1D vector_atomic_kernel = [](BufferFloat3 buffer) noexcept {
        buffer.atomic(0u).x.fetch_add(1.f);// Atomic add to x component
    };

    // Test atomic operations on matrix elements
    Kernel1D matrix_atomic_kernel = [](BufferFloat2x2 buffer) noexcept {
        buffer.atomic(0u)[1].x.fetch_add(1.f);// Atomic add to [1][0] element
    };

    // Test atomic operations on nested array elements
    Kernel1D array_atomic_kernel = [](BufferVar<std::array<std::array<float4, 3u>, 5u>> buffer) noexcept {
        buffer.atomic(0u)[1][2][3].fetch_add(1.f);// Atomic add to specific array element
    };

    // Test atomic operations on struct members
    Kernel1D struct_atomic_kernel = [](BufferVar<Something> buffer) noexcept {
        auto a = buffer.atomic(0u);
        a.v.x.fetch_max(1.f);// Atomic max on struct member

        // Test shared memory atomics
        Shared<float> s{16};
        s.atomic(0).compare_exchange(0.f, 1.f);// CAS on shared memory
    };

    // Validate float atomic addition
    float result = 0.f;
    stream << atomic_float_buffer.copy_from(luisa::span{&result, 1})
           << add_shader(atomic_float_buffer).dispatch(1024u)
           << atomic_float_buffer.copy_to(luisa::span{&result, 1})
           << synchronize();
    LUISA_INFO("Atomic float result: {}.", result);
    boost::ut::expect(static_cast<bool>(result == 1024.f))
        << "Atomic float operation failed.";

    {
        constexpr auto n = 512u;
        auto vec_buf = device.create_buffer<float3>(1u);
        float3 vec_init = make_float3(0.f);
        auto vec_shader = device.compile(vector_atomic_kernel);
        float3 vec_result{};
        stream << vec_buf.copy_from(luisa::span{&vec_init, 1})
               << vec_shader(vec_buf).dispatch(n)
               << vec_buf.copy_to(luisa::span{&vec_result, 1})
               << synchronize();
        LUISA_INFO("Vector atomic result: x={}, y={}, z={}", vec_result.x, vec_result.y, vec_result.z);
        boost::ut::expect(static_cast<bool>(vec_result.x == static_cast<float>(n)))
            << "Vector atomic fetch_add on .x failed: expected " << n << " got " << vec_result.x;
        boost::ut::expect(static_cast<bool>(vec_result.y == 0.f))
            << "Vector atomic .y should remain 0";
        boost::ut::expect(static_cast<bool>(vec_result.z == 0.f))
            << "Vector atomic .z should remain 0";
    }

    {
        constexpr auto n = 256u;
        auto mat_buf = device.create_buffer<float2x2>(1u);
        float2x2 mat_init = float2x2::fill(0.f);
        auto mat_shader = device.compile(matrix_atomic_kernel);
        float2x2 mat_result{};
        stream << mat_buf.copy_from(luisa::span{&mat_init, 1})
               << mat_shader(mat_buf).dispatch(n)
               << mat_buf.copy_to(luisa::span{&mat_result, 1})
               << synchronize();
        LUISA_INFO("Matrix atomic result: [0]=({},{}), [1]=({},{})",
                   mat_result.cols[0].x, mat_result.cols[0].y,
                   mat_result.cols[1].x, mat_result.cols[1].y);
        boost::ut::expect(static_cast<bool>(mat_result.cols[1].x == static_cast<float>(n)))
            << "Matrix atomic fetch_add on [1].x failed: expected " << n;
        boost::ut::expect(static_cast<bool>(mat_result.cols[0].x == 0.f && mat_result.cols[0].y == 0.f))
            << "Matrix atomic: col 0 should remain zero";
        boost::ut::expect(static_cast<bool>(mat_result.cols[1].y == 0.f))
            << "Matrix atomic: [1].y should remain zero";
    }

    {
        constexpr auto n = 128u;
        using ArrayT = std::array<std::array<float4, 3u>, 5u>;
        auto arr_buf = device.create_buffer<ArrayT>(1u);
        ArrayT arr_init{};
        std::memset(&arr_init, 0, sizeof(ArrayT));
        auto arr_shader = device.compile(array_atomic_kernel);
        ArrayT arr_result{};
        stream << arr_buf.copy_from(luisa::span{&arr_init, 1})
               << arr_shader(arr_buf).dispatch(n)
               << arr_buf.copy_to(luisa::span{&arr_result, 1})
               << synchronize();
        float target = arr_result[1][2].w;
        LUISA_INFO("Array atomic result [1][2].w: {}", target);
        boost::ut::expect(static_cast<bool>(target == static_cast<float>(n)))
            << "Array atomic fetch_add on [1][2][3] failed: expected " << n;
        boost::ut::expect(static_cast<bool>(arr_result[0][0].x == 0.f))
            << "Array atomic: [0][0].x should remain zero";
        boost::ut::expect(static_cast<bool>(arr_result[1][2].x == 0.f))
            << "Array atomic: [1][2].x should remain zero";
    }

    {
        constexpr auto n = 64u;
        auto struct_buf = device.create_buffer<Something>(1u);
        Something s_init{};
        s_init.x = 0u;
        s_init.v = make_float3(0.f);
        auto struct_shader = device.compile(struct_atomic_kernel);
        Something s_result{};
        stream << struct_buf.copy_from(luisa::span{&s_init, 1})
               << struct_shader(struct_buf).dispatch(n)
               << struct_buf.copy_to(luisa::span{&s_result, 1})
               << synchronize();
        LUISA_INFO("Struct atomic result: x={}, v=({},{},{})", s_result.x, s_result.v.x, s_result.v.y, s_result.v.z);
        boost::ut::expect(static_cast<bool>(s_result.v.x == 1.f))
            << "Struct atomic fetch_max on .v.x failed: expected 1.0";
        boost::ut::expect(static_cast<bool>(s_result.v.y == 0.f && s_result.v.z == 0.f))
            << "Struct atomic: .v.y and .v.z should remain zero";
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_atomic(device);
}

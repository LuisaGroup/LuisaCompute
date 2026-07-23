// Test for atomic operations on buffers and shared memory.
//
// This test verifies various atomic operation types:
// - exchange and compare-exchange
// - integer add, subtract, and bitwise operations
// - signed, unsigned, and floating-point min/max
// - returned old values and final stored values
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
#include <array>
#include <cmath>
#include <cstdint>
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

void test_raw_buffer_atomic_matrix(Device &device) {

    constexpr size_t uint_value_count = 10u;
    constexpr size_t int_value_count = 2u;
    constexpr size_t float_value_count = 7u;

    std::array<uint32_t, uint_value_count> uint_values{
        0x10203040u,
        17u,
        23u,
        10u,
        10u,
        0xf0ccaa55u,
        0x10010010u,
        0xffff0000u,
        0xfffffff0u,
        17u};
    std::array<int32_t, int_value_count> int_values{7, -4};
    std::array<float, float_value_count> float_values{
        8.5f, -3.5f, 7.0f, 1.5f, 5.5f, 8.5f, -3.5f};

    auto uint_buffer = device.create_buffer<uint32_t>(uint_value_count);
    auto int_buffer = device.create_buffer<int32_t>(int_value_count);
    auto float_buffer = device.create_buffer<float>(float_value_count);
    auto uint_old_buffer = device.create_buffer<uint32_t>(uint_value_count);
    auto int_old_buffer = device.create_buffer<int32_t>(int_value_count);
    auto float_old_buffer = device.create_buffer<float>(float_value_count);

    Kernel1D atomic_matrix_kernel = [](BufferUInt uint_buffer,
                                       BufferInt int_buffer,
                                       BufferFloat float_buffer,
                                       BufferUInt uint_old_buffer,
                                       BufferInt int_old_buffer,
                                       BufferFloat float_old_buffer) noexcept {
        uint_old_buffer.write(0u, uint_buffer.atomic(0u).exchange(0xa0b0c0d0u));
        uint_old_buffer.write(1u, uint_buffer.atomic(1u).compare_exchange(17u, 99u));
        uint_old_buffer.write(2u, uint_buffer.atomic(2u).compare_exchange(24u, 100u));
        uint_old_buffer.write(3u, uint_buffer.atomic(3u).fetch_add(7u));
        uint_old_buffer.write(4u, uint_buffer.atomic(4u).fetch_sub(3u));
        uint_old_buffer.write(5u, uint_buffer.atomic(5u).fetch_and(0x0ff00ff0u));
        uint_old_buffer.write(6u, uint_buffer.atomic(6u).fetch_or(0x01101001u));
        uint_old_buffer.write(7u, uint_buffer.atomic(7u).fetch_xor(0x0ff00ff0u));
        uint_old_buffer.write(8u, uint_buffer.atomic(8u).fetch_min(17u));
        uint_old_buffer.write(9u, uint_buffer.atomic(9u).fetch_max(0xfffffff0u));

        int_old_buffer.write(0u, int_buffer.atomic(0u).fetch_min(-4));
        int_old_buffer.write(1u, int_buffer.atomic(1u).fetch_max(7));

        float_old_buffer.write(0u, float_buffer.atomic(0u).exchange(-1.25f));
        float_old_buffer.write(
            1u, float_buffer.atomic(1u).compare_exchange(-3.5f, 4.25f));
        float_old_buffer.write(
            2u, float_buffer.atomic(2u).compare_exchange(8.0f, 9.0f));
        float_old_buffer.write(3u, float_buffer.atomic(3u).fetch_add(2.25f));
        float_old_buffer.write(4u, float_buffer.atomic(4u).fetch_sub(1.25f));
        float_old_buffer.write(5u, float_buffer.atomic(5u).fetch_min(-2.25f));
        float_old_buffer.write(6u, float_buffer.atomic(6u).fetch_max(6.75f));
    };
    auto atomic_matrix_shader = device.compile(atomic_matrix_kernel);

    std::array<uint32_t, uint_value_count> uint_old_values{};
    std::array<int32_t, int_value_count> int_old_values{};
    std::array<float, float_value_count> float_old_values{};
    auto stream = device.create_stream();
    stream << uint_buffer.copy_from(luisa::span{uint_values})
           << int_buffer.copy_from(luisa::span{int_values})
           << float_buffer.copy_from(luisa::span{float_values})
           << atomic_matrix_shader(
                  uint_buffer, int_buffer, float_buffer,
                  uint_old_buffer, int_old_buffer, float_old_buffer)
                  .dispatch(1u)
           << uint_buffer.copy_to(luisa::span{uint_values})
           << int_buffer.copy_to(luisa::span{int_values})
           << float_buffer.copy_to(luisa::span{float_values})
           << uint_old_buffer.copy_to(luisa::span{uint_old_values})
           << int_old_buffer.copy_to(luisa::span{int_old_values})
           << float_old_buffer.copy_to(luisa::span{float_old_values})
           << synchronize();

    constexpr std::array<uint32_t, uint_value_count> expected_uint_values{
        0xa0b0c0d0u,
        99u,
        23u,
        17u,
        7u,
        0xf0ccaa55u & 0x0ff00ff0u,
        0x10010010u | 0x01101001u,
        0xffff0000u ^ 0x0ff00ff0u,
        17u,
        0xfffffff0u};
    constexpr std::array<uint32_t, uint_value_count> expected_uint_old_values{
        0x10203040u,
        17u,
        23u,
        10u,
        10u,
        0xf0ccaa55u,
        0x10010010u,
        0xffff0000u,
        0xfffffff0u,
        17u};
    constexpr std::array<int32_t, int_value_count> expected_int_values{-4, 7};
    constexpr std::array<int32_t, int_value_count> expected_int_old_values{7, -4};
    constexpr std::array<float, float_value_count> expected_float_values{
        -1.25f, 4.25f, 7.0f, 3.75f, 4.25f, -2.25f, 6.75f};
    constexpr std::array<float, float_value_count> expected_float_old_values{
        8.5f, -3.5f, 7.0f, 1.5f, 5.5f, 8.5f, -3.5f};

    for (size_t i = 0u; i < uint_value_count; ++i) {
        expect(uint_old_values[i] == expected_uint_old_values[i])
            << "Unexpected uint atomic old value at operation " << i;
        expect(uint_values[i] == expected_uint_values[i])
            << "Unexpected uint atomic final value at operation " << i;
    }
    for (size_t i = 0u; i < int_value_count; ++i) {
        expect(int_old_values[i] == expected_int_old_values[i])
            << "Unexpected signed atomic old value at operation " << i;
        expect(int_values[i] == expected_int_values[i])
            << "Unexpected signed atomic final value at operation " << i;
    }
    for (size_t i = 0u; i < float_value_count; ++i) {
        expect(std::abs(float_old_values[i] -
                        expected_float_old_values[i]) < 1e-6f)
            << "Unexpected float atomic old value at operation " << i;
        expect(std::abs(float_values[i] - expected_float_values[i]) < 1e-6f)
            << "Unexpected float atomic final value at operation " << i;
    }
}

void test_shared_compare_exchange(Device &device) {
    constexpr auto thread_count = 64u;
    auto old_values_buffer = device.create_buffer<uint>(thread_count);
    auto final_value_buffer = device.create_buffer<uint>(1u);

    Kernel1D shared_compare_exchange = [](BufferUInt old_values,
                                          BufferUInt final_value) noexcept {
        set_block_size(thread_count, 1u, 1u);
        Shared<uint> shared_value{1u};
        auto lane = thread_id().x;
        $if (lane == 0u) {
            shared_value.write(0u, 0u);
        };
        sync_block();

        // Exactly one lane may replace zero. All losing lanes must observe the
        // winning lane's nonzero value as the returned old value.
        auto old = shared_value.atomic(0u).compare_exchange(0u, lane + 1u);
        old_values.write(lane, old);
        sync_block();
        $if (lane == 0u) {
            final_value.write(0u, shared_value.read(0u));
        };
    };

    auto shader = device.compile(shared_compare_exchange);
    std::array<uint, thread_count> old_values{};
    uint final_value = 0u;
    auto stream = device.create_stream();
    stream << shader(old_values_buffer, final_value_buffer).dispatch(thread_count)
           << old_values_buffer.copy_to(luisa::span{old_values})
           << final_value_buffer.copy_to(luisa::span{&final_value, 1u})
           << synchronize();

    auto winner_count = 0u;
    auto winner_value = 0u;
    for (auto lane = 0u; lane < thread_count; lane++) {
        if (old_values[lane] == 0u) {
            winner_count++;
            winner_value = lane + 1u;
        }
    }
    expect(winner_count == 1u)
        << "shared compare-exchange must have exactly one successful lane";
    expect(final_value == winner_value)
        << "shared compare-exchange final value must come from the winning lane";
    for (auto old : old_values) {
        expect(old == 0u || old == winner_value)
            << "losing shared compare-exchange lanes must observe the winner";
    }
}

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
    test_raw_buffer_atomic_matrix(device);
    test_shared_compare_exchange(device);
    test_atomic(device);
}

// Test for constant arrays in compute kernels.
// This test verifies dynamically indexed constant struct values with an
// independent, exact host oracle.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test struct for constant buffer data
struct Foo {
    luisa::uint a;// Index field
    float b;      // Constant payload
    float c;
    float d;
};
LUISA_STRUCT(Foo, a, b, c, d) {};

[[nodiscard]] int test_constant(Device &device) {
    constexpr auto element_count = 37u;
    auto output = device.create_buffer<Foo>(element_count);
    auto stream = device.create_stream();

    Kernel1D kernel = [](BufferVar<Foo> result) noexcept {
        Foo foo_data[4]{
            {1, 2.0f, 3.0f, 4.0f},
            {5, 6.0f, 7.0f, 8.0f},
            {9, 10.0f, 11.0f, 12.0f},
            {13, 14.0f, 15.0f, 16.0f}};
        Constant<Foo> foo(foo_data, 4);
        auto index = (dispatch_x() * 3u + 1u) % 4u;
        result.write(dispatch_x(), foo.read(index));
    };

    auto shader = device.compile(kernel);
    luisa::vector<Foo> host_output(element_count);
    stream << shader(output).dispatch(element_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    constexpr Foo expected[4]{
        {1, 2.0f, 3.0f, 4.0f},
        {5, 6.0f, 7.0f, 8.0f},
        {9, 10.0f, 11.0f, 12.0f},
        {13, 14.0f, 15.0f, 16.0f}};
    auto all_correct = true;
    for (auto i = 0u; i < element_count; ++i) {
        auto expected_value = expected[(i * 3u + 1u) % 4u];
        auto value = host_output[i];
        if (value.a != expected_value.a ||
            value.b != expected_value.b ||
            value.c != expected_value.c ||
            value.d != expected_value.d) {
            LUISA_WARNING(
                "Constant mismatch at {}: got ({}, {}, {}, {}), expected ({}, {}, {}, {}).",
                i, value.a, value.b, value.c, value.d,
                expected_value.a, expected_value.b, expected_value.c, expected_value.d);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "dynamically indexed constant structs must exactly match the host values";
    return all_correct ? 0 : 1;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    return test_constant(device);
}

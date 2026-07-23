// Test for DSL syntactic sugar features
// This test demonstrates the simplified syntax macros that make
// kernel code more concise and readable.
//
// Sugar features tested:
// - $ prefix for Var types ($int, $float, $uint, etc.)
// - $for loop macro
// - $if/$elif/$else conditional macros
// - $loop macro
// - $switch/$case/$default macros
// - $break macro
// - $array for local arrays
// - $shared for shared memory
// - $constant for constant values
// - $buffer for buffer types

#include <cmath>
#include <vector>

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test structure for DSL struct handling
struct Test {
    int3 something;
    float a;
};

// Register the structure with the DSL
LUISA_STRUCT(Test, something, a) {};

// Type alias using the $ sugar syntax
using $Test = Var<Test>;

[[nodiscard]] int test_dsl_sugar(Device &device) {

    constexpr auto element_count = 32u;
    auto input = device.create_buffer<float>(element_count);
    auto output = device.create_buffer<float4>(element_count);

    // Create constant vector
    std::vector<int> const_vector{1, 2, 3, 4};

    // Callable using sugar syntax for parameter types ($int, $float)
    Callable callable = [&]($int a, $int b, $float c) noexcept {
        $constant int_consts = const_vector;
        return cast<float>(int_consts[a]) + b.cast<float>() * c;
    };

    // Kernel using sugar syntax throughout
    Kernel1D kernel = [&]($buffer<float> buffer_float,
                          $buffer<float4> output_buffer) noexcept {
        set_block_size(element_count, 1u, 1u);

        // $constant for constant declarations
        $constant float_consts = {1.0f, 2.0f};
        $constant int_consts = const_vector;

        // $shared for shared memory
        $shared<float> shared_floats{element_count};

        // $array for local array
        $array<float, 5> array;

        // $ prefix for automatic type deduction (becomes $int)
        $ v_int = 1;
        static_assert(std::is_same_v<decltype(v_int), $int>);

        $ index = dispatch_x();
        $ v_float = buffer_float.read(index);
        shared_floats[thread_x()] = v_float;
        sync_block();
        $ shared_copy = shared_floats[thread_x()];

        // $for loop and $array sugar
        $ array_sum = 0.0f;
        $for (array_index, 5) {
            array[array_index] = cast<float>(array_index) + shared_copy;
            array_sum += array[array_index];
        };

        $ call_ret = callable(v_int, 3, shared_copy);

        $ v_float_copy = shared_copy;

        // Arithmetic operations
        $ z = shared_copy + float_consts[0];

        // Vector operations
        $ v_vec = make_float3(1.0f);
        $ v2 = make_float3(2.0f) - v_vec * 2.0f;
        v2 *= 5.0f + shared_copy;

        $float2 w{cast<float>(v_int), shared_copy};
        w *= float2{1.2f};

        // $if/$elif/$else sugar syntax
        $int branch = 0;
        $if (index % 2u == 0u) {
            branch = 10;
        }
        $elif (index % 4u == 1u) {
            branch = 20;
        }
        $else {
            branch = 30;
        };

        // $loop and $break sugar
        $loop {
            branch += 7;
            $break;
        };

        // $switch/$case/$default sugar
        $switch (cast<int>(index % 3u)) {
            $case (0) {
                branch += 100;
            };
            $case (1) {
                branch += 200;
            };
            $default {
                branch -= 300;
            };
        };

        $int3 s{cast<int>(index), branch, cast<int>(array_sum)};

        // Struct variable with sugar syntax
        $Test vvt{s, v_float_copy};
        $Test vt{vvt};

        $ xx = 1.0f;

        $ vt_copy = vt;
        $ c = 0.5f + vt.a * 1.0f;

        output_buffer.write(index,
                            make_float4(z,
                                        call_ret,
                                        c,
                                        array_sum + cast<float>(branch + vt.something.x)));
    };

    auto shader = device.compile(kernel);
    luisa::vector<float> host_input(element_count);
    luisa::vector<float4> host_output(element_count);
    for (auto i = 0u; i < element_count; ++i) {
        host_input[i] = static_cast<float>(i) * 0.25f;
    }
    auto stream = device.create_stream();
    stream << input.copy_from(luisa::span{host_input})
           << shader(input, output).dispatch(element_count)
           << output.copy_to(luisa::span{host_output})
           << synchronize();

    auto all_correct = true;
    for (auto i = 0u; i < element_count; ++i) {
        auto value = host_input[i];
        auto branch = i % 2u == 0u ? 10 : i % 4u == 1u ? 20 :
                                                         30;
        branch += 7;
        switch (i % 3u) {
            case 0u: branch += 100; break;
            case 1u: branch += 200; break;
            default: branch -= 300; break;
        }
        auto expected = make_float4(
            value + 1.0f,
            2.0f + 3.0f * value,
            0.5f + value,
            10.0f + 5.0f * value + static_cast<float>(branch + static_cast<int>(i)));
        auto actual = host_output[i];
        if (std::abs(actual.x - expected.x) > 1e-5f ||
            std::abs(actual.y - expected.y) > 1e-5f ||
            std::abs(actual.z - expected.z) > 1e-5f ||
            std::abs(actual.w - expected.w) > 1e-5f) {
            LUISA_WARNING(
                "DSL sugar mismatch at {}: got ({}, {}, {}, {}), expected ({}, {}, {}, {}).",
                i, actual.x, actual.y, actual.z, actual.w,
                expected.x, expected.y, expected.z, expected.w);
            all_correct = false;
            break;
        }
    }
    expect(all_correct) << "DSL sugar control flow, arrays, shared memory, callable, struct, and buffers must match the host oracle";
    return all_correct ? 0 : 1;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    return test_dsl_sugar(device);
}

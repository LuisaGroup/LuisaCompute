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

#include <iostream>

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
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

void test_dsl_sugar(Device &device) {

    // Create buffers for testing
    Buffer<float4> buffer = device.create_buffer<float4>(1024u);
    Buffer<float> float_buffer = device.create_buffer<float>(1024u);

    // Create constant vector
    std::vector<int> const_vector{1, 2, 3, 4};

    // Callable using sugar syntax for parameter types ($int, $float)
    Callable callable = [&]($int a, $int b, $float c) noexcept {
        $constant int_consts = const_vector;
        return cast<float>(int_consts[a]) + b.cast<float>() * c;
    };

    // Kernel using sugar syntax throughout
    Kernel1D kernel = [&]($buffer<float> buffer_float, $uint count) noexcept {
        // $constant for constant declarations
        $constant float_consts = {1.0f, 2.0f};
        $constant int_consts = const_vector;

        // $shared for shared memory
        $shared<float4> shared_floats{16};

        // $array for local array
        $array<float, 5> array;

        // $ prefix for automatic type deduction (becomes $int)
        $ v_int = 10;
        static_assert(std::is_same_v<decltype(v_int), $int>);

        // $for loop sugar
        $for (x, 1) {
            array[x] = cast<float>(v_int);
        };

        // Buffer operations
        $ v_float = buffer_float.read(count);
        $ call_ret = callable(10, v_int, v_float);

        $ v_float_copy = v_float;

        // Arithmetic operations
        $ z = -1 + v_int * v_float + 1.0f;
        z += 1;

        // Vector operations
        $ v_vec = make_float3(1.0f);
        $ v2 = make_float3(2.0f) - v_vec * 2.0f;
        v2 *= 5.0f + v_float;

        $float2 w{cast<float>(v_int), v_float};
        w *= float2{1.2f};

        // $if/$elif/$else sugar syntax
        $if (w.x < 5) {
        }
        $elif (w.x > 0) {
        }
        $else {
        };

        // $loop and $break sugar
        $loop {
            $break;
        };

        // $switch/$case/$default sugar
        $switch (123) {
            $case (1) {
            };
            $default {
            };
        };

        $int x = cast<int>(w.x);
        $int3 s{x, x, x};

        // Struct variable with sugar syntax
        $Test vvt{s, v_float_copy};
        $Test vt{vvt};

        $ xx = 1.0f;

        $ vt_copy = vt;
        $ c = 0.5f + vt.a * 1.0f;

        // Buffer access
        $ vec4 = buffer->read(10);           // indexing into captured buffer (with literal)
        $ another_vec4 = buffer->read(v_int);// indexing into captured buffer (with Var)
    };

    auto shader = device.compile(kernel);
    expect(true) << "DSL sugar kernel compiled successfully";
    luisa::unique_ptr<Command> command = shader(float_buffer, 12u).dispatch(1024u);
    ShaderDispatchCommand *launch_command = static_cast<ShaderDispatchCommand *>(command.get());
    expect(launch_command != nullptr) << "dispatch command created";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_dsl_sugar(device);
}

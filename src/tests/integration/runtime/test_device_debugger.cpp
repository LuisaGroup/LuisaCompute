// Device debugger test demonstrating programmable breakpoints
// and custom trap functions for GPU debugging.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Test structure with vector types
struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

// Custom trap function with parameters for debugging
void my_trap(auto p, auto s, auto v, auto coord) {
    // You may do something with printing here
    LUISA_INFO("my_trap: p = {}, s = {}, v = {}, coord = {}", p, s, v, coord);
    // Please place a break point here with your IDE
}

// Custom trap function without parameters
void foo() {
    // Please place a break point here with your IDE
}

void test_device_debugger(Device &device) {

    log_level_verbose();

    // Define a kernel with debug breakpoints
    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == 1) {
            // Programmable break point with __debugbreak-like intrinsics
            $debug_break(coord);
        };
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            Var<MyStruct> s;
            s.a = v;
            s.b = coord;
            // Break point with custom trap function (dispatch_id is always captured for convenience)
            $debug_break_on(s, v, coord, my_trap(dispatch_id, s, v, coord));
            $outline_with_name("my_logger") {
                device_log("s = {} at {}", s, dispatch_id());
            };
            // Custom trap function without parameters (useful for use with interactive debuggers)
            $debug_break_on(foo());
        };
    };

    // Compile and dispatch the kernel
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();
    stream << shader().dispatch(128u, 128u)
           << synchronize();
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_device_debugger(device);
}

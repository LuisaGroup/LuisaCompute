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

struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

void test_printer(Device &device) {

    log_level_verbose();

    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == 1) {
            device_log("hello {} {}", coord, make_float3x3());
        };
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            Var<MyStruct> s;
            s.a = v;
            s.b = coord;
            $outline {
                device_log("s = {} at {}", s, dispatch_id());
            };
        };
    };
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();
    // stream.set_log_callback([](auto&& str){
    //     LUISA_WARNING("device: {}", str);
    // });
    stream << shader().dispatch(128u, 128u)
           << synchronize();
    expect(true) << "printer dispatch completed";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_printer(device);
}

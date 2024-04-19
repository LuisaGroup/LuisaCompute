//
// Created by Mike Smith on 2022/11/26.
//
#include "common/config.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {
int use_printer(Device &device) {
    log_level_verbose();

    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            device_log("v = ({}, {})", v.x, v.y);
        };
    };
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();
    stream << shader().dispatch(128u, 128u)
           << synchronize();
    return 0;
}
}// namespace luisa::test

TEST_SUITE("example") {
    TEST_CASE("use_printer") {
        Context context{luisa::test::argv()[0]};

        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::use_printer(device) == 0);
            }
        }
    }
}

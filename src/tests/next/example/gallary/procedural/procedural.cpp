/**
 * @file: tests/next/example/gallary/render/procedural.cpp
 * @author: sailing-innocent
 * @date: 2023-07-28
 * @brief: the basic pcg example
*/

#include "common/config.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>
using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int procedural(Device &device) {
    return 0;
}

}// namespace luisa::test

TEST_SUITE("procedural") {
    TEST_CASE("procedural") {
        Context context{luisa::test::argv()[0]};
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::procedural(device) == 0);
            }
        }
    }
}

//
// Created by Mike Smith on 2021/2/27.
//

#include "common/config.h"

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_dsl(Device &device) {
    // TODO: @saili i don't understand what the original test case means
    return 0;
}
}// namespace luisa::test

TEST_SUITE("feat") {
    TEST_CASE("dsl") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::supported_backends_count(); i++) {
            luisa::string device_name = luisa::test::supported_backends()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::test_dsl(device) == 0);
            }
        }
    }
}
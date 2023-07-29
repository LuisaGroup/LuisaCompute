/**
 * @file test/feat/test_buffer_view.cpp
 * @author sailing-innocent
 * @date 2023/07/29
 * @brief the buffer view test case
*/

#include "common/config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

template<typename T_Buffer>
int test_buffer_view(Device &device, int literal_size, int align_size = 4) {
    return 0;
}
}// namespace luisa::test

TEST_SUITE("common") {
    TEST_CASE("bufferview::float2") {
        Context context{luisa::test::argv()[0]};
        for (auto i = 0; i < luisa::test::supported_backends_count(); i++) {
            luisa::string device_name = luisa::test::supported_backends()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::test_buffer_view<float2>(device, 2, 2) == 0);
            }
        }
    }// TEST_CASE("bufferview::float2")
}// TEST_SUITE("common")
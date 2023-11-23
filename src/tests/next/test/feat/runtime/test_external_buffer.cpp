/**
 * @file test/feat/common/test_external_buffer.cpp
 * @author sailing-innocent
 * @date 2023/11/02
 * @brief test import_external_buffer
*/

#include "common/config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_external_buffer(Device &device) {
    constexpr uint n = 10u;
    Buffer<int> a = device.create_buffer<int>(n);
    Stream stream = device.create_stream();
    luisa::vector<int> data_init(n, 1);
    luisa::vector<int> data_result(n, 0);

    stream << a.copy_from(data_init.data());
    stream << synchronize();

    auto b = device.import_external_buffer<int>(a.native_handle(), n);

    stream << b.copy_to(data_result.data());
    stream << synchronize();

    for (uint idx = 0u; idx < n; idx++) {
        CHECK_MESSAGE(data_result[idx] == 1, "failed when ", idx);
    }

    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    TEST_CASE("external_buffer") {
        Context context{luisa::test::argv()[0]};
        Device device = context.create_device("dx");
        REQUIRE(luisa::test::test_external_buffer(device) == 0);
    }
}
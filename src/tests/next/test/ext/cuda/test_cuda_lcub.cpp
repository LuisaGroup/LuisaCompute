/**
 * @file src/tests/next/tset/ext/dx/test_dml.cpp
 * @author sailing-innocent, on MuGdxy's previous work on 7/26/2023.
 * @date 2023/11/03
 * @brief the directML test suite
*/

#include "common/config.h"
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <luisa/backends/ext/cuda/lcub/device_scan.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::cuda::lcub;

namespace luisa::test {

int test_lcub_scan(Device &device) {
    return 0;
}

}// namespace luisa::test

TEST_SUITE("ext_cuda") {
    TEST_CASE("cuda_lcub") {
        Context context{luisa::test::argv()[0]};
        Device device = context.create_device("cuda");
        REQUIRE(luisa::test::test_lcub_scan(device) == 0);
    }
}

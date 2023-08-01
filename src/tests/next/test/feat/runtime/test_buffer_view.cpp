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
    LUISA_TEST_CASE_WITH_DEVICE("buffer_view", luisa::test::test_buffer_view<float4>(device, 4, 4) == 0);
}// TEST_SUITE("common")
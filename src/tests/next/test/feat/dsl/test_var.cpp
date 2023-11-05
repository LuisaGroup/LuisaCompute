/**
 * @file test/feat/dsl/test_var.cpp
 * @author sailing-innocent
 * @date 2023/11/04
 * @brief the var of dsl 
*/

#include "common/config.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
namespace luisa::test {

int test_var(Device &device) {
    ulong a = 0;
    CHECK(sizeof(a) == 8);
    ushort b = 0;
    CHECK(sizeof(b) == 2);
    uint c = 0;
    CHECK(sizeof(c) == 4);
    return 0;
}

}// namespace luisa::test

TEST_SUITE("dsl") {
    LUISA_TEST_CASE_WITH_DEVICE("dsl_var", luisa::test::test_var(device) == 0);
}
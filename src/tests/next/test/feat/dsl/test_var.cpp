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
#include "stdio.h"

using namespace luisa;
using namespace luisa::compute;
namespace luisa::test {

int test_var(Device &device) {
    uint64_t a = 1;
    CHECK(sizeof(a) == 8);
    printf("a = %llu\n", a);
    printf("a = %llx\n", a);
    a <<= 32;
    printf("a = %llu\n", a);
    printf("a = %llx\n", a);
    a = a + 1;
    printf("a = %llu\n", a);
    printf("a = %llx\n", a);
    uint64_t bp = a & 0x00000000ffffffff;
    printf("bp = %llu\n", bp);
    printf("bp = %llx\n", bp);
    uint64_t up = a & 0xffffffff00000000 >> 32;
    printf("up = %llu\n", up);
    printf("up = %llx\n", up);

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
/**
 * @file test/feat/dsl/test_var.cpp
 * @author sailing-innocent
 * @date 2023/11/04
 * @brief the var of dsl 
*/

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>
#include "stdio.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_var(Device &device) {
    uint64_t a = 1;
    boost::ut::expect(static_cast<bool>(sizeof(a) == 8));
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
    boost::ut::expect(static_cast<bool>(sizeof(b) == 2));
    uint c = 0;
    boost::ut::expect(static_cast<bool>(sizeof(c) == 4));
    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_var(device);
}

/**
 * @file test/feat/ast/test_calc.cpp
 * @author sailing-innocent 
 * @date 2023/08/04 based on by Mike Smith version on 2021/2/27.
 * @brief the calculation in ast shaders
*/

#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int test_ast(Device &device) {
    Stream stream = device.create_stream();
    Buffer<int> buf = device.create_buffer<int>(10);
    constexpr auto sentinel = -0x13579;
    luisa::vector<int> v(10, sentinel);
    Kernel1D k1 = [&] {
        buf->write(1, 42);
    };
    auto s = device.compile(k1);
    stream << buf.copy_from(luisa::span{v})
           << s().dispatch(1u)
           << buf.copy_to(luisa::span{v})
           << synchronize();

    for (auto i = 0u; i < 10u; i++) {
        if (i == 1) {
            expect(static_cast<bool>(v[i] == 42));
        } else {
            expect(static_cast<bool>(v[i] == sentinel));
        }
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    expect(test_ast(device) == 0);
}

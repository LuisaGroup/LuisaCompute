//
// Created by Mike Smith on 2021/2/27.
//

#include "common/config.h"

#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
namespace luisa::test {
int test_ast(Device &device) {
    Stream stream = device.create_stream();
    // CHECK(Type::of<Buffer<int>>()->description() == doctest::Contains("buffer<int>"));
    Buffer<int> buf = device.create_buffer<int>(10);
    Kernel1D k1 = [&] {
        buf->write(1, 42);
    };
    Shader1D<> s = device.compile(k1);
    stream << s().dispatch(1u);
    stream << synchronize();

    luisa::vector<int> v(10);
    stream << buf.copy_to(v.data());
    stream << synchronize();

    for (auto i = 0u; i < 10u; i++) {
        if (i == 1) {
            CHECK(v[i] == 42);
        } else {
            CHECK(v[i] == 0);
        }
    }

    return 0;
}
}// namespace luisa::test

TEST_SUITE("ast") {
    LUISA_TEST_CASE_WITH_DEVICE("ast_basic", luisa::test::test_ast(device) == 0);
}
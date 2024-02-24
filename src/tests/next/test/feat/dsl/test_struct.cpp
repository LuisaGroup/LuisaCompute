/**
 * @file test/feat/common/test_struct.cpp
 * @author sailing-innocent
 * @date 2024-02-24
 * @brief test suite for `LUISA_STRUCT`
*/

#include "common/config.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

struct Dummy {
    int a;
};

}// namespace luisa::test

LUISA_STRUCT(luisa::test::Dummy, a) {};

namespace luisa::test {

int test_luisa_struct(Device &device) {
    auto stream = device.create_stream();
    constexpr uint N = 64;
    Buffer<int> int_buf = device.create_buffer<int>(N);

    Kernel1D kernel_def = [&](const Var<Dummy> &dummy_var, BufferVar<int> int_buf_var) noexcept {
        set_block_size(32u);
        UInt index = dispatch_id().x;
        int_buf_var.write(index, dummy_var.a);
    };

    Dummy dummy{1};
    CHECK_MESSAGE(dummy.a == 1, "dummy.a should be 1");

    auto shader = device.compile(kernel_def);
    stream << shader(dummy, int_buf).dispatch(N);

    std::vector<int> int_result(N);
    stream << int_buf.copy_to(int_result.data());
    stream.synchronize();

    for (auto i = 0u; i < N; i++) {
        CHECK_MESSAGE(int_result[i] == 1, "int_result[i] should be 1");
    }

    return 0;
}

}// namespace luisa::test

TEST_SUITE("dsl") {
    LUISA_TEST_CASE_WITH_DEVICE("luisa_struct", luisa::test::test_luisa_struct(device) == 0);
}
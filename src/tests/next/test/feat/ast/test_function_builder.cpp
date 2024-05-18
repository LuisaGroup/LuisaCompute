/**
 * @file test_function_builder.cpp
 * @brief Build up AST through Function Builder and Device Interface
 * @author sailing-innocent
 * @date 2024-05-18
 */
#include "common/config.h"
#include <luisa/runtime/device.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/rhi/command_encoder.h>

#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
namespace luisa::test {

int test_function_builder(Device &device) {
    // device.impl(): the DeviceInterface
    // device.impl_shared(): the shared_ptr<DeviceInterface>
    Buffer<int> buf = device.create_buffer<int>(10);

    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto f = FuncBuilder::define_kernel([&] {
        auto &cur = *FuncBuilder::current();
        cur.set_block_size(make_uint3(256, 1, 1));
        auto arg0 = cur.buffer(Type::of<Buffer<int>>());
        cur.call(
            CallOp::BUFFER_WRITE,
            {arg0,
             cur.literal(Type::of<uint>(), 1u),
             cur.literal(Type::of<int>(), 42)});
    });
    ShaderOption option{
        .name = "test_function_builder_shader",
    };
    auto shader_create_info = device.impl()->create_shader(
        option, f->function());
    auto shader = device.load_shader<1, Buffer<int>>("test_function_builder_shader");

    auto stream = device.create_stream();
    stream << shader(buf).dispatch(1u);
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
    LUISA_TEST_CASE_WITH_DEVICE("ast_function_builder", luisa::test::test_function_builder(device) == 0);
}
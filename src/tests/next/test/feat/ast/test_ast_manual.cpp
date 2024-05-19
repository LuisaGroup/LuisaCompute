/**
 * @file test_ast_manual.cpp
 * @brief The Manural AST Builder, used for Extending AST
 * @author sailing-innocent
 * @date 2024-05-19
 */

#include "common/config.h"
#include <luisa/runtime/device.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/variable.h>
#include <luisa/runtime/rhi/command_encoder.h>

#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
namespace luisa::test {
int test_ast_manual(Device &device) {
    Stream stream = device.create_stream();
    constexpr int N = 10;
    // init and upload buffer
    Buffer<int> buf0 = device.create_buffer<int>(N);
    Buffer<int> buf1 = device.create_buffer<int>(N);
    auto buf0_view = buf0.view();
    auto buf1_view = buf1.view();
    // construct shader manually
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;
    auto f = FuncBuilder::define_kernel([] {
        auto &cur = *FuncBuilder::current();
        ///. set block size
        cur.set_block_size(make_uint3(256, 1, 1));
        /// bind args
        auto arg0 = cur.buffer(Type::of<Buffer<int>>());
        auto arg1 = cur.buffer(Type::of<Buffer<int>>());
        /// load index
        auto idx_uint3 = cur.dispatch_id();
        auto idx = cur.local(Type::of<uint>());
        // idx = idx_uint3.x;
        uint64_t swizzle_code = 0ull;
        cur.assign(idx, cur.swizzle(Type::of<uint>(), idx_uint3, 1, swizzle_code));
        /// local res = buf0[idx] + buf1[idx];
        // mark buf0 as read
        cur.mark_variable_usage(arg0->variable().uid(), Usage::READ);
        // mark buf1 as readwrite
        cur.mark_variable_usage(arg1->variable().uid(), Usage::READ_WRITE);
        // load num1 from buf0
        auto num1 = cur.call(Type::of<int>(), CallOp::BUFFER_READ, {arg0, idx});
        // load num2 from buf1
        auto num2 = cur.call(Type::of<int>(), CallOp::BUFFER_READ, {arg1, idx});
        // res = num1 + num2
        auto res = cur.local(Type::of<int>());
        cur.assign(res, cur.binary(Type::of<int>(), BinaryOp::ADD, num1, num2));
        /// write res back to buf1
        cur.call(CallOp::BUFFER_WRITE, {arg1, idx, res});
    });
    Function f_func = f->function();

    ShaderCreationInfo shader_create_info = device.impl()->create_shader({.name = "ast_manual_shader"}, f_func);
    ComputeDispatchCmdEncoder encoder{
        shader_create_info.handle,
        2,
        ShaderDispatchCmdEncoder::compute_uniform_size(f_func.unbound_arguments()),
    };
    encoder.encode_buffer(buf0_view.handle(), buf0_view.offset_bytes(), buf0_view.size_bytes());
    encoder.encode_buffer(buf1_view.handle(), buf1_view.offset_bytes(), buf1_view.size_bytes());

    // upload data
    luisa::vector<int> v0(N);
    luisa::vector<int> v1(N);
    for (int i = 0; i < N; i++) {
        v0[i] = i;
        v1[i] = 1;
    }
    stream << buf0.copy_from(v0.data());
    stream << buf1.copy_from(v1.data());
    stream << synchronize();

    encoder.set_dispatch_size(make_uint3(N, 1, 1));
    stream << std::move(encoder).build();

    luisa::vector<int> v1_after(N);
    stream << buf1.copy_to(v1_after.data());
    stream << synchronize();

    for (int i = 0; i < N; i++) {
        CHECK(v1_after[i] == v0[i] + v1[i]);
    }
    device.impl()->destroy_shader(shader_create_info.handle);
    return 0;
}

}// namespace luisa::test

TEST_SUITE("ast") {
    LUISA_TEST_CASE_WITH_DEVICE("ast_manual", luisa::test::test_ast_manual(device) == 0);
}
// Test for CallOp::ASYNC_COPY manual AST construction and validation.

#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include "ut/ut.hpp"

#ifdef LUISA_ASYNC_COPY_HLSL_CODEGEN_TEST
#include <hlsl_codegen.h>
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {
    using FuncBuilder = luisa::compute::detail::FunctionBuilder;

    auto kernel_builder = FuncBuilder::define_kernel([&]() {
        auto &cur = *FuncBuilder::current();
        cur.set_block_size(uint3(64u, 1u, 1u));

        // Shared array that serves as both source and destination.
        auto shared = cur.shared(Type::array(Type::of<uint32_t>(), 64u));

        // Build lvalue references to source and destination elements.
        auto zero = cur.literal(Type::of<uint32_t>(), 0u);
        auto one = cur.literal(Type::of<uint32_t>(), 1u);
        auto dst = cur.access(Type::of<uint32_t>(), shared, zero);
        auto src = cur.access(Type::of<uint32_t>(), shared, one);

        // ASYNC_COPY arguments: scope, dst, src, elem_bytes, num_elements, stride, event.
        auto scope = cur.literal(Type::of<uint32_t>(), 2u);
        auto elem_bytes = cur.literal(Type::of<uint32_t>(), 4u);
        auto num_elements = cur.literal(Type::of<uint32_t>(), 1u);
        auto stride = cur.literal(Type::of<uint32_t>(), 4u);
        auto event = cur.literal(Type::of<uint32_t>(), 0u);

        cur.call(Type::of<uint32_t>(), CallOp::ASYNC_COPY,
                 {scope, dst, src, elem_bytes, num_elements, stride, event});
    });

    auto kernel = Function(kernel_builder.get());

    // The kernel should report that it uses the new builtin.
    expect(kernel.propagated_builtin_callables().test(CallOp::ASYNC_COPY));

#ifdef LUISA_ASYNC_COPY_HLSL_CODEGEN_TEST
    // Verify the HLSL backend emits the SPIR-V intrinsic in SPIR-V mode.
    lc::hlsl::CodegenUtility util;
    auto code = util.Codegen(kernel, "", 0u, true);
    auto hlsl = luisa::string{code.result.view()};
    expect(hlsl.find("__builtin_spirv_group_async_copy") != luisa::string::npos);
    expect(hlsl.find("[[vk::ext_instruction(4434") != luisa::string::npos);
    expect(hlsl.find("SPV_KHR_untyped_pointers") != luisa::string::npos);
#endif
}

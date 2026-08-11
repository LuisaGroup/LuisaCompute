// Test for CallOp::ASYNC_COPY manual AST construction.
// This covers the currently supported front-end contract only: dst/src are
// 64-bit device addresses (e.g. produced by ADDRESS_OF / BUFFER_ADDRESS), and
// the op may be emitted either as a statement (void) or with a uint event
// return type. No backend claims are made here.

#include "ut/ut.hpp"

#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {
    "async_copy_builtin_propagates"_test = [] {
        using FuncBuilder = luisa::compute::detail::FunctionBuilder;

        auto kernel_builder = FuncBuilder::define_kernel([&]() {
            auto &cur = *FuncBuilder::current();
            cur.set_block_size(uint3(64u, 1u, 1u));

            // Shared array that serves as the destination.
            auto shared = cur.shared(Type::array(Type::of<uint32_t>(), 64u));

            // dst is an lvalue of the shared-memory destination; src is the
            // 64-bit device address of the global source.
            auto zero = cur.literal(Type::of<uint32_t>(), 0u);
            auto one = cur.literal(Type::of<uint32_t>(), 1u);
            auto dst_elem = cur.access(Type::of<uint32_t>(), shared, zero);
            auto src_elem = cur.access(Type::of<uint32_t>(), shared, one);
            auto src = cur.call(Type::of<ulong>(), CallOp::ADDRESS_OF, {src_elem});

            // ASYNC_COPY arguments: scope, dst_lvalue, src_addr, elem_bytes, num_elements, stride, event.
            auto scope = cur.literal(Type::of<uint32_t>(), 2u);
            auto elem_bytes = cur.literal(Type::of<uint32_t>(), 4u);
            auto num_elements = cur.literal(Type::of<uint32_t>(), 1u);
            auto stride = cur.literal(Type::of<uint32_t>(), 4u);
            auto event = cur.literal(Type::of<uint32_t>(), 0u);

            [[maybe_unused]] auto async_copy_call = cur.call(
                Type::of<uint32_t>(), CallOp::ASYNC_COPY,
                {scope, dst_elem, src, elem_bytes, num_elements, stride, event});
        });

        auto kernel = Function(kernel_builder.get());

        // The kernel should report that it uses the new builtin.
        expect(kernel.propagated_builtin_callables().test(CallOp::ASYNC_COPY));
    };
}

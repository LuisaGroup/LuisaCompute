#include <luisa/dsl/stmt.h>
#include <luisa/dsl/dispatch_indirect.h>

namespace luisa::compute {

namespace detail {

void IndirectDispatchBufferExprProxy::set_dispatch_count(Expr<uint> count) const noexcept {
    Expr<IndirectDispatchBuffer>{_buffer}.set_dispatch_count(count);
}

void IndirectDispatchBufferExprProxy::set_kernel(
    Expr<uint> offset,
    Expr<uint3> block_size,
    Expr<uint3> dispatch_size, Expr<uint> kernel_id) const noexcept {
    Expr<IndirectDispatchBuffer>{_buffer}
        .set_kernel(offset, block_size, dispatch_size, kernel_id);
}

}// namespace detail

Expr<IndirectDispatchBuffer>::Expr(const IndirectDispatchBuffer &buffer) noexcept
    : _expression{detail::FunctionBuilder::current()->buffer_binding(
          Type::of<IndirectDispatchBuffer>(), buffer.handle(), 0u, buffer.size_bytes())} {}

void Expr<IndirectDispatchBuffer>::set_dispatch_count(Expr<uint> count) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::INDIRECT_SET_DISPATCH_COUNT,
        {_expression, count.expression()});
}

void Expr<IndirectDispatchBuffer>::set_kernel(
    Expr<uint> offset,
    Expr<uint3> block_size,
    Expr<uint3> dispatch_size, Expr<uint> kernel_id) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::INDIRECT_SET_DISPATCH_KERNEL,
        {_expression,
         offset.expression(),
         block_size.expression(),
         dispatch_size.expression(),
         kernel_id.expression()});
}

}// namespace luisa::compute

#include <dsl/stmt.h>
#include <dsl/dispatch_indirect.h>

namespace luisa::compute {

namespace detail {

void IndirectDispatchBufferExprProxy::clear() const noexcept {
    Expr<IndirectDispatchBuffer>{_buffer}.clear();
}

void IndirectDispatchBufferExprProxy::dispatch_kernel(Expr<uint> kernel_id,
                                                      Expr<uint3> block_size,
                                                      Expr<uint3> dispatch_size) const noexcept {
    Expr<IndirectDispatchBuffer>{_buffer}
        .dispatch_kernel(kernel_id, block_size, dispatch_size);
}

}// namespace detail

Expr<IndirectDispatchBuffer>::Expr(const IndirectDispatchBuffer &buffer) noexcept
    : _expression{detail::FunctionBuilder::current()->buffer_binding(
          Type::of<IndirectKernelDispatch>(), buffer.handle(), 0u, buffer.size_bytes())} {}

void Expr<IndirectDispatchBuffer>::clear() const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::INDIRECT_CLEAR_DISPATCH_BUFFER,
        {_expression});
}

void Expr<IndirectDispatchBuffer>::dispatch_kernel(Expr<uint> kernel_id,
                                                   Expr<uint3> block_size,
                                                   Expr<uint3> dispatch_size) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::INDIRECT_EMPLACE_DISPATCH_KERNEL,
        {_expression,
         block_size.expression(),
         dispatch_size.expression(),
         kernel_id.expression()});
}

}// namespace luisa::compute

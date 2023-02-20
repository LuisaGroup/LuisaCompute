#pragma once

#include <dsl/expr.h>
#include <dsl/var.h>
#include <dsl/struct.h>
#include <runtime/dispatch_buffer.h>

namespace luisa::compute {

template<>
struct LC_DSL_API Expr<IndirectDispatchBuffer> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept : _expression{expr} {}
    Expr(const IndirectDispatchBuffer &buffer) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    void clear() const noexcept;
    void dispatch_kernel(Expr<uint> kernel_id,
                         Expr<uint3> block_size,
                         Expr<uint3> dispatch_size) const noexcept;
};

Expr(const IndirectDispatchBuffer &) -> Expr<IndirectDispatchBuffer>;

template<>
struct Var<IndirectDispatchBuffer> : public Expr<IndirectDispatchBuffer> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<IndirectDispatchBuffer>{
              detail::FunctionBuilder::current()->buffer(
                  Type::of<IndirectKernelDispatch>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

using IndirectDispatchBufferVar = Var<IndirectDispatchBuffer>;

namespace detail {

class LC_DSL_API IndirectDispatchBufferExprProxy {

private:
    IndirectDispatchBuffer _buffer;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(IndirectDispatchBufferExprProxy)

public:
    void clear() const noexcept;
    void dispatch_kernel(Expr<uint> kernel_id,
                         Expr<uint3> block_size,
                         Expr<uint3> dispatch_size) const noexcept;
};

}// namespace detail

}// namespace luisa::compute

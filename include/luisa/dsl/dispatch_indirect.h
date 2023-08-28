#pragma once

#include <luisa/dsl/expr.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/struct.h>
#include <luisa/runtime/dispatch_buffer.h>

namespace luisa::compute {

template<>
struct LC_DSL_API Expr<IndirectDispatchBuffer> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept : _expression{expr} {}
    Expr(const IndirectDispatchBuffer &buffer) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    void set_dispatch_count(Expr<uint> count) const noexcept;
    void set_kernel(Expr<uint> offset, Expr<uint3> block_size, Expr<uint3> dispatch_size, Expr<uint> kernel_id) const noexcept;
    void set_kernel(Expr<uint> offset, Expr<uint3> block_size, Expr<uint3> dispatch_size) const noexcept {
        set_kernel(offset, block_size, dispatch_size, 0u);
    }
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

Expr(const IndirectDispatchBuffer &) -> Expr<IndirectDispatchBuffer>;

template<>
struct Var<IndirectDispatchBuffer> : public Expr<IndirectDispatchBuffer> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<IndirectDispatchBuffer>{
              detail::FunctionBuilder::current()->buffer(
                  Type::of<IndirectDispatchBuffer>())} {}
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
    void set_dispatch_count(Expr<uint> count) const noexcept;
    void set_kernel(Expr<uint> offset, Expr<uint3> block_size, Expr<uint3> dispatch_size, Expr<uint> kernel_id) const noexcept;
    void set_kernel(Expr<uint> offset, Expr<uint3> block_size, Expr<uint3> dispatch_size) const noexcept {
        set_kernel(offset, block_size, dispatch_size, 0u);
    }
};

}// namespace detail

}// namespace luisa::compute

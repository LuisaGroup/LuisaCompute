#pragma once

#ifndef LC_DISABLE_DSL

#include <dsl/expr.h>
#include <dsl/var.h>
#include <dsl/struct.h>
#include <runtime/custom_struct.h>

LUISA_CUSTOM_STRUCT(DispatchArgs1D);
LUISA_CUSTOM_STRUCT(DispatchArgs2D);
LUISA_CUSTOM_STRUCT(DispatchArgs3D);

namespace luisa::compute {

LC_DSL_API void clear_dispatch_buffer(Expr<Buffer<DispatchArgs1D>> buffer);
LC_DSL_API void clear_dispatch_buffer(Expr<Buffer<DispatchArgs2D>> buffer);
LC_DSL_API void clear_dispatch_buffer(Expr<Buffer<DispatchArgs3D>> buffer);

LC_DSL_API void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs1D>> buffer,
    Expr<uint> block_size,
    Expr<uint> dispatch_size,
    Expr<uint> kernel_id);

LC_DSL_API void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs2D>> buffer,
    Expr<uint2> block_size,
    Expr<uint2> dispatch_size,
    Expr<uint> kernel_id);

LC_DSL_API void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs3D>> buffer,
    Expr<uint3> block_size,
    Expr<uint3> dispatch_size,
    Expr<uint> kernel_id);

template<size_t N, typename... Args>
inline void emplace_dispatch_kernel(
    Expr<Buffer<DispatchArgs1D>> buffer, Kernel<N, Args...> const &kernel, Expr<uint> dispatch_size, Expr<uint> kernel_id) {
    if constexpr (N == 1) {
        emplace_dispatch_kernel(
            buffer,
            kernel.function()->block_size().x,
            dispatch_size,
            kernel_id);
    } else if constexpr (N == 2) {
        emplace_dispatch_kernel(
            buffer,
            kernel.function()->block_size().xy(),
            dispatch_size,
            kernel_id);
    }else{
        emplace_dispatch_kernel(
            buffer,
            kernel.function()->block_size().xyz(),
            dispatch_size,
            kernel_id);
    }
}

}// namespace luisa::compute

#endif

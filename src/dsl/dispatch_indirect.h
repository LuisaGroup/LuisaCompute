#pragma once

#ifndef LC_DISABLE_DSL

#include <dsl/expr.h>
#include <dsl/var.h>
#include <dsl/struct.h>
#include <runtime/custom_struct.h>

LUISA_CUSTOM_STRUCT(DispatchArgs);

namespace luisa::compute {

LC_DSL_API void clear_dispatch_buffer(Expr<DispatchArgsBuffer buffer);

LC_DSL_API void emplace_dispatch_kernel(
    Expr<DispatchArgsBuffer buffer,
    Expr<uint> block_size,
    Expr<uint> dispatch_size,
    Expr<uint> kernel_id);

}// namespace luisa::compute

#endif

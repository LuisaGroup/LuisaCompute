#pragma once

#include <core/stl/functional.h>
#include <ast/function_builder.h>
#include <dsl/rtx/hit.h>

namespace luisa::compute {

class LC_DSL_API RayQuery {

private:
    const Expression *_expr;

public:
    RayQuery(const CallExpr *func) noexcept;
    RayQuery(RayQuery &&) noexcept = default;
    RayQuery(RayQuery const &) noexcept = delete;
    using Callback = luisa::move_only_function<void(Var<Hit> &&)>;
    Var<Hit> proceed(const Callback &triangle_callback, const Callback &prim_callback) noexcept;
};

LC_DSL_API void commit_triangle() noexcept;
LC_DSL_API void commit_primitive(Expr<float> distance) noexcept;

}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQuery, "LC_RayQuery")

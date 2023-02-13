#pragma once

#ifndef LC_DISABLE_DSL

#include <core/stl/functional.h>
#include <ast/function_builder.h>
#include <dsl/hit.h>

namespace luisa::compute {

class RayQuery;

template<>
struct detail::TypeDesc<RayQuery> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "LC_RayQuery"sv;
    }
};

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

#endif

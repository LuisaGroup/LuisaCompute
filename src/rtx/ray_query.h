#pragma once
#include <rtx/hit.h>
#include <ast/function_builder.h>
#include <core/stl/functional.h>
namespace luisa::compute {
class RayQuery;
template<>
struct detail::TypeDesc<RayQuery> {
    static constexpr luisa::string_view description() noexcept {
        using namespace std::string_view_literals;
        return "LC_RayQuery"sv;
    }
};
class LC_RUNTIME_API RayQuery {
    const Expression *_expr;

public:
    RayQuery(const CallExpr *func) noexcept;
    RayQuery(RayQuery &&) noexcept = default;
    RayQuery(RayQuery const &) noexcept = delete;
#ifndef LC_DISABLE_DSL
    using Callback = luisa::move_only_function<void(Var<Hit> &&)>;
    Var<Hit> proceed(const Callback &triangle_callback, const Callback &prim_callback) noexcept;
#endif
};
#ifndef LC_DISABLE_DSL
LC_RUNTIME_API void commit_triangle() noexcept;
LC_RUNTIME_API void commit_primitive(Expr<float> distance) noexcept;
#endif
}// namespace luisa::compute
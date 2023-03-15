#pragma once

#include <core/stl/functional.h>
#include <ast/function_builder.h>
#include <dsl/rtx/hit.h>

namespace luisa::compute {

class LC_DSL_API RayQuery {

private:
    const Expression *_expr;

public:
    explicit RayQuery(const CallExpr *func) noexcept;
    RayQuery(RayQuery &&) noexcept = default;
    RayQuery(RayQuery const &) noexcept = delete;
    [[nodiscard]] Var<bool> proceed() const noexcept;
    [[nodiscard]] Var<Hit> triangle_candidate() const noexcept;
    [[nodiscard]] Var<Hit> procedural_candidate() const noexcept;
    [[nodiscard]] Var<Hit> committed_hit() const noexcept;
    [[nodiscard]] Var<bool> is_candidate_triangle() const noexcept;
    [[nodiscard]] Var<bool> is_candidate_procedural() const noexcept;
    void commit_triangle() const noexcept;
    void commit_procedural(Expr<float> distance) const noexcept;
};

}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::RayQuery, "LC_RayQuery")

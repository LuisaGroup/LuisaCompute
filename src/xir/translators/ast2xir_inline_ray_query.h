#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/statement.h>

namespace luisa::compute::xir::detail {

struct InlineRayQueryASTLoop {
    const RefExpr *query{nullptr};
    const ScopeStmt *surface_handler{nullptr};
    const ScopeStmt *procedural_handler{nullptr};
    luisa::vector<const CommentStmt *> dispatch_comments;
};

using InlineRayQueryASTLoopMap =
    luisa::unordered_map<const LoopStmt *, InlineRayQueryASTLoop>;

// Proves the exact frontend `$while (query.proceed())` shape for every marked
// loop in one function. The result is transactional: false always leaves an
// empty map, so AST-to-XIR cannot partially structure a malformed function.
[[nodiscard]] bool analyze_inline_ray_query_ast(
    const ScopeStmt *function_body,
    InlineRayQueryASTLoopMap &matches) noexcept;

}// namespace luisa::compute::xir::detail

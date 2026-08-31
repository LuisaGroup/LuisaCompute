#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/function.h>
#include <luisa/xir/translators/xir2ast.h>

namespace luisa::compute::xir {

/// Translate a coroutine continuation function from restructured XIR back
/// to an AST Callable. The function must already have been through the
/// coroutine pipeline (coro-cfg-distill → coro-split → coro-materialize →
/// reg2mem → restructure_cfg). Frame accesses (GEP/Load/Store) are
/// translated to struct member access expressions.
[[nodiscard]] LUISA_XIR_API luisa::shared_ptr<const ASTFunctionBuilder>
xir_to_ast_translate_continuation(const FunctionDefinition &function,
                                  const XIR2ASTConfig &config = {}) noexcept;

/// Translate a set of continuations from the same immutable XIR module. Root
/// value/CFG state remains isolated, while ordinary callable dependencies are
/// translated once and shared by pointer identity across the returned ASTs.
/// XIR2ASTConfig::verify_same_module_once replaces per-function verification
/// with one stronger whole-module verification performed synchronously before
/// translation starts.
[[nodiscard]] LUISA_XIR_API
luisa::vector<luisa::shared_ptr<const ASTFunctionBuilder>>
xir_to_ast_translate_continuations(
    luisa::span<const FunctionDefinition *const> functions,
    const XIR2ASTConfig &config = {}) noexcept;

}// namespace luisa::compute::xir

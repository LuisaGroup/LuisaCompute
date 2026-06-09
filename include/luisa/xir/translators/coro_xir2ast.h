#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/translators/xir2ast.h>

namespace luisa::compute::xir {

/// Check whether a function definition represents a coroutine continuation.
/// A continuation is a CallableFunction whose first argument is a reference
/// (the frame struct passed by the caller).
[[nodiscard]] LUISA_XIR_API bool is_continuation(const FunctionDefinition &f) noexcept;

/// Translate a coroutine continuation function from restructured XIR back
/// to an AST Callable. The function must already have been through the
/// coroutine pipeline (coro-cfg-distill → coro-split → coro-materialize →
/// reg2mem → restructure_cfg). Frame accesses (GEP/Load/Store) are
/// translated to struct member access expressions.
[[nodiscard]] LUISA_XIR_API luisa::shared_ptr<const ASTFunctionBuilder>
xir_to_ast_translate_continuation(const FunctionDefinition &function,
                                  const XIR2ASTConfig &config = {}) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute {
class Function;
class ExternalFunction;
}// namespace luisa::compute

namespace luisa::compute::xir {

class AST2XIRContext;

struct AST2XIRConfig {
    // Preserve a canonical DSL `$while (query.proceed())` candidate dispatch
    // directly as RayQueryLoopInst. Disabling this is a diagnostic oracle for
    // the legacy generic-loop plus reconstruction route.
    bool preserve_inline_ray_query_loops{true};
};

using ASTFunction = compute::Function;
using ASTExternalFunction = compute::ExternalFunction;

[[nodiscard]] LUISA_XIR_API AST2XIRContext *ast_to_xir_translate_begin(const AST2XIRConfig &config) noexcept;
// Returns the exact XIR function owned by the supplied AST builder. Keeping
// this provenance is important for clients whose root function may become
// indistinguishable from its dependencies after optimization (for example, a
// coroutine whose every suspend is unreachable).
LUISA_XIR_API Function *ast_to_xir_translate_add_function(
    AST2XIRContext *ctx, const ASTFunction &f) noexcept;
void LUISA_XIR_API ast_to_xir_translate_add_external_function(AST2XIRContext *ctx, const ASTExternalFunction &f) noexcept;
[[nodiscard]] LUISA_XIR_API luisa::unique_ptr<Module> ast_to_xir_translate_finalize(AST2XIRContext *ctx) noexcept;

[[nodiscard]] LUISA_XIR_API luisa::unique_ptr<Module> ast_to_xir_translate(const ASTFunction &kernel, const AST2XIRConfig &config) noexcept;

}// namespace luisa::compute::xir

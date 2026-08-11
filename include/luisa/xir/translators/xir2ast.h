#pragma once

#include <cstddef>

#include <luisa/ast/function_builder.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class XIR2ASTContext;

struct XIR2ASTTranslationStatistics {
    size_t function_translations{0u};
    size_t function_cache_hits{0u};
    size_t value_binding_insertions{0u};
    size_t value_map_checkpoint_count{0u};
    // Exact number of branch-local bindings inspected and erased while
    // restoring checkpoints. Retained prefix bindings are never visited.
    size_t value_map_rollback_work{0u};
    size_t peak_value_map_size{0u};
};

struct XIR2ASTConfig {
    bool strict{true};
    luisa::span<const compute::Function::Binding> bound_arguments{};
    XIR2ASTTranslationStatistics *statistics{nullptr};
    // Diagnostic oracle: retain the former full-map snapshot at every
    // structured scope and compare it with the incremental rollback result.
    bool verify_value_map_checkpoints{false};
};

using ASTFunctionBuilder = compute::detail::FunctionBuilder;

[[nodiscard]] LUISA_XIR_API XIR2ASTContext *xir_to_ast_translate_begin(const XIR2ASTConfig &config) noexcept;
void LUISA_XIR_API xir_to_ast_translate_add_function(XIR2ASTContext *ctx, const FunctionDefinition &f) noexcept;
[[nodiscard]] LUISA_XIR_API luisa::shared_ptr<const ASTFunctionBuilder> xir_to_ast_translate_finalize(XIR2ASTContext *ctx) noexcept;

void LUISA_XIR_API xir_to_ast_normalize_module(Module *module) noexcept;
[[nodiscard]] LUISA_XIR_API luisa::shared_ptr<const ASTFunctionBuilder> xir_to_ast_translate(const FunctionDefinition &function, const XIR2ASTConfig &config) noexcept;

}// namespace luisa::compute::xir

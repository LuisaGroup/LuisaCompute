#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class XIR2ASTContext;

struct XIR2ASTConfig {
    bool strict{true};
    luisa::span<const compute::Function::Binding> bound_arguments{};
};

using ASTFunctionBuilder = compute::detail::FunctionBuilder;

[[nodiscard]] LUISA_XIR_API XIR2ASTContext *xir_to_ast_translate_begin(const XIR2ASTConfig &config) noexcept;
void LUISA_XIR_API xir_to_ast_translate_add_function(XIR2ASTContext *ctx, const FunctionDefinition &f) noexcept;
[[nodiscard]] LUISA_XIR_API luisa::shared_ptr<const ASTFunctionBuilder> xir_to_ast_translate_finalize(XIR2ASTContext *ctx) noexcept;

void LUISA_XIR_API xir_to_ast_normalize_module(Module *module) noexcept;
[[nodiscard]] LUISA_XIR_API luisa::shared_ptr<const ASTFunctionBuilder> xir_to_ast_translate(const FunctionDefinition &function, const XIR2ASTConfig &config) noexcept;

class LUISA_XIR_API XIR2AST {
public:
    [[nodiscard]] static luisa::shared_ptr<const ASTFunctionBuilder> build(const KernelFunction *kernel) noexcept;
    [[nodiscard]] static luisa::shared_ptr<const ASTFunctionBuilder> build(const CallableFunction *callable) noexcept;
};

}// namespace luisa::compute::xir

#pragma once

#include <luisa/core/logging.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>

namespace luisa::compute::backend_detail {

[[nodiscard]] inline luisa::shared_ptr<const xir::ASTFunctionBuilder>
lower_autodiff_to_ast(Function kernel) noexcept {
    auto module = xir::ast_to_xir_translate(kernel, {});
    auto inline_info = xir::inline_all_pass_run_on_module(module.get());
    auto autodiff_info = xir::autodiff_pass_run_on_module(module.get());
    xir::xir_to_ast_normalize_module(module.get());
    LUISA_VERBOSE(
        "XIR AutoDiff lowering: inlined {} call(s), transformed {} scope(s), "
        "removed {} instruction(s).",
        inline_info.inlined_call_count,
        autodiff_info.transformed_scope_count,
        autodiff_info.removed_instruction_count);
    auto config = xir::XIR2ASTConfig{
        .bound_arguments = kernel.bound_arguments()};
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
            return xir::xir_to_ast_translate(
                *static_cast<xir::FunctionDefinition *>(function), config);
        }
    }
    LUISA_ERROR_WITH_LOCATION(
        "XIR AutoDiff lowering did not produce a kernel definition.");
}

}// namespace luisa::compute::backend_detail

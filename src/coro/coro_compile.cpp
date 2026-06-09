#include <stdexcept>

#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/coro_xir2ast.h>

namespace luisa::compute::detail {

CoroutineCompileResult compile_coroutine_pipeline(
    const luisa::shared_ptr<const FunctionBuilder> &builder) {

    CoroutineCompileResult result{};

    auto ast_func = Function{builder.get()};
    xir::AST2XIRConfig config{};
    auto module = xir::ast_to_xir_translate(ast_func, config);
    if (!module) {
        throw std::runtime_error(
            "Coroutine compilation failed: AST→XIR translation returned null module");
    }

    xir::Function *coro_func = nullptr;
    for (auto *f : module->function_list()) {
        if (f->isa<xir::CallableFunction>() && f->definition() != nullptr) {
            auto *def = f->definition();
            bool has_coro = false;
            def->traverse_instructions([&](xir::Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() ==
                    xir::DerivedInstructionTag::CORO_SUSPEND) {
                    has_coro = true;
                }
            });
            if (has_coro) {
                coro_func = f;
                break;
            }
        }
    }
    if (!coro_func) {
        throw std::runtime_error(
            "Coroutine compilation failed: no coroutine function found in XIR module");
    }

    // Phase 1: Coroutine-specific passes (directly on AST→XIR output)
    auto cfg = xir::coro_cfg_distill_pass_run_on_function(coro_func);
    if (cfg.scopes.empty()) {
        throw std::runtime_error(
            "Coroutine compilation failed: coro-cfg-distill found no scopes");
    }

    auto split_count = xir::coro_split_pass_run_on_module_with_cfg(module.get(), cfg);
    if (split_count == 0u) {
        throw std::runtime_error(
            "Coroutine compilation failed: coro-split produced no callables");
    }

    auto materialize_info = xir::coro_materialize_pass_run_on_module(module.get());
    if (materialize_info.callable_count == 0u) {
        throw std::runtime_error(
            "Coroutine compilation failed: coro-materialize found no callables");
    }

    (void)xir::coro_reg2mem_pass_run_on_module(module.get());

    // Phase 2: Phi elimination on continuations
    (void)xir::reg2mem_pass_run_on_module(module.get());

    // Phase 3: Restructure for xir2ast translation
    (void)xir::restructure_cfg_pass_run_on_module(module.get());

    (void)xir::dce_pass_run_on_module(module.get());
    // NOTE: simplify_cfg_pass corrupts the structured CFG produced by
    // restructure_cfg, causing XIR2AST translation to crash in _predeclare_allocas.
    // Temporarily skipped until the root cause is resolved.
    // (void)xir::simplify_cfg_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());

    result.graph = coro::CoroGraph::from_module(*module, materialize_info, cfg);
    result.frame_desc.from_materialize_info(materialize_info);

    // Translate each continuation callable from XIR back to AST.
    // Nodes in the graph map 1:1 to the callables created by coro-split.
    for (size_t i = 0u; i < result.graph.node_count(); ++i) {
        auto &node = result.graph.node(i);
        if (node.callable != nullptr) {
            auto ast = xir::xir_to_ast_translate_continuation(
                *node.callable);
            if (ast) {
                result.subroutines.push_back(std::move(ast));
            }
        }
    }

    return result;
}

} // namespace luisa::compute::detail

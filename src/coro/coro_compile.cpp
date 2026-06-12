#include <stdexcept>

#include <luisa/ast/function.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/type.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/coro_xir2ast.h>

namespace luisa::compute::detail {

namespace {

[[nodiscard]] xir::PassPipeline create_coro_pre_distill_pipeline() noexcept {
    xir::PassPipeline p;
    p.add("algebraic-simplify", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::algebraic_simplify_pass_run_on_module(m, {}, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    return p;
}

}// namespace

CoroutineCompileResult compile_coroutine_pipeline(
    const luisa::shared_ptr<const FunctionBuilder> &builder) {

    CoroutineCompileResult result{};

    auto ast_func = Function{builder.get()};
    xir::AST2XIRConfig config{};
    auto module = xir::ast_to_xir_translate(ast_func, config);
    if (!module) {
        throw std::runtime_error(
            "Coroutine compilation failed: AST->XIR translation returned null module");
    }

    xir::Function *coro_func = nullptr;
    for (auto *f : module->function_list()) {
        if (f->isa<xir::CallableFunction>() && f->definition() != nullptr) {
            auto *def = f->definition();
            bool has_coro = false;
            def->traverse_instructions([&](xir::Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::CORO_SUSPEND) { has_coro = true; }
            });
            if (has_coro) { coro_func = f; break; }
        }
    }
    if (!coro_func) {
        throw std::runtime_error("Coroutine compilation failed: no coroutine function found in XIR module");
    }

    (void)xir::destructure_cfg_pass_run_on_module(module.get());
    auto pre_distill_pipeline = create_coro_pre_distill_pipeline();
    auto pre_distill_stats = pre_distill_pipeline.run(module.get());
    pre_distill_stats.log("Coroutine pre-distill optimization");

    coro_func = nullptr;
    for (auto *f : module->function_list()) {
        if (f->isa<xir::CallableFunction>() && f->definition() != nullptr) {
            auto *def = f->definition();
            bool has_coro = false;
            def->traverse_instructions([&](xir::Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::CORO_SUSPEND) { has_coro = true; }
            });
            if (has_coro) { coro_func = f; break; }
        }
    }
    if (!coro_func) { throw std::runtime_error("coro_func lost after destructure_cfg"); }

    auto cfg = xir::coro_cfg_distill_pass_run_on_function(coro_func);
    if (cfg.scopes.empty()) { throw std::runtime_error("coro-cfg-distill found no scopes"); }
    luisa::vector<const Type *> frame_fields;
    frame_fields.push_back(Type::of<uint>());// [0] token
    frame_fields.push_back(Type::of<uint>());// [1] skip_flag
    for (auto &value : cfg.frame_values) { frame_fields.push_back(value.type); }
    auto *frame_type = Type::structure(frame_fields);

    auto split_info = xir::coro_split_pass_run_on_module_with_cfg_and_frame_info(module.get(), cfg, frame_type);
    if (split_info.subroutines.empty()) { throw std::runtime_error("coro-split produced no callables"); }

    auto materialize_info = xir::coro_materialize_pass_run_on_module_with_cfg(module.get(), cfg, split_info);
    if (materialize_info.callable_count == 0u) { throw std::runtime_error("coro-materialize found no callables"); }

    (void)xir::coro_reg2mem_pass_run_on_split(split_info);
    (void)xir::destructure_cfg_pass_run_on_module(module.get());
    (void)xir::simplify_cfg_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());
    (void)xir::restructure_cfg_pass_run_on_module(module.get());
    (void)xir::dce_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());

    result.graph = coro::CoroGraph::from_module(*module, materialize_info, cfg, split_info);
    result.frame_desc.from_materialize_info(materialize_info);

    for (auto &subroutine : split_info.subroutines) {
        if (subroutine.callable != nullptr) {
            auto ast = xir::xir_to_ast_translate_continuation(*subroutine.callable);
            if (ast) { result.subroutines.push_back(std::move(ast)); }
        }
    }

    result.trigger_tokens.resize(cfg.scopes.size(), 0u);
    for (size_t i = 0u; i < cfg.scopes.size(); ++i) {
        result.trigger_tokens[i] = cfg.scopes[i].trigger_token;
    }

    return result;
}

} // namespace luisa::compute::detail

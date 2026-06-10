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
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_reg2mem.h>
#include <luisa/xir/passes/coro_split.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
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

    luisa::unordered_set<luisa::string> seen;
    luisa::vector<std::pair<luisa::string, const Type *>> regs;
    coro_func->definition()->traverse_instructions([&](xir::Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() != xir::DerivedInstructionTag::ALLOCA) { return; }
        auto *alloca = static_cast<xir::AllocaInst *>(inst);
        if (!alloca->is_local()) { return; }
        auto name_opt = alloca->name();
        if (!name_opt.has_value()) { return; }
        luisa::string name(name_opt.value());
        if (seen.insert(name).second) {
            regs.push_back({std::move(name), alloca->type()});
        }
    });
    luisa::vector<const Type *> frame_fields;
    frame_fields.push_back(Type::of<uint>());// [0] token
    frame_fields.push_back(Type::of<uint>());// [1] skip_flag
    for (auto &reg : regs) { frame_fields.push_back(reg.second); }
    auto *frame_type = Type::structure(frame_fields);

    auto split_count = xir::coro_split_pass_run_on_module_with_cfg_and_frame(module.get(), cfg, frame_type);
    if (split_count == 0u) { throw std::runtime_error("coro-split produced no callables"); }

    auto materialize_info = xir::coro_materialize_pass_run_on_module(module.get());
    if (materialize_info.callable_count == 0u) { throw std::runtime_error("coro-materialize found no callables"); }

    (void)xir::coro_reg2mem_pass_run_on_module(module.get());
    (void)xir::destructure_cfg_pass_run_on_module(module.get());
    (void)xir::simplify_cfg_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());
    (void)xir::restructure_cfg_pass_run_on_module(module.get());
    (void)xir::dce_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());

    result.graph = coro::CoroGraph::from_module(*module, materialize_info, cfg);
    result.frame_desc.from_materialize_info(materialize_info);

    for (size_t i = 0u; i < result.graph.node_count(); ++i) {
        auto &node = result.graph.node(i);
        if (node.callable != nullptr) {
            auto ast = xir::xir_to_ast_translate_continuation(*node.callable);
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

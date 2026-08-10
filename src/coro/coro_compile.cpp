#include <luisa/core/logging.h>

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
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/coro_xir2ast.h>
#include <luisa/xir/verifier.h>

namespace luisa::compute::detail {

namespace {

void verify_coro_xir_or_error(
    const xir::Module *module, luisa::string_view stage,
    const xir::XIRVerificationOptions &options = {}) noexcept {
    auto verification = xir::xir_verify_module(module, options);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at coroutine {}: {} ({} error(s) total).",
            stage, verification.errors.front().message, verification.errors.size());
    }
}

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
    // Coro scope/token metadata must describe the executable CFG, not the
    // front-end statement list. In particular, an AST suspend may live in a
    // constant-dead arm while a later live suspend keeps its (now sparse)
    // token. Fold the branch and erase the unreachable suspend/resume pair
    // before distilling scopes so every distilled token has a lowered callable.
    p.add("simplify-cfg", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::simplify_cfg_pass_run_on_module(m, &r);
        return i.changed();
    });
    p.add("dce", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::dce_pass_run_on_module(m, &r);
        return i.changed();
    });
    p.add("trace-gep", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::trace_gep_pass_run_on_module(m);
        r.set("traced_gep", i.traced_gep_count);
        r.set("removed_noop_gep", i.removed_noop_gep_count);
        return i.changed();
    });
    p.add("sroa", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::sroa_pass_run_on_module(m, {.decompose_vectors = true}, &r);
        return i.changed();
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
    LUISA_ASSERT(module != nullptr,
                 "Coroutine compilation failed: AST->XIR translation returned null module");
    verify_coro_xir_or_error(module.get(), "AST translation");

    xir::Function *coro_func = nullptr;
    for (auto *f : module->function_list()) {
        if (f->isa<xir::CallableFunction>() && f->definition() != nullptr) {
            auto *def = f->definition();
            bool has_coro = false;
            def->traverse_instructions([&](xir::Instruction *inst) noexcept {
                if (inst->derived_instruction_tag() == xir::DerivedInstructionTag::CORO_SUSPEND) { has_coro = true; }
            });
            if (has_coro) {
                coro_func = f;
                break;
            }
        }
    }
    LUISA_ASSERT(coro_func != nullptr,
                 "Coroutine compilation failed: no coroutine function found in XIR module");

    // Coro cfg distill/split/materialize intentionally accept only raw CFG.
    // Destructure converts structured SwitchInst nodes to IndexedBranchInst.
    auto destructure_info = xir::destructure_cfg_pass_run_on_module(module.get());
    if (!destructure_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine destructuring failed (errors={}, leaked_blocks={}).",
            destructure_info.error_count, destructure_info.leaked_block_count);
    }
    auto pre_distill_pipeline = create_coro_pre_distill_pipeline();
    auto pre_distill_stats = pre_distill_pipeline.run(module.get());
    verify_coro_xir_or_error(module.get(), "pre-distill optimization");
    pre_distill_stats.log("Coroutine pre-distill optimization");

    // Keep the coroutine owner's identity across optimization. Its surviving
    // token set is allowed to be a strict subset of the front-end token set,
    // including the empty set. Re-discovering the owner by searching for a
    // remaining CoroSuspendInst would reject that valid degenerate case.
    LUISA_ASSERT(
        coro_func->definition() != nullptr &&
            coro_func->parent_module() == module.get(),
        "Coroutine source definition was lost during pre-distill optimization.");

    auto cfg = xir::coro_cfg_distill_pass_run_on_function(coro_func);
    LUISA_ASSERT(
        cfg.succeeded(),
        "coro-cfg-distill rejected its input (structured={}, invalid_input={}, invalid_cfg={})",
        cfg.structured_cfg_error_count, cfg.invalid_input_error_count,
        cfg.invalid_cfg_error_count);
    LUISA_ASSERT(!cfg.scopes.empty(), "coro-cfg-distill found no scopes");
    luisa::vector<const Type *> frame_fields;
    auto frame_alignment = Type::of<uint>()->alignment();
    for (auto i = 0u; i < CoroFrameDesc::reserved_field_count; i++) {
        frame_fields.push_back(Type::of<uint>());
    }
    for (auto &value : cfg.frame_values) {
        frame_fields.push_back(value.type);
        frame_alignment = std::max(frame_alignment, value.type->alignment());
    }
    auto *frame_type = Type::structure(frame_alignment, frame_fields);

    auto split_info = xir::coro_split_pass_run_on_module_with_cfg_and_frame_info(module.get(), cfg, frame_type);
    if (!split_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine split rejected its input (structured={}, invalid_cfg={}).",
            split_info.structured_cfg_error_count, split_info.invalid_cfg_error_count);
    }
    LUISA_ASSERT(!split_info.subroutines.empty(), "coro-split produced no callables");

    auto materialize_info = xir::coro_materialize_pass_run_on_module_with_cfg(module.get(), cfg, split_info);
    if (!materialize_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine materialization rejected its input (structured={}, invalid_input={}).",
            materialize_info.structured_cfg_error_count,
            materialize_info.invalid_input_error_count);
    }
    LUISA_ASSERT(materialize_info.callable_count != 0u, "coro-materialize found no callables");

    (void)xir::coro_reg2mem_pass_run_on_split(split_info);
    destructure_info = xir::destructure_cfg_pass_run_on_module(module.get());
    if (!destructure_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine post-materialization destructuring failed (errors={}, leaked_blocks={}).",
            destructure_info.error_count, destructure_info.leaked_block_count);
    }
    (void)xir::simplify_cfg_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());
    auto restructure_info = xir::restructure_cfg_pass_run_on_module(module.get());
    if (!restructure_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine restructuring failed (irreducible={}, unstructured={}, invalid={}, iteration_limit={}).",
            restructure_info.irreducible_region_count,
            restructure_info.unstructured_branch_count,
            restructure_info.invalid_construct_count,
            restructure_info.iteration_limit_count);
    }
    (void)xir::dce_pass_run_on_module(module.get());
    (void)xir::reg2mem_pass_run_on_module(module.get());
    verify_coro_xir_or_error(module.get(), "codegen handoff", {.require_no_phi = true});

    result.graph = coro::CoroGraph::from_module(*module, materialize_info, cfg, split_info);
    result.frame_desc.from_materialize_info(materialize_info);

    // Keep continuation code and its routing token as one atomic relation.
    // Silently skipping a failed XIR->AST translation and then independently
    // rebuilding tokens from cfg.scopes would shift every later token onto the
    // wrong callable. Materialization guarantees exactly one split callable per
    // distilled scope; enforce that contract and project both vectors together
    // in scope order.
    LUISA_ASSERT(
        split_info.subroutines.size() == cfg.scopes.size(),
        "Coroutine lowering produced {} split callable(s) for {} distilled scope(s).",
        split_info.subroutines.size(), cfg.scopes.size());
    luisa::vector<const xir::CoroSplitInfo::Subroutine *> subroutines_by_scope(
        cfg.scopes.size(), nullptr);
    for (auto &subroutine : split_info.subroutines) {
        LUISA_ASSERT(
            subroutine.scope_index < subroutines_by_scope.size(),
            "Coroutine lowering produced out-of-range scope index {} (scope count {}).",
            subroutine.scope_index, subroutines_by_scope.size());
        LUISA_ASSERT(
            subroutines_by_scope[subroutine.scope_index] == nullptr,
            "Coroutine lowering produced duplicate callable metadata for scope {}.",
            subroutine.scope_index);
        LUISA_ASSERT(
            subroutine.callable != nullptr &&
                subroutine.callable->definition() != nullptr &&
                subroutine.trigger_token ==
                    cfg.scopes[subroutine.scope_index].trigger_token,
            "Coroutine lowering produced incomplete or inconsistent callable metadata for scope {}.",
            subroutine.scope_index);
        subroutines_by_scope[subroutine.scope_index] = &subroutine;
    }

    result.subroutines.reserve(subroutines_by_scope.size());
    result.trigger_tokens.reserve(subroutines_by_scope.size());
    for (size_t scope_index = 0u;
         scope_index < subroutines_by_scope.size(); ++scope_index) {
        auto *subroutine = subroutines_by_scope[scope_index];
        LUISA_ASSERT(
            subroutine != nullptr,
            "Coroutine lowering did not materialize distilled scope {}.",
            scope_index);
        auto ast = xir::xir_to_ast_translate_continuation(
            *subroutine->callable);
        LUISA_ASSERT(
            ast != nullptr,
            "Coroutine XIR->AST translation failed for scope {} (trigger token {}).",
            scope_index, subroutine->trigger_token);
        LUISA_ASSERT(
            result.graph.node(scope_index).token ==
                subroutine->trigger_token,
            "Coroutine graph/callable token mismatch at scope {}.",
            scope_index);
        result.subroutines.emplace_back(std::move(ast));
        result.trigger_tokens.emplace_back(subroutine->trigger_token);
    }
    LUISA_ASSERT(
        !result.trigger_tokens.empty() &&
            result.trigger_tokens.front() == 0u &&
            result.subroutines.size() == result.trigger_tokens.size(),
        "Coroutine lowering lost the entry continuation or callable/token pairing.");

    return result;
}

}// namespace luisa::compute::detail

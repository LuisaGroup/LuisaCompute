#include <algorithm>

#include <luisa/core/clock.h>
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
#include <luisa/xir/passes/lower_irreducible_cfg.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/coro_xir2ast.h>
#include <luisa/xir/verifier.h>

#include <cstdlib>

namespace luisa::compute::detail {

namespace {

struct OrdinaryCallableSnapshot {
    xir::Function *function;
    xir::BasicBlock *body_block;
    luisa::vector<const xir::BasicBlock *> blocks;
    luisa::vector<const xir::Instruction *> instructions;
};

[[nodiscard]] bool environment_flag_enabled(const char *name) noexcept {
    if (auto value = std::getenv(name)) {
        return luisa::string_view{value} == "1";
    }
    return false;
}

[[nodiscard]] bool verify_coro_pass_domain_enabled() noexcept {
    return environment_flag_enabled("LUISA_CORO_VERIFY_PASS_DOMAIN");
}

[[nodiscard]] bool verify_intermediate_xir_enabled() noexcept {
    return environment_flag_enabled("LUISA_XIR_VERIFY_INTERMEDIATE");
}

class CoroutineCompilePhaseProfiler {

private:
    bool _enabled;
    luisa::Clock _total_clock;
    luisa::Clock _phase_clock;

public:
    CoroutineCompilePhaseProfiler() noexcept
        : _enabled{environment_flag_enabled(
              "LUISA_CORO_PROFILE_COMPILATION")} {}

    void checkpoint(luisa::string_view phase) noexcept {
        if (_enabled) {
            LUISA_INFO("Coroutine compilation phase '{}': {:.3f} ms",
                       phase, _phase_clock.toc());
            _phase_clock.tic();
        }
    }

    void checkpoint_split(luisa::string_view first_phase,
                          double first_milliseconds,
                          luisa::string_view second_phase) noexcept {
        if (_enabled) {
            auto total_milliseconds = _phase_clock.toc();
            auto bounded_first = std::clamp(
                first_milliseconds, 0.0, total_milliseconds);
            LUISA_INFO("Coroutine compilation phase '{}': {:.3f} ms",
                       first_phase, bounded_first);
            LUISA_INFO("Coroutine compilation phase '{}': {:.3f} ms",
                       second_phase,
                       total_milliseconds - bounded_first);
            _phase_clock.tic();
        }
    }

    void finish() noexcept {
        if (_enabled) {
            LUISA_INFO("Coroutine compilation total: {:.3f} ms",
                       _total_clock.toc());
        }
    }
};

[[nodiscard]] luisa::vector<OrdinaryCallableSnapshot>
snapshot_ordinary_callables(xir::Module *module,
                            const xir::Function *coroutine) noexcept {
    luisa::vector<OrdinaryCallableSnapshot> snapshots;
    for (auto *function : module->function_list()) {
        if (function == coroutine || function->definition() == nullptr) {
            continue;
        }
        auto *definition = function->definition();
        OrdinaryCallableSnapshot snapshot{
            .function = function,
            .body_block = definition->body_block()};
        for (auto *block : definition->basic_blocks()) {
            snapshot.blocks.emplace_back(block);
            for (auto *instruction : block->instructions()) {
                snapshot.instructions.emplace_back(instruction);
            }
        }
        snapshots.emplace_back(std::move(snapshot));
    }
    return snapshots;
}

void verify_ordinary_callables_unchanged(
    luisa::span<const OrdinaryCallableSnapshot> snapshots,
    luisa::string_view stage) noexcept {
    for (auto &&snapshot : snapshots) {
        LUISA_ASSERT(
            snapshot.function->parent_module() != nullptr &&
                snapshot.function->definition() != nullptr,
            "Coroutine lowering removed an ordinary callable dependency.");
        auto *definition = snapshot.function->definition();
        luisa::vector<const xir::BasicBlock *> blocks;
        luisa::vector<const xir::Instruction *> instructions;
        for (auto *block : definition->basic_blocks()) {
            blocks.emplace_back(block);
            for (auto *instruction : block->instructions()) {
                instructions.emplace_back(instruction);
            }
        }
        LUISA_ASSERT(
            definition->body_block() == snapshot.body_block &&
                blocks == snapshot.blocks &&
                instructions == snapshot.instructions,
            "Coroutine CFG passes mutated ordinary callable '{}'. Only the "
            "source coroutine and materialized continuations belong to the "
            "coroutine pass domain (first observed after {}).",
            snapshot.function->name().value_or("<unnamed>"), stage);
    }
}

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

[[nodiscard]] xir::PassPipeline create_coro_pre_distill_pipeline(
    xir::Function *coroutine) noexcept {
    LUISA_ASSERT(coroutine != nullptr && coroutine->definition() != nullptr,
                 "Coroutine pre-distill optimization requires a definition.");
    xir::PassPipeline p;
    p.add("algebraic-simplify", [coroutine](xir::Module *, xir::PassReport &r) {
        auto i = xir::algebraic_simplify_pass_run_on_function(coroutine);
        r.set("simplified_inst", i.simplified_inst_count);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [coroutine](xir::Module *, xir::PassReport &r) {
        auto i = xir::const_fold_pass_run_on_function(coroutine);
        r.set("folded_inst", i.folded_inst_count);
        return i.folded_inst_count > 0u;
    });
    // Coro scope/token metadata must describe the executable CFG, not the
    // front-end statement list. In particular, an AST suspend may live in a
    // constant-dead arm while a later live suspend keeps its (now sparse)
    // token. Fold the branch and erase the unreachable suspend/resume pair
    // before distilling scopes so every distilled token has a lowered callable.
    p.add("simplify-cfg", [coroutine](xir::Module *, xir::PassReport &r) {
        auto i = xir::simplify_cfg_pass_run_on_function(coroutine);
        r.set("folded_constant_cond_br", i.folded_constant_cond_br_count);
        r.set("folded_switch", i.folded_switch_count);
        r.set("threaded_empty_block", i.threaded_empty_block_count);
        r.set("merged_straight_line", i.merged_straight_line_count);
        r.set("removed_unreachable_block", i.removed_unreachable_block_count);
        r.set("straight_line_scan", i.straight_line_scan_count);
        r.set("straight_line_block_visit", i.straight_line_block_visit_count);
        return i.changed();
    });
    p.add("dce", [coroutine](xir::Module *, xir::PassReport &r) {
        auto i = xir::dce_pass_run_on_function(coroutine);
        r.set("removed_inst", i.removed_inst_count);
        r.set("removed_block", i.removed_block_count);
        r.set("inserted_terminator", i.inserted_terminator_count);
        r.set("dead_code_instruction_scan", i.dead_code_instruction_scan_count);
        r.set("dead_code_worklist_pop", i.dead_code_worklist_pop_count);
        return i.changed();
    });
    p.add("trace-gep", [coroutine](xir::Module *, xir::PassReport &r) {
        auto i = xir::trace_gep_pass_run_on_function(coroutine);
        r.set("traced_gep", i.traced_gep_count);
        r.set("removed_noop_gep", i.removed_noop_gep_count);
        return i.changed();
    });
    p.add("sroa", [coroutine](xir::Module *, xir::PassReport &r) {
        auto i = xir::sroa_pass_run_on_function(
            coroutine, {.decompose_vectors = true});
        r.set("decomposed_alloca", i.decomposed_alloca_count);
        r.set("inserted_alloca", i.inserted_alloca_count);
        return i.changed();
    });
    return p;
}

}// namespace

CoroutineCompileResult compile_coroutine_pipeline(
    const luisa::shared_ptr<const FunctionBuilder> &builder) {

    CoroutineCompileResult result{};
    CoroutineCompilePhaseProfiler profiler;

    auto ast_func = Function{builder.get()};
    xir::AST2XIRConfig config{};
    auto *translation = xir::ast_to_xir_translate_begin(config);
    auto *coro_func = xir::ast_to_xir_translate_add_function(
        translation, ast_func);
    auto module = xir::ast_to_xir_translate_finalize(translation);
    LUISA_ASSERT(module != nullptr,
                 "Coroutine compilation failed: AST->XIR translation returned null module");
    LUISA_ASSERT(
        coro_func != nullptr && coro_func->definition() != nullptr &&
            coro_func->parent_module() == module.get(),
        "Coroutine compilation failed: AST->XIR translation lost root-function provenance.");
    profiler.checkpoint("AST-to-XIR translation");
    verify_coro_xir_or_error(module.get(), "AST translation");
    profiler.checkpoint("input verification");
    profiler.checkpoint("coroutine root provenance");

    // Coro cfg distill/split/materialize intentionally accept only raw CFG.
    // A ray-query candidate loop is not coroutine scheduling control flow: no
    // suspension is permitted inside its synchronous candidate callbacks.
    // Outline those callbacks and replace the loop by one atomic pipeline
    // instruction before destructuring the surrounding CFG:
    //
    //   RayQueryLoopInst -> RayQueryPipelineInst -> raw coroutine CFG.
    //
    // Keeping this boundary high-level is backend-neutral. After coroutine
    // materialization, XIR-to-AST reconstructs RayQueryStmt and each backend
    // remains free to choose callback pipelines (HIP/fallback) or an inline
    // query loop (native SPIR-V). The transform is module-transactional, so a
    // rejected handler shape cannot leave a partially outlined module.
    auto ray_query_info =
        xir::lower_ray_query_loop_pass_run_on_module(module.get());
    if (!ray_query_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine ray-query normalization rejected {} unsupported "
            "ray-query loop(s).",
            ray_query_info.error_count);
    }
    profiler.checkpoint("ray-query normalization");
    auto ordinary_callable_snapshots =
        verify_coro_pass_domain_enabled() ?
            snapshot_ordinary_callables(module.get(), coro_func) :
            luisa::vector<OrdinaryCallableSnapshot>{};
    profiler.checkpoint("pass-domain snapshot");
    // Destructure then converts the remaining structured constructs to the raw
    // CFG expected by coroutine distillation.
    auto destructure_info =
        xir::destructure_cfg_pass_run_on_function(coro_func);
    if (!destructure_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine destructuring failed (errors={}, leaked_blocks={}).",
            destructure_info.error_count, destructure_info.leaked_block_count);
    }
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "source destructuring");
    }
    profiler.checkpoint("source destructuring");
    auto pre_distill_pipeline =
        create_coro_pre_distill_pipeline(coro_func);
    auto pre_distill_stats = pre_distill_pipeline.run(module.get());
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "pre-distill optimization");
    }
    pre_distill_stats.log("Coroutine pre-distill optimization");
    profiler.checkpoint("pre-distill optimization");
    if (verify_intermediate_xir_enabled()) {
        verify_coro_xir_or_error(module.get(), "pre-distill optimization");
        profiler.checkpoint("intermediate verification");
    }

    // Keep the coroutine owner's identity across optimization. Its surviving
    // token set is allowed to be a strict subset of the front-end token set,
    // including the empty set. Re-discovering the owner by searching for a
    // remaining CoroSuspendInst would reject that valid degenerate case.
    LUISA_ASSERT(
        coro_func->definition() != nullptr &&
            coro_func->parent_module() == module.get(),
        "Coroutine source definition was lost during pre-distill optimization.");

    auto cfg = xir::coro_cfg_distill_pass_run_on_function(coro_func);
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "CFG distillation");
    }
    LUISA_ASSERT(
        cfg.succeeded(),
        "coro-cfg-distill rejected its input (structured={}, invalid_input={}, invalid_cfg={})",
        cfg.structured_cfg_error_count, cfg.invalid_input_error_count,
        cfg.invalid_cfg_error_count);
    LUISA_ASSERT(!cfg.scopes.empty(), "coro-cfg-distill found no scopes");
    profiler.checkpoint("CFG distillation");
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
    profiler.checkpoint("frame layout");

    auto split_info = xir::coro_split_pass_run_on_module_with_cfg_and_frame_info(module.get(), cfg, frame_type);
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "coroutine splitting");
    }
    if (!split_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine split rejected its input (structured={}, invalid_cfg={}).",
            split_info.structured_cfg_error_count, split_info.invalid_cfg_error_count);
    }
    LUISA_ASSERT(!split_info.subroutines.empty(), "coro-split produced no callables");
    profiler.checkpoint("coroutine splitting");

    auto materialize_info = xir::coro_materialize_pass_run_on_module_with_cfg(module.get(), cfg, split_info);
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "continuation materialization");
    }
    if (!materialize_info.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Coroutine materialization rejected its input (structured={}, invalid_input={}).",
            materialize_info.structured_cfg_error_count,
            materialize_info.invalid_input_error_count);
    }
    LUISA_ASSERT(materialize_info.callable_count != 0u, "coro-materialize found no callables");
    profiler.checkpoint("continuation materialization");

    result.graph = coro::CoroGraph::from_module(
        *module, materialize_info, cfg, split_info);
    result.frame_desc.from_materialize_info(materialize_info);
    profiler.checkpoint("graph and frame metadata");

    // Split callables are now the complete code-generation domain. The source
    // coroutine is an analysis artifact whose ordinary CFG ends at each
    // CoroSuspendInst; blocks owned beyond those semantic edges are therefore
    // deliberately disconnected in the ordinary CFG. Passing that source
    // definition to whole-module simplify/restructure would ask non-coroutine
    // passes to normalize blocks outside their executable domain.
    //
    // Detach it from the module transaction, but retain ownership until every
    // cfg/split metadata consumer below has finished. This also realizes the
    // ownership contract documented by generic XIR-to-AST normalization: that
    // generic path preserves source coroutines, whereas this compile path
    // replaces one source coroutine by its generated callables.
    LUISA_ASSERT(
        coro_func->use_list().empty(),
        "Coroutine source function still has {} IR user(s) after materialization.",
        coro_func->use_list().count_size());
    auto source_coroutine_owner = coro_func->remove_self();
    LUISA_ASSERT(
        source_coroutine_owner != nullptr,
        "Coroutine source function was not linked in its module.");
    profiler.checkpoint("source detachment");

    (void)xir::coro_reg2mem_pass_run_on_split(split_info);
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "coroutine reg2mem");
    }
    profiler.checkpoint("coroutine frame spilling");
    // Ordinary callable dependencies are not part of the coroutine state
    // machine. Keep their original structured AST-to-XIR form intact and
    // normalize only the generated continuations. Besides avoiding a
    // destructive CFG round trip, this preserves one canonical helper body
    // for XIR-to-AST instead of repeatedly normalizing shader-graph code that
    // is independent of coroutine scheduling.
    for (auto &subroutine : split_info.subroutines) {
        destructure_info = xir::destructure_cfg_pass_run_on_function(
            subroutine.callable);
        if (!destructure_info.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "Coroutine continuation {} post-materialization "
                "destructuring failed (errors={}, leaked_blocks={}).",
                subroutine.scope_index,
                destructure_info.error_count,
                destructure_info.leaked_block_count);
        }
        (void)xir::simplify_cfg_pass_run_on_function(
            subroutine.callable);
        (void)xir::reg2mem_pass_run_on_function(
            subroutine.callable);
    }
    profiler.checkpoint("continuation destructuring");
    // Splitting at a suspend boundary can cut paths inside an otherwise
    // reducible source loop. A continuation scope may consequently contain a
    // residual cyclic SCC with several entry nodes even though the original
    // structured coroutine was reducible. Normalize exactly the generated
    // continuations before generic restructuring. The lowering routes every
    // entry edge through a selector and one dispatcher; it never clones the
    // shader body and therefore has linear CFG/code-size cost.
    for (auto &subroutine : split_info.subroutines) {
        auto irreducible_info =
            xir::lower_irreducible_cfg_pass_run_on_function(
                subroutine.callable);
        if (!irreducible_info.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "Coroutine irreducible-CFG lowering failed "
                "(remaining={}, errors={}).",
                irreducible_info.remaining_irreducible_region_count,
                irreducible_info.error_count);
        }
    }
    profiler.checkpoint("irreducible-CFG lowering");
    for (auto &subroutine : split_info.subroutines) {
        auto restructure_info =
            xir::restructure_cfg_pass_run_on_function(
                subroutine.callable,
                {.mutation_mode =
                     xir::RestructureCFGMutationMode::
                         IN_PLACE_DISCARDABLE});
        if (!restructure_info.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "Coroutine continuation {} restructuring failed "
                "(irreducible={}, unstructured={}, invalid={}, "
                "iteration_limit={}).",
                subroutine.scope_index,
                restructure_info.irreducible_region_count,
                restructure_info.unstructured_branch_count,
                restructure_info.invalid_construct_count,
                restructure_info.iteration_limit_count);
        }
        (void)xir::dce_pass_run_on_function(
            subroutine.callable);
        (void)xir::reg2mem_pass_run_on_function(
            subroutine.callable);
    }
    profiler.checkpoint("continuation restructuring");
    if (!ordinary_callable_snapshots.empty()) {
        verify_ordinary_callables_unchanged(
            ordinary_callable_snapshots, "continuation normalization");
        profiler.checkpoint("pass-domain verification");
    }
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
    luisa::vector<const xir::FunctionDefinition *> continuation_definitions;
    continuation_definitions.reserve(subroutines_by_scope.size());
    for (size_t scope_index = 0u;
         scope_index < subroutines_by_scope.size(); ++scope_index) {
        auto *subroutine = subroutines_by_scope[scope_index];
        LUISA_ASSERT(
            subroutine != nullptr,
            "Coroutine lowering did not materialize distilled scope {}.",
            scope_index);
        LUISA_ASSERT(
            result.graph.node(scope_index).token ==
                subroutine->trigger_token,
            "Coroutine graph/callable token mismatch at scope {}.",
            scope_index);
        continuation_definitions.emplace_back(subroutine->callable);
        result.trigger_tokens.emplace_back(subroutine->trigger_token);
    }
    xir::XIR2ASTTranslationStatistics xir_to_ast_statistics;
    auto continuation_asts =
        xir::xir_to_ast_translate_continuations(
            luisa::span{continuation_definitions},
            {.statistics = &xir_to_ast_statistics,
             .verify_value_map_checkpoints =
                 environment_flag_enabled(
                     "LUISA_XIR2AST_VERIFY_VALUE_MAP_CHECKPOINTS"),
             .verify_same_module_once = true});
    LUISA_ASSERT(
        xir_to_ast_statistics.whole_module_verification_count == 1u &&
            xir_to_ast_statistics.function_verification_count == 0u,
        "Coroutine XIR-to-AST handoff must verify exactly one immutable "
        "whole-module boundary and perform no redundant per-function "
        "verification.");
    LUISA_ASSERT(
        continuation_asts.size() == subroutines_by_scope.size(),
        "Coroutine XIR->AST batch translation returned {} AST(s) for {} scope(s).",
        continuation_asts.size(), subroutines_by_scope.size());
    for (size_t scope_index = 0u;
         scope_index < continuation_asts.size(); ++scope_index) {
        LUISA_ASSERT(
            continuation_asts[scope_index] != nullptr,
            "Coroutine XIR->AST translation failed for scope {} (trigger token {}).",
            scope_index, result.trigger_tokens[scope_index]);
        result.subroutines.emplace_back(
            std::move(continuation_asts[scope_index]));
    }
    LUISA_ASSERT(
        !result.trigger_tokens.empty() &&
            result.trigger_tokens.front() == 0u &&
            result.subroutines.size() == result.trigger_tokens.size(),
        "Coroutine lowering lost the entry continuation or callable/token pairing.");
    profiler.checkpoint_split(
        "output verification",
        xir_to_ast_statistics.verification_milliseconds,
        "XIR-to-AST continuation translation");
    if (environment_flag_enabled("LUISA_CORO_PROFILE_COMPILATION")) {
        LUISA_INFO(
            "Coroutine XIR-to-AST work: functions={} cache_hits={} "
            "module_verifications={} function_verifications={} "
            "verification_ms={:.3f} "
            "value_bindings={} checkpoints={} rollback_work={} "
            "peak_value_map_size={}.",
            xir_to_ast_statistics.function_translations,
            xir_to_ast_statistics.function_cache_hits,
            xir_to_ast_statistics.whole_module_verification_count,
            xir_to_ast_statistics.function_verification_count,
            xir_to_ast_statistics.verification_milliseconds,
            xir_to_ast_statistics.value_binding_insertions,
            xir_to_ast_statistics.value_map_checkpoint_count,
            xir_to_ast_statistics.value_map_rollback_work,
            xir_to_ast_statistics.peak_value_map_size);
    }
    profiler.finish();

    return result;
}

}// namespace luisa::compute::detail

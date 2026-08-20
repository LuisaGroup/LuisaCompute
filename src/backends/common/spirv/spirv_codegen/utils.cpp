#include "utils.h"
#include "dialect.h"
#include "pointer_legalization.h"
#include "structural_closure.h"

#include "../../backend_print_code.h"
#include <cstdlib>
#include <fstream>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/cvp.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/div_rem_pairs.h>
#include <luisa/xir/passes/early_cse.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/lower_ray_query_to_loop.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/reassociate.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/scalarizer.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/simplify_libcalls.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>

namespace luisa::compute::spirv {

SpirvInactivePayloadCleanupInfo
clear_spirv_codegen_inactive_block_payloads(xir::Module *module) noexcept {
    LUISA_ASSERT(module != nullptr,
                 "SPIR-V inactive-payload cleanup requires a module.");
    SpirvInactivePayloadCleanupInfo info;
    for (auto *function : module->function_list()) {
        auto *definition = function->definition();
        if (definition == nullptr || definition->body_block() == nullptr) {
            continue;
        }

        // Use the same structural closure as exact dialect validation and
        // physical planning. Membership distinguishes a true orphan (outside
        // exact emission) from a disconnected raw role block (identity emitted
        // at opt0, but payload proven dead at this post-restructure boundary).
        luisa::unordered_set<xir::BasicBlock *> structural_closure;
        for (auto *block :
             lc::spirv::collect_spirv_codegen_structural_closure(definition)) {
            structural_closure.emplace(
                const_cast<xir::BasicBlock *>(block));
        }

        // Ordinary reachability follows every encoded CFG edge without folding
        // constants. A raw structured merge/role may belong to the closure yet
        // have no ordinary executable predecessor. Restructure has already
        // made every executable transfer explicit, so its identity must
        // survive but its payload is dead and must not pin reg2mem spill slots.
        luisa::unordered_set<xir::BasicBlock *> ordinary_reachable;
        luisa::vector<xir::BasicBlock *> ordinary_worklist;
        auto enqueue_ordinary = [&](xir::BasicBlock *block) noexcept {
            if (block == nullptr) { return; }
            LUISA_ASSERT(structural_closure.contains(block),
                         "SPIR-V ordinary CFG escaped the structural closure.");
            if (ordinary_reachable.emplace(block).second) {
                ordinary_worklist.emplace_back(block);
            }
        };
        enqueue_ordinary(definition->body_block());
        while (!ordinary_worklist.empty()) {
            auto *block = ordinary_worklist.back();
            ordinary_worklist.pop_back();
            LUISA_ASSERT(block->is_terminated(),
                         "SPIR-V ordinary CFG contains an unterminated block.");
            for (auto *operand_use : block->terminator()->operand_uses()) {
                if (auto *value = operand_use->value();
                    value != nullptr && value->isa<xir::BasicBlock>()) {
                    enqueue_ordinary(static_cast<xir::BasicBlock *>(value));
                }
            }
        }

        luisa::unordered_set<xir::BasicBlock *> inactive;
        for (auto *block : definition->basic_blocks()) {
            if (!ordinary_reachable.contains(block)) {
                inactive.emplace(block);
            }
        }
        if (inactive.empty()) { continue; }

        // Phi predecessor identity is stored outside the ordinary operand list.
        // Detach inactive incoming edges before destroying their instructions.
        for (auto *block : ordinary_reachable) {
            block->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                if (!instruction->isa<xir::PhiInst>()) { return; }
                auto *phi = static_cast<xir::PhiInst *>(instruction);
                for (size_t i = phi->incoming_count(); i-- > 0u;) {
                    if (inactive.contains(phi->incoming(i).block)) {
                        phi->remove_incoming(i);
                        info.removed_phi_incoming_count++;
                    }
                }
            });
        }

        // Keep removed instructions alive until every inactive block has been
        // detached; such blocks may refer to values in one another. Any use in
        // the ordinary-live CFG is malformed cross-boundary SSA.
        luisa::vector<ManagedPtr<xir::Instruction>> removed;
        for (auto *block : inactive) {
            block->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                for (auto *use : instruction->use_list()) {
                    auto *user = use->user();
                    if (user != nullptr && user->isa<xir::Instruction>()) {
                        auto *user_block =
                            static_cast<xir::Instruction *>(user)->parent_block();
                        LUISA_ASSERT(user_block != nullptr &&
                                         inactive.contains(user_block),
                                     "SPIR-V inactive value is used by the live structural closure.");
                    }
                }
            });
        }
        for (auto *block : inactive) {
            auto already_empty_unreachable =
                block->instructions().count_size() == 1u &&
                block->instructions().front()->isa<xir::UnreachableInst>();
            if (already_empty_unreachable) { continue; }
            while (!block->instructions().empty()) {
                removed.emplace_back(
                    block->instructions().back()->remove_self());
                info.removed_instruction_count++;
            }
            xir::XIRBuilder builder;
            builder.set_insertion_point(block);
            builder.unreachable_(
                "SPIR-V legalization cleared an inactive block payload");
            info.cleared_block_count++;
            if (structural_closure.contains(block)) {
                info.cleared_disconnected_role_block_count++;
            } else {
                info.cleared_true_orphan_block_count++;
            }
        }
    }
    return info;
}

xir::PassPipeline
create_spirv_codegen_post_restructure_pipeline() noexcept {
    xir::PassPipeline pipeline;
    pipeline.add("clear-spirv-inactive-payloads", [](xir::Module *m,
                                                     xir::PassReport &r) {
        auto i = clear_spirv_codegen_inactive_block_payloads(m);
        r.set("cleared_inactive_block", i.cleared_block_count);
        r.set("cleared_true_orphan_block",
              i.cleared_true_orphan_block_count);
        r.set("cleared_disconnected_role_block",
              i.cleared_disconnected_role_block_count);
        r.set("removed_instruction", i.removed_instruction_count);
        r.set("removed_phi_incoming", i.removed_phi_incoming_count);
        return i.cleared_block_count > 0u ||
               i.removed_phi_incoming_count > 0u;
    });
    pipeline.add("mem2reg-post-restructure", [](xir::Module *m,
                                                xir::PassReport &r) {
        auto i = xir::mem2reg_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("audit-reg2mem-spills", [](xir::Module *m,
                                            xir::PassReport &r) {
        auto i = xir::audit_reg2mem_spills_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V post-restructure SSA recovery left {} reg2mem spill "
                "marker(s): {} PHI alloca(s), {} cross-block alloca(s), and "
                "{} invalid placement(s) or kind(s).",
                i.remaining_spill_count(),
                i.remaining_phi_spill_count,
                i.remaining_cross_block_spill_count,
                i.remaining_invalid_spill_count);
        }
        return false;
    });
    return pipeline;
}

namespace {

[[nodiscard]] bool optional_optimization_enabled() noexcept {
    if (auto env = std::getenv("LUISA_XIR_DISABLE_OPTIMIZATION")) {
        return luisa::string_view{env} != "1";
    }
    return true;
}

// The scalarizer is an optional optimization, not a legalization step.
// Keeping vector operations intact is the default because it preserves
// vector-level CSE and produces smaller SPIR-V for representative workloads.
// An explicit environment setting has process-wide precedence over the
// per-shader option, including an explicit "0" that disables it.
[[nodiscard]] bool scalarizer_enabled(
    const ShaderOption &option) noexcept {
    if (auto *env =
            std::getenv("LUISA_XIR_ENABLE_SCALARIZER")) {
        return luisa::string_view{env} == "1";
    }
    return option.enable_scalarizer;
}

[[nodiscard]] bool has_autodiff_scope(const xir::Module *module) noexcept {
    auto found = false;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
                found |= inst->derived_instruction_tag() ==
                         xir::DerivedInstructionTag::AUTODIFF_SCOPE;
            });
        }
    }
    return found;
}

void dump_xir_module(const xir::Module *module,
                     luisa::string_view filename) noexcept {
    std::ofstream f{luisa::string{filename}.c_str()};
    f << xir::xir_to_text_translate(module, true);
    auto flat_filename = luisa::format("{}.flat", filename);
    std::ofstream flat{flat_filename.c_str()};
    flat << xir::xir_to_flat_text_translate(module, true);
}

void dump_xir_stage(const xir::Module *module, uint64_t kernel_hash,
                    luisa::string_view stage) noexcept {
    if (luisa::compute::backend_print_code_enabled()) {
        auto filename = stage.empty() ?
                            luisa::format("kernel.{:016x}.xir", kernel_hash) :
                            luisa::format("kernel.{:016x}.{}.xir", kernel_hash, stage);
        dump_xir_module(module, filename);
    }
}

void verify_xir_or_error(
    const xir::Module *module, luisa::string_view stage,
    const xir::XIRVerificationOptions &options = {}) noexcept {
    auto verification = xir::xir_verify_module(module, options);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at SPIR-V {}: {} ({} error(s) total).",
            stage, verification.errors.front().message,
            verification.errors.size());
    }
}

[[nodiscard]] xir::PassPipeline make_structured_optimization_pipeline(
    const ShaderOption &option) noexcept {
    auto algebraic_options = xir::AlgebraicSimplifyOptions{
        .enable_fast_math = option.enable_fast_math};
    xir::PassPipeline pipeline;
    // This phase still owns structured Loop/If/Switch roles. Generic DCE is
    // intentionally deferred to the final explicitly destructured interval;
    // otherwise a constant Loop.prepare condition can be rewritten before its
    // owner is lowered.
    if (scalarizer_enabled(option)) {
        pipeline.add(
            "scalarizer",
            [](xir::Module *m,
               xir::PassReport &r) {
                auto i =
                    xir::scalarizer_pass_run_on_module(
                        m, &r);
                return i.scalarized_inst_count > 0u;
            });
    }
    pipeline.add("trace-gep", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::trace_gep_pass_run_on_module(m);
        r.set("traced_gep", i.traced_gep_count);
        r.set("removed_noop_gep", i.removed_noop_gep_count);
        return i.changed();
    });
    pipeline.add("local-store-forward", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::local_store_forward_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    pipeline.add("local-load-elimination", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::local_load_elimination_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    pipeline.add("algebraic-simplify", [algebraic_options](
                                           xir::Module *m,
                                           xir::PassReport &r) {
        auto i = xir::algebraic_simplify_pass_run_on_module(
            m, algebraic_options, &r);
        return i.simplified_inst_count > 0u;
    });
    pipeline.add("simplify-libcalls", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::simplify_libcalls_pass_run_on_module(m, &r);
        return i.simplified_count > 0u;
    });
    pipeline.add("reassociate", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::reassociate_pass_run_on_module(m, &r);
        return i.reassociated_inst_count > 0u;
    });
    pipeline.add("const-fold", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    pipeline.add("cvp", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::cvp_pass_run_on_module(m, &r);
        return i.replaced_inst_count > 0u;
    });
    pipeline.add("indvar-simplify", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::indvar_simplify_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("div-rem-pairs", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::div_rem_pairs_pass_run_on_module(m, &r);
        return i.merged_pair_count > 0u;
    });
    pipeline.add("promote-ref-arg", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::promote_ref_arg_pass_run_on_module(m, &r);
        return i.promoted_ref_arg_count > 0u;
    });
    pipeline.add("sroa", [](xir::Module *m, xir::PassReport &r) {
        auto options = xir::SROAOptions{
            .decompose_vectors = true,
            .decompose_matrices = false,
            .aggressive = false};
        auto i = xir::sroa_pass_run_on_module(m, options, &r);
        return i.changed();
    });
    pipeline.add("gvn", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::gvn_pass_run_on_module(m, &r);
        return i.changed();
    });
    pipeline.add("dead-store-elimination", [](xir::Module *m,
                                              xir::PassReport &r) {
        auto i = xir::dead_store_elimination_pass_run_on_module(m, &r);
        return i.eliminated_store_count > 0u;
    });
    return pipeline;
}

void add_lower_ray_query_to_loop(xir::PassPipeline &pipeline) noexcept {
    pipeline.add("lower-ray-query-to-loop", [](xir::Module *m,
                                                    xir::PassReport &r) {
        auto i = xir::lower_ray_query_to_loop_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V XIR legalization rejected {} ray-query loop(s).",
                i.error_count);
        }
        return i.lowered_ray_query_loop_count > 0u;
    });
}

void add_destructure_cfg(xir::PassPipeline &pipeline) noexcept {
    pipeline.add("destructure-cfg", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::destructure_cfg_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V XIR destructuring failed (errors={}, leaked_blocks={}).",
                i.error_count, i.leaked_block_count);
        }
        return i.changed();
    });
}

void add_promote_readonly_ref_args(xir::PassPipeline &pipeline) noexcept {
    pipeline.add("promote-readonly-ref-args",
                 [](xir::Module *m, xir::PassReport &r) {
                     auto i = xir::promote_ref_arg_pass_run_on_module(m, &r);
                     return i.promoted_ref_arg_count > 0u;
                 });
}

void add_inline_spirv_pointer_args(xir::PassPipeline &pipeline) noexcept {
    pipeline.add("inline-spirv-pointer-args",
                 [](xir::Module *m, xir::PassReport &r) {
                     auto i = lc::spirv::legalize_spirv_pointer_arguments(m);
                     r.set("planned_pointer_call",
                           i.planned_pointer_call_count);
                     r.set("blocking_function",
                           i.blocking_function_count);
                     r.set("destructured_blocking_function",
                           i.destructured_blocking_function_count);
                     r.set("destructured_blocking_switch",
                           i.destructured_switch_count);
                     r.set("pruned_unreachable_callable",
                           i.pruned_unreachable_callable_count);
                     r.set("inlined_call",
                           i.inline_info.inlined_call_count);
                     r.set(
                         "consumed_call_site_diagnostic_metadata",
                         i.inline_info
                             .consumed_call_site_diagnostic_metadata_count);
                     r.set("skipped_structured_call",
                           i.inline_info.skipped_structured_call_count);
                     r.set("skipped_constrained_call",
                           i.inline_info.skipped_constrained_call_count);
                     r.set("skipped_metadata_call",
                           i.inline_info.skipped_metadata_call_count);
                     r.set("skipped_declaration_call",
                           i.inline_info.skipped_declaration_call_count);
                     r.set("rejected_malformed_call",
                           i.inline_info.rejected_malformed_call_count);
                     r.set("skipped_recursive_callable",
                           i.inline_info
                               .skipped_recursive_callable_count);
                     r.set("remaining_pointer_call",
                           i.remaining_pointer_call_count);
                     r.set("argument_usage_analysis",
                           i.argument_usage_analysis_count);
                     r.set("indexed_call_site",
                           i.indexed_call_site_count);
                     r.set("argument_usage_structural_closure",
                           i.argument_usage_structural_closure_count);
                     r.set("argument_usage_instruction_scan",
                           i.argument_usage_instruction_scan_count);
                     r.set("argument_usage_call_dependency",
                           i.argument_usage_call_dependency_count);
                     r.set("argument_usage_worklist_pop",
                           i.argument_usage_worklist_pop_count);
                     r.set("argument_usage_dependency_visit",
                           i.argument_usage_dependency_visit_count);
                     r.set("inline_summary_function",
                           i.inline_info.call_site_summary_function_count);
                     r.set(
                         "inline_summary_instruction_scan",
                         i.inline_info
                             .call_site_summary_instruction_scan_count);
                     r.set("inline_cached_apply",
                           i.inline_info.call_site_cached_apply_count);
                     r.set(
                         "inline_revalidated_apply",
                         i.inline_info
                             .call_site_revalidated_apply_count);
                     r.set(
                         "inline_clone_layout_function",
                         i.inline_info
                             .call_site_clone_layout_function_count);
                     r.set(
                         "inline_clone_layout_value",
                         i.inline_info
                             .call_site_clone_layout_value_count);
                     r.set(
                         "inline_dense_resolver_apply",
                         i.inline_info
                             .call_site_dense_resolver_apply_count);
                     r.set(
                         "inline_dense_resolver_fallback",
                         i.inline_info
                             .call_site_dense_resolver_fallback_count);
                     r.set(
                         "ordinary_inline_summary_function",
                         i.inline_info
                             .inline_pass_summary_function_count);
                     r.set(
                         "ordinary_inline_summary_instruction_scan",
                         i.inline_info
                             .inline_pass_summary_instruction_scan_count);
                     r.set(
                         "ordinary_inline_clone_layout_function",
                         i.inline_info
                             .inline_pass_clone_layout_function_count);
                     r.set(
                         "ordinary_inline_clone_layout_value",
                         i.inline_info
                             .inline_pass_clone_layout_value_count);
                     r.set(
                         "ordinary_inline_dense_resolver_apply",
                         i.inline_info
                             .inline_pass_dense_resolver_apply_count);
                     r.set(
                         "ordinary_inline_dense_resolver_fallback",
                         i.inline_info
                             .inline_pass_dense_resolver_fallback_count);
                     r.set(
                         "ordinary_inline_caller_barrier_function",
                         i.inline_info
                             .inline_pass_caller_barrier_function_count);
                     r.set(
                         "ordinary_inline_caller_barrier_instruction_scan",
                         i.inline_info
                             .inline_pass_caller_barrier_instruction_scan_count);
                     r.set(
                         "ordinary_inline_caller_barrier_cache_hit",
                         i.inline_info
                             .inline_pass_caller_barrier_cache_hit_count);
                     r.set(
                         "inline_recursion_function",
                         i.inline_info
                             .recursion_analysis_function_count);
                     r.set(
                         "inline_recursion_call_use_visit",
                         i.inline_info
                             .recursion_analysis_call_use_visit_count);
                     r.set("inline_recursion_edge",
                           i.inline_info
                               .recursion_analysis_edge_count);
                     r.set(
                         "inline_recursion_vertex_visit",
                         i.inline_info
                             .recursion_analysis_vertex_visit_count);
                     r.set(
                         "inline_recursion_edge_visit",
                         i.inline_info
                             .recursion_analysis_edge_visit_count);
                     if (!i.succeeded()) {
                         LUISA_ERROR_WITH_LOCATION(
                             "SPIR-V reference-argument legalization failed: {}",
                             i.diagnostic);
                     }
                     return i.inline_info.changed() ||
                            i.destructured_switch_count > 0u;
                 });
}

void add_reg2mem(xir::PassPipeline &pipeline, luisa::string name) noexcept {
    pipeline.add(std::move(name), [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::reg2mem_pass_run_on_module(m, &r);
        return i.changed();
    });
}

void add_restructure_cfg(xir::PassPipeline &pipeline) noexcept {
    pipeline.add("restructure-cfg", [](xir::Module *m, xir::PassReport &r) {
        // AST-to-XIR produces a fresh, exclusively owned legalization module.
        // A failed pass terminates code generation below, so the module is
        // discarded and does not need transactional shadow/replay. Successful
        // inputs still receive the same complete pre/post boundary checks.
        auto i = xir::restructure_cfg_pass_run_on_module(
            m, &r,
            {.mutation_mode =
                 xir::RestructureCFGMutationMode::
                     IN_PLACE_DISCARDABLE});
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V XIR restructuring failed (irreducible={}, "
                "unstructured={}, invalid={}, iteration_limit={}).",
                i.irreducible_region_count, i.unstructured_branch_count,
                i.invalid_construct_count, i.iteration_limit_count);
        }
        return i.changed();
    });
}

void add_fix_self_referential(xir::PassPipeline &pipeline) noexcept {
    pipeline.add("fix-self-referential", [](xir::Module *m,
                                            xir::PassReport &r) {
        auto i = xir::fix_self_referential_pass_run_on_module(m, &r);
        if (!i.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V XIR legalization left {} unresolved self-reference(s).",
                i.unresolved_count);
        }
        return i.fixed_count > 0u;
    });
}

[[nodiscard]] xir::PassPipeline make_pre_autodiff_legalization_pipeline(
    bool optimize, const ShaderOption &option) noexcept {
    auto optimization_options = xir::OptimizationPipelineOptions{
        .enable_fast_math = option.enable_fast_math};
    xir::PassPipeline pipeline;
    add_lower_ray_query_to_loop(pipeline);
    add_destructure_cfg(pipeline);

    // Autodiff requires a whole-program body. Multi-block callables are only
    // safe to inline after structured control flow has been destructured.
    pipeline.add("inline-all", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::inline_all_pass_run_on_module(
            m, {.allow_autodiff_scope_in_caller = true}, &r);
        return i.changed();
    });
    if (optimize) {
        pipeline.add_sequence(
            "post-inline-cleanup",
            xir::create_post_inline_cleanup_pipeline(optimization_options));
        pipeline.add("dead-arg-elim", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dead_arg_elim_pass_run_on_module(m, &r);
            return i.removed_arg_count > 0u;
        });
        pipeline.add("simplify-cfg", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::simplify_cfg_pass_run_on_module(m, &r);
            return i.changed();
        });
    }
    add_reg2mem(pipeline, "reg2mem-pre-restructure");
    add_restructure_cfg(pipeline);
    // Autodiff consumes Phi-free structured XIR, so this boundary deliberately
    // stays in typed spill-memory form instead of immediately running mem2reg.
    // The markers must survive autodiff and SROA until final legalization.
    add_reg2mem(pipeline, "reg2mem-post-restructure");
    add_fix_self_referential(pipeline);
    return pipeline;
}

[[nodiscard]] xir::PassPipeline make_autodiff_lowering_pipeline() noexcept {
    xir::PassPipeline pipeline;
    pipeline.add("autodiff", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::autodiff_pass_run_on_module(m);
        r.set("transformed_scope_count", i.transformed_scope_count);
        r.set("removed_instruction_count", i.removed_instruction_count);
        return i.changed();
    });
    pipeline.add("scalarizer", [](xir::Module *m, xir::PassReport &r) {
        auto i = xir::scalarizer_pass_run_on_module(m, &r);
        return i.scalarized_inst_count > 0u;
    });
    pipeline.add("sroa", [](xir::Module *m, xir::PassReport &r) {
        auto options = xir::SROAOptions{.decompose_vectors = true};
        auto i = xir::sroa_pass_run_on_module(m, options, &r);
        return i.changed();
    });
    // Pre-autodiff legalization reconstructed structured roles. Do not run
    // generic DCE or mem2reg on them here. Autodiff output deliberately stays
    // Phi-free through the upcoming final destructure/restructure interval;
    // immediate mem2reg would only synthesize Phis that final reg2mem must
    // lower again. SROA preserves typed spill provenance, and this reg2mem
    // tags any new Phi spills for the final legalization boundary to recover
    // and audit.
    add_reg2mem(pipeline, "reg2mem");
    return pipeline;
}

[[nodiscard]] xir::PassPipeline make_spirv_legalization_pipeline(
    bool optimize, const ShaderOption &option) noexcept {
    auto algebraic_options = xir::AlgebraicSimplifyOptions{
        .enable_fast_math = option.enable_fast_math};
    auto optimization_options = xir::OptimizationPipelineOptions{
        .enable_fast_math = option.enable_fast_math};
    xir::PassPipeline pipeline;

    // Everything outside the `optimize` blocks is part of the SPIR-V input
    // language contract and therefore cannot be disabled for debugging.
    add_lower_ray_query_to_loop(pipeline);
    add_promote_readonly_ref_args(pipeline);
    if (optimize) {
        pipeline.add("licm", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::licm_pass_run_on_module(m, &r);
            return i.hoisted_count > 0u;
        });
    }
    add_destructure_cfg(pipeline);
    add_inline_spirv_pointer_args(pipeline);

    if (optimize) {
        // This ordering is intentional: inlining structured multi-block
        // callables before destructure_cfg changes their control ownership.
        pipeline.add("inline", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::inline_pass_run_on_module(m, &r);
            return i.changed();
        });
        pipeline.add_sequence(
            "post-inline-cleanup",
            xir::create_post_inline_cleanup_pipeline(optimization_options));
        pipeline.add("dead-arg-elim", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dead_arg_elim_pass_run_on_module(m, &r);
            return i.removed_arg_count > 0u;
        });
        pipeline.add("mem2reg", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::mem2reg_pass_run_on_module(m, &r);
            return i.changed();
        });
        pipeline.add("algebraic-simplify", [algebraic_options](
                                               xir::Module *m,
                                               xir::PassReport &r) {
            auto i = xir::algebraic_simplify_pass_run_on_module(
                m, algebraic_options, &r);
            return i.simplified_inst_count > 0u;
        });
        pipeline.add("const-fold", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::const_fold_pass_run_on_module(m, &r);
            return i.folded_inst_count > 0u;
        });
        // Reassociate after inlining exposes new CSE opportunities.
        pipeline.add("reassociate", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::reassociate_pass_run_on_module(m, &r);
            return i.reassociated_inst_count > 0u;
        });
        pipeline.add("sccp", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::sccp_pass_run_on_module(m, &r);
            return i.changed();
        });
        pipeline.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.changed();
        });
        pipeline.add("early-cse", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::early_cse_pass_run_on_module(m, &r);
            return i.eliminated_inst_count > 0u;
        });
        pipeline.add("local-store-forward", [](xir::Module *m,
                                               xir::PassReport &r) {
            auto i = xir::local_store_forward_pass_run_on_module(m, &r);
            return i.removed_load_count > 0u;
        });
        pipeline.add("local-load-elimination", [](xir::Module *m,
                                                  xir::PassReport &r) {
            auto i = xir::local_load_elimination_pass_run_on_module(m, &r);
            return i.removed_load_count > 0u;
        });
        pipeline.add("dead-store-elimination", [](xir::Module *m,
                                                  xir::PassReport &r) {
            auto i = xir::dead_store_elimination_pass_run_on_module(m, &r);
            return i.eliminated_store_count > 0u;
        });
        pipeline.add("gvn", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::gvn_pass_run_on_module(m, &r);
            return i.changed();
        });
        // After destructure_cfg, all structured regions are lowered to plain
        // CFG. if_conversion can safely convert eligible diamonds to select
        // instructions, enabling downstream GVN/CSE to value-number across them.
        pipeline.add("if-conversion", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::if_conversion_pass_run_on_module(m, &r);
            return i.changed();
        });
        pipeline.add("phi-cleanup", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::phi_cleanup_pass_run_on_module(m, &r);
            return i.removed_phi_count > 0u;
        });
        pipeline.add("unused-callable-removal", [](xir::Module *m,
                                                   xir::PassReport &r) {
            auto i = xir::unused_callable_removal_pass_run_on_module(m, &r);
            return i.removed_callable_count > 0u;
        });
        pipeline.add("simplify-cfg", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::simplify_cfg_pass_run_on_module(m, &r);
            return i.changed();
        });
    }

    // restructure_cfg currently accepts phi-free CFG. This reg2mem is a
    // temporary boundary adapter, not the codegen representation.
    add_reg2mem(pipeline, "reg2mem-pre-restructure");
    add_restructure_cfg(pipeline);

    // Restructuring may leave true orphans and disconnected raw role blocks
    // behind. In this final ordinary legalization (after any deliberate
    // Phi-free autodiff interval), clear only their proven-inactive payloads,
    // retain every block identity, then recover SSA and audit immediately.
    // This boundary is deliberately independent of the optimization toggle
    // and contains no generic DCE/CFG simplification: those passes can change
    // the prepare form or erase one of a structured construct's role arms.
    // Native OpPhi emission, rather than spirv-opt, owns SSA reconstruction.
    pipeline.add_sequence(
        "post-restructure-boundary",
        create_spirv_codegen_post_restructure_pipeline());

    add_fix_self_referential(pipeline);
    return pipeline;
}

void run_pipeline(xir::Module *module, const xir::PassPipeline &pipeline,
                  luisa::string_view name) noexcept {
    auto stats = pipeline.run(module);
    LUISA_VERBOSE("{} done in {} ms.", name, stats.total_ms);
    stats.log(name);
}

}// namespace

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(
    Function kernel, const ShaderOption &option) noexcept
    -> luisa::unique_ptr<xir::Module> {
    Clock translate_clock;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { xir_module->set_location(option.name); }
    verify_xir_or_error(xir_module.get(), "AST translation");
    dump_xir_stage(xir_module.get(), kernel.hash(), {});
    LUISA_VERBOSE("XIR translation done in {} ms.", translate_clock.toc());

    auto optimize = optional_optimization_enabled();
    if (optimize) {
        auto pipeline = make_structured_optimization_pipeline(option);
        run_pipeline(xir_module.get(), pipeline,
                     "SPIR-V structured optimization");
        verify_xir_or_error(xir_module.get(), "structured optimization");
        dump_xir_stage(xir_module.get(), kernel.hash(), "structured_opt");
    }

    if (has_autodiff_scope(xir_module.get())) {
        auto pre_autodiff =
            make_pre_autodiff_legalization_pipeline(optimize, option);
        run_pipeline(xir_module.get(), pre_autodiff,
                     "SPIR-V pre-autodiff legalization");
        verify_xir_or_error(
            xir_module.get(), "pre-autodiff legalization",
            {.require_unique_merge_blocks = true,
             .require_canonical_break_continue_targets = true});
        dump_xir_stage(xir_module.get(), kernel.hash(), "pre_ad");

        auto autodiff = make_autodiff_lowering_pipeline();
        run_pipeline(xir_module.get(), autodiff,
                     "SPIR-V autodiff lowering");
        verify_xir_or_error(xir_module.get(), "autodiff lowering");
        dump_xir_stage(xir_module.get(), kernel.hash(), "ad");
    }

    auto legalization = make_spirv_legalization_pipeline(optimize, option);
    run_pipeline(xir_module.get(), legalization, "SPIR-V XIR legalization");
    // Preserve the exact handoff IR when source dumping is enabled, including
    // invalid modules. This makes a fail-closed dialect diagnostic actionable
    // without changing the validation or emission boundary.
    dump_xir_stage(xir_module.get(), kernel.hash(), "norm");
    auto dialect = lc::spirv::validate_spirv_xir_codegen_dialect(
        xir_module.get(),
        {.release_assertions_are_no_op =
             !option.enable_debug_info});
    if (!dialect.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at SPIR-V codegen handoff: {} "
            "({} diagnostic(s) total).",
            dialect.diagnostics.front().message,
            dialect.diagnostics.size());
    }
    return xir_module;
}

}// namespace luisa::compute::spirv

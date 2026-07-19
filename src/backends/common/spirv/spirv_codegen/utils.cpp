#include "utils.h"

#include "../../backend_print_code.h"
#include <cstdlib>
#include <fstream>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>
#include <luisa/xir/passes/lower_switch.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/cvp.h>
#include <luisa/xir/passes/dead_arg_elim.h>
#include <luisa/xir/passes/div_rem_pairs.h>
#include <luisa/xir/passes/early_return_elimination.h>
#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/lower_break_continue.h>
#include <luisa/xir/passes/reassociate.h>
#include <luisa/xir/passes/scalarizer.h>
#include <luisa/xir/passes/simplify_libcalls.h>
#include <luisa/xir/passes/trace_gep.h>
#include <luisa/xir/passes/transpose_gep.h>
#include <luisa/xir/passes/fix_self_referential.h>
#include <luisa/xir/passes/scalar_evolution.h>
#include <luisa/xir/passes/alias_analysis.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/outline.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/verifier.h>

namespace luisa::compute::spirv {

namespace {

const bool LUISA_XIR_DISABLE_NORMALIZE_CFG = [] {
    if (auto env = getenv("LUISA_XIR_DISABLE_NORMALIZE_CFG")) {
        return luisa::string_view{env} == "1";
    }
    return false;
}();

const bool LUISA_XIR_DISABLE_RESTRUCTURE_CFG = [] {
    if (auto env = getenv("LUISA_XIR_DISABLE_RESTRUCTURE_CFG")) {
        return luisa::string_view{env} == "1";
    }
    return false;
}();

const bool LUISA_XIR_DISABLE_OPTIMIZATION = [] {
    if (auto env = getenv("LUISA_XIR_DISABLE_OPTIMIZATION")) {
        return luisa::string_view{env} == "1";
    }
    return false;
}();

[[nodiscard]] bool has_autodiff_scope(xir::Module *module) noexcept {
    auto found = false;
    for (auto f : module->function_list()) {
        if (auto def = f->definition()) {
            def->traverse_instructions([&](xir::Instruction *inst) noexcept {
                found |= inst->derived_instruction_tag() == xir::DerivedInstructionTag::AUTODIFF_SCOPE;
            });
        }
    }
    return found;
}

void dump_xir_module(const xir::Module *module, luisa::string_view filename) noexcept {
    std::ofstream f{luisa::string{filename}.c_str()};
    f << xir::xir_to_text_translate(module, true);
    auto flat_filename = luisa::format("{}.flat", filename);
    std::ofstream flat{flat_filename.c_str()};
    flat << xir::xir_to_flat_text_translate(module, true);
}

void verify_xir_or_error(const xir::Module *module, luisa::string_view stage,
                         const xir::XIRVerificationOptions &options = {}) noexcept {
    auto verification = xir::xir_verify_module(module, options);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at SPIR-V {}: {} ({} error(s) total).",
            stage, verification.errors.front().message, verification.errors.size());
    }
}

}// namespace

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept -> luisa::unique_ptr<xir::Module> {
    Clock translate_clk;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { xir_module->set_location(option.name); }
    verify_xir_or_error(xir_module.get(), "AST translation");

    if (luisa::compute::backend_print_code_enabled()) {
        auto filename = luisa::format("kernel.{:016x}.xir", kernel.hash());
        dump_xir_module(xir_module.get(), filename);
    }

    LUISA_VERBOSE("XIR translation done in {} ms.", translate_clk.toc());

    // Pipeline invariants:
    //   Phase A runs on structured-CFG alloca-form (ast2xir output).
    //   destructure_cfg: structured -> unstructured.
    //   Phase B runs SSA opts on unstructured CFG; mem2reg legal here.
    //   reg2mem before restructure_cfg: restructure_cfg requires phi-free input.
    //   restructure_cfg: unstructured -> structured.
    //   reg2mem after restructure_cfg: eliminates any remaining phis so that
    //   SPIR-V codegen doesn't need OpPhi (planned for future optimization).
    //   Typed-buffer scalar-to-vector fusion is deliberately disabled: XIR
    //   BUFFER_READ/WRITE access types must match the buffer element type.

    if (!LUISA_XIR_DISABLE_OPTIMIZATION) {
        Clock opt_clk;
        auto algebraic_options = xir::AlgebraicSimplifyOptions{.enable_fast_math = option.enable_fast_math};
        auto opt_options = xir::OptimizationPipelineOptions{.enable_fast_math = option.enable_fast_math};

        xir::PassPipeline phase_a;
        phase_a.add("scalarizer", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::scalarizer_pass_run_on_module(m, &r);
            return i.scalarized_inst_count > 0u;
        });
        phase_a.add("trace-gep", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::trace_gep_pass_run_on_module(m);
            return i.traced_gep_count > 0u;
        });
        phase_a.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.removed_inst_count > 0u || i.removed_block_count > 0u;
        });
        phase_a.add("local-store-forward", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::local_store_forward_pass_run_on_module(m, &r);
            return i.removed_load_count > 0u;
        });
        phase_a.add("local-load-elimination", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::local_load_elimination_pass_run_on_module(m, &r);
            return i.removed_load_count > 0u;
        });
        phase_a.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.removed_inst_count > 0u || i.removed_block_count > 0u;
        });
        phase_a.add("algebraic-simplify", [algebraic_options](xir::Module *m, xir::PassReport &r) {
            auto i = xir::algebraic_simplify_pass_run_on_module(m, algebraic_options, &r);
            return i.simplified_inst_count > 0u;
        });
        phase_a.add("simplify-libcalls", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::simplify_libcalls_pass_run_on_module(m, &r);
            return i.simplified_count > 0u;
        });
        phase_a.add("reassociate", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::reassociate_pass_run_on_module(m, &r);
            return i.reassociated_inst_count > 0u;
        });
        phase_a.add("const-fold", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::const_fold_pass_run_on_module(m, &r);
            return i.folded_inst_count > 0u;
        });
        phase_a.add("cvp", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::cvp_pass_run_on_module(m, &r);
            return i.replaced_inst_count > 0u;
        });
        phase_a.add("div-rem-pairs", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::div_rem_pairs_pass_run_on_module(m, &r);
            return i.merged_pair_count > 0u;
        });
        phase_a.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.removed_inst_count > 0u || i.removed_block_count > 0u;
        });
        phase_a.add("promote-ref-arg", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::promote_ref_arg_pass_run_on_module(m, &r);
            return i.promoted_ref_arg_count > 0u;
        });
        phase_a.add("sroa", [](xir::Module *m, xir::PassReport &r) {
            xir::SROAOptions sroa_opts;
            sroa_opts.decompose_vectors = true;
            sroa_opts.decompose_matrices = false;
            sroa_opts.aggressive = false;
            auto i = xir::sroa_pass_run_on_module(m, sroa_opts, &r);
            return i.decomposed_alloca_count > 0u;
        });
        phase_a.add("gvn", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::gvn_pass_run_on_module(m, &r);
            return i.replaced_inst_count > 0u || i.removed_inst_count > 0u;
        });
        phase_a.add("dead-store-elimination", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dead_store_elimination_pass_run_on_module(m, &r);
            return i.eliminated_store_count > 0u;
        });
        phase_a.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.removed_inst_count > 0u || i.removed_block_count > 0u;
        });
        auto phase_a_stats = phase_a.run(xir_module.get());
        verify_xir_or_error(xir_module.get(), "phase A");
        LUISA_VERBOSE("SPIR-V Phase A done in {} ms.", phase_a_stats.total_ms);
        phase_a_stats.log("SPIR-V Phase A");

        auto has_ad_scope = has_autodiff_scope(xir_module.get());
        if (has_ad_scope) {
            xir::PassPipeline pre_autodiff;
            if (!LUISA_XIR_DISABLE_NORMALIZE_CFG && !LUISA_XIR_DISABLE_RESTRUCTURE_CFG) {
                pre_autodiff.add("lower-ray-query-loop-to-loop", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::lower_ray_query_loop_to_loop_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR pre-autodiff normalization rejected {} ray-query loop(s).",
                            i.error_count);
                    }
                    return i.lowered_ray_query_loop_count > 0u;
                });
                pre_autodiff.add("lower-switch", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::lower_switch_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR pre-autodiff normalization rejected {} switch(es).",
                            i.rejected_switch_count);
                    }
                    return i.lowered_switch_count > 0u;
                });
                pre_autodiff.add("destructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::destructure_cfg_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR pre-autodiff destructuring failed (errors={}, leaked_blocks={}).",
                            i.error_count, i.leaked_block_count);
                    }
                    return i.destructured_if_count > 0u ||
                           i.destructured_loop_count > 0u ||
                           i.destructured_simple_loop_count > 0u;
                });
                pre_autodiff.add("inline-all", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::inline_all_pass_run_on_module(
                        m, {.allow_autodiff_scope_in_caller = true}, &r);
                    return i.inlined_call_count > 0u;
                });
                pre_autodiff.add_fixed_point(
                    "post-inline-cleanup",
                    xir::create_post_inline_cleanup_pipeline(opt_options), 1u);
                pre_autodiff.add("dead-arg-elim", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dead_arg_elim_pass_run_on_module(m, &r);
                    return i.removed_arg_count > 0u;
                });
                pre_autodiff.add("dce", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dce_pass_run_on_module(m, &r);
                    return i.removed_inst_count > 0u || i.removed_block_count > 0u;
                });
                pre_autodiff.add("simplify-cfg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::simplify_cfg_pass_run_on_module(m, &r);
                    return i.folded_constant_cond_br_count > 0u ||
                           i.folded_switch_count > 0u ||
                           i.threaded_empty_block_count > 0u ||
                           i.merged_straight_line_count > 0u ||
                           i.removed_unreachable_block_count > 0u;
                });
                pre_autodiff.add("reg2mem-pre", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::reg2mem_pass_run_on_module(m, &r);
                    return i.lowered_phi_count > 0u ||
                           i.lowered_cross_block_value_count > 0u;
                });
                pre_autodiff.add("restructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::restructure_cfg_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR pre-autodiff restructuring failed (irreducible={}, unstructured={}, invalid={}, iteration_limit={}).",
                            i.irreducible_region_count, i.unstructured_branch_count,
                            i.invalid_construct_count, i.iteration_limit_count);
                    }
                    return i.restructured_loop_count > 0u || i.restructured_if_count > 0u;
                });
                pre_autodiff.add("dce", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dce_pass_run_on_module(m, &r);
                    return i.removed_inst_count > 0u || i.removed_block_count > 0u;
                });
                pre_autodiff.add("reg2mem-post", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::reg2mem_pass_run_on_module(m, &r);
                    return i.lowered_phi_count > 0u ||
                           i.lowered_cross_block_value_count > 0u;
                });
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "SPIR-V XIR autodiff requires CFG normalization and restructuring: "
                    "multi-block callable inlining must run after destructure_cfg. "
                    "Unset LUISA_XIR_DISABLE_NORMALIZE_CFG and "
                    "LUISA_XIR_DISABLE_RESTRUCTURE_CFG.");
            }
            auto pre_autodiff_stats = pre_autodiff.run(xir_module.get());
            if (luisa::compute::backend_print_code_enabled()) {
                auto filename = luisa::format("kernel.{:016x}.pre_ad.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }
            verify_xir_or_error(
                xir_module.get(), "pre-autodiff normalization",
                {.require_no_phi = !LUISA_XIR_DISABLE_NORMALIZE_CFG &&
                                   !LUISA_XIR_DISABLE_RESTRUCTURE_CFG,
                 .require_unique_merge_blocks = !LUISA_XIR_DISABLE_NORMALIZE_CFG &&
                                                !LUISA_XIR_DISABLE_RESTRUCTURE_CFG});
            LUISA_VERBOSE("SPIR-V pre-autodiff normalization done in {} ms.", pre_autodiff_stats.total_ms);
            pre_autodiff_stats.log("SPIR-V pre-autodiff");
        }
        xir::PassPipeline autodiff;
        autodiff.add("autodiff", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::autodiff_pass_run_on_module(m);
            r.set("transformed_scope_count", i.transformed_scope_count);
            r.set("removed_instruction_count", i.removed_instruction_count);
            return i.transformed_scope_count > 0u || i.removed_instruction_count > 0u;
        });
        autodiff.add("scalarizer", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::scalarizer_pass_run_on_module(m, &r);
            return i.scalarized_inst_count > 0u;
        });
        autodiff.add("sroa", [](xir::Module *m, xir::PassReport &r) {
            xir::SROAOptions sroa_opts;
            sroa_opts.decompose_vectors = true;
            auto i = xir::sroa_pass_run_on_module(m, sroa_opts, &r);
            return i.decomposed_alloca_count > 0u;
        });
        autodiff.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.removed_inst_count > 0u || i.removed_block_count > 0u;
        });
        // Reverse-mode replay may use a primal value produced only in the
        // corresponding forward branch. Materialize such cross-block values
        // as local tape slots before verifying or normalizing the AD result.
        autodiff.add("reg2mem", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::reg2mem_pass_run_on_module(m, &r);
            return i.lowered_phi_count > 0u ||
                   i.lowered_cross_block_value_count > 0u;
        });
        auto autodiff_stats = autodiff.run(xir_module.get());
        if (luisa::compute::backend_print_code_enabled()) {
            auto filename = luisa::format("kernel.{:016x}.ad.xir", kernel.hash());
            dump_xir_module(xir_module.get(), filename);
        }
        verify_xir_or_error(xir_module.get(), "autodiff lowering");
        LUISA_VERBOSE("SPIR-V autodiff lowering done in {} ms.", autodiff_stats.total_ms);
        autodiff_stats.log("SPIR-V autodiff");

        if (!LUISA_XIR_DISABLE_NORMALIZE_CFG) {
            xir::PassPipeline norm;
            norm.add("lower-ray-query-loop-to-loop", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::lower_ray_query_loop_to_loop_pass_run_on_module(m, &r);
                if (!i.succeeded()) {
                    LUISA_ERROR_WITH_LOCATION(
                        "SPIR-V XIR normalization rejected {} ray-query loop(s).",
                        i.error_count);
                }
                return i.lowered_ray_query_loop_count > 0u;
            });

            if (!LUISA_XIR_DISABLE_RESTRUCTURE_CFG) {
                norm.add("lower-switch", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::lower_switch_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR normalization rejected {} switch(es).",
                            i.rejected_switch_count);
                    }
                    return i.lowered_switch_count > 0u;
                });
                // Keep structured loop transforms out of the production pipeline
                // until they preserve loop ownership, PHIs, and break/continue.
                norm.add("licm", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::licm_pass_run_on_module(m, &r);
                    return i.hoisted_count > 0u;
                });
                norm.add("destructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::destructure_cfg_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR destructuring failed (errors={}, leaked_blocks={}).",
                            i.error_count, i.leaked_block_count);
                    }
                    return i.destructured_if_count > 0u ||
                           i.destructured_loop_count > 0u ||
                           i.destructured_simple_loop_count > 0u;
                });
                norm.add("inline", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::inline_pass_run_on_module(m, &r);
                    return i.inlined_call_count > 0u;
                });
                norm.add_fixed_point(
                    "post-inline-cleanup",
                    xir::create_post_inline_cleanup_pipeline(opt_options), 1u);
                norm.add("dead-arg-elim", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dead_arg_elim_pass_run_on_module(m, &r);
                    return i.removed_arg_count > 0u;
                });
                norm.add("post-inline-dce", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dce_pass_run_on_module(m, &r);
                    return i.removed_inst_count > 0u || i.removed_block_count > 0u;
                });
                norm.add("mem2reg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::mem2reg_pass_run_on_module(m, &r);
                    return i.promoted_alloca_count > 0u;
                });
                norm.add("algebraic-simplify", [algebraic_options](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::algebraic_simplify_pass_run_on_module(m, algebraic_options, &r);
                    return i.simplified_inst_count > 0u;
                });
                norm.add("const-fold", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::const_fold_pass_run_on_module(m, &r);
                    return i.folded_inst_count > 0u;
                });
                // SCCP: fixed loop-carried phi unsoundness (UNDEFINED was TOP,
                // and visit_arithmetic had an operand-order bug that missed BOTTOM).
                norm.add("sccp", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::sccp_pass_run_on_module(m, &r);
                    return i.folded_inst_count > 0u || i.removed_branch_count > 0u;
                });
                norm.add("dce", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dce_pass_run_on_module(m, &r);
                    return i.removed_inst_count > 0u || i.removed_block_count > 0u;
                });
                norm.add("local-store-forward", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::local_store_forward_pass_run_on_module(m, &r);
                    return i.removed_load_count > 0u;
                });
                norm.add("local-load-elimination", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::local_load_elimination_pass_run_on_module(m, &r);
                    return i.removed_load_count > 0u;
                });
                norm.add("dead-store-elimination", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dead_store_elimination_pass_run_on_module(m, &r);
                    return i.eliminated_store_count > 0u;
                });
                norm.add("dce", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dce_pass_run_on_module(m, &r);
                    return i.removed_inst_count > 0u || i.removed_block_count > 0u;
                });
                norm.add("gvn", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::gvn_pass_run_on_module(m, &r);
                    return i.replaced_inst_count > 0u || i.removed_inst_count > 0u;
                });
                norm.add("if-conversion", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::if_conversion_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR if-conversion rejected {} structured function(s).",
                            i.structured_cfg_error_count);
                    }
                    return i.converted_diamond_count > 0u;
                });
                norm.add("phi-cleanup", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::phi_cleanup_pass_run_on_module(m, &r);
                    return i.removed_phi_count > 0u;
                });
                norm.add("unused-callable-removal", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::unused_callable_removal_pass_run_on_module(m, &r);
                    return i.removed_callable_count > 0u;
                });
                norm.add("simplify-cfg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::simplify_cfg_pass_run_on_module(m, &r);
                    return i.folded_constant_cond_br_count > 0u ||
                           i.folded_switch_count > 0u ||
                           i.threaded_empty_block_count > 0u ||
                           i.merged_straight_line_count > 0u ||
                           i.removed_unreachable_block_count > 0u;
                });
                norm.add("reg2mem-pre", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::reg2mem_pass_run_on_module(m, &r);
                    return i.lowered_phi_count > 0u ||
                           i.lowered_cross_block_value_count > 0u;
                });
                norm.add("restructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::restructure_cfg_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR restructuring failed (irreducible={}, unstructured={}, invalid={}, iteration_limit={}).",
                            i.irreducible_region_count, i.unstructured_branch_count,
                            i.invalid_construct_count, i.iteration_limit_count);
                    }
                    return i.restructured_loop_count > 0u || i.restructured_if_count > 0u;
                });
                norm.add("mem2reg-post-restructure", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::mem2reg_pass_run_on_module(m, &r);
                    return i.promoted_alloca_count > 0u;
                });
                norm.add("dce", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::dce_pass_run_on_module(m, &r);
                    return i.removed_inst_count > 0u || i.removed_block_count > 0u;
                });
                norm.add("reg2mem-mid", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::reg2mem_pass_run_on_module(m, &r);
                    return i.lowered_phi_count > 0u ||
                           i.lowered_cross_block_value_count > 0u;
                });
                norm.add_fixed_point("phase-c", xir::create_post_restructure_cleanup_pipeline(opt_options), 3u);
                norm.add("fix-self-referential", [](xir::Module *m, xir::PassReport &r) {
                    auto i = xir::fix_self_referential_pass_run_on_module(m, &r);
                    if (!i.succeeded()) {
                        LUISA_ERROR_WITH_LOCATION(
                            "SPIR-V XIR normalization left {} unresolved self-reference(s).",
                            i.unresolved_count);
                    }
                    return i.fixed_count > 0u;
                });
            }
            auto norm_stats = norm.run(xir_module.get());
            verify_xir_or_error(
                xir_module.get(), "CFG normalization",
                {.require_no_phi = !LUISA_XIR_DISABLE_RESTRUCTURE_CFG,
                 .require_unique_merge_blocks = !LUISA_XIR_DISABLE_RESTRUCTURE_CFG});
            LUISA_VERBOSE("SPIR-V CFG normalization done in {} ms.", norm_stats.total_ms);
            norm_stats.log("SPIR-V CFG normalization");

            if (luisa::compute::backend_print_code_enabled()) {
                auto filename = luisa::format("kernel.{:016x}.norm.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }
        }

        if (luisa::compute::backend_print_code_enabled()) {
            auto filename = luisa::format("kernel.{:016x}.opt.xir", kernel.hash());
            dump_xir_module(xir_module.get(), filename);
        }

        LUISA_VERBOSE("XIR optimization done in {} ms.", opt_clk.toc());

        if (luisa::compute::backend_print_code_enabled()) {
            auto filename = luisa::format("kernel.{:016x}.opt.rq.xir", kernel.hash());
            dump_xir_module(xir_module.get(), filename);
        }
    }// if (!LUISA_XIR_DISABLE_OPTIMIZATION)
    verify_xir_or_error(
        xir_module.get(), "codegen handoff",
        {.require_no_phi = true,
         .require_unique_merge_blocks = !LUISA_XIR_DISABLE_OPTIMIZATION &&
                                        !LUISA_XIR_DISABLE_NORMALIZE_CFG &&
                                        !LUISA_XIR_DISABLE_RESTRUCTURE_CFG});
    return xir_module;
}

}// namespace luisa::compute::spirv

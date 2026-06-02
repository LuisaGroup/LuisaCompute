#include "utils.h"

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
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/loop_unroll.h>
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
#include <luisa/xir/passes/loop_rotation.h>
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

namespace luisa::compute::spirv {

namespace {

const bool LUISA_SPIRV_SHOULD_DUMP_XIR = [] {
    if (auto env = getenv("LUISA_DUMP_SOURCE")) {
        return luisa::string_view{env} == "1";
    }
    return false;
}();


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

void dump_xir_module(const xir::Module *module, luisa::string_view filename) noexcept {
    std::ofstream f{luisa::string{filename}.c_str()};
    f << xir::xir_to_text_translate(module, true);
    auto flat_filename = luisa::format("{}.flat", filename);
    std::ofstream flat{flat_filename.c_str()};
    flat << xir::xir_to_flat_text_translate(module, true);
}

}// namespace

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept -> luisa::unique_ptr<xir::Module> {
    Clock translate_clk;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { xir_module->set_location(option.name); }

    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
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
    //   SPIR-V codegen now emits OpPhi directly; post-reg2mem no longer required.

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
        auto i = xir::sroa_pass_run_on_module(m, {}, &r);
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
    bool inlined_anything = false;
    phase_a.add("inline", [&inlined_anything](xir::Module *m, xir::PassReport &r) {
        auto i = xir::inline_pass_run_on_module(m, &r);
        if (i.inlined_call_count > 0u) { inlined_anything = true; }
        return i.inlined_call_count > 0u;
    });
    auto phase_a_stats = phase_a.run(xir_module.get());
    LUISA_VERBOSE("SPIR-V Phase A done in {} ms.", phase_a_stats.total_ms);
    phase_a_stats.log("SPIR-V Phase A");

    if (inlined_anything) {
        auto post_inline = xir::create_post_inline_cleanup_pipeline(opt_options);
        auto post_inline_stats = post_inline.run(xir_module.get());
        LUISA_VERBOSE("SPIR-V post-inline cleanup done in {} ms.", post_inline_stats.total_ms);
        post_inline_stats.log("SPIR-V post-inline cleanup");

        xir::PassPipeline post_inline_extra;
        post_inline_extra.add("dead-arg-elim", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dead_arg_elim_pass_run_on_module(m, &r);
            return i.removed_arg_count > 0u;
        });
        post_inline_extra.add("dce", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::dce_pass_run_on_module(m, &r);
            return i.removed_inst_count > 0u || i.removed_block_count > 0u;
        });
        auto post_inline_extra_stats = post_inline_extra.run(xir_module.get());
        LUISA_VERBOSE("SPIR-V post-inline extra done in {} ms.", post_inline_extra_stats.total_ms);
        post_inline_extra_stats.log("SPIR-V post-inline extra");
    }

    if (!LUISA_XIR_DISABLE_NORMALIZE_CFG) {
        xir::PassPipeline norm;
        norm.add("lower-ray-query-loop-to-loop", [](xir::Module *m, xir::PassReport &r) {
            auto i = xir::lower_ray_query_loop_to_loop_pass_run_on_module(m, &r);
            return i.lowered_ray_query_loop_count > 0u;
        });

        if (!LUISA_XIR_DISABLE_RESTRUCTURE_CFG) {
            norm.add("destructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::destructure_cfg_pass_run_on_module(m, &r);
                return i.destructured_if_count > 0u ||
                       i.destructured_loop_count > 0u ||
                       i.destructured_simple_loop_count > 0u;
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
            // SCCP is disabled because it incorrectly folds loop-carried phi nodes
            // created by mem2reg on unstructured CFG, causing scalar loop bodies
            // (e.g., in ONNX Gemm operators with local-array inputs) to be eliminated.
            // SPIRV-Tools optimizer (level 2) performs its own constant propagation,
            // so disabling XIR-level SCCP does not lose correctness.
            // TODO: Investigate a proper fix in sccp_pass for loop-carried phis.
            // norm.add("sccp", [](xir::Module *m, xir::PassReport &r) {
            //     auto i = xir::sccp_pass_run_on_module(m, &r);
            //     return i.folded_inst_count > 0u || i.removed_branch_count > 0u;
            // });
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
                return i.lowered_phi_count > 0u;
            });
            norm.add("restructure-cfg", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::restructure_cfg_pass_run_on_module(m, &r);
                return i.restructured_loop_count > 0u || i.restructured_if_count > 0u;
            });
            norm.add("dce", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::dce_pass_run_on_module(m, &r);
                return i.removed_inst_count > 0u || i.removed_block_count > 0u;
            });
            norm.add("reg2mem-mid", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::reg2mem_pass_run_on_module(m, &r);
                return i.lowered_phi_count > 0u;
            });
            norm.add_fixed_point("phase-c", xir::create_post_restructure_cleanup_pipeline(opt_options), 1u);
            norm.add("fix-self-referential", [](xir::Module *m, xir::PassReport &r) {
                auto i = xir::fix_self_referential_pass_run_on_module(m, &r);
                return i.fixed_count > 0u;
            });
        }
        auto norm_stats = norm.run(xir_module.get());
        LUISA_VERBOSE("SPIR-V CFG normalization done in {} ms.", norm_stats.total_ms);
        norm_stats.log("SPIR-V CFG normalization"); 

        if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
            auto filename = luisa::format("kernel.{:016x}.norm.xir", kernel.hash());
            dump_xir_module(xir_module.get(), filename);
        }
    }

    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.xir", kernel.hash());
        dump_xir_module(xir_module.get(), filename);
    }

    LUISA_VERBOSE("XIR optimization done in {} ms.", opt_clk.toc());

    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.rq.xir", kernel.hash());
        dump_xir_module(xir_module.get(), filename);
    }
    return xir_module;
}

}// namespace luisa::compute::spirv

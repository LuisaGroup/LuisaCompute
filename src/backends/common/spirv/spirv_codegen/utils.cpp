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
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/restructure_cfg.h>

namespace luisa::compute::spirv {

namespace {

const bool LUISA_SPIRV_SHOULD_DUMP_XIR = [] {
    if (auto env = getenv("LUISA_DUMP_SOURCE")) {
        return luisa::string_view{env} == "1";
    }
    return false;
}();

const bool LUISA_SPIRV_DUMP_OPT_STATS = [] {
    if (auto env = getenv("LUISA_SPIRV_DUMP_OPT_STATS")) {
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

    // Pipeline invariants:
    //   Phase A runs on structured-CFG alloca-form (ast2xir output).
    //   destructure_cfg: structured -> unstructured.
    //   Phase B runs SSA opts on unstructured CFG; mem2reg legal here.
    //   reg2mem before restructure_cfg: restructure_cfg requires phi-free input.
    //   restructure_cfg: unstructured -> structured.
    //   Phase C ends with reg2mem: SPIR-V emit rejects PhiInst.

    Clock opt_clk;
    Clock pass_clk;

    // Phase A
    auto dceA1_info = xir::dce_pass_run_on_module(xir_module.get());
    pass_clk.tic();
    auto storeA_info = xir::local_store_forward_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.store-forward: {} ms (forwarded {})", pass_clk.toc(), storeA_info.removed_load_count);
    pass_clk.tic();
    auto loadA_info = xir::local_load_elimination_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.load-elim: {} ms (eliminated {})", pass_clk.toc(), loadA_info.removed_load_count);
    pass_clk.tic();
    auto dceA2_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.dce2: {} ms", pass_clk.toc());
    pass_clk.tic();
    auto algA_info = xir::algebraic_simplify_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.algebraic-simplify: {} ms (simplified {})", pass_clk.toc(), algA_info.simplified_inst_count);
    pass_clk.tic();
    auto cfA_info = xir::const_fold_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.const-fold: {} ms (folded {})", pass_clk.toc(), cfA_info.folded_inst_count);
    pass_clk.tic();
    auto dceA3_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.dce3: {} ms", pass_clk.toc());
    pass_clk.tic();
    auto promote_arg_info = xir::promote_ref_arg_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.promote-ref-arg: {} ms (promoted {})", pass_clk.toc(), promote_arg_info.promoted_ref_arg_count);
    pass_clk.tic();
    auto sroa_info = xir::sroa_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.sroa: {} ms (decomposed {} into {})", pass_clk.toc(), sroa_info.decomposed_alloca_count, sroa_info.inserted_alloca_count);
    pass_clk.tic();
    auto loop_unroll_info = xir::loop_unroll_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.loop-unroll: {} ms (unrolled {})", pass_clk.toc(), loop_unroll_info.unrolled_loop_count);
    pass_clk.tic();
    auto dceA4_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  A.dce4: {} ms", pass_clk.toc());

    pass_clk.tic();
    auto inline_info = xir::inline_all_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  inline-all: {} ms (inlined {}, removed {})", pass_clk.toc(), inline_info.inlined_call_count, inline_info.removed_callable_count);
    if (inline_info.inlined_call_count > 0) {
        pass_clk.tic();
        xir::dce_pass_run_on_module(xir_module.get());
        xir::local_store_forward_pass_run_on_module(xir_module.get());
        xir::local_load_elimination_pass_run_on_module(xir_module.get());
        xir::dce_pass_run_on_module(xir_module.get());
        xir::algebraic_simplify_pass_run_on_module(xir_module.get());
        xir::const_fold_pass_run_on_module(xir_module.get());
        xir::dce_pass_run_on_module(xir_module.get());
        xir::sroa_pass_run_on_module(xir_module.get());
        xir::dce_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  post-inline-cleanup: {} ms", pass_clk.toc());
    }

    xir::DestructureCFGInfo destructure_cfg_info{};
    xir::SimplifyCFGInfo simplify_cfg_info{};
    xir::RestructureCFGInfo restructure_cfg_info{};
    xir::Mem2RegInfo mem2regB_info{};
    xir::Reg2MemInfo reg2mem_pre_info{};
    xir::UnusedCallableRemovalInfo unused_callable_info{};
    xir::LowerRayQueryLoopToLoopInfo rq_to_loop_info{};

    if (!LUISA_XIR_DISABLE_NORMALIZE_CFG) {
        pass_clk.tic();
        rq_to_loop_info = xir::lower_ray_query_loop_to_loop_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  lower-ray-query-loop-to-loop: {} ms (lowered {})", pass_clk.toc(), rq_to_loop_info.lowered_ray_query_loop_count);

        if (!LUISA_XIR_DISABLE_RESTRUCTURE_CFG) {
        pass_clk.tic();
        destructure_cfg_info = xir::destructure_cfg_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  destructure-cfg: {} ms", pass_clk.toc());

        if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
            auto filename = luisa::format("kernel.{:016x}.after_destructure.xir", kernel.hash());
            dump_xir_module(xir_module.get(), filename);
        }

        // Phase B
        pass_clk.tic();
        mem2regB_info = xir::mem2reg_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.mem2reg: {} ms (promoted {} alloca(s), {} phi(s))", pass_clk.toc(), mem2regB_info.promoted_alloca_count, mem2regB_info.inserted_phi_count);
        pass_clk.tic();
        auto algB_info = xir::algebraic_simplify_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.algebraic-simplify: {} ms (simplified {})", pass_clk.toc(), algB_info.simplified_inst_count);
        pass_clk.tic();
        auto cfB_info = xir::const_fold_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.const-fold: {} ms (folded {})", pass_clk.toc(), cfB_info.folded_inst_count);
        pass_clk.tic();
        auto dceB1_info = xir::dce_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.dce1: {} ms", pass_clk.toc());
        pass_clk.tic();
        auto storeB_info = xir::local_store_forward_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.store-forward: {} ms (forwarded {})", pass_clk.toc(), storeB_info.removed_load_count);
        pass_clk.tic();
        auto loadB_info = xir::local_load_elimination_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.load-elim: {} ms (eliminated {})", pass_clk.toc(), loadB_info.removed_load_count);
        pass_clk.tic();
        auto dceB2_info = xir::dce_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  B.dce2: {} ms", pass_clk.toc());

        pass_clk.tic();
        unused_callable_info = xir::unused_callable_removal_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  unused-callable-removal: {} ms (removed {})", pass_clk.toc(), unused_callable_info.removed_callable_count);

        pass_clk.tic();
        simplify_cfg_info = xir::simplify_cfg_pass_run_on_module(xir_module.get());
        if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  simplify-cfg: {} ms", pass_clk.toc());

        LUISA_VERBOSE("XIR CFG normalization done:\n"
                      "    destructured {} if(s), {} loop(s), {} simple loop(s), {} break(s), {} continue(s), {} ray query loop(s)->loop(s),\n"
                      "    simplified: folded {} constant cond_br(s), threaded {} empty block(s), removed {} unreachable block(s).",
                      destructure_cfg_info.destructured_if_count,
                      destructure_cfg_info.destructured_loop_count,
                      destructure_cfg_info.destructured_simple_loop_count,
                      destructure_cfg_info.destructured_break_count,
                      destructure_cfg_info.destructured_continue_count,
                      rq_to_loop_info.lowered_ray_query_loop_count,
                      simplify_cfg_info.folded_constant_cond_br_count,
                      simplify_cfg_info.threaded_empty_block_count,
                      simplify_cfg_info.removed_unreachable_block_count);

        if (!LUISA_XIR_DISABLE_RESTRUCTURE_CFG) {
            if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
                auto filename = luisa::format("kernel.{:016x}.before_reg2mem.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }
            pass_clk.tic();
            reg2mem_pre_info = xir::reg2mem_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  reg2mem-pre: {} ms (lowered {} phi(s), {} cross-block value(s))", pass_clk.toc(), reg2mem_pre_info.lowered_phi_count, reg2mem_pre_info.lowered_cross_block_value_count);

            if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
                auto filename = luisa::format("kernel.{:016x}.after_reg2mem.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }

            pass_clk.tic();
            restructure_cfg_info = xir::restructure_cfg_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  restructure-cfg: {} ms", pass_clk.toc());
            LUISA_VERBOSE("XIR CFG restructuring done: restructured {} loop(s), {} if(s); {} irreducible region(s) remained.",
                          restructure_cfg_info.restructured_loop_count,
                          restructure_cfg_info.restructured_if_count,
                          restructure_cfg_info.irreducible_region_count);

            if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
                auto filename = luisa::format("kernel.{:016x}.after_restructure.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }

            pass_clk.tic();
            xir::dce_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  post-restructure-dce: {} ms", pass_clk.toc());

            pass_clk.tic();
            auto reg2mem_mid_info = xir::reg2mem_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  reg2mem-mid: {} ms (lowered {} phi(s), {} cross-block value(s))", pass_clk.toc(), reg2mem_mid_info.lowered_phi_count, reg2mem_mid_info.lowered_cross_block_value_count);

            if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
                auto filename = luisa::format("kernel.{:016x}.after_reg2mem_mid.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }

            // Phase C
            pass_clk.tic();
            auto dceC1_info = xir::dce_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  C.dce1: {} ms", pass_clk.toc());
            pass_clk.tic();
            auto storeC_info = xir::local_store_forward_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  C.store-forward: {} ms (forwarded {})", pass_clk.toc(), storeC_info.removed_load_count);
            pass_clk.tic();
            auto loadC_info = xir::local_load_elimination_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  C.load-elim: {} ms (eliminated {})", pass_clk.toc(), loadC_info.removed_load_count);
            pass_clk.tic();
            auto cfC_info = xir::const_fold_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  C.const-fold: {} ms (folded {})", pass_clk.toc(), cfC_info.folded_inst_count);
            pass_clk.tic();
            auto dceC2_info = xir::dce_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  C.dce2: {} ms", pass_clk.toc());

            pass_clk.tic();
            auto reg2mem_post_info = xir::reg2mem_pass_run_on_module(xir_module.get());
            if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  reg2mem-post: {} ms (lowered {} phi(s), {} cross-block value(s))", pass_clk.toc(), reg2mem_post_info.lowered_phi_count, reg2mem_post_info.lowered_cross_block_value_count);

            if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
                auto filename = luisa::format("kernel.{:016x}.after_reg2mem_post.xir", kernel.hash());
                dump_xir_module(xir_module.get(), filename);
            }
        }
        }// !LUISA_XIR_DISABLE_RESTRUCTURE_CFG

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

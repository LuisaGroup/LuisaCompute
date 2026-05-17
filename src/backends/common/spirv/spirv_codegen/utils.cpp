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
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/loop_unroll.h>

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

}// namespace

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept -> luisa::unique_ptr<xir::Module> {
    Clock translate_clk;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { xir_module->set_location(option.name); }

    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }

    // XIR optimization pipeline:
    //   DCE → algebraic-simplify → const-fold → DCE → store-forward → load-elim → DCE →
    //   [inline → const-fold → DCE] → promote-ref-arg →
    //   sroa → loop-unroll → mem2reg → reg2mem → DCE
    //
    // Note: inline pass currently disabled due to edge cases with
    // block wiring and callee removal. Implementation is complete
    // in src/xir/passes/inline.cpp.

    Clock opt_clk;

    // Phase 1: early cleanup + algebraic simplify + const-fold
    Clock pass_clk;
    auto dce1_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  dce1: {} ms", pass_clk.toc());

    pass_clk.tic();
    auto alg_simplify_info = xir::algebraic_simplify_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  algebraic-simplify: {} ms (simplified {})", pass_clk.toc(), alg_simplify_info.simplified_inst_count);

    pass_clk.tic();
    auto const_fold1_info = xir::const_fold_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  const-fold1: {} ms (folded {})", pass_clk.toc(), const_fold1_info.folded_inst_count);

    pass_clk.tic();
    auto dce2_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  dce2: {} ms", pass_clk.toc());

    // Phase 2: memory optimization
    pass_clk.tic();
    auto store_forward_info = xir::local_store_forward_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  store-forward: {} ms (forwarded {})", pass_clk.toc(), store_forward_info.removed_load_count);

    pass_clk.tic();
    auto load_elim_info = xir::local_load_elimination_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  load-elim: {} ms (eliminated {})", pass_clk.toc(), load_elim_info.removed_load_count);

    pass_clk.tic();
    auto dce3_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  dce3: {} ms", pass_clk.toc());

    // Phase 3: callable inlining (disabled, slot reserved)
    // auto inline_info = xir::inline_pass_run_on_module(xir_module.get());
    // pass_clk.tic();
    // auto const_fold2_info = xir::const_fold_pass_run_on_module(xir_module.get());
    // auto dce_inline_info = xir::dce_pass_run_on_module(xir_module.get());
    xir::InlineInfo inline_info{};
    xir::ConstFoldInfo const_fold2_info{};
    xir::DCEInfo dce_inline_info{};

    // Phase 4: argument promotion + SROA + mem2reg
    pass_clk.tic();
    auto promote_arg_info = xir::promote_ref_arg_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  promote-ref-arg: {} ms (promoted {})", pass_clk.toc(), promote_arg_info.promoted_ref_arg_count);

    pass_clk.tic();
    auto sroa_info = xir::sroa_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  sroa: {} ms (decomposed {} into {})", pass_clk.toc(), sroa_info.decomposed_alloca_count, sroa_info.inserted_alloca_count);

    pass_clk.tic();
    auto loop_unroll_info = xir::loop_unroll_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  loop-unroll: {} ms (unrolled {})", pass_clk.toc(), loop_unroll_info.unrolled_loop_count);

    pass_clk.tic();
    auto mem2reg_info = xir::mem2reg_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  mem2reg: {} ms (promoted {} alloca(s), {} phi(s))", pass_clk.toc(), mem2reg_info.promoted_alloca_count, mem2reg_info.inserted_phi_count);

    pass_clk.tic();
    auto reg2mem_info = xir::reg2mem_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  reg2mem: {} ms", pass_clk.toc());

    pass_clk.tic();
    auto dce4_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_DUMP_OPT_STATS) LUISA_INFO("  dce4: {} ms", pass_clk.toc());

    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }

    LUISA_VERBOSE("XIR optimization done in {} ms:\n"
                  "    folded {} constant instruction(s),\n"
                  "    forwarded {} store instruction(s),\n"
                  "    eliminated {} load instruction(s),\n"
                  "    inlined {} call(s) and removed {} callable(s),\n"
                  "    sroa decomposed {} alloca(s) into {} scalar alloca(s),\n"
                  "    promoted {} alloca instruction(s) with {} load and {} store instruction(s) removed and {} phi node(s) inserted,\n"
                  "    removed {} + {} + {} + {} + {} = {} dead instruction(s) and {} + {} + {} + {} + {} = {} dead block(s),\n"
                  "    promoted {} reference argument(s).",
                  opt_clk.toc(),
                  const_fold1_info.folded_inst_count + const_fold2_info.folded_inst_count,
                  store_forward_info.removed_load_count,
                  load_elim_info.removed_load_count,
                  inline_info.inlined_call_count, inline_info.removed_callable_count,
                  sroa_info.decomposed_alloca_count, sroa_info.inserted_alloca_count,
                  mem2reg_info.promoted_alloca_count, mem2reg_info.removed_load_count, mem2reg_info.removed_store_count, mem2reg_info.inserted_phi_count,
                  dce1_info.removed_inst_count, dce2_info.removed_inst_count, dce3_info.removed_inst_count, dce_inline_info.removed_inst_count, dce4_info.removed_inst_count,
                  dce1_info.removed_inst_count + dce2_info.removed_inst_count + dce3_info.removed_inst_count + dce_inline_info.removed_inst_count + dce4_info.removed_inst_count,
                  dce1_info.removed_block_count, dce2_info.removed_block_count, dce3_info.removed_block_count, dce_inline_info.removed_block_count, dce4_info.removed_block_count,
                  dce1_info.removed_block_count + dce2_info.removed_block_count + dce3_info.removed_block_count + dce_inline_info.removed_block_count + dce4_info.removed_block_count,
                  promote_arg_info.promoted_ref_arg_count);

    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.rq.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }
    return xir_module;
}

}// namespace luisa::compute::spirv

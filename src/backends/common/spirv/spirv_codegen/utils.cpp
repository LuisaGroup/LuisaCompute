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

namespace luisa::compute::spirv {

namespace {

const bool LUISA_SPIRV_SHOULD_DUMP_XIR = [] {
    if (auto env = getenv("LUISA_DUMP_XIR")) {
        return luisa::string_view{env} == "1";
    }
    return false;
}();

}

[[nodiscard]] auto luisa_spirv_backend_translate_ast_to_xir(Function kernel, const ShaderOption &option) noexcept -> luisa::unique_ptr<xir::Module> {
    Clock translate_clk;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { xir_module->set_location(option.name); }
    LUISA_VERBOSE("AST to XIR translation done in {} ms.", translate_clk.toc());

    // dump for debugging
    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }

    // run some simple optimization passes on XIR
    Clock opt_clk;
    auto dce1_info = xir::dce_pass_run_on_module(xir_module.get());
    auto store_forward_info = xir::local_store_forward_pass_run_on_module(xir_module.get());
    auto load_elim_info = xir::local_load_elimination_pass_run_on_module(xir_module.get());
    auto dce2_info = xir::dce_pass_run_on_module(xir_module.get());
    auto promote_arg_info = xir::promote_ref_arg_pass_run_on_module(xir_module.get());
    auto mem2reg_info = xir::mem2reg_pass_run_on_module(xir_module.get());
    auto dce3_info = xir::dce_pass_run_on_module(xir_module.get());
    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }
    LUISA_VERBOSE("XIR optimization done in {} ms:\n"
                  "    forwarded {} store instruction(s),\n"
                  "    eliminated {} load instruction(s),\n"
                  "    promoted {} alloca instruction(s) with {} load and {} store instruction(s) removed and {} phi node(s) inserted,\n"
                  "    removed {} + {} + {} = {} dead instruction(s) and {} + {} + {} = {} dead block(s),\n"
                  "    promoted {} reference argument(s).",
                  opt_clk.toc(),
                  store_forward_info.removed_load_count,
                  load_elim_info.removed_load_count,
                  mem2reg_info.promoted_alloca_count, mem2reg_info.removed_load_count, mem2reg_info.removed_store_count, mem2reg_info.inserted_phi_count,
                  dce1_info.removed_inst_count, dce2_info.removed_inst_count, dce3_info.removed_inst_count,
                  dce1_info.removed_inst_count + dce2_info.removed_inst_count + dce3_info.removed_inst_count,
                  dce1_info.removed_block_count, dce2_info.removed_block_count, dce3_info.removed_block_count,
                  dce1_info.removed_block_count + dce2_info.removed_block_count + dce3_info.removed_block_count,
                  promote_arg_info.promoted_ref_arg_count);

    // dump for debugging
    if (LUISA_SPIRV_SHOULD_DUMP_XIR) {
        auto filename = luisa::format("kernel.{:016x}.opt.rq.xir", kernel.hash());
        std::ofstream f{filename.c_str()};
        f << xir::xir_to_text_translate(xir_module.get(), true);
    }
    return xir_module;
}

}// namespace luisa::compute::spirv

#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

#include "helpers.h"

namespace luisa::compute::xir {

LoopFusionInfo loop_fusion_pass_run_on_function(Function *function) noexcept {
    LoopFusionInfo info;
    if (function == nullptr || function->definition() == nullptr) { return info; }
    if (contains_structured_control_flow(function->definition())) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop fusion rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return info;
    }
    // TODO: Discover adjacent canonical natural loops in plain CFG and verify
    // memory/SSA dependences before enabling fusion.
    return info;
}

LoopFusionInfo loop_fusion_pass_run_on_module(Module *module,
                                              PassReport *report) noexcept {
    LoopFusionInfo info;
    for (auto *function : module->function_list()) {
        auto function_info = loop_fusion_pass_run_on_function(function);
        info.fused_loop_count += function_info.fused_loop_count;
        info.structured_cfg_error_count += function_info.structured_cfg_error_count;
    }
    if (report != nullptr) {
        report->set("fused_loop_count", info.fused_loop_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir

#include <luisa/xir/passes/loop_rotation.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void run(FunctionDefinition *def, LoopRotationInfo &info) noexcept {
    if (def == nullptr) { return; }
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop rotation rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    // TODO: Rotate only verifier-backed natural loops in plain CFG.
}

}// namespace detail

LoopRotationInfo loop_rotation_pass_run_on_function(FunctionDefinition *def) noexcept {
    LoopRotationInfo info;
    detail::run(def, info);
    return info;
}

LoopRotationInfo loop_rotation_pass_run_on_module(Module *module,
                                                  PassReport *report) noexcept {
    LoopRotationInfo info;
    for (auto *function : module->function_list()) {
        detail::run(function->definition(), info);
    }
    if (report != nullptr) {
        report->set("rotated_loop_count", info.rotated_loop_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir

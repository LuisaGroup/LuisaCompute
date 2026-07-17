#include <luisa/xir/passes/loop_vectorization.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>

#include "helpers.h"

namespace luisa::compute::xir {

LoopVectorizationInfo loop_vectorization_pass_run_on_function(Function *function) noexcept {
    LoopVectorizationInfo info;
    if (function == nullptr || function->definition() == nullptr) { return info; }
    if (contains_structured_control_flow(function->definition())) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop vectorization rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return info;
    }
    // TODO: Implement vectorization on verifier-backed natural loops in plain
    // CFG, including reductions, remainder handling, and memory dependence.
    return info;
}

LoopVectorizationInfo loop_vectorization_pass_run_on_module(Module *module,
                                                            PassReport *report) noexcept {
    LoopVectorizationInfo info;
    for (auto *function : module->function_list()) {
        auto function_info = loop_vectorization_pass_run_on_function(function);
        info.vectorized_loop_count += function_info.vectorized_loop_count;
        info.created_vector_inst_count += function_info.created_vector_inst_count;
        info.structured_cfg_error_count += function_info.structured_cfg_error_count;
    }
    if (report != nullptr) {
        report->set("vectorized_loop_count", info.vectorized_loop_count);
        report->set("created_vector_inst_count", info.created_vector_inst_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir

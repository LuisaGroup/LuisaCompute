#include <luisa/xir/passes/loop_unroll.h>
#include <luisa/core/logging.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

static void run(Function *function, LoopUnrollInfo &info,
                const LoopUnrollOptions &) noexcept {
    if (function == nullptr || function->definition() == nullptr) { return; }
    auto *def = function->definition();
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop unroll rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    // TODO: Implement verifier-backed natural-loop discovery and cloning for
    // plain CFG. Until then the unstructured input is accepted but unchanged.
}

}// namespace detail

LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function,
                                                LoopUnrollOptions options) noexcept {
    LoopUnrollInfo info;
    detail::run(function, info, options);
    return info;
}

LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module,
                                              LoopUnrollOptions options) noexcept {
    LoopUnrollInfo info;
    for (auto *function : module->function_list()) {
        detail::run(function, info, options);
    }
    return info;
}

}// namespace luisa::compute::xir

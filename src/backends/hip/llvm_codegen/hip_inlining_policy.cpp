#include "hip_inlining_policy.h"

#include <llvm/Config/llvm-config.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/ModuleInliner.h>

namespace luisa::compute::hip {

void configure_hip_cgscc_canonicalization_inlining(
    llvm::PipelineTuningOptions &options) noexcept {
    // LLVM accepts a modeled cost of zero because the final comparison is
    // cost < max(1, threshold). Thus threshold zero denotes the complete
    // non-positive-cost fixed point, rather than disabling inlining.
    options.InlinerThreshold = 0;
}

llvm::InlineParams hip_module_priority_inline_params(
    llvm::OptimizationLevel level) noexcept {
#if LLVM_VERSION_MAJOR >= 23
    auto params = llvm::getInlineParamsFromOptLevel(
        level.getSpeedupLevel());
#else
    auto params = llvm::getInlineParams(
        level.getSpeedupLevel(), level.getSizeLevel());
#endif
    // Deferral compensates for bottom-up SCC visitation. The module inliner
    // uses a global priority queue, so deferral has no semantic role and would
    // only make the selected edge set depend on a policy for the wrong order.
    params.EnableDeferral = false;
    return params;
}

void add_hip_module_priority_inliner(
    llvm::ModulePassManager &pipeline,
    llvm::OptimizationLevel level) noexcept {
    pipeline.addPass(llvm::ModuleInlinerPass{
        hip_module_priority_inline_params(level),
        llvm::InliningAdvisorMode::Default,
        llvm::ThinOrFullLTOPhase::None});
}

}// namespace luisa::compute::hip

#include "loop_unswitch.h"

#include <luisa/xir/instructions/branch.h>

#include "warp_uniformity.h"

namespace luisa::compute::simd::schedule {

namespace {

[[nodiscard]] bool is_varying_candidate(
    const xir::ConditionalBranchInst *branch,
    const void *context) noexcept {
    auto *uniformity = static_cast<
        const WarpUniformityAnalysis *>(context);
    return branch != nullptr && uniformity != nullptr &&
           uniformity->classify(branch->condition()) ==
               ValueClass::varying;
}

}// namespace

SIMDLoopUnswitchInfo
unswitch_invariant_varying_loop_condition(
    xir::Function *function,
    bool enable_guarded_dynamic_trip) noexcept {
    WarpUniformityAnalysis uniformity;
    uniformity.analyze(function);
    auto unswitch = xir::loop_unswitch_pass_run_on_function(
        function,
        {.max_loop_instruction_count = 48u,
         .max_unswitched_loop_count = 1u,
         .enable_guarded_dynamic_trip = enable_guarded_dynamic_trip,
         .candidate_filter = is_varying_candidate,
         .candidate_filter_context = &uniformity});
    auto cleanup = unswitch.changed() ?
                       xir::simplify_cfg_pass_run_on_function(function) :
                       xir::SimplifyCFGInfo{};
    return {
        .unswitch = unswitch,
        .cleanup = cleanup,
    };
}

}// namespace luisa::compute::simd::schedule

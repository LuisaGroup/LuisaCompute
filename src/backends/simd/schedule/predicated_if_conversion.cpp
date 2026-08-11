#include "predicated_if_conversion.h"

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

PredicatedIfConversionInfo predicate_small_varying_diamonds(
    xir::Function *function) noexcept {
    WarpUniformityAnalysis uniformity;
    uniformity.analyze(function);
    auto if_conversion = xir::if_conversion_pass_run_on_function(
        function,
        {.max_arm_instruction_count = 4u,
         .max_total_instruction_count = 6u,
         .max_live_out_register_units = 4u,
         .max_speculation_cost = 12u,
         .candidate_filter = is_varying_candidate,
         .candidate_filter_context = &uniformity});
    auto select_factoring = if_conversion.changed() ?
                                xir::select_factor_pass_run_on_function(
                                    function) :
                                xir::SelectFactorInfo{};
    return {
        .if_conversion = if_conversion,
        .select_factoring = select_factoring,
    };
}

}// namespace luisa::compute::simd::schedule

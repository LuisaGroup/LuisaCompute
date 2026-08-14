#pragma once

#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/select_factor.h>

namespace luisa::compute::xir {
class Function;
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

struct PredicatedIfConversionInfo {
    xir::IfConversionInfo if_conversion{};
    xir::SelectFactorInfo select_factoring{};
    size_t refinement_round_count{0u};
    size_t forwarded_phi_count{0u};
    size_t removed_forwarding_block_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return if_conversion.changed() ||
               removed_forwarding_block_count != 0u ||
               select_factoring.changed();
    }
};

// If-converts only small, inexpensive diamonds whose condition is genuinely
// varying. Optional refinement collapses transparent select/Phi forwarding
// blocks so a bounded enclosing diamond can be reconsidered, then matching
// arithmetic is factored back through the generated selects. Warp- and
// cohort-uniform conditions retain scalar control flow. The caller supplies
// the target policy's weighted speculation ceiling; safety and structural
// limits remain internal to the pass.
[[nodiscard]] PredicatedIfConversionInfo predicate_small_varying_diamonds(
    xir::Function *function,
    bool enable_refinement = true,
    size_t max_speculation_cost = 12u) noexcept;

}// namespace luisa::compute::simd::schedule

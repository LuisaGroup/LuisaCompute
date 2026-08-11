#pragma once

#include <luisa/xir/passes/loop_unswitch.h>
#include <luisa/xir/passes/simplify_cfg.h>

namespace luisa::compute::xir {
class Function;
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

struct SIMDLoopUnswitchInfo {
    xir::LoopUnswitchInfo unswitch{};
    xir::SimplifyCFGInfo cleanup{};

    [[nodiscard]] bool changed() const noexcept {
        return unswitch.changed() || cleanup.changed();
    }
};

// Hoists one lane-varying but loop-invariant condition out of a positive
// constant-trip, read-only natural loop. This splits the packet once before
// the two loop versions instead of diverging and reconverging on every
// iteration.
[[nodiscard]] SIMDLoopUnswitchInfo
unswitch_invariant_varying_loop_condition(
    xir::Function *function) noexcept;

}// namespace luisa::compute::simd::schedule

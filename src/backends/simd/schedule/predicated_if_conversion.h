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

    [[nodiscard]] bool changed() const noexcept {
        return if_conversion.changed() || select_factoring.changed();
    }
};

// If-converts only small, inexpensive diamonds whose condition is genuinely
// varying, then factors matching arithmetic back through the generated
// selects. Warp- and cohort-uniform conditions retain scalar control flow.
[[nodiscard]] PredicatedIfConversionInfo predicate_small_varying_diamonds(
    xir::Function *function) noexcept;

}// namespace luisa::compute::simd::schedule

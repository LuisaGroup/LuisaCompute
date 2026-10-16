#pragma once

#include <cstddef>

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct SelectFactorInfo {
    size_t factored_select_count{0u};
    size_t removed_arithmetic_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return factored_select_count != 0u;
    }
};

// Factors one-use matching arithmetic producers through a select:
//
//   select(f(a), f(b), c) -> f(select(a, b, c))
//
// The implementation also permits n-ary f when exactly one corresponding
// operand differs. Operations must be explicitly classified as total, pure
// arithmetic and the rewrite must be component-wise for a vector condition.
// Annotated instructions are retained so instruction-local metadata is never
// discarded.
[[nodiscard]] LUISA_XIR_API SelectFactorInfo
select_factor_pass_run_on_function(Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API SelectFactorInfo
select_factor_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

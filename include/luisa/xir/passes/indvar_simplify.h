#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct IndVarSimplifyInfo {
    size_t simplified_iv_count{0u};
    size_t removed_dead_iv_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return simplified_iv_count != 0u ||
               removed_dead_iv_count != 0u;
    }
};

// Strength reduction is plain-CFG-only and retains annotated candidates when
// their uses cannot be replaced by one unique metadata owner. Null/bodyless
// functions are unchanged no-ops.
[[nodiscard]] LUISA_XIR_API IndVarSimplifyInfo indvar_simplify_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API IndVarSimplifyInfo indvar_simplify_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

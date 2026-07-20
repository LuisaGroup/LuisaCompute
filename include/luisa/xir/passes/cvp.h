#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

// Correlated Value Propagation pass.
// Leverages structured control flow (IfInst) to propagate values
// that are known to be equal to a constant along conditional branches.
//
// For each IfInst where the condition is BINARY_EQUAL(var, constant):
//   - In the true block: var == constant, so replace uses of var with constant.
// For each IfInst where the condition is BINARY_NOT_EQUAL(var, constant):
//   - In the false block: var == constant, so replace uses of var with constant.
//
// Uses the dominator tree to restrict replacements to blocks dominated
// by the target block.

struct CVPInfo {
    size_t replaced_inst_count{0u};
};

[[nodiscard]] LUISA_XIR_API CVPInfo cvp_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API CVPInfo cvp_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

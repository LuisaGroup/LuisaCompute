#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct DeadArgElimInfo {
    size_t removed_arg_count{0u};
};

// Removes only unannotated unused arguments from unconstrained internal
// definitions and updates every validated call site transactionally.
// Argument-local metadata is an ABI constraint because no unique replacement
// owner exists after removal.
[[nodiscard]] LUISA_XIR_API DeadArgElimInfo dead_arg_elim_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API DeadArgElimInfo dead_arg_elim_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

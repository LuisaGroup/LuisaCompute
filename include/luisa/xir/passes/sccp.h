#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct SCCPInfo {
    size_t folded_inst_count{0u};
    size_t removed_branch_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return folded_inst_count != 0u ||
               removed_branch_count != 0u;
    }
};

// Annotated value-producing instructions are not replaced with module-uniqued
// constants. Terminator metadata is transferred to replacement branches.
// Null inputs are no-ops.

[[nodiscard]] LUISA_XIR_API SCCPInfo sccp_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API SCCPInfo sccp_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

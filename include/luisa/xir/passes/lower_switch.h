#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Module;
class Function;
class PassReport;

struct LowerSwitchInfo {
    size_t lowered_switch_count{0u};
    size_t rejected_switch_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return rejected_switch_count == 0u; }
};

// Lowers SwitchInst to a structured IfInst cascade. A structured zero-case
// switch is deliberately left unchanged: replacing it with BranchInst would
// erase its lexical merge frame. Such a rejection is reported through
// rejected_switch_count/succeeded(), and the complete function/module request
// is left unchanged. An unstructured zero-case switch (null merge marker) may
// be lowered to BranchInst.
[[nodiscard]] LUISA_XIR_API LowerSwitchInfo lower_switch_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LowerSwitchInfo lower_switch_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

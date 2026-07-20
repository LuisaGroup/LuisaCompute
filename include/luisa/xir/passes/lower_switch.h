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

[[nodiscard]] LUISA_XIR_API LowerSwitchInfo lower_switch_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LowerSwitchInfo lower_switch_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Module;
class Function;

struct LowerBreakContinueInfo {
    size_t lowered_break_count{0u};
    size_t lowered_continue_count{0u};
    size_t rejected_break_count{0u};
    size_t rejected_continue_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return lowered_break_count != 0u ||
               lowered_continue_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return rejected_break_count == 0u && rejected_continue_count == 0u;
    }
};

[[nodiscard]] LUISA_XIR_API LowerBreakContinueInfo lower_break_continue_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LowerBreakContinueInfo lower_break_continue_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

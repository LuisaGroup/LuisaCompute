#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct DeadArgElimInfo {
    size_t removed_arg_count{0u};
};

[[nodiscard]] LUISA_XIR_API DeadArgElimInfo dead_arg_elim_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API DeadArgElimInfo dead_arg_elim_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

struct CSEInfo {
    size_t eliminated_inst_count{0u};
};

[[nodiscard]] LUISA_XIR_API CSEInfo cse_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API CSEInfo cse_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

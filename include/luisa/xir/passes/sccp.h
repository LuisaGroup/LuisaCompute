#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

struct SCCPInfo {
    size_t folded_inst_count{0u};
    size_t removed_branch_count{0u};
};

[[nodiscard]] LUISA_XIR_API SCCPInfo sccp_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API SCCPInfo sccp_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

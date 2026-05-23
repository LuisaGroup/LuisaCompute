#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class Function;

struct GVNInfo {
    size_t replaced_inst_count = 0u;
    size_t removed_inst_count = 0u;
};

[[nodiscard]] LUISA_XIR_API GVNInfo gvn_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API GVNInfo gvn_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

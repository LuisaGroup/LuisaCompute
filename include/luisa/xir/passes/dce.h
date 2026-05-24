#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct DCEInfo {
    size_t removed_inst_count{0u};
    size_t removed_block_count{0u};
};

[[nodiscard]] LUISA_XIR_API DCEInfo dce_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API DCEInfo dce_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

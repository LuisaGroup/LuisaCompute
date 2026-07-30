#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/outline.h>

namespace luisa::compute::xir {

// Region outlining is not implemented yet. The pass detects OutlineInst,
// leaves structured control flow unchanged, and reports an explicit failure
// instead of silently claiming success.

struct OutlineInfo {
    size_t outlined_func_count{0u};
    size_t unsupported_outline_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return unsupported_outline_count == 0u; }
};

LUISA_XIR_API OutlineInfo outline_pass_run_on_function(Module *module, Function *function) noexcept;
/// Total entry point: a null module is an unchanged successful no-op.
LUISA_XIR_API OutlineInfo outline_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

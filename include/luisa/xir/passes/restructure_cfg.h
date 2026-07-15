#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct RestructureCFGInfo {
    size_t restructured_loop_count{0u};
    size_t restructured_if_count{0u};
    size_t irreducible_region_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return irreducible_region_count == 0u; }
};

// Converts reducible plain CFG regions into structured control flow. A function
// containing an irreducible (multi-entry) cyclic SCC is rejected before any IR
// mutation and reported through irreducible_region_count.
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

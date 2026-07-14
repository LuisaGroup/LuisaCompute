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
};

[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API RestructureCFGInfo restructure_cfg_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

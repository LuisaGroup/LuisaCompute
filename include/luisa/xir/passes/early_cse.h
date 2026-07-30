#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Module;
class Function;
class PassReport;

struct EarlyCSEInfo {
    size_t eliminated_inst_count{0u};
};

/// Eliminates same-block duplicates of proven-pure instructions. Annotated
/// duplicates are retained as distinct metadata owners; null inputs are no-ops.
[[nodiscard]] LUISA_XIR_API EarlyCSEInfo early_cse_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API EarlyCSEInfo early_cse_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

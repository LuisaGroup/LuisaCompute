#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class ReturnInst;
class Module;
class Function;
class PassReport;

struct EarlyReturnEliminationInfo {
    size_t removed_return_count{0u};
};

[[nodiscard]] LUISA_XIR_API EarlyReturnEliminationInfo early_return_elimination_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API EarlyReturnEliminationInfo early_return_elimination_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

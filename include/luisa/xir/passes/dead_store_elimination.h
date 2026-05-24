#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class PassReport;

class Function;
class Module;

struct DeadStoreEliminationInfo {
    size_t eliminated_store_count{0u};
};

[[nodiscard]] LUISA_XIR_API DeadStoreEliminationInfo dead_store_elimination_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API DeadStoreEliminationInfo dead_store_elimination_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class PassReport;

class Function;
class Module;

struct DeadStoreEliminationInfo {
    size_t eliminated_store_count{0u};
};

/// Eliminates only overwritten stores to proven thread-local allocation
/// locations. Stores carrying instruction-local metadata are retained when
/// there is no unique replacement metadata owner. Null inputs are no-ops.
[[nodiscard]] LUISA_XIR_API DeadStoreEliminationInfo dead_store_elimination_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API DeadStoreEliminationInfo dead_store_elimination_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

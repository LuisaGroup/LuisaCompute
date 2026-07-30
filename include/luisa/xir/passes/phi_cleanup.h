#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;
class Function;

struct PhiCleanupInfo {
    size_t removed_phi_count{0u};
};

/// Removes only live trivial Phis whose instruction-local metadata is empty;
/// dead Phis may always be erased. Null inputs are no-ops.
[[nodiscard]] LUISA_XIR_API PhiCleanupInfo phi_cleanup_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API PhiCleanupInfo phi_cleanup_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

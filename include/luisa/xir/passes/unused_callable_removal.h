#pragma once

#include <luisa/xir/module.h>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class CallableFunction;

struct UnusedCallableRemovalInfo {
    size_t removed_callable_count{0u};
};

// Removes defined, unconstrained callables unreachable from kernels. Bodyless
// declaration-like and signature-constrained callables are retained because
// their external visibility cannot be disproved. References in disconnected
// owned blocks are preserved because this pass does not delete those blocks.
// Unused recursive SCCs are also retained until an SCC-aware removal is
// implemented.
[[nodiscard]] LUISA_XIR_API UnusedCallableRemovalInfo unused_callable_removal_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

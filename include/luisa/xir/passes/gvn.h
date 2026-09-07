#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

class Function;

struct GVNInfo {
    size_t replaced_inst_count = 0u;
    size_t removed_inst_count = 0u;
    // A dominated equivalent can still be a bad coroutine replacement: using
    // a pre-suspend leader in a continuation turns rematerialized computation
    // into frame state. Such candidates are intentionally left distinct.
    size_t rejected_cross_suspend_count = 0u;
    [[nodiscard]] bool changed() const noexcept {
        return replaced_inst_count != 0u ||
               removed_inst_count != 0u;
    }
};

/// Performs dominance-aware value numbering of proven-pure instructions.
/// Raw coroutine definitions are analyzed over their augmented
/// suspend-to-resume CFG, but a replacement is confined to a region that does
/// not cross a suspension. This prevents GVN from increasing frame liveness.
/// Annotated duplicates remain as distinct metadata owners; null inputs are
/// no-ops.
[[nodiscard]] LUISA_XIR_API GVNInfo gvn_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API GVNInfo gvn_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

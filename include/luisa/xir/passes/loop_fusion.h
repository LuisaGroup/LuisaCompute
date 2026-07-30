#pragma once

#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PassReport;

struct LoopFusionInfo {
    size_t fused_loop_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool changed() const noexcept { return fused_loop_count != 0u; }
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

// Unstructured-CFG-only: structured functions are rejected without mutation.
// Adjacent plain-CFG loops with matching constant trip counts, no buffer
// write/read dependence, and no cross-loop SSA value flow are fused into a
// single loop whose body is the concatenation of both bodies. Fusion is
// rejected before mutation when metadata on a deleted block/header value has
// no unique replacement owner.
[[nodiscard]] LUISA_XIR_API LoopFusionInfo loop_fusion_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LoopFusionInfo loop_fusion_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

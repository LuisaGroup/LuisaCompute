#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

struct InlineInfo {
    size_t inlined_call_count{0u};
    size_t removed_callable_count{0u};
    size_t skipped_recursive_callable_count{0u};
    size_t skipped_structured_call_count{0u};
};

// Single-block callees can be inlined into structured callers without changing
// their CFG. Multi-block inlining is unstructured-CFG-only; calls involving a
// structured caller or callee are reported and left unchanged.
[[nodiscard]] LUISA_XIR_API InlineInfo inline_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API InlineInfo inline_all_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

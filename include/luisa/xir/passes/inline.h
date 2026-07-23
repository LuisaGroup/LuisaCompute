#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;
class CallInst;

struct InlineInfo {
    size_t inlined_call_count{0u};
    size_t removed_callable_count{0u};
    size_t skipped_recursive_callable_count{0u};
    size_t skipped_structured_call_count{0u};
    size_t rejected_malformed_call_count{0u};
};

struct InlineOptions {
    bool allow_autodiff_scope_in_caller{false};
};

// Single-block callees can be inlined into structured callers without changing
// their CFG. By default, multi-block inlining is unstructured-CFG-only. The
// opt-in option permits only a retained caller-side autodiff scope after the
// caller and callee's ordinary structured CFG has already been destructured.
[[nodiscard]] LUISA_XIR_API InlineInfo inline_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API InlineInfo inline_all_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API InlineInfo inline_all_pass_run_on_module(Module *module, InlineOptions options, PassReport *report = nullptr) noexcept;

[[nodiscard]] LUISA_XIR_API InlineInfo
inline_call_sites_pass_run_on_module(
    Module *module, luisa::span<CallInst *const> call_sites,
    InlineOptions options = {}, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

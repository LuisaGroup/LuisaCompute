#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// This pass inlines callable functions into their callers.
// Callables with a single call site are always inlined.
// Callables with multiple call sites and small bodies are also inlined.

struct InlineInfo {
    size_t inlined_call_count{0u};
    size_t removed_callable_count{0u};
};

[[nodiscard]] LUISA_XIR_API InlineInfo inline_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

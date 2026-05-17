#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// Loop unrolling pass.
// Unrolls loops with constant iteration counts ≤ 16.

struct LoopUnrollInfo {
    size_t unrolled_loop_count{0u};
};

[[nodiscard]] LUISA_XIR_API LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

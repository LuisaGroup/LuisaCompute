#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// Unstructured-CFG-only loop unrolling pass. Callers must run an explicit CFG
// destructuring pass first. Structured control flow is rejected and left
// unchanged; see LoopUnrollInfo::structured_cfg_error_count. Plain-CFG loops
// discovered by natural-loop analysis are fully unrolled when their constant
// trip count fits max_trip_count; larger loops can optionally have their
// first iterations peeled (partial unrolling).

struct LoopUnrollOptions {
    size_t max_trip_count{256};       // Maximum trip count to fully unroll (raised from 16 for compute loops)
    bool unroll_pure_only{false};     // If true, skip loops with buffer writes
    // Partial unrolling (peeling): when non-zero and a loop's constant trip
    // count exceeds max_trip_count, the first partial_unroll_factor iterations
    // are emitted as straight-line code and the remaining loop (with trip
    // count reduced by the peeled amount) is kept. Zero disables peeling.
    size_t partial_unroll_factor{0u};
};

struct LoopUnrollInfo {
    size_t unrolled_loop_count{0u};
    size_t partially_unrolled_loop_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

[[nodiscard]] LUISA_XIR_API LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function, LoopUnrollOptions options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module, LoopUnrollOptions options = {}) noexcept;

}// namespace luisa::compute::xir

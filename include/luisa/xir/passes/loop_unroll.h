#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

// Unstructured-CFG-only loop unrolling pass. Callers must run an explicit CFG
// destructuring pass first. Structured control flow is rejected and left
// unchanged; see LoopUnrollInfo::structured_cfg_error_count. Plain CFG is
// currently accepted unchanged pending verifier-backed natural-loop support.

struct LoopUnrollOptions {
    size_t max_trip_count{256};       // Maximum trip count to unroll (raised from 16 for compute loops)
    bool unroll_pure_only{false};     // If true, skip loops with buffer writes
};

struct LoopUnrollInfo {
    size_t unrolled_loop_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

[[nodiscard]] LUISA_XIR_API LoopUnrollInfo loop_unroll_pass_run_on_function(Function *function, LoopUnrollOptions options = {}) noexcept;
[[nodiscard]] LUISA_XIR_API LoopUnrollInfo loop_unroll_pass_run_on_module(Module *module, LoopUnrollOptions options = {}) noexcept;

}// namespace luisa::compute::xir

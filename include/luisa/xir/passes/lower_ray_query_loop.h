#pragma once

#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>

namespace luisa::compute::xir {

// Compatibility surface for code written before the pass was named after its
// output representation. New code should include
// lower_ray_query_to_pipeline.h and use the canonical names below. Keep the
// legacy result layout so callers built against the old API can still observe
// the handler-local scratch optimization added on the next branch.
struct RayQueryLoopLowerInfo {
    size_t lowered_loop_count{0u};
    size_t localized_alloca_count{0u};
    size_t error_count{0u};
    [[nodiscard]] bool succeeded() const noexcept {
        return error_count == 0u;
    }
};

[[nodiscard, deprecated(
                 "Use lower_ray_query_to_pipeline_pass_run_on_function instead.")]]
LUISA_XIR_API RayQueryLoopLowerInfo
lower_ray_query_loop_pass_run_on_function(Function *function) noexcept;

[[nodiscard, deprecated(
                 "Use lower_ray_query_to_pipeline_pass_run_on_module instead.")]]
LUISA_XIR_API RayQueryLoopLowerInfo
lower_ray_query_loop_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>

namespace luisa::compute::xir {

// Compatibility surface for code written before the pass was named after its
// output representation. New code should include
// lower_ray_query_to_pipeline.h and use the canonical names below.
using RayQueryLoopLowerInfo [[deprecated(
    "Use LowerRayQueryToPipelineInfo instead.")]] =
    LowerRayQueryToPipelineInfo;

[[nodiscard, deprecated(
                 "Use lower_ray_query_to_pipeline_pass_run_on_function instead.")]]
LUISA_XIR_API LowerRayQueryToPipelineInfo
lower_ray_query_loop_pass_run_on_function(Function *function) noexcept;

[[nodiscard, deprecated(
                 "Use lower_ray_query_to_pipeline_pass_run_on_module instead.")]]
LUISA_XIR_API LowerRayQueryToPipelineInfo
lower_ray_query_loop_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

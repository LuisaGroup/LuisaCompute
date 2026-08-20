#pragma once

#include <cstddef>
#include <limits>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Module;
class Function;
class PassReport;

class RayQueryLoopInst;
class RayQueryPipelineInst;

// This pass lowers ray query loops into ray query pipelines.
//
// A ray query loop is a control flow structure:
// RayQueryLoop {
//   /* dispatch_block */
//   RayQueryDispatch(object)
//     -> merge_block
//     -> on_surface_candidate_block {
//       /* on surface candidate block */
//       br dispatch_block
//     }
//     -> on_procedural_candidate_block {
//       /* on procedural candidate block */
//       br dispatch_block
//     }
// }
// /* merge_block */
// { ... }
//
// A ray query pipeline is a high-level instruction that takes a
// query object, an on-surface function, an on-procedural function,
// and a list of captured arguments (the context):
// RayQueryPipeline(query_object, on_surface_func, on_procedural_func, captured_args...),
// where the signature of on_*_func is (query_object, captured_args...) -> void.
//
// This pass lowers ray query loops into ray query pipelines in three steps:
// 1. Compute the context of the ray query loop, i.e., the captured arguments.
// 2. Outline the on-surface and on-procedural candidate blocks into functions.
// 3. Create a ray query pipeline instruction to replace the ray query loop.
//
// Note: to minimize the number of captured arguments, this pass should be run
// after other optimization passes. A DCE pass is also recommended after this pass.

struct LowerRayQueryToPipelineInfo {
    size_t lowered_loop_count{0u};
    size_t error_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return error_count == 0u; }
};

struct LowerRayQueryToPipelineOptions {
    // The default preserves the established fallback/CUDA/HIP behavior. A
    // backend may use zero to select only callbacks whose sole argument is
    // the ray-query object itself, leaving captured loops in structured form
    // for a different lowering.
    size_t max_captured_argument_count{
        std::numeric_limits<size_t>::max()};
    // Optional output for callers using the selective overload. The pass
    // resets this value to zero and reports structurally valid loops retained
    // because their callback capture ABI exceeds the bound above. Keeping the
    // counter here preserves the established LowerRayQueryToPipelineInfo ABI.
    size_t *skipped_loop_count{nullptr};
};

// Every loop in a function is preflighted before outlining. Unsupported handler
// shapes (for example, overlapping handlers, cross-handler PHIs, or nested
// ray-query loops inside a handler) are reported through error_count/succeeded();
// if any loop is rejected, the complete function is left unchanged.
// Keep the established overloads as exported ABI entry points. The explicit
// option overloads let selective consumers add a stricter capture policy
// without changing existing source or binary callers.
[[nodiscard]] LUISA_XIR_API LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_function(
    Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_function(
    Function *function, LowerRayQueryToPipelineOptions options) noexcept;
[[nodiscard]] LUISA_XIR_API LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_module(
    Module *module, PassReport *report,
    LowerRayQueryToPipelineOptions options) noexcept;

}// namespace luisa::compute::xir

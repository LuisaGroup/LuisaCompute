#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

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

struct RayQueryLoopLowerInfo {
    size_t lowered_loop_count{0u};
};

[[nodiscard]] LUISA_XIR_API RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir


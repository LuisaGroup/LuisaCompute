#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class PassReport;

class Module;
class Function;

struct LowerRayQueryToLoopInfo {
    size_t lowered_ray_query_loop_count{0u};
    size_t error_count{0u};
    [[nodiscard]] bool succeeded() const noexcept { return error_count == 0u; }
};

// This is a structured-to-structured lowering. Inputs whose dispatch/handler
// entry contains PHI nodes, whose surface/procedural handler regions overlap,
// or whose handler exits/merge markers cannot be retargeted without changing
// structured semantics are rejected before any mutation. Every loop in a
// function is preflighted first; one rejection leaves the complete function
// unchanged and is reported through error_count/succeeded().
[[nodiscard]] LUISA_XIR_API LowerRayQueryToLoopInfo lower_ray_query_to_loop_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API LowerRayQueryToLoopInfo lower_ray_query_to_loop_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Function;
class Module;
class PassReport;

struct ReconstructRayQueryLoopInfo {
    size_t reconstructed_ray_query_loop_count{0u};
    size_t ignored_loop_count{0u};
    size_t error_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return reconstructed_ray_query_loop_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return error_count == 0u;
    }
};

// Reconstructs RayQueryLoopInst from either the canonical LoopInst proceed shell
// produced by lower_ray_query_to_loop or the canonical pre-mem2reg SimpleLoopInst
// emitted for the native DSL `$while (query.proceed())` form. The frontend form
// must immediately dispatch the same query with a surface/procedural if/else;
// break/continue, shared post-dispatch payload, nested PROCEED, escaped guard
// temporaries, and SSA loop-carried PHIs are rejected rather than guessed.
// Ordinary loops are ignored. Any loop containing RAY_QUERY_OBJECT_PROCEED but
// not satisfying a complete supported shape is a malformed near-match and is
// rejected without mutation. Every candidate in the function/module is
// preflighted before the first edit.
[[nodiscard]] LUISA_XIR_API ReconstructRayQueryLoopInfo
reconstruct_ray_query_loop_pass_run_on_function(
    Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API ReconstructRayQueryLoopInfo
reconstruct_ray_query_loop_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

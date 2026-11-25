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
// produced by lower_ray_query_to_loop or a pre/post-mem2reg SimpleLoopInst
// emitted for native DSL `$while (query.proceed())` or manual
// `$loop { if (!query.proceed()) break; ... }` forms. Any structured or CFG
// body without nested RayQuery loops, inner PROCEED, returns, or candidate-kind
// conflicts (e.g., surface ops reachable in a procedural arm or vice versa) is
// specialized into separate surface and procedural handler regions, turning
// `$break` into RAY_QUERY_OBJECT_TERMINATE when the query is not proceeded again
// after the loop. Ordinary loops are ignored. Any loop containing
// RAY_QUERY_OBJECT_PROCEED that cannot be safely reconstructed is rejected
// atomically before the first edit.
[[nodiscard]] LUISA_XIR_API ReconstructRayQueryLoopInfo
reconstruct_ray_query_loop_pass_run_on_function(
    Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API ReconstructRayQueryLoopInfo
reconstruct_ray_query_loop_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

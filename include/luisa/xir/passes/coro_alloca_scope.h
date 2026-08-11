#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Function;

// Contracts local-allocation lifetimes in the augmented coroutine CFG. For
// each local alloca, the pass follows every derived pointer use and computes
// the nearest common dominator including suspend -> resume transfer edges.
// Moving across a block boundary is allowed only after a forward must-analysis
// proves that every observation is definitely initialized after each dynamic
// execution of the proposed lifetime start. The proof combines static
// subaggregate coverage with exact pointer-projection versions; reexecuting a
// GEP kills its exact version so a store from an earlier loop iteration cannot
// initialize a later, potentially different runtime index.
//
// A proved alloca is moved to the latest legal point in that block. It then
// acts as an explicit lifetime start during frame liveness: storage from an
// earlier continuation iteration is undefined, not an implicit input to a
// partial store in the new lifetime. Same-block motion does not change dynamic
// lifetime count and therefore needs no definite-initialization proof.
//
// Phi pointer uses are retained conservatively because they occur on incoming
// edges rather than at the Phi's textual block position. Unreachable uses,
// malformed ownership, and a non-dominating original allocation likewise
// leave that allocation unchanged. The pass moves existing instructions and
// preserves their metadata.
struct CoroAllocaScopeInfo {
    size_t semantic_block_count{0u};
    size_t semantic_edge_count{0u};
    size_t scanned_local_alloca_count{0u};
    size_t contracted_alloca_count{0u};
    size_t cross_block_contraction_count{0u};
    size_t intra_block_contraction_count{0u};
    size_t rejected_phi_use_count{0u};
    size_t rejected_unreachable_use_count{0u};
    size_t rejected_non_dominating_alloca_count{0u};
    size_t definite_initialization_proof_count{0u};
    size_t rejected_prior_lifetime_observation_count{0u};
    size_t definite_initialization_block_evaluation_count{0u};
    size_t invalid_semantic_cfg_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return contracted_alloca_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API CoroAllocaScopeInfo
coro_alloca_scope_pass_run_on_function(Function *function) noexcept;

}// namespace luisa::compute::xir

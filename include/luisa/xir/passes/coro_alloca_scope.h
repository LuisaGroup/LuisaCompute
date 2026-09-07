#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Function;

struct CoroAllocaScopeOptions {
    // Expensive compiler-validation oracle. Every snapshot-order query is
    // compared with a linear walk over the current intrusive instruction list.
    bool verify_instruction_order{false};
};

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
// A second finite domain recognizes counted arrays without materializing their
// unused suffix. It proves Prefix(A, C): all elements below an unsigned counter
// are initialized. Only C = 0 and the atomic abstract transition
// A[C] = value; C = C + 1 can establish/preserve that invariant; CFG joins use
// Must intersection, counter overflow is rejected, and reads must select either
// a statically initialized sentinel or an index proved less than C. Pointer
// escape, unknown counter mutation, and unsupported arithmetic fail closed.
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
// preserves their metadata. Placement queries use an immutable per-block
// ordinal snapshot: earlier contractions cannot reorder the observation,
// definition, or insertion instructions of an unprocessed valid candidate.
// Current intrusive-list adjacency is consulted separately where intervening
// moved nodes are semantically observable.
struct CoroAllocaScopeInfo {
    size_t semantic_block_count{0u};
    size_t semantic_edge_count{0u};
    size_t scanned_local_alloca_count{0u};
    size_t contracted_alloca_count{0u};
    size_t cross_block_contraction_count{0u};
    size_t intra_block_contraction_count{0u};
    size_t delayed_first_definition_count{0u};
    size_t cross_block_first_definition_delay_count{0u};
    size_t intra_block_first_definition_delay_count{0u};
    size_t rejected_phi_use_count{0u};
    size_t rejected_unreachable_use_count{0u};
    size_t rejected_non_dominating_alloca_count{0u};
    size_t definite_initialization_proof_count{0u};
    size_t guarded_initialization_proof_count{0u};
    size_t initialized_prefix_proof_count{0u};
    size_t rejected_prior_lifetime_observation_count{0u};
    size_t definite_initialization_block_evaluation_count{0u};
    size_t guarded_initialization_state_evaluation_count{0u};
    size_t initialized_prefix_block_evaluation_count{0u};
    size_t predicate_widening_count{0u};
    size_t instruction_order_query_count{0u};
    size_t placement_user_inspection_count{0u};
    size_t invalid_semantic_cfg_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return contracted_alloca_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API CoroAllocaScopeInfo
coro_alloca_scope_pass_run_on_function(
    Function *function,
    const CoroAllocaScopeOptions &options = {}) noexcept;

}// namespace luisa::compute::xir

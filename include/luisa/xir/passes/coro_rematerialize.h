#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Function;

// Promotes coroutine local state back to reconstructible SSA expressions.
// The pass accepts a local allocation only when:
//
// - it has only unannotated whole-object stores, loads, and GEP projections;
// - every stored expression is replayable under the coroutine replay cost
//   model;
// - sparse must-reaching-value analysis over the semantic CFG obtained by
//   adding suspend(token) -> resume(token) edges proves one exact value at
//   every load; and
// - every reconstructed GEP projection also fits the replay cost model; and
// - every removed load is unannotated.
//
// The reaching-value lattice distinguishes pending, undefined, one exact SSA
// value, and conflict. Entry contributes undefined and loop backedges
// participate in the same fixed point, so loop-carried state cannot be
// mistaken for a scope-local temporary. Direct loads become the proven value.
// Loads through GEP chains become extracts at the original load position.
// Later DCE can remove the dead local storage, while coroutine splitting
// rematerializes the pure SSA DAG in each continuation instead of carrying it
// in the frame. Invalid token graphs and null/declaration inputs are
// conservative no-ops.
struct CoroRematerializeInfo {
    size_t semantic_block_count{0u};
    size_t semantic_edge_count{0u};
    size_t scanned_alloca_count{0u};
    size_t replayable_single_store_count{0u};
    size_t replayable_multi_store_count{0u};
    size_t reaching_dataflow_alloca_count{0u};
    size_t reaching_dataflow_block_evaluation_count{0u};
    size_t promoted_multi_store_alloca_count{0u};
    size_t unresolved_load_count{0u};
    size_t rejected_projected_replay_cost_count{0u};
    size_t promoted_alloca_count{0u};
    size_t replaced_load_count{0u};
    size_t inserted_extract_count{0u};
    size_t initializer_replay_instruction_cost{0u};
    size_t promoted_state_bytes{0u};
    size_t invalid_semantic_cfg_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return replaced_load_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API CoroRematerializeInfo
coro_rematerialize_local_state_pass_run_on_function(
    Function *function) noexcept;

}// namespace luisa::compute::xir

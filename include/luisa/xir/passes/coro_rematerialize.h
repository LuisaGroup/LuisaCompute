#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Function;

// Promotes immutable local state back to reconstructible SSA expressions.
// The pass accepts a local allocation only when:
//
// - it has one unannotated whole-object store and no other writes or escapes;
// - the stored expression is replayable under the coroutine replay cost model;
// - that store dominates every load in the semantic CFG obtained by adding
//   suspend(token) -> resume(token) edges; and
// - every reconstructed GEP projection also fits the replay cost model; and
// - every removed load is unannotated.
//
// Direct loads become the stored value. Loads through GEP chains become
// extracts at the original load position. Later DCE can remove the dead local
// storage, while coroutine splitting rematerializes the pure SSA DAG in each
// continuation instead of carrying it in the frame. Invalid token graphs and
// null/declaration inputs are conservative no-ops.
struct CoroRematerializeInfo {
    size_t semantic_block_count{0u};
    size_t semantic_edge_count{0u};
    size_t scanned_alloca_count{0u};
    size_t replayable_single_store_count{0u};
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

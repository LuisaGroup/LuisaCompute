#pragma once

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;

// Rewrites a statically indexed projection of a local aggregate snapshot
//
//   extract(load(p), i, j, ...)
//
// into
//
//   load(gep(p, i, j, ...)).
//
// The projected load is inserted at the original aggregate load, rather than
// at the extract, so an intervening store cannot change the observed value.
// Only local allocations and fully constant pointer/extract paths are
// accepted. This makes splitting one aggregate load into several field loads
// race-free and bounds IR growth by the number of distinct live projections.
// Identical unannotated projections of one snapshot are value-numbered.
// Dynamic paths and annotated intermediate extracts are conservative
// boundaries. Every function-owned block is considered, including coroutine
// resume roots disconnected from the entry by CoroSuspend. Null inputs are
// no-ops.

struct DeferLocalAggregateLoadInfo {
    size_t aggregate_load_count{0u};
    size_t candidate_extract_count{0u};
    size_t rewritten_extract_count{0u};
    size_t inserted_gep_count{0u};
    size_t inserted_load_count{0u};
    size_t reused_projection_count{0u};
    size_t removed_aggregate_load_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return rewritten_extract_count != 0u ||
               inserted_gep_count != 0u ||
               inserted_load_count != 0u ||
               removed_aggregate_load_count != 0u;
    }
};

[[nodiscard]] LUISA_XIR_API DeferLocalAggregateLoadInfo
defer_local_aggregate_load_pass_run_on_function(
    Function *function) noexcept;

[[nodiscard]] LUISA_XIR_API DeferLocalAggregateLoadInfo
defer_local_aggregate_load_pass_run_on_module(
    Module *module, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir

#pragma once

#include <cstdint>

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::tile::bridge::tirx {

// Relative issue-work coefficients, NOT measured nanoseconds or a promise of
// occupancy. Device calibration can replace these priors without changing
// candidate legality or the solver. Unknown register allocation is represented
// by live fragment scalars, never reported as a measured hardware register count.
struct ExecutionCostModel {
    double matrix_issue{1.0};
    double shared_fragment_transfer{2.0};
    double independent_element{0.0078125};
    double subgroup_setup{8.0};
    uint32_t preferred_subgroups{4u};
    uint32_t preferred_fragment_scalars_per_lane{32u};
};

struct PlannerOptions {
    bool enabled{true};
    bool retain_accumulators{true};
    // Zero lets the solver choose. A nonzero value is an exact tuning
    // constraint, checked against the target before any code is generated.
    uint32_t threads_per_group{0u};
    // A compiler search/code-size budget, not a claimed hardware register limit.
    uint32_t max_fragment_scalars_per_lane{64u};
    // Bound compilation work independently of target capacity. If an automatic
    // search would exceed this many widths, require a larger budget or an exact
    // thread count instead of silently truncating the supposedly exact search.
    uint32_t max_thread_candidates{32u};
    // Maximum independent values loaded before their stores in a cooperative
    // copy. One retains the scalar load/store sequence; no async engine or
    // vector-alignment promise is implied. The emitter checks each domain.
    uint32_t max_copy_batch{1u};
    ExecutionCostModel cost;
};

struct ExecutionLimits {
    uint32_t max_threads{256u};
    uint32_t subgroup_size{32u};
    uint64_t shared_memory_bytes{0u};
};

struct MatrixWorkload {
    uint64_t rows{0u};
    uint64_t columns{0u};
    uint64_t contraction{0u};
    uint64_t executions{1u};
    // A proved closed recurrence: C -> MMA -> D -> yield C, with no other
    // observation of C/D inside this many loop iterations. Zero disables it.
    uint64_t accumulator_iterations{0u};
};

struct GroupWorkload {
    uint64_t programs{0u};
    uint64_t independent_elements{0u};
    uint64_t max_independent_elements{1u};
    uint64_t shared_memory_bytes{0u};
    luisa::vector<MatrixWorkload> matrices;
};

struct MatrixDistribution {
    // A zero subgroup grid selects the existing one-atom-at-a-time reference
    // realization, including uniform subgroup tails. Otherwise the exact map is
    // (sg_m * atom_rows + local_m, sg_n * atom_columns + local_n).
    uint32_t subgroups_m{0u};
    uint32_t subgroups_n{0u};
    uint64_t atom_rows{1u};
    uint64_t atom_columns{1u};
    bool persistent_accumulator{false};

    [[nodiscard]] bool rectangular() const noexcept { return subgroups_m != 0u && subgroups_n != 0u; }
};

struct PlanCost {
    double matrix_issues{0.0};
    double shared_fragment_transfers{0.0};
    uint64_t fragment_scalars_per_lane{0u};
    double score{0.0};
};

struct GroupPlan {
    luisa::string name;
    uint32_t threads{1u};
    uint64_t shared_memory_bytes{0u};
    uint64_t candidates_considered{0u};
    uint64_t candidates_rejected{0u};
    uint32_t max_copy_batch{1u};
    uint64_t batched_copy_operations{0u};
    luisa::vector<MatrixDistribution> matrices;
    PlanCost cost;
    bool optimized{false};
};

struct PlanningResult {
    GroupPlan plan;
    luisa::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Exhaustive discrete solver for the currently implemented group realization
// family: thread count x rectangular subgroup factorizations x resident atom
// blocks. Covers every exact rectangular factorization in the supplied bounds,
// not all possible GPU programs. The reference realization is always a candidate.
// No compiler IR, JIT, measurements, or mutable global state is needed to rank
// candidates. MatrixWorkloads must come from proved semantic MMA contracts.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API PlanningResult plan_group(
    const GroupWorkload &workload, const ExecutionLimits &limits,
    const PlannerOptions &options = {}) noexcept;

// Independent, integer-only coverage/resource check, also applied by codegen.
// Cost-model coefficients cannot grant legality to a candidate.
[[nodiscard]] LUISA_TILE_TIRX_BRIDGE_API bool verify_matrix_distribution(
    const MatrixWorkload &workload, const MatrixDistribution &distribution,
    uint32_t threads, uint32_t subgroup_size) noexcept;

}// namespace luisa::compute::tile::bridge::tirx

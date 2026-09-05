#include <algorithm>
#include <cmath>
#include <limits>

#include <luisa/tile/bridge/tirx/planner.h>

namespace luisa::compute::tile::bridge::tirx {

namespace {

[[nodiscard]] bool valid_matrix(const MatrixWorkload &workload) noexcept {
    return workload.rows != 0u && workload.columns != 0u && workload.contraction != 0u &&
           workload.rows % 8u == 0u && workload.columns % 8u == 0u && workload.contraction % 8u == 0u &&
           workload.rows / 8u <= std::numeric_limits<uint64_t>::max() / (workload.columns / 8u) &&
           (!workload.has_direct_output || workload.accumulator_iterations != 0u) &&
           (workload.accumulator_iterations == 0u || workload.executions % workload.accumulator_iterations == 0u);
}

[[nodiscard]] double aspect_excess(uint64_t rows, uint64_t columns) noexcept {
    auto large = static_cast<double>(std::max(rows, columns));
    auto small = static_cast<double>(std::min(rows, columns));
    return large / small - 1.0;
}

[[nodiscard]] PlanCost reference_matrix_cost(const MatrixWorkload &workload, const MatrixDistribution &distribution,
                                             uint32_t threads, const ExecutionLimits &limits, const PlannerOptions &options) noexcept {
    auto atoms = static_cast<double>(workload.rows / 8u) * static_cast<double>(workload.columns / 8u);
    auto steps = static_cast<double>(workload.contraction / 8u);
    auto groups = threads / limits.subgroup_size;
    PlanCost result;
    result.matrix_issues = atoms * steps * static_cast<double>(workload.executions);
    auto inputs = distribution.rectangular() ?
                      static_cast<double>(groups) * static_cast<double>(distribution.atom_rows + distribution.atom_columns) :
                      2.0 * atoms;
    // One C read and one D write per atom, plus A/B transfers per K atom.
    auto accumulator_executions = distribution.persistent_accumulator ?
                                      static_cast<double>(workload.executions) / static_cast<double>(workload.accumulator_iterations) :
                                      static_cast<double>(workload.executions);
    result.shared_fragment_transfers = inputs * steps * static_cast<double>(workload.executions);
    if (distribution.direct_accumulator_store) {
        // The global output still exists. Its logical element work remains in
        // GroupWorkload::independent_elements; do not call the transfer free.
        result.direct_fragment_stores = atoms * accumulator_executions;
    } else {
        result.shared_fragment_transfers += 2.0 * atoms * accumulator_executions;
    }
    result.fragment_scalars_per_lane = distribution.rectangular() ?
                                           (64u / limits.subgroup_size) * (distribution.atom_rows * distribution.atom_columns + distribution.atom_rows + distribution.atom_columns) :
                                           3u * (64u / limits.subgroup_size);
    auto &model = options.cost;
    auto issue_work = result.matrix_issues * model.matrix_issue + result.shared_fragment_transfers * model.shared_fragment_transfer;
    auto state_pressure = std::max(1.0, static_cast<double>(result.fragment_scalars_per_lane) / model.preferred_fragment_scalars_per_lane);
    // This is a tunable parallelism prior, not an occupancy calculation. The
    // actual register file, scheduler residency, and launch cost need profiling.
    auto useful_groups = std::min(static_cast<double>(model.preferred_subgroups), atoms);
    auto parallelism = std::max(1.0, useful_groups / groups);
    result.score = issue_work * state_pressure * parallelism;
    return result;
}

[[nodiscard]] PlanCost mpp_matrix_cost(const MatrixWorkload &workload, const MatrixDistribution &distribution,
                                       uint32_t threads, const ExecutionLimits &limits, const PlannerOptions &options) noexcept {
    auto atoms = static_cast<double>(workload.rows / 8u) * static_cast<double>(workload.columns / 8u);
    auto steps = static_cast<double>(workload.contraction / 8u);
    auto executions = static_cast<double>(workload.executions);
    auto groups = threads / limits.subgroup_size;
    PlanCost result;
    if (!distribution.rectangular()) { return result; }

    result.matrix_issues = atoms * steps * executions;
    result.metal_mpp_operations = static_cast<double>(groups) * executions;
    result.memory_fragment_reads = static_cast<double>(groups) *
                                   static_cast<double>(distribution.atom_rows + distribution.atom_columns) *
                                   steps * executions;
    // These are unique logical footprints within one group. The MPP operation
    // may issue duplicate requests across subgroups; memory_fragment_reads
    // accounts for those separately. A/B are intentionally asymmetric because
    // row-major RHS rows have the less favorable reuse direction in this
    // realization. This is a ranking prior, not an alias/cache proof.
    result.lhs_footprint_fragments = static_cast<double>(workload.rows / 8u) * steps * executions;
    result.rhs_footprint_fragments = static_cast<double>(workload.columns / 8u) * steps * executions;
    auto accumulator_executions = distribution.persistent_accumulator ?
                                      executions / static_cast<double>(workload.accumulator_iterations) :
                                      executions;
    if (!workload.overwrites_accumulator) {
        result.accumulator_initializations = atoms * accumulator_executions;
    }
    if (distribution.direct_accumulator_store) {
        result.direct_fragment_stores = atoms * accumulator_executions;
    } else {
        result.shared_fragment_transfers = atoms * accumulator_executions;
    }

    auto local_rows = distribution.atom_rows;
    auto local_columns = distribution.atom_columns;
    result.tile_aspect_fragments = atoms * executions * aspect_excess(workload.rows, workload.columns);
    if (local_rows > local_columns) {
        result.local_row_aspect_issues = result.matrix_issues * aspect_excess(local_rows, local_columns);
    } else if (local_columns > local_rows) {
        result.local_column_aspect_issues = result.matrix_issues * aspect_excess(local_rows, local_columns);
    }
    // Only the output cooperative tensor is explicitly live in generated MSL;
    // A/B remain memory operands. This is logical state, not a register count.
    result.fragment_scalars_per_lane = (64u / limits.subgroup_size) *
                                       distribution.atom_rows * distribution.atom_columns;

    auto &model = options.cost;
    auto issue_work = result.matrix_issues * model.matrix_issue +
                      result.shared_fragment_transfers * model.shared_fragment_transfer +
                      result.direct_fragment_stores * model.metal_mpp_output_fragment +
                      result.metal_mpp_operations * model.metal_mpp_tensor_operation +
                      result.memory_fragment_reads * model.metal_mpp_memory_fragment +
                      result.lhs_footprint_fragments * model.metal_mpp_lhs_footprint +
                      result.rhs_footprint_fragments * model.metal_mpp_rhs_footprint +
                      result.accumulator_initializations * model.metal_mpp_accumulator_init +
                      result.tile_aspect_fragments * model.metal_mpp_tile_aspect +
                      result.local_row_aspect_issues * model.metal_mpp_local_row_aspect +
                      result.local_column_aspect_issues * model.metal_mpp_local_column_aspect;
    auto state_pressure = std::max(1.0, static_cast<double>(result.fragment_scalars_per_lane) /
                                            model.preferred_fragment_scalars_per_lane);
    // The rectangular distribution assigns disjoint tensor work to `groups`
    // subgroups that execute concurrently. Its critical path is aggregate
    // issue work divided by that explicit parallel width, not the aggregate
    // itself. Outer machine saturation is priced once in plan_group.
    result.score = issue_work * state_pressure / static_cast<double>(groups);
    return result;
}

[[nodiscard]] PlanCost matrix_cost(const MatrixWorkload &workload, const MatrixDistribution &distribution,
                                   uint32_t threads, const ExecutionLimits &limits, const PlannerOptions &options,
                                   MatrixCostBasis cost_basis) noexcept {
    switch (cost_basis) {
        case MatrixCostBasis::SIMDGROUP_REFERENCE:
            return reference_matrix_cost(workload, distribution, threads, limits, options);
        case MatrixCostBasis::METAL_MPP_MEMORY:
            return mpp_matrix_cost(workload, distribution, threads, limits, options);
    }
    return {};
}

[[nodiscard]] bool valid_options(const PlannerOptions &options) noexcept {
    auto &cost = options.cost;
    for (auto coefficient : {cost.matrix_issue, cost.shared_fragment_transfer, cost.independent_element, cost.subgroup_setup,
                             cost.subgroup_reduction_scalar_round, cost.subgroup_reduction_collective,
                             cost.subgroup_reduction_group_setup,
                             cost.metal_mpp_memory_fragment, cost.metal_mpp_lhs_footprint, cost.metal_mpp_rhs_footprint,
                             cost.metal_mpp_output_fragment, cost.metal_mpp_tensor_operation, cost.metal_mpp_accumulator_init,
                             cost.metal_mpp_tile_aspect, cost.metal_mpp_local_row_aspect, cost.metal_mpp_local_column_aspect,
                             cost.metal_mpp_independent_element, cost.metal_mpp_group_setup}) {
        if (!std::isfinite(coefficient) || coefficient < 0.0) { return false; }
    }
    return cost.preferred_subgroups != 0u && cost.preferred_fragment_scalars_per_lane != 0u &&
           cost.preferred_concurrent_programs != 0u && cost.metal_mpp_concurrent_subgroups != 0u &&
           options.max_fragment_scalars_per_lane >= 6u && options.max_thread_candidates != 0u &&
           options.max_reduction_striped_scalars_per_worker != 0u &&
           options.max_copy_batch != 0u && options.max_copy_batch <= 16u;
}

struct Alternative {
    MatrixDistribution distribution;
    PlanCost cost;
    uint64_t released_bytes{0u};
};

struct PartialPlan {
    luisa::vector<MatrixDistribution> matrices;
    PlanCost cost;
    uint64_t released_bytes{0u};
};

void add_cost(PlanCost &destination, const PlanCost &source) noexcept {
    destination.matrix_issues += source.matrix_issues;
    destination.shared_fragment_transfers += source.shared_fragment_transfers;
    destination.direct_fragment_stores += source.direct_fragment_stores;
    destination.metal_mpp_operations += source.metal_mpp_operations;
    destination.memory_fragment_reads += source.memory_fragment_reads;
    destination.lhs_footprint_fragments += source.lhs_footprint_fragments;
    destination.rhs_footprint_fragments += source.rhs_footprint_fragments;
    destination.accumulator_initializations += source.accumulator_initializations;
    destination.tile_aspect_fragments += source.tile_aspect_fragments;
    destination.local_row_aspect_issues += source.local_row_aspect_issues;
    destination.local_column_aspect_issues += source.local_column_aspect_issues;
    destination.independent_elements += source.independent_elements;
    destination.fragment_scalars_per_lane = std::max(destination.fragment_scalars_per_lane,
                                                     source.fragment_scalars_per_lane);
    destination.score += source.score;
}

// Exact Pareto dominance for the additive objective and one shared-capacity
// constraint. Keeping only the fastest choice per operation would be wrong:
// a slightly slower resident realization can be the only combination that fits.
void insert_frontier(luisa::vector<PartialPlan> &frontier, PartialPlan candidate) {
    for (auto &previous : frontier) {
        if (previous.released_bytes >= candidate.released_bytes && previous.cost.score <= candidate.cost.score) { return; }
    }
    std::erase_if(frontier, [&](const PartialPlan &previous) {
        return candidate.released_bytes >= previous.released_bytes && candidate.cost.score <= previous.cost.score;
    });
    frontier.emplace_back(std::move(candidate));
}

}// namespace

bool verify_matrix_distribution(const MatrixWorkload &workload, const MatrixDistribution &distribution,
                                uint32_t threads, uint32_t subgroup_size) noexcept {
    if (!valid_matrix(workload) || subgroup_size != 32u || threads < subgroup_size || threads % subgroup_size != 0u) { return false; }
    if (distribution.persistent_accumulator && (!distribution.rectangular() || workload.accumulator_iterations == 0u)) { return false; }
    if (distribution.direct_accumulator_store && (!distribution.persistent_accumulator || !workload.has_direct_output)) { return false; }
    if (!distribution.rectangular()) {
        return distribution.subgroups_m == 0u && distribution.subgroups_n == 0u &&
               distribution.atom_rows == 1u && distribution.atom_columns == 1u;
    }
    auto groups = threads / subgroup_size;
    return distribution.subgroups_m <= groups && groups % distribution.subgroups_m == 0u &&
           distribution.subgroups_n == groups / distribution.subgroups_m &&
           workload.rows / 8u % distribution.subgroups_m == 0u && workload.columns / 8u % distribution.subgroups_n == 0u &&
           distribution.atom_rows == workload.rows / 8u / distribution.subgroups_m &&
           distribution.atom_columns == workload.columns / 8u / distribution.subgroups_n;
}

PlanningResult plan_group(const GroupWorkload &workload, const ExecutionLimits &limits, const PlannerOptions &options,
                          MatrixCostBasis cost_basis) noexcept {
    PlanningResult result;
    auto &plan = result.plan;
    plan.programs = workload.programs;
    plan.cost_basis = cost_basis;
    if (limits.max_threads == 0u || limits.subgroup_size != 32u || !valid_options(options)) {
        result.error = "invalid group planner limits or cost coefficients";
        return result;
    }
    if (options.threads_per_group > limits.max_threads) {
        result.error = "requested group thread count exceeds target capacity";
        return result;
    }
    for (auto &matrix : workload.matrices) {
        if (!valid_matrix(matrix)) {
            result.error = "matrix planner requires a proved positive 8x8 FP32 atom domain";
            return result;
        }
    }
    plan.shared_memory_bytes = workload.shared_memory_bytes;
    plan.max_copy_batch = options.enabled ? options.max_copy_batch : 1u;
    auto reference_threads = std::min<uint64_t>(std::max<uint64_t>(1u, workload.max_independent_elements), limits.max_threads);
    if (!workload.matrices.empty() && limits.max_threads >= limits.subgroup_size) {
        reference_threads = std::min<uint64_t>((reference_threads + limits.subgroup_size - 1u) / limits.subgroup_size,
                                               limits.max_threads / limits.subgroup_size) *
                            limits.subgroup_size;
    }
    plan.threads = options.threads_per_group != 0u ? options.threads_per_group : static_cast<uint32_t>(reference_threads);
    plan.matrices.resize(workload.matrices.size());
    // Ordinary element domains retain their existing realization in this
    // version. Do not pretend a matrix cost model predicts reductions/scans.
    if (!options.enabled || workload.matrices.empty() || limits.max_threads < limits.subgroup_size ||
        (options.threads_per_group != 0u && options.threads_per_group % limits.subgroup_size != 0u)) {
        if (workload.shared_memory_bytes > limits.shared_memory_bytes) { result.error = "cooperative Tile storage exceeds target shared-memory capacity"; }
        return result;
    }

    auto best = std::numeric_limits<double>::infinity();
    auto first = options.threads_per_group != 0u ? options.threads_per_group / limits.subgroup_size : 1u;
    auto last = options.threads_per_group != 0u ? first : limits.max_threads / limits.subgroup_size;
    if (last - first + 1u > options.max_thread_candidates) {
        result.error = "group planner thread-search budget exceeded; increase the budget or constrain threads_per_group";
        return result;
    }
    for (auto groups = first; groups <= last; groups++) {
        auto threads = groups * limits.subgroup_size;
        luisa::vector<uint32_t> factors;
        for (auto factor = 1u; factor <= groups / factor; factor++) {
            if (groups % factor == 0u) {
                factors.emplace_back(factor);
                if (factor != groups / factor) { factors.emplace_back(groups / factor); }
            }
        }
        std::sort(factors.begin(), factors.end());
        luisa::vector<PartialPlan> frontier(1u);
        for (auto &matrix : workload.matrices) {
            luisa::vector<Alternative> alternatives;
            plan.candidates_considered++;
            // Enumerating the integer subgroup factors is an exact solver for
            // this finite family. Deriving local factors by division prunes
            // uncovered/overlapping assignments before any floating-point score.
            for (auto gm : factors) {
                auto gn = groups / gm;
                plan.candidates_considered++;
                MatrixDistribution candidate{gm, gn, matrix.rows / 8u / gm, matrix.columns / 8u / gn};
                candidate.persistent_accumulator = options.retain_accumulators && matrix.accumulator_iterations != 0u;
                candidate.direct_accumulator_store = candidate.persistent_accumulator && options.direct_accumulator_store && matrix.has_direct_output;
                if (!verify_matrix_distribution(matrix, candidate, threads, limits.subgroup_size)) {
                    plan.candidates_rejected++;
                    continue;
                }
                // Metal MPP's matrix descriptor requires at least one local
                // output dimension to be a multiple of 16. Since atom extents
                // are measured in 8x8 units, two odd local factors are not a
                // legal MPP realization even though they cover the Tile.
                if (cost_basis == MatrixCostBasis::METAL_MPP_MEMORY &&
                    candidate.atom_rows % 2u != 0u && candidate.atom_columns % 2u != 0u) {
                    plan.candidates_rejected++;
                    continue;
                }
                // Avoid overflow as well as unbounded generated unrolled code.
                auto scalar_budget = options.max_fragment_scalars_per_lane / (64u / limits.subgroup_size);
                if (candidate.atom_rows > scalar_budget || candidate.atom_columns > scalar_budget ||
                    candidate.atom_rows * candidate.atom_columns + candidate.atom_rows + candidate.atom_columns > scalar_budget) {
                    plan.candidates_rejected++;
                    continue;
                }
                auto estimate = matrix_cost(matrix, candidate, threads, limits, options, cost_basis);
                auto released = uint64_t{0u};
                if (candidate.persistent_accumulator) {
                    auto bytes_per_element = candidate.direct_accumulator_store ? 8u : 4u;
                    if (matrix.rows > std::numeric_limits<uint64_t>::max() / matrix.columns / bytes_per_element) {
                        result.error = "accumulator storage size overflow";
                        return result;
                    }
                    released = matrix.rows * matrix.columns * bytes_per_element;
                }
                alternatives.emplace_back(Alternative{candidate, estimate, released});
            }
            // The MPP target has no scalar/reference fallback under an explicit
            // MPP realization request. Keeping it as a scored alternative would
            // let the solver return a plan that codegen must reject later.
            if (cost_basis == MatrixCostBasis::SIMDGROUP_REFERENCE) {
                alternatives.emplace_back(Alternative{{}, matrix_cost(matrix, {}, threads, limits, options, cost_basis), 0u});
            }
            luisa::vector<PartialPlan> next;
            for (auto &partial : frontier) {
                for (auto &alternative : alternatives) {
                    auto candidate = partial;
                    if (alternative.released_bytes > workload.shared_memory_bytes - candidate.released_bytes) {
                        result.error = "invalid accumulator storage accounting in group workload";
                        return result;
                    }
                    candidate.released_bytes += alternative.released_bytes;
                    candidate.matrices.emplace_back(alternative.distribution);
                    add_cost(candidate.cost, alternative.cost);
                    insert_frontier(next, std::move(candidate));
                }
            }
            frontier = std::move(next);
        }
        for (auto &candidate : frontier) {
            auto shared_bytes = workload.shared_memory_bytes - candidate.released_bytes;
            candidate.cost.independent_elements = static_cast<double>(workload.independent_elements);
            if (cost_basis == MatrixCostBasis::METAL_MPP_MEMORY) {
                candidate.cost.score += candidate.cost.independent_elements * options.cost.metal_mpp_independent_element /
                                            static_cast<double>(groups) +
                                        options.cost.metal_mpp_group_setup;
                // Logical programs compete for subgroup slots. Fractional
                // waves intentionally remain a smooth ranking prior rather
                // than claiming a queried residency boundary.
                candidate.cost.concurrent_waves = std::max(
                    1.0, static_cast<double>(workload.programs) * static_cast<double>(groups) /
                             static_cast<double>(options.cost.metal_mpp_concurrent_subgroups));
            } else {
                candidate.cost.score += candidate.cost.independent_elements * options.cost.independent_element +
                                        groups * options.cost.subgroup_setup;
                candidate.cost.concurrent_waves = std::max(
                    1.0, static_cast<double>(workload.programs) /
                             static_cast<double>(options.cost.preferred_concurrent_programs));
            }
            candidate.cost.kernel_score = candidate.cost.score * candidate.cost.concurrent_waves;
            if (shared_bytes > limits.shared_memory_bytes || !std::isfinite(candidate.cost.kernel_score)) {
                plan.candidates_rejected++;
                continue;
            }
            if (candidate.cost.kernel_score < best) {
                best = candidate.cost.kernel_score;
                plan.threads = threads;
                plan.matrices = std::move(candidate.matrices);
                plan.shared_memory_bytes = shared_bytes;
                plan.cost = candidate.cost;
                plan.optimized = true;
            }
        }
    }
    if (!std::isfinite(best)) {
        result.error = cost_basis == MatrixCostBasis::METAL_MPP_MEMORY ?
                           "no legal Metal MPP group plan: each local matrix requires M or N to be a multiple of 16 and must fit shared-memory capacity and cost bounds" :
                           "no finite group plan fits target shared-memory capacity and cost bounds";
    }
    return result;
}

}// namespace luisa::compute::tile::bridge::tirx

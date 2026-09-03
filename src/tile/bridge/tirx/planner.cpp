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
           (workload.accumulator_iterations == 0u || workload.executions % workload.accumulator_iterations == 0u);
}

[[nodiscard]] PlanCost matrix_cost(const MatrixWorkload &workload, const MatrixDistribution &distribution,
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
    result.shared_fragment_transfers = inputs * steps * static_cast<double>(workload.executions) + 2.0 * atoms * accumulator_executions;
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

[[nodiscard]] bool valid_options(const PlannerOptions &options) noexcept {
    auto &cost = options.cost;
    for (auto coefficient : {cost.matrix_issue, cost.shared_fragment_transfer, cost.independent_element, cost.subgroup_setup}) {
        if (!std::isfinite(coefficient) || coefficient < 0.0) { return false; }
    }
    return cost.preferred_subgroups != 0u && cost.preferred_fragment_scalars_per_lane != 0u &&
           options.max_fragment_scalars_per_lane >= 6u && options.max_thread_candidates != 0u;
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

PlanningResult plan_group(const GroupWorkload &workload, const ExecutionLimits &limits, const PlannerOptions &options) noexcept {
    PlanningResult result;
    auto &plan = result.plan;
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
                if (!verify_matrix_distribution(matrix, candidate, threads, limits.subgroup_size)) {
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
                auto estimate = matrix_cost(matrix, candidate, threads, limits, options);
                auto released = uint64_t{0u};
                if (candidate.persistent_accumulator) {
                    if (matrix.rows > std::numeric_limits<uint64_t>::max() / matrix.columns / 4u) {
                        result.error = "accumulator storage size overflow";
                        return result;
                    }
                    released = matrix.rows * matrix.columns * 4u;
                }
                alternatives.emplace_back(Alternative{candidate, estimate, released});
            }
            alternatives.emplace_back(Alternative{{}, matrix_cost(matrix, {}, threads, limits, options), 0u});
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
                    candidate.cost.matrix_issues += alternative.cost.matrix_issues;
                    candidate.cost.shared_fragment_transfers += alternative.cost.shared_fragment_transfers;
                    candidate.cost.fragment_scalars_per_lane = std::max(candidate.cost.fragment_scalars_per_lane, alternative.cost.fragment_scalars_per_lane);
                    candidate.cost.score += alternative.cost.score;
                    insert_frontier(next, std::move(candidate));
                }
            }
            frontier = std::move(next);
        }
        for (auto &candidate : frontier) {
            auto shared_bytes = workload.shared_memory_bytes - candidate.released_bytes;
            candidate.cost.score += static_cast<double>(workload.independent_elements) * options.cost.independent_element +
                                    groups * options.cost.subgroup_setup;
            if (shared_bytes > limits.shared_memory_bytes) {
                plan.candidates_rejected++;
                continue;
            }
            if (candidate.cost.score < best) {
                best = candidate.cost.score;
                plan.threads = threads;
                plan.matrices = std::move(candidate.matrices);
                plan.shared_memory_bytes = shared_bytes;
                plan.cost = candidate.cost;
                plan.optimized = true;
            }
        }
    }
    if (!std::isfinite(best)) { result.error = "no finite group plan fits target shared-memory capacity and cost bounds"; }
    return result;
}

}// namespace luisa::compute::tile::bridge::tirx

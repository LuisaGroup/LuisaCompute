// Test the finite execution-mapping solver independently of JIT and devices.
// Covers exact atom ownership, integer constraints, all thread-count choices,
// multiple MMA contracts, resource limits, bad cost models, and reference mode.
#include "ut/ut.hpp"

#include <luisa/tile/bridge/tirx/planner.h>
#include <luisa/tile/bridge/tirx/layout.h>

#include <algorithm>
#include <limits>

using namespace luisa;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] GroupWorkload workload(uint64_t m, uint64_t n, uint64_t k) {
    return GroupWorkload{64u, (m + n) * k, std::max({m * n, m * k, k * n}), 4u * (2u * m * n + m * k + k * n), {{m, n, k, 7u}}};
}

void check_coverage(const MatrixWorkload &matrix, const MatrixDistribution &mapping, uint32_t threads) {
    expect(verify_matrix_distribution(matrix, mapping, threads, 32u));
    auto rows = matrix.rows / 8u;
    auto columns = matrix.columns / 8u;
    vector<uint32_t> visits(rows * columns, 0u);
    auto groups = threads / 32u;
    for (auto group = 0u; group < groups; group++) {
        if (mapping.rectangular()) {
            auto row0 = (group / mapping.subgroups_n) * mapping.atom_rows;
            auto column0 = (group % mapping.subgroups_n) * mapping.atom_columns;
            for (auto i = uint64_t{0u}; i < mapping.atom_rows; i++) {
                for (auto j = uint64_t{0u}; j < mapping.atom_columns; j++) {
                    expect(row0 + i < rows && column0 + j < columns);
                    if (row0 + i < rows && column0 + j < columns) { visits[(row0 + i) * columns + column0 + j]++; }
                }
            }
        } else {
            for (auto atom = uint64_t{group}; atom < visits.size(); atom += groups) { visits[atom]++; }
        }
    }
    expect(std::all_of(visits.begin(), visits.end(), [](auto count) { return count == 1u; }));
}

[[gnu::noinline]] tvm::ffi::Map<tvm::ffi::String, tvm::PrimExpr> native_placement(
    const tvm::tirx::Layout &layout, uint64_t row, uint64_t column, const MatrixWorkload &work) {
    return layout->Apply({tvm::IntImm::Int64(static_cast<int64_t>(row)), tvm::IntImm::Int64(static_cast<int64_t>(column))},
                         {tvm::IntImm::Int64(static_cast<int64_t>(work.rows / 8u)), tvm::IntImm::Int64(static_cast<int64_t>(work.columns / 8u))});
}

void check_native_correspondence(const MatrixWorkload &work, const MatrixDistribution &mapping) {
    if (!mapping.rectangular()) { return; }
    auto native = matrix_distribution_layout(work, mapping);
    expect(native.ok()) << native.error;
    if (!native) { return; }
    for (auto i = uint64_t{0u}; i < work.rows / 8u; i++) {
        for (auto j = uint64_t{0u}; j < work.columns / 8u; j++) {
            auto placement = native_placement(native.value, i, j, work);
            auto subgroup = placement.Get("warpid");
            auto fragment = placement.Get("m");
            expect(subgroup.has_value() && fragment.has_value());
            auto sg = subgroup ? subgroup->as<tvm::IntImmNode>() : nullptr;
            auto f = fragment ? fragment->as<tvm::IntImmNode>() : nullptr;
            expect(sg != nullptr && f != nullptr);
            if (sg == nullptr || f == nullptr) { continue; }
            expect(eq(static_cast<uint64_t>(sg->value), (i / mapping.atom_rows) * mapping.subgroups_n + j / mapping.atom_columns));
            expect(eq(static_cast<uint64_t>(f->value), (i % mapping.atom_rows) * mapping.atom_columns + j % mapping.atom_columns));
            auto inverse = matrix_atom_coordinates(work, mapping, tvm::IntImm::Int64(sg->value), tvm::IntImm::Int64(f->value));
            expect(inverse.ok()) << inverse.error;
            if (!inverse) { continue; }
            expect(inverse.value[0].as<tvm::IntImmNode>() != nullptr && inverse.value[1].as<tvm::IntImmNode>() != nullptr);
            if (auto row = inverse.value[0].as<tvm::IntImmNode>()) { expect(eq(row->value, static_cast<int64_t>(i))); }
            if (auto column = inverse.value[1].as<tvm::IntImmNode>()) { expect(eq(column->value, static_cast<int64_t>(j))); }
        }
    }
}

void test_exact_solver_and_coverage() {
    ExecutionLimits limits{256u, 32u, 1024u * 1024u};
    for (auto m : {8u, 16u, 24u, 32u, 64u}) {
        for (auto n : {8u, 16u, 24u, 32u, 64u}) {
            auto work = workload(m, n, 32u);
            auto result = plan_group(work, limits);
            expect(result.ok()) << result.error;
            if (!result) { continue; }
            expect(result.plan.optimized);
            expect(result.plan.candidates_considered > 8u);
            expect(result.plan.threads <= limits.max_threads);
            expect(result.plan.cost.fragment_scalars_per_lane <= 64u);
            check_coverage(work.matrices[0], result.plan.matrices[0], result.plan.threads);
            check_native_correspondence(work.matrices[0], result.plan.matrices[0]);
            // Independently constrain each hardware-width choice. No omitted
            // thread count may beat the unconstrained solution.
            for (auto threads = 32u; threads <= limits.max_threads; threads += 32u) {
                PlannerOptions options;
                options.threads_per_group = threads;
                auto forced = plan_group(work, limits, options);
                expect(forced.ok());
                if (!forced) { continue; }
                expect(eq(forced.plan.threads, threads));
                expect(result.plan.cost.score <= forced.plan.cost.score);
                check_coverage(work.matrices[0], forced.plan.matrices[0], threads);
            }
        }
    }
}

void test_constraints_and_reference() {
    auto work = workload(32, 64, 32);
    ExecutionLimits limits{256u, 32u, 32768u};
    auto planned = plan_group(work, limits);
    expect(planned.ok()) << planned.error;
    if (planned) {
        expect(eq(planned.plan.threads, 128u));
        expect(planned.plan.matrices[0].rectangular());
        expect(eq(planned.plan.matrices[0].atom_rows * planned.plan.matrices[0].atom_columns, 8ull));
        expect(eq(planned.plan.cost.fragment_scalars_per_lane, 28ull));
    }
    PlannerOptions reference;
    reference.enabled = false;
    auto original = plan_group(work, limits, reference);
    expect(original.ok());
    expect(!original.plan.optimized);
    expect(eq(original.plan.threads, 256u));
    expect(!original.plan.matrices[0].rectangular());
    reference.threads_per_group = 48u;
    auto scalar = plan_group(work, limits, reference);
    expect(scalar.ok());
    expect(eq(scalar.plan.threads, 48u));
    expect(!scalar.plan.optimized);
    reference.threads_per_group = 257u;
    expect(!plan_group(work, limits, reference).ok());
    limits.shared_memory_bytes = work.shared_memory_bytes - 1u;
    expect(!plan_group(work, limits).ok());
    limits.shared_memory_bytes = work.shared_memory_bytes;
    expect(plan_group(work, limits).ok());
    limits.max_threads = 31u;
    auto no_atom = plan_group(work, limits);
    expect(no_atom.ok());
    expect(!no_atom.plan.optimized);
    expect(eq(no_atom.plan.threads, 31u));

    limits.max_threads = std::numeric_limits<uint32_t>::max();
    auto unbounded = plan_group(work, limits);
    expect(!unbounded.ok());
    expect(unbounded.error.find("search budget") != string::npos);
    PlannerOptions one_width;
    one_width.threads_per_group = 32u;
    expect(plan_group(work, limits, one_width).ok());
    one_width.threads_per_group = 1u << 29u;
    // Factor enumeration is sqrt-bounded even when a synthetic target and
    // an exact request supply a very large width. Never run this on a device.
    expect(plan_group(work, limits, one_width).ok());
    one_width.max_thread_candidates = 0u;
    expect(!plan_group(work, limits, one_width).ok());
    one_width.max_thread_candidates = 32u;
    one_width.max_copy_batch = 0u;
    expect(!plan_group(work, limits, one_width).ok());
    one_width.max_copy_batch = 17u;
    expect(!plan_group(work, limits, one_width).ok());
    one_width.max_copy_batch = 4u;
    auto batched = plan_group(work, limits, one_width);
    expect(batched.ok());
    expect(eq(batched.plan.max_copy_batch, 4u));
    one_width.enabled = false;
    auto unbatched = plan_group(work, limits, one_width);
    expect(unbatched.ok());
    expect(eq(unbatched.plan.max_copy_batch, 1u));

    expect(!verify_matrix_distribution(work.matrices[0], {1u, 4u, 4u, 3u}, 128u, 32u));
    expect(!verify_matrix_distribution(work.matrices[0], {0u, 4u, 1u, 1u}, 128u, 32u));
    expect(!verify_matrix_distribution(work.matrices[0], {}, 31u, 32u));
    expect(!verify_matrix_distribution(work.matrices[0], {}, 128u, 16u));
}

void test_multiple_contracts_and_model_separation() {
    auto work = workload(32, 64, 32);
    work.matrices.push_back({16u, 24u, 16u, 3u});
    ExecutionLimits limits{256u, 32u, 32768u};
    for (auto preferred : {1u, 2u, 4u, 8u}) {
        PlannerOptions options;
        options.cost.preferred_subgroups = preferred;
        options.cost.shared_fragment_transfer = preferred * 0.5;
        auto result = plan_group(work, limits, options);
        expect(result.ok());
        if (!result) { continue; }
        expect(eq(result.plan.matrices.size(), size_t{2u}));
        for (auto i = 0u; i < 2u; i++) { check_coverage(work.matrices[i], result.plan.matrices[i], result.plan.threads); }
    }
    PlannerOptions bad;
    bad.cost.matrix_issue = std::numeric_limits<double>::quiet_NaN();
    expect(!plan_group(work, limits, bad).ok());
    bad.cost.matrix_issue = -1.0;
    expect(!plan_group(work, limits, bad).ok());
    bad.cost.matrix_issue = 1.0;
    bad.cost.preferred_subgroups = 0u;
    expect(!plan_group(work, limits, bad).ok());
    bad.cost.preferred_subgroups = 4u;
    bad.max_fragment_scalars_per_lane = 5u;
    expect(!plan_group(work, limits, bad).ok());
    work.matrices[0].rows = 31u;
    expect(!plan_group(work, limits).ok());
}

void test_capacity_requires_slower_resident_choice() {
    auto work = workload(32, 64, 32);
    work.matrices[0].accumulator_iterations = 7u;
    auto result_bytes = uint64_t{32u * 64u * 4u};
    ExecutionLimits limits{256u, 32u, work.shared_memory_bytes - result_bytes};
    PlannerOptions options;
    options.threads_per_group = 32u;
    options.max_fragment_scalars_per_lane = 128u;
    options.cost.preferred_fragment_scalars_per_lane = 1u;
    options.cost.shared_fragment_transfer = 0.0;
    auto fits = plan_group(work, limits, options);
    expect(fits.ok()) << fits.error;
    if (fits) {
        expect(fits.plan.matrices[0].persistent_accumulator);
        expect(eq(fits.plan.shared_memory_bytes, limits.shared_memory_bytes));
    }
    options.retain_accumulators = false;
    expect(!plan_group(work, limits, options).ok());
    options.retain_accumulators = true;
    work.matrices.push_back(work.matrices[0]);
    work.shared_memory_bytes *= 2u;
    limits.shared_memory_bytes *= 2u;
    auto joint = plan_group(work, limits, options);
    expect(joint.ok()) << joint.error;
    if (joint) {
        expect(joint.plan.matrices[0].persistent_accumulator && joint.plan.matrices[1].persistent_accumulator);
        expect(eq(joint.plan.shared_memory_bytes, limits.shared_memory_bytes));
    }
}

void test_direct_output_requires_proof_and_releases_both_buffers() {
    auto work = workload(64u, 64u, 32u);
    auto &matrix = work.matrices[0];
    matrix.accumulator_iterations = 7u;
    matrix.has_direct_output = true;
    ExecutionLimits limits{256u, 32u, 16384u};
    PlannerOptions options;
    options.threads_per_group = 256u;
    auto direct = plan_group(work, limits, options);
    expect(direct.ok()) << direct.error;
    if (direct) {
        expect(direct.plan.matrices[0].persistent_accumulator && direct.plan.matrices[0].direct_accumulator_store);
        expect(eq(direct.plan.shared_memory_bytes, 16384ull));
        expect(eq(direct.plan.cost.direct_fragment_stores, 64.0));
        check_coverage(matrix, direct.plan.matrices[0], 256u);
        check_native_correspondence(matrix, direct.plan.matrices[0]);
        auto unproved = matrix;
        unproved.has_direct_output = false;
        expect(!verify_matrix_distribution(unproved, direct.plan.matrices[0], 256u, 32u));
        auto not_resident = direct.plan.matrices[0];
        not_resident.persistent_accumulator = false;
        expect(!verify_matrix_distribution(matrix, not_resident, 256u, 32u));
    }
    options.direct_accumulator_store = false;
    expect(!plan_group(work, limits, options).ok());
    limits.shared_memory_bytes = 32768u;
    auto shared = plan_group(work, limits, options);
    expect(shared.ok()) << shared.error;
    if (shared) {
        expect(!shared.plan.matrices[0].direct_accumulator_store);
        expect(eq(shared.plan.shared_memory_bytes, 32768ull));
        expect(eq(shared.plan.cost.direct_fragment_stores, 0.0));
        expect(shared.plan.cost.shared_fragment_transfers > direct.plan.cost.shared_fragment_transfers);
    }
    options.direct_accumulator_store = true;
    matrix.accumulator_iterations = 0u;
    expect(!plan_group(work, limits, options).ok());
}

[[nodiscard]] PlanningResult mpp_plan(uint64_t m, uint64_t n, uint64_t k, bool overwrite = true) {
    GroupWorkload work;
    work.programs = 64u;
    work.max_independent_elements = m * n;
    work.shared_memory_bytes = 8u * m * n;
    work.matrices.push_back({m, n, k, 1u, 1u, true, overwrite});
    PlannerOptions options;
    options.threads_per_group = 128u;
    return plan_group(work, ExecutionLimits{128u, 32u, 0u}, options,
                      MatrixCostBasis::METAL_MPP_MEMORY);
}

void test_mpp_cost_basis_and_shape_ranking() {
    auto square_512 = mpp_plan(64u, 64u, 512u);
    auto tall_512 = mpp_plan(128u, 32u, 512u);
    auto square_1024 = mpp_plan(64u, 64u, 1024u);
    auto tall_1024 = mpp_plan(128u, 32u, 1024u);
    for (auto result : {&square_512, &tall_512, &square_1024, &tall_1024}) {
        expect(result->ok()) << result->error;
        if (!*result) { continue; }
        expect(result->plan.cost_basis == MatrixCostBasis::METAL_MPP_MEMORY);
        expect(eq(result->plan.programs, 64ull));
        expect(result->plan.matrices[0].rectangular());
        expect(result->plan.cost.metal_mpp_operations > 0.0);
        expect(result->plan.cost.memory_fragment_reads > 0.0);
        expect(eq(result->plan.cost.accumulator_initializations, 0.0));
        expect(eq(result->plan.cost.kernel_score, result->plan.cost.score));
    }
    // The versioned MPP prior crosses over as K grows: a balanced group tile
    // wins while descriptor/setup work dominates, then the row-major RHS
    // footprint makes a tall tile preferable. These are ranking contracts, not
    // nanosecond predictions or legality rules.
    expect(square_512.plan.cost.score < tall_512.plan.cost.score);
    expect(tall_1024.plan.cost.score < square_1024.plan.cost.score);
    expect(square_1024.plan.cost.score != tall_1024.plan.cost.score);

    auto accumulate = mpp_plan(64u, 64u, 1024u, false);
    expect(accumulate.ok()) << accumulate.error;
    if (accumulate) {
        expect(accumulate.plan.cost.accumulator_initializations > 0.0);
        expect(accumulate.plan.cost.score > square_1024.plan.cost.score);
    }

    GroupWorkload work;
    work.programs = 1u;
    work.shared_memory_bytes = 8u * 64u * 64u;
    work.matrices.push_back({64u, 64u, 1024u, 1u, 1u, true, true});
    PlannerOptions bad;
    bad.threads_per_group = 128u;
    bad.cost.metal_mpp_rhs_footprint = std::numeric_limits<double>::quiet_NaN();
    expect(!plan_group(work, ExecutionLimits{128u, 32u, 0u}, bad,
                       MatrixCostBasis::METAL_MPP_MEMORY)
                .ok());
    bad.cost.metal_mpp_rhs_footprint = 0.375;
    bad.cost.preferred_concurrent_programs = 0u;
    expect(!plan_group(work, ExecutionLimits{128u, 32u, 0u}, bad,
                       MatrixCostBasis::METAL_MPP_MEMORY)
                .ok());
    bad.cost.preferred_concurrent_programs = 64u;
    bad.cost.metal_mpp_concurrent_subgroups = 0u;
    expect(!plan_group(work, ExecutionLimits{128u, 32u, 0u}, bad,
                       MatrixCostBasis::METAL_MPP_MEMORY)
                .ok());
}

void test_mpp_subgroup_critical_path_and_machine_waves() {
    GroupWorkload work;
    work.programs = 1u;
    work.independent_elements = 1024u;
    work.max_independent_elements = 1024u;
    work.shared_memory_bytes = 8u * 32u * 32u;
    work.matrices.push_back({32u, 32u, 32u, 1u, 1u, true, true});
    PlannerOptions narrow_options;
    narrow_options.threads_per_group = 64u;
    PlannerOptions wide_options;
    wide_options.threads_per_group = 256u;
    auto limits = ExecutionLimits{256u, 32u, 0u};
    auto narrow = plan_group(work, limits, narrow_options, MatrixCostBasis::METAL_MPP_MEMORY);
    auto wide = plan_group(work, limits, wide_options, MatrixCostBasis::METAL_MPP_MEMORY);
    expect(narrow.ok()) << narrow.error;
    expect(wide.ok()) << wide.error;
    if (!narrow || !wide) { return; }
    expect(wide.plan.cost.score < narrow.plan.cost.score);
    expect(eq(narrow.plan.cost.concurrent_waves, 1.0));
    expect(eq(wide.plan.cost.concurrent_waves, 1.0));
    expect(wide.plan.cost.kernel_score < narrow.plan.cost.kernel_score);

    // The same per-program critical path is not a whole-device throughput
    // prediction. Once many programs saturate the target prior, a wider group
    // consumes proportionally more subgroup slots and therefore more waves.
    work.programs = 1024u;
    narrow = plan_group(work, limits, narrow_options, MatrixCostBasis::METAL_MPP_MEMORY);
    wide = plan_group(work, limits, wide_options, MatrixCostBasis::METAL_MPP_MEMORY);
    expect(narrow.ok()) << narrow.error;
    expect(wide.ok()) << wide.error;
    if (!narrow || !wide) { return; }
    expect(narrow.plan.cost.concurrent_waves < wide.plan.cost.concurrent_waves);
    expect(narrow.plan.cost.kernel_score < wide.plan.cost.kernel_score);

    // Eight 8x8 local matrices cover 32x16 geometrically, but violate MPP's
    // descriptor rule. The unconstrained solver must choose a narrower legal
    // cohort instead of returning a plan that fails during code generation.
    GroupWorkload descriptor_work;
    descriptor_work.programs = 1u;
    descriptor_work.max_independent_elements = 32u * 16u;
    descriptor_work.shared_memory_bytes = 8u * 32u * 16u;
    descriptor_work.matrices.push_back({32u, 16u, 32u, 1u, 1u, true, true});
    PlannerOptions automatic;
    auto descriptor = plan_group(descriptor_work, limits, automatic, MatrixCostBasis::METAL_MPP_MEMORY);
    expect(descriptor.ok()) << descriptor.error;
    if (descriptor) {
        expect(descriptor.plan.threads < 256u);
        auto &mapping = descriptor.plan.matrices[0];
        expect(mapping.atom_rows % 2u == 0u || mapping.atom_columns % 2u == 0u);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_planner_exact_thread_search_and_atom_coverage"_test = [] { test_exact_solver_and_coverage(); };
    "tile_planner_constraints_and_reference"_test = [] { test_constraints_and_reference(); };
    "tile_planner_multiple_contracts_and_model_separation"_test = [] { test_multiple_contracts_and_model_separation(); };
    "tile_planner_pareto_capacity_beats_local_greedy_choice"_test = [] { test_capacity_requires_slower_resident_choice(); };
    "tile_planner_direct_output_proof_and_storage_accounting"_test = [] { test_direct_output_requires_proof_and_releases_both_buffers(); };
    "tile_planner_mpp_cost_basis_and_shape_ranking"_test = [] { test_mpp_cost_basis_and_shape_ranking(); };
    "tile_planner_mpp_subgroup_critical_path_and_machine_waves"_test = [] { test_mpp_subgroup_critical_path_and_machine_waves(); };
}

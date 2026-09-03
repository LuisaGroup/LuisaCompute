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

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_planner_exact_thread_search_and_atom_coverage"_test = [] { test_exact_solver_and_coverage(); };
    "tile_planner_constraints_and_reference"_test = [] { test_constraints_and_reference(); };
    "tile_planner_multiple_contracts_and_model_separation"_test = [] { test_multiple_contracts_and_model_separation(); };
    "tile_planner_pareto_capacity_beats_local_greedy_choice"_test = [] { test_capacity_requires_slower_resident_choice(); };
}

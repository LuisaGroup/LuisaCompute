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
    bad.max_fragment_scalars_per_lane = 64u;
    bad.max_reduction_striped_scalars_per_worker = 0u;
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
    // A closed scalar DAG releases only its own additional storage. Its
    // arithmetic remains in independent_elements, even for direct output.
    matrix.epilogue_storage_bytes = 4096u;
    work.shared_memory_bytes += 4096u;
    limits.shared_memory_bytes = 16384u;
    auto epilogue = plan_group(work, limits, options);
    expect(epilogue.ok()) << epilogue.error;
    if (epilogue) {
        expect(eq(epilogue.plan.shared_memory_bytes, 16384ull));
        expect(eq(epilogue.plan.cost.independent_elements, direct.plan.cost.independent_elements));
    }
    options.direct_accumulator_store = false;
    expect(!plan_group(work, limits, options));
    options.direct_accumulator_store = true;
    matrix.epilogue_storage_bytes = std::numeric_limits<uint64_t>::max();
    expect(!plan_group(work, limits, options));
    matrix.epilogue_storage_bytes = 4096u;
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

void test_backend_cost_policy() {
    class Policy final : public AnalyticExecutionCostPolicy {
    public:
        mutable uint32_t calls{0u};
        bool invalid{false};
        ExecutionCostModel coefficients(const ExecutionLimits &, MatrixCostBasis,
                                        const ExecutionCostModel &prior) const noexcept override {
            calls++;
            auto model = prior;
            model.matrix_issue = invalid ? std::numeric_limits<double>::quiet_NaN() : 10.0;
            model.preferred_subgroups = 1u;
            return model;
        }
    } policy;
    auto work = workload(32u, 64u, 32u);
    auto limits = ExecutionLimits{256u, 32u, 1024u * 1024u};
    PlannerOptions options;
    options.cost_policy = &policy;
    auto customized = plan_group(work, limits, options);
    expect(eq(policy.calls, 1u));
    PlannerOptions explicit_profile;
    explicit_profile.cost.matrix_issue = 10.0;
    explicit_profile.cost.preferred_subgroups = 1u;
    auto expected = plan_group(work, limits, explicit_profile);
    expect(customized.ok() && expected.ok());
    if (customized && expected) {
        expect(eq(customized.plan.threads, expected.plan.threads));
        expect(eq(customized.plan.cost.score, expected.plan.cost.score));
        check_coverage(work.matrices[0], customized.plan.matrices[0], customized.plan.threads);
    }
    // A backend's preferred score cannot waive resource limits or invalid
    // coefficients, and the caller's prior is not mutated by calibration.
    expect(eq(options.cost.matrix_issue, ExecutionCostModel{}.matrix_issue));
    expect(!plan_group(work, ExecutionLimits{256u, 32u, 0u}, options));
    policy.invalid = true;
    expect(!plan_group(work, limits, options));
}

void test_realization_fragment_state_budget() {
    // Independent pixel-domain enumeration: the MPP emitter keeps one MxN
    // output tensor, not the reference emitter's A/B fragment arrays. Cover
    // exact limits, ragged subgroup factorizations and both aspect directions.
    auto limits = ExecutionLimits{256u, 32u, 1u << 24u};
    for (auto m : {8u, 16u, 24u, 32u, 48u, 64u, 96u, 128u}) {
        for (auto n : {8u, 16u, 24u, 32u, 48u, 64u, 96u, 128u}) {
            auto work = workload(m, n, 32u);
            for (auto threads : {32u, 64u, 96u, 128u, 192u, 256u}) {
                for (auto budget : {3u, 4u, 5u, 6u, 16u, 31u, 32u, 47u, 48u, 63u, 64u, 96u, 128u}) {
                    auto expected = false;
                    for (auto gm = 1u; gm <= threads / 32u; gm++) {
                        if (threads / 32u % gm != 0u) { continue; }
                        auto gn = threads / 32u / gm;
                        if (m % gm != 0u || n % gn != 0u) { continue; }
                        auto local_m = m / gm;
                        auto local_n = n / gn;
                        expected |= local_m % 8u == 0u && local_n % 8u == 0u &&
                                    (local_m % 16u == 0u || local_n % 16u == 0u) &&
                                    local_m * local_n / 32u <= budget;
                    }
                    PlannerOptions options;
                    options.threads_per_group = threads;
                    options.max_fragment_scalars_per_lane = budget;
                    auto result = plan_group(work, limits, options, MatrixCostBasis::METAL_MPP_MEMORY);
                    expect(eq(result.ok(), expected)) << m << n << threads << budget << result.error;
                    if (result) {
                        expect(eq(result.plan.cost.fragment_scalars_per_lane, uint64_t{m * n / threads}));
                        expect(verify_matrix_distribution(work.matrices[0], result.plan.matrices[0], threads, 32u));
                    }
                }
            }
        }
    }
    auto work = workload(32u, 64u, 32u);
    PlannerOptions options;
    options.threads_per_group = 32u;
    // Keep the pressure prior neutral here: this checks admission, not the
    // profitability of large explicit fragments on a particular GPU.
    options.cost.preferred_fragment_scalars_per_lane = 128u;
    options.max_fragment_scalars_per_lane = 87u;
    auto reference = plan_group(work, limits, options);
    expect(reference.ok() && !reference.plan.matrices[0].rectangular());
    options.max_fragment_scalars_per_lane = 88u;
    reference = plan_group(work, limits, options);
    expect(reference.ok() && reference.plan.matrices[0].rectangular());
    if (reference) { expect(eq(reference.plan.cost.fragment_scalars_per_lane, 88ull)); }
    // Correcting a software-state budget cannot waive physical shared capacity.
    options.max_fragment_scalars_per_lane = 64u;
    expect(!plan_group(work, {32u, 32u, work.shared_memory_bytes - 1u}, options, MatrixCostBasis::METAL_MPP_MEMORY));
}

void test_realized_matrix_work() {
    GroupWorkload work;
    work.programs = 1u;
    work.max_independent_elements = 2048u;
    work.shared_memory_bytes = 16384u;
    work.matrices.push_back({32u, 64u, 32u, 3u, 3u, true, false, 17.0 / 3.0, 6144u, 4096u});
    work.independent_elements = 6144u + 4096u + 17u;
    PlannerOptions options;
    options.threads_per_group = 32u;
    options.max_fragment_scalars_per_lane = 128u;
    options.cost.preferred_fragment_scalars_per_lane = 128u;
    ExecutionLimits limits{32u, 32u, work.shared_memory_bytes};
    for (auto basis : {MatrixCostBasis::SIMDGROUP_REFERENCE, MatrixCostBasis::METAL_MPP_MEMORY}) {
        for (auto retain : {false, true}) {
            for (auto direct : {false, true}) {
                options.retain_accumulators = retain;
                options.direct_accumulator_store = direct;
                auto result = plan_group(work, limits, options, basis);
                expect(result.ok()) << result.error;
                if (!result) { continue; }
                expect(result.plan.matrices[0].rectangular());
                auto elided = (retain ? 6144u : 0u) + (retain && direct ? 4096u : 0u);
                expect(eq(result.plan.cost.elided_independent_elements, static_cast<double>(elided)));
                expect(eq(result.plan.cost.independent_elements, static_cast<double>(work.independent_elements - elided)));
                expect(eq(result.plan.cost.nominal_matrix_issues, 32.0 * 12.0));
                auto physical = basis == MatrixCostBasis::METAL_MPP_MEMORY ? 32.0 * 17.0 / 8.0 : 32.0 * 12.0;
                expect(std::abs(result.plan.cost.matrix_issues - physical) < 1e-10);
                expect(eq(result.plan.cost.direct_fragment_stores, retain && direct ? 32.0 : 0.0));
            }
        }
    }
    // A direct-only overwrite proof must not discount initialization if the
    // candidate retains the scalar fill and loads C instead.
    auto &matrix = work.matrices[0];
    matrix.executions = matrix.accumulator_iterations = 1u;
    matrix.overwrites_accumulator = true;
    matrix.recurrence_elements = 2048u;
    options.direct_accumulator_store = false;
    auto shared = plan_group(work, limits, options, MatrixCostBasis::METAL_MPP_MEMORY);
    expect(shared.ok() && shared.plan.cost.accumulator_initializations == 32.0);
    options.direct_accumulator_store = true;
    auto direct = plan_group(work, limits, options, MatrixCostBasis::METAL_MPP_MEMORY);
    expect(direct.ok() && direct.plan.cost.accumulator_initializations == 0.0);
    // Invalid accounting cannot subtract another operation's work or wrap.
    matrix.direct_output_elements = work.independent_elements;
    expect(!plan_group(work, limits, options));
    matrix.direct_output_elements = 4096u;
    for (auto invalid : {-1.0, 33.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        matrix.mean_contraction = invalid;
        expect(!plan_group(work, limits, options));
    }
}

void test_realized_work_pareto_objective() {
    // Independent enumeration of two atom/reference choices per operation.
    // The two identical output footprints have different recurrence counts
    // and K work, so released storage alone does not determine scalar savings.
    for (auto scalar_cost : {0.0, 0.1, 1.0, 10.0}) {
        for (auto required_resident : {0u, 1u, 2u}) {
            GroupWorkload work;
            work.programs = 1u;
            work.max_independent_elements = 2048u;
            work.shared_memory_bytes = 32768u;
            work.matrices = {{32u, 64u, 8u, 1u, 1u, true, true, 0.0, 2048u, 4096u},
                             {32u, 64u, 32u, 7u, 7u, true, false, 0.0, 14336u, 4096u}};
            work.independent_elements = 24576u + 19u;
            PlannerOptions options;
            options.threads_per_group = 32u;
            options.max_fragment_scalars_per_lane = 88u;
            options.cost.preferred_fragment_scalars_per_lane = 8u;
            // Neutralize the separate subgroup-count prior: this oracle
            // enumerates state pressure and retained work, not that prior.
            options.cost.preferred_subgroups = 1u;
            options.cost.independent_element = scalar_cost;
            ExecutionLimits limits{32u, 32u, work.shared_memory_bytes - required_resident * 16384u};
            auto best = std::numeric_limits<double>::infinity();
            for (auto choices = 0u; choices < 4u; choices++) {
                auto resident = (choices & 1u) + ((choices >> 1u) & 1u);
                if (resident < required_resident) { continue; }
                auto score = 19.0 * scalar_cost + 8.0;
                for (auto index = 0u; index < 2u; index++) {
                    auto &matrix = work.matrices[index];
                    auto steps = static_cast<double>(matrix.contraction / 8u * matrix.executions);
                    if ((choices & (1u << index)) != 0u) {
                        score += (32.0 * steps + 2.0 * (12.0 * steps + 32.0)) * 11.0;
                    } else {
                        score += 32.0 * steps + 2.0 * (64.0 * steps + 64.0 * matrix.executions) +
                                 (matrix.recurrence_elements + matrix.direct_output_elements) * scalar_cost;
                    }
                }
                best = std::min(best, score);
            }
            auto planned = plan_group(work, limits, options);
            expect(planned.ok()) << planned.error;
            if (planned) {
                expect(std::abs(planned.plan.cost.kernel_score - best) < 1e-8)
                    << scalar_cost << required_resident << planned.plan.cost.kernel_score << best;
            }
        }
    }
}

void test_reduction_access_service_policy() {
    ReductionCandidate candidate;
    candidate.scalar_rounds = 12.0;
    candidate.reductions = 2u;
    candidate.subgroups_per_program = 4u;
    candidate.programs_per_group = 1u;
    candidate.payload_accesses_known = true;
    candidate.payload_accesses_per_worker = {64.0, 32.0, 96.0, 16.0};
    ExecutionCostModel model;
    AnalyticExecutionCostPolicy policy;
    auto historical = policy.reduction_score(candidate, model);
    expect(eq(historical, 44.0));
    // Deliberately synthetic units test independent service coefficients, not
    // a calibrated GPU profile. The default remains exactly the old score.
    model.subgroup_reduction_global_access_byte = 0.25;
    model.subgroup_reduction_private_access_byte = 0.0625;
    expect(eq(policy.reduction_score(candidate, model), 75.0));
    candidate.payload_accesses_known = false;
    expect(eq(policy.reduction_score(candidate, model), historical));
    PlannerOptions options;
    for (auto invalid : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        options.cost.subgroup_reduction_global_access_byte = invalid;
        expect(!plan_group(workload(32u, 64u, 32u), ExecutionLimits{256u, 32u, 1u << 20u}, options));
        options.cost.subgroup_reduction_global_access_byte = 0.0;
        options.cost.subgroup_reduction_private_access_byte = invalid;
        expect(!plan_group(workload(32u, 64u, 32u), ExecutionLimits{256u, 32u, 1u << 20u}, options));
        options.cost.subgroup_reduction_private_access_byte = 0.0;
    }
}

void test_reduction_machine_cost() {
    ReductionCandidate candidate;
    candidate.programs = 1024u;
    candidate.threadgroups = 1024u;
    candidate.subgroups_per_program = 4u;
    candidate.programs_per_group = 1u;
    candidate.scalar_rounds = 12.0;
    candidate.reductions = 2u;
    candidate.payload_accesses_known = true;
    candidate.payload_accesses_per_worker = {64.0, 32.0, 96.0, 16.0};
    candidate.payload_accesses_per_program = {128.0, 64.0, 192.0, 32.0};
    ExecutionCostModel prior;
    AnalyticExecutionCostPolicy legacy;
    auto original = legacy.reduction_cost(candidate, prior);
    expect(eq(original.program_score, legacy.reduction_score(candidate, prior)));
    expect(eq(original.concurrent_waves, 16.0));
    expect(eq(original.kernel_score, original.program_score * 16.0));
    // Synthetic arithmetic, deliberately unrelated to the M1 Max fit.
    auto model = ReductionServiceModel{512u, 3.0, 2.0, 5.0, 0.25, 0.125, 0.0625};
    ServiceExecutionCostPolicy policy{model};
    expect(policy.valid());
    auto cost = policy.reduction_cost(candidate, prior);
    expect(eq(cost.program_score, 71.0));
    expect(eq(cost.concurrent_waves, 8.0));
    expect(eq(cost.kernel_score, 49735.0));
    prior.preferred_concurrent_programs = 1u;
    expect(eq(policy.reduction_cost(candidate, prior).kernel_score, cost.kernel_score));
    // Five programs packed three per group launch six subgroups, not five.
    candidate.programs = 5u;
    candidate.threadgroups = 2u;
    candidate.programs_per_group = 3u;
    candidate.subgroups_per_program = 1u;
    model.concurrent_subgroups = 4u;
    expect(eq(ServiceExecutionCostPolicy{model}.reduction_cost(candidate, prior).concurrent_waves, 1.5));
    // Two cooperating subgroups per program: the sixth physical program
    // replays an input row but does not write an extra output row.
    candidate.subgroups_per_program = 2u;
    auto packed_cost = ServiceExecutionCostPolicy{model}.reduction_cost(candidate, prior);
    expect(eq(packed_cost.concurrent_waves, 3.0));
    expect(eq(packed_cost.kernel_score, 440.0));
    candidate.programs = 6u;
    expect(eq(ServiceExecutionCostPolicy{model}.reduction_cost(candidate, prior).kernel_score, 456.0));
    candidate.payload_accesses_known = false;
    expect(!std::isfinite(policy.reduction_cost(candidate, prior).kernel_score));
    model.concurrent_subgroups = 0u;
    expect(!ServiceExecutionCostPolicy{model}.valid());
    model.concurrent_subgroups = 512u;
    for (auto member : {&ReductionServiceModel::dispatch, &ReductionServiceModel::scalar_round,
                        &ReductionServiceModel::collective, &ReductionServiceModel::global_program_byte,
                        &ReductionServiceModel::global_worker_byte, &ReductionServiceModel::private_worker_byte}) {
        for (auto invalid : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            auto bad = model;
            bad.*member = invalid;
            ServiceExecutionCostPolicy rejected{bad};
            expect(!rejected.valid());
            expect(!std::isfinite(rejected.reduction_cost(candidate, prior).kernel_score));
        }
    }
}

void test_collective_reference_width() {
    ExecutionLimits limits{256u, 32u, 32768u};
    GroupWorkload work;
    work.programs = 3u;
    work.max_independent_elements = 1u;
    auto scalar = plan_group(work, limits);
    expect(scalar.ok());
    expect(eq(scalar.plan.threads, 1u));
    for (auto outputs : {uint64_t{1}, uint64_t{3}, uint64_t{100}, UINT64_MAX}) {
        work.max_collective_outputs = outputs;
        auto collective = plan_group(work, limits);
        expect(collective.ok());
        expect(eq(collective.plan.threads, static_cast<uint32_t>(std::min(outputs, uint64_t{8}) * 32u)));
        // Width admission is not a newly calibrated profitability model.
        expect(!collective.plan.optimized);
    }
    PlannerOptions exact;
    exact.threads_per_group = 1u;
    auto fallback = plan_group(work, limits, exact);
    expect(fallback.ok());
    expect(eq(fallback.plan.threads, 1u));
    auto narrow = plan_group(work, ExecutionLimits{16u, 32u, 32768u});
    expect(narrow.ok());
    expect(eq(narrow.plan.threads, 1u));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_planner_exact_thread_search_and_atom_coverage"_test = [] { test_exact_solver_and_coverage(); };
    "tile_planner_constraints_and_reference"_test = [] { test_constraints_and_reference(); };
    "tile_planner_collective_reference_width"_test = [] { test_collective_reference_width(); };
    "tile_planner_multiple_contracts_and_model_separation"_test = [] { test_multiple_contracts_and_model_separation(); };
    "tile_planner_pareto_capacity_beats_local_greedy_choice"_test = [] { test_capacity_requires_slower_resident_choice(); };
    "tile_planner_direct_output_proof_and_storage_accounting"_test = [] { test_direct_output_requires_proof_and_releases_both_buffers(); };
    "tile_planner_mpp_cost_basis_and_shape_ranking"_test = [] { test_mpp_cost_basis_and_shape_ranking(); };
    "tile_planner_mpp_subgroup_critical_path_and_machine_waves"_test = [] { test_mpp_subgroup_critical_path_and_machine_waves(); };
    "tile_planner_realization_fragment_state_budget"_test = [] { test_realization_fragment_state_budget(); };
    "tile_planner_realized_matrix_work"_test = [] { test_realized_matrix_work(); };
    "tile_planner_realized_work_pareto_objective"_test = [] { test_realized_work_pareto_objective(); };
    "tile_planner_backend_cost_policy"_test = [] { test_backend_cost_policy(); };
    "tile_planner_reduction_access_service_policy"_test = [] { test_reduction_access_service_policy(); };
    "tile_planner_reduction_machine_cost"_test = [] { test_reduction_machine_cost(); };
}

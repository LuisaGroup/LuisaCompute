// Test backend-supplied XIR execution capabilities and scheduling.
// Covers physical GPU packet widths, resource admission before costs, CPU
// parity, and static resource analysis against independently inspected XIR.

#include "ut/ut.hpp"
#include <luisa/ast/type.h>
#include <luisa/core/mathematics.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/dsl.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/verifier.h>
#include <cmath>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
namespace bx = tile::bridge::xir;

[[nodiscard]] tile::Kernel row_fixture(uint32_t width, bool reduction = true, uint32_t rows = 17u) {
    using namespace tile;
    return tile_kernel("target_info_rows", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
               auto m = axis("m", 1), n = axis("n", width);
               for (auto &nest : parallel(shape(rows))) {
                   auto origin = coord(nest.index(), 0);
                   auto x = input.tile(origin, shape(m, n)).load();
                   auto y = x * 2.0f + 1.0f;
                   if (reduction) { y = y + reduce(x, n, add); }
                   output(origin, shape(m, n)).store(y);
               }
           })
        .capture(tensor_shape(rows, width), tensor_shape(rows, width));
}

template<typename T>
[[nodiscard]] tile::Kernel copy_fixture(uint32_t width, uint32_t rows = 17u) {
    using namespace tile;
    return tile_kernel("target_info_copy", [=](TensorView<const T, 2> input, TensorView<T, 2> output) {
               auto m = axis("m", 1), n = axis("n", width);
               for (auto &nest : parallel(shape(rows))) {
                   auto origin = coord(nest.index(), 0);
                   auto x = input.tile(origin, shape(m, n)).load();
                   output(origin, shape(m, n)).store(x);
               }
           })
        .capture(tensor_shape(rows, width), tensor_shape(rows, width));
}

void expect_same_resources(const bx::ExecutionResources &actual, const bx::ExecutionResources &expected) {
    expect(eq(actual.snapshot_bytes_per_worker, expected.snapshot_bytes_per_worker));
    expect(eq(actual.snapshot_allocations, expected.snapshot_allocations));
}

[[nodiscard]] bx::NativeFunction check_resources(const tile::Kernel &kernel, const bx::LowerOptions &options = {}) {
    expect(kernel.valid());
    if (!kernel.valid()) { return {}; }
    auto analysis = bx::analyze_resources(kernel.function(), options);
    expect(analysis.ok()) << analysis.error;
    if (!analysis) { return {}; }
    auto lowered = bx::lower(kernel.function(), options);
    expect(lowered.ok()) << lowered.error;
    if (!lowered) { return {}; }
    // Do not use the estimator or the lowerer's counters to obtain the oracle.
    // Sum the concrete allocation types in the actual pre-cleanup XIR graph.
    bx::ExecutionResources observed;
    lowered.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
        if (!instruction->isa<xir::AllocaInst>()) { return; }
        auto allocation = static_cast<xir::AllocaInst *>(instruction);
        expect(allocation->is_local());
        observed.snapshot_allocations++;
        observed.snapshot_bytes_per_worker += allocation->type()->size();
    });
    expect_same_resources(analysis.resources, observed);
    expect_same_resources(lowered.resources, observed);
    expect(xir::xir_verify_module(lowered.module.get(), {.require_reachable_blocks = true}).succeeded());
    return lowered;
}

template<typename T>
void test_scalar_snapshot_sizes() {
    for (auto lanes : {1u, 32u, 64u}) {
        for (auto width : {31u, 65u, 129u}) {
            // Current packet admission requires a nonunit local axis at least
            // one packet wide. Small complete-program Tiles remain covered.
            if (lanes > 1u && width < lanes) { continue; }
            auto kernel = copy_fixture<T>(width);
            auto lowered = check_resources(kernel, {.block_size = 128u, .local_lanes = lanes});
            if (!lowered) { continue; }
            auto materialized = width > 64u || (lanes > 1u && width >= lanes);
            auto count = lanes > 1u && width >= lanes ? ceil_div(width, lanes) : width;
            expect(eq(lowered.resources.snapshot_allocations, materialized ? uint64_t{1u} : uint64_t{0u}));
            expect(eq(lowered.resources.snapshot_bytes_per_worker, materialized ? static_cast<uint64_t>(count) * sizeof(T) : uint64_t{0u}));
        }
    }
}

struct RecordingCostPolicy final : bx::ExecutionCostPolicy {
    bool prefer_large_blocks{false};
    bool prefer_complete_programs{false};
    mutable vector<bx::ExecutionWork> observed_work;
    mutable vector<bx::ExecutionPlan> observed_candidates;
    [[nodiscard]] bx::ExecutionCost evaluate(bx::ExecutionTarget, const bx::ExecutionPlan &candidate,
                                             const bx::ExecutionWork &work, const bx::ExecutionCostModel &) const noexcept override {
        observed_work.emplace_back(work);
        observed_candidates.emplace_back(candidate);
        if (prefer_complete_programs) { return {.score = candidate.local_lanes == 1u ? 0.0 : 1.0}; }
        auto width_cost = prefer_large_blocks ? 2048u - candidate.block_size : candidate.block_size;
        return {.score = static_cast<double>(work.critical_packets) * 4096.0 + width_cost};
    }
};

struct MockGpuTargetInfo final : bx::ExecutionTargetInfo {
    uint32_t physical_width{32u};
    vector<uint32_t> proposed_blocks{32u, 64u, 128u};
    bool local_distribution{true};
    bool reject_all{false};
    uint32_t rejected_block{0u};
    uint64_t snapshot_budget{262144u};
    uint32_t constrained_block{0u};
    uint64_t constrained_block_budget{0u};
    mutable vector<uint32_t> admission_checks;
    mutable vector<bx::ExecutionWork> schedule_inputs;
    RecordingCostPolicy policy;

    // Zero CPU counts are intentional: a GPU must not inherit thread-pool
    // validation, home chunks, or caller-thread scheduling.
    [[nodiscard]] bx::ExecutionTarget target() const noexcept override { return {physical_width, 0u, 0u}; }
    [[nodiscard]] vector<uint32_t> block_sizes() const noexcept override { return proposed_blocks; }
    [[nodiscard]] bool supports_local_distribution() const noexcept override { return local_distribution; }
    [[nodiscard]] bool accepts(const bx::ExecutionPlan &candidate) const noexcept override {
        admission_checks.emplace_back(candidate.block_size);
        return !reject_all && candidate.block_size != rejected_block;
    }
    [[nodiscard]] bx::ExecutionResourceLimits resource_limits(const bx::ExecutionPlan &candidate) const noexcept override {
        return {.max_snapshot_bytes_per_worker = candidate.block_size == constrained_block ? constrained_block_budget : snapshot_budget};
    }
    [[nodiscard]] bx::ExecutionWork schedule(const bx::ExecutionPlan &, bx::ExecutionWork work) const noexcept override {
        schedule_inputs.emplace_back(work);
        // A deliberately simple three-way GPU scheduling mock, not a measured
        // hardware model. CPU task fields remain zero and are never consumed.
        work.active_workers = 3u;
        work.critical_packets = ceil_div(work.packet_count, uint64_t{3u});
        work.critical_blocks = ceil_div(work.block_count, uint64_t{3u});
        return work;
    }
    [[nodiscard]] const bx::ExecutionCostPolicy &cost_policy() const noexcept override { return policy; }
};

void expect_same_plan(const bx::ExecutionPlan &a, const bx::ExecutionPlan &b) {
    expect(eq(a.block_size, b.block_size));
    expect(a.root_axis_order == b.root_axis_order);
    expect(a.root_axis_tiles == b.root_axis_tiles);
    expect(eq(a.dispatch_size, b.dispatch_size));
    expect(eq(a.local_lanes, b.local_lanes));
    expect(eq(a.blocks_per_task, b.blocks_per_task));
    expect_same_resources(a.resources, b.resources);
    expect(eq(a.resource_limits.max_snapshot_bytes_per_worker, b.resource_limits.max_snapshot_bytes_per_worker));
    auto same_cost = [](double x, double y) { expect(std::abs(x - y) < 1e-12); };
    same_cost(a.cost.arithmetic_work, b.cost.arithmetic_work);
    same_cost(a.cost.memory_work, b.cost.memory_work);
    same_cost(a.cost.dispatch_work, b.cost.dispatch_work);
    same_cost(a.cost.imbalance_work, b.cost.imbalance_work);
    same_cost(a.cost.score, b.cost.score);
    same_cost(a.cost.task_dispatch_work, b.cost.task_dispatch_work);
    same_cost(a.cost.activation_work, b.cost.activation_work);
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    "tile_xir_gpu_physical_packet_abi_and_ragged_rows"_test = [] {
        for (auto packet : {32u, 64u}) {
            for (auto width : {65u, 129u}) {
                for (auto reduction : {false, true}) {
                    auto kernel = row_fixture(width, reduction);
                    MockGpuTargetInfo info;
                    info.physical_width = packet;
                    auto planned = bx::plan(kernel.function(), info, {.block_size = 2u * packet, .local_lanes = packet});
                    expect(planned.ok()) << planned.error;
                    if (!planned) { continue; }
                    expect(eq(planned.candidates.size(), size_t{1u}));
                    expect(eq(planned.selected.local_lanes, packet));
                    expect(eq(planned.selected.dispatch_size, 17u * packet));
                    auto lowered = bx::lower(kernel.function(), {.block_size = planned.selected.block_size,
                                                                 .root_axis_order = planned.selected.root_axis_order,
                                                                 .local_lanes = planned.selected.local_lanes});
                    expect(lowered.ok()) << lowered.error;
                    if (!lowered) { continue; }
                    expect(eq(lowered.required_packet_width, packet));
                    expect(eq(lowered.dispatch_size, planned.selected.dispatch_size));
                    expect(eq(lowered.function->block_size().x, planned.selected.block_size));
                    // Attribute payloads use double storage, but these FP32
                    // literals must not introduce any FP64 XIR requirement.
                    for (auto constant : lowered.module->constant_list()) {
                        expect(!constant->type()->is_float64());
                    }
                    expect(xir::xir_verify_module(lowered.module.get(), {.require_reachable_blocks = true}).succeeded());
                    auto complete_program = bx::lower(kernel.function(), {.block_size = 2u * packet});
                    expect(complete_program.ok()) << complete_program.error;
                    if (complete_program) {
                        expect(eq(complete_program.required_packet_width, 0u));
                        expect(eq(complete_program.dispatch_size, 17u));
                        expect(xir::xir_verify_module(complete_program.module.get(), {.require_reachable_blocks = true}).succeeded());
                    }
                }
            }
        }
    };

    "tile_xir_backend_candidates_are_filtered_deduplicated_and_scheduled"_test = [] {
        auto kernel = row_fixture(65u);
        MockGpuTargetInfo info;
        info.proposed_blocks = {128u, 0u, 32u, 16u, 64u, 32u, UINT32_MAX};
        info.rejected_block = 64u;
        auto planned = bx::plan(kernel.function(), info, {.max_candidates = 3u});
        expect(planned.ok()) << planned.error;
        if (!planned) { return; }
        expect(info.admission_checks == vector<uint32_t>{32u, 64u, 128u});
        expect(eq(planned.candidates.size(), size_t{2u}));
        expect(eq(planned.selected.block_size, 32u));
        expect(eq(info.schedule_inputs.size(), size_t{2u}));
        expect(eq(info.policy.observed_work.size(), info.schedule_inputs.size()));
        for (size_t i = 0u; i < info.schedule_inputs.size(); i++) {
            const auto &before = info.schedule_inputs[i];
            expect(eq(before.packet_count, uint64_t{1u}));
            expect(eq(before.block_count, uint64_t{1u}));
            expect(eq(before.task_count, uint64_t{0u}));
            expect(eq(before.active_workers, 0u));
            expect(eq(before.blocks_per_task, 0u));
            expect(eq(before.critical_packets, uint64_t{0u}));
            expect(eq(before.critical_blocks, uint64_t{0u}));
            expect(eq(before.critical_tasks, uint64_t{0u}));
            if (i >= info.policy.observed_work.size()) { continue; }
            const auto &after = info.policy.observed_work[i];
            expect(eq(after.active_workers, 3u));
            expect(eq(after.critical_packets, uint64_t{1u}));
            expect(eq(after.critical_blocks, uint64_t{1u}));
            expect(eq(after.task_count, uint64_t{0u}));
            expect(eq(after.blocks_per_task, 0u));
            expect(eq(after.critical_tasks, uint64_t{0u}));
            expect(std::abs(after.arithmetic_per_packet - before.arithmetic_per_packet) < 1e-12);
            expect(std::abs(after.memory_per_packet - before.memory_per_packet) < 1e-12);
        }
    };

    "tile_xir_custom_cost_cannot_legalize_backend_rejections"_test = [] {
        auto kernel = row_fixture(65u);
        MockGpuTargetInfo info;
        RecordingCostPolicy override_policy;
        override_policy.prefer_large_blocks = true;
        auto options = bx::PlannerOptions{.cost_policy = &override_policy};
        auto planned = bx::plan(kernel.function(), info, options);
        expect(planned.ok()) << planned.error;
        if (planned) { expect(eq(planned.selected.block_size, 128u)); }
        expect(info.policy.observed_work.empty());
        expect(eq(override_policy.observed_work.size(), size_t{3u}));
        override_policy.observed_work.clear();
        info.reject_all = true;
        auto rejected = bx::plan(kernel.function(), info, options);
        expect(!rejected.ok());
        expect(rejected.error.find("rejected every") != string::npos);
        expect(eq(rejected.rejected.size(), size_t{3u}));
        for (const auto &rejection : rejected.rejected) { expect(!rejection.reason.empty()); }
        expect(override_policy.observed_work.empty());
        info.reject_all = false;
        options.block_size = 33u;
        rejected = bx::plan(kernel.function(), info, options);
        expect(!rejected.ok());
        expect(rejected.error.find("block width") != string::npos);
        expect(override_policy.observed_work.empty());
        options.block_size = 0u;
        info.proposed_blocks = {0u, 16u, 33u, UINT32_MAX};
        rejected = bx::plan(kernel.function(), info, options);
        expect(!rejected.ok());
        expect(rejected.error.find("no legal block widths") != string::npos);
        expect(override_policy.observed_work.empty());
    };

    "tile_xir_disabled_local_distribution_falls_back_or_reports_error"_test = [] {
        auto kernel = row_fixture(129u);
        MockGpuTargetInfo info;
        info.local_distribution = false;
        auto automatic = bx::plan(kernel.function(), info, {.local_lanes = 0u});
        expect(automatic.ok()) << automatic.error;
        if (automatic) {
            expect(eq(automatic.candidates.size(), size_t{3u}));
            for (const auto &candidate : automatic.candidates) { expect(eq(candidate.local_lanes, 1u)); }
        }
        auto forced = bx::plan(kernel.function(), info, {.local_lanes = 32u});
        expect(!forced.ok());
        expect(forced.error.find("local-axis distribution") != string::npos);
        for (auto options : {bx::PlannerOptions{.blocks_per_task = 1u}, bx::PlannerOptions{.search_task_grain = true}}) {
            auto grain = bx::plan(kernel.function(), info, options);
            expect(!grain.ok());
            expect(grain.error.find("task-grain") != string::npos);
        }
        for (auto packet : {0u, 3u}) {
            info.physical_width = packet;
            auto invalid = bx::plan(kernel.function(), info);
            expect(!invalid.ok());
            expect(invalid.error.find("invalid XIR target") != string::npos);
        }
    };

    "tile_xir_thread_pool_target_info_preserves_legacy_overload"_test = [] {
        auto kernel = row_fixture(65u, true, 257u);
        auto default_target = bx::plan(kernel.function(), {});
        expect(default_target.ok()) << default_target.error;
        bx::ExecutionTarget target{8u, 8u, 32u};
        bx::ThreadPoolExecutionTargetInfo info{target};
        expect(info.supports_task_grain());
        for (auto options : {bx::PlannerOptions{},
                             bx::PlannerOptions{.block_size = 64u, .local_lanes = 0u},
                             bx::PlannerOptions{.block_size = 32u, .local_lanes = 0u, .search_task_grain = true},
                             bx::PlannerOptions{.block_size = 32u, .local_lanes = 8u, .blocks_per_task = 3u}}) {
            auto legacy = bx::plan(kernel.function(), target, options);
            auto explicit_info = bx::plan(kernel.function(), info, options);
            expect(legacy.ok() && explicit_info.ok()) << legacy.error << explicit_info.error;
            if (!legacy || !explicit_info) { continue; }
            expect_same_plan(legacy.selected, explicit_info.selected);
            expect(eq(legacy.candidates.size(), explicit_info.candidates.size()));
            for (size_t i = 0u; i < legacy.candidates.size() && i < explicit_info.candidates.size(); i++) {
                expect_same_plan(legacy.candidates[i], explicit_info.candidates[i]);
            }
        }
        auto work = info.schedule({.block_size = 32u, .dispatch_size = 257u},
                                  {.arithmetic_per_packet = 3.0, .memory_per_packet = 7.0, .packet_count = 33u, .block_count = 9u});
        expect(eq(work.task_count, uint64_t{9u}));
        expect(eq(work.active_workers, 8u));
        expect(eq(work.blocks_per_task, 1u));
        expect(eq(work.critical_packets, uint64_t{5u}));
        expect(eq(work.critical_blocks, uint64_t{2u}));
        expect(eq(work.critical_tasks, uint64_t{2u}));
    };

    "tile_xir_invalid_root_mapping_is_recoverable_before_resource_admission"_test = [] {
        auto check_invalid = [](const tile::Kernel &kernel, const vector<uint32_t> &order,
                                const vector<uint32_t> &tiles, string_view diagnostic) {
            expect(kernel.valid());
            if (!kernel.valid()) { return; }
            for (auto lanes : {1u, 32u}) {
                MockGpuTargetInfo info;
                auto planned = bx::plan(kernel.function(), info, {.block_size = 64u, .root_axis_order = order, .local_lanes = lanes, .root_axis_tiles = tiles});
                expect(!planned.ok());
                expect(planned.error == diagnostic) << planned.error;
                expect(planned.candidates.empty());
                expect(info.admission_checks.empty());
                expect(info.schedule_inputs.empty());
                expect(info.policy.observed_work.empty());
                bx::LowerOptions options{.block_size = 64u,
                                         .root_axis_order = order,
                                         .local_lanes = lanes,
                                         .root_axis_tiles = tiles};
                auto analysis = bx::analyze_resources(kernel.function(), options);
                expect(!analysis.ok());
                expect(analysis.error == diagnostic) << analysis.error;
                auto lowered = bx::lower(kernel.function(), options);
                expect(!lowered.ok());
                expect(lowered.error == diagnostic) << lowered.error;
                expect(lowered.module == nullptr);
                expect(lowered.function == nullptr);
            }
        };
        constexpr auto divisors = "XIR root axis tiles must be positive divisors of the original extents";
        constexpr auto rank = "XIR root axis tiles must specify every original axis";
        constexpr auto permutation = "XIR root axis order must be a complete permutation";
        auto rows = copy_fixture<float>(65u);
        // Prime extent 17 exposes nondividing factors before the internal
        // mixed-radix mapper can assert, even for a forced packet realization.
        check_invalid(rows, {}, {2u}, divisors);
        check_invalid(rows, {}, {0u}, divisors);
        check_invalid(rows, {}, {1u, 1u}, rank);
        check_invalid(rows, {1u}, {}, permutation);
        check_invalid(rows, {0u, 0u}, {}, permutation);
        using namespace tile;
        auto grid = tile_kernel("resource_root_grid", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
                        auto a = axis("a", 3), b = axis("b", 17);
                        auto m = axis("m", 1), n = axis("n", 65);
                        for (auto &nest : parallel(shape(a, b))) {
                            auto origin = coord(nest.index(a) * 17 + nest.index(b), 0);
                            output(origin, shape(m, n)).store(input[origin, shape(m, n)]);
                        }
                    }).capture(tensor_shape(51, 65), tensor_shape(51, 65));
        check_invalid(grid, {}, {1u}, rank);
        check_invalid(grid, {}, {1u, 1u, 1u}, rank);
        check_invalid(grid, {}, {1u, 2u}, divisors);
        check_invalid(grid, {}, {1u, 0u}, divisors);
        check_invalid(grid, {0u}, {1u, 17u}, permutation);
        check_invalid(grid, {0u, 0u}, {1u, 17u}, permutation);
        check_invalid(grid, {0u, 2u}, {1u, 17u}, permutation);
    };

    "tile_xir_resource_budget_boundary_and_backend_dependent_limits"_test = [] {
        auto kernel = copy_fixture<float>(129u);
        constexpr auto bytes = uint64_t{129u * sizeof(float)};
        auto analysis = bx::analyze_resources(kernel.function(), {.max_local_bytes = 0u});
        expect(analysis.ok()) << analysis.error;
        if (!analysis) { return; }
        // Analysis reports demand independently of a caller's budget.
        expect_same_resources(analysis.resources, {bytes, 1u});
        for (auto budget : {bytes - 1u, bytes, bytes + 1u}) {
            MockGpuTargetInfo info;
            info.snapshot_budget = budget;
            auto planned = bx::plan(kernel.function(), info, {.block_size = 64u});
            expect(eq(planned.ok(), budget >= bytes)) << planned.error;
            if (planned) {
                expect_same_resources(planned.selected.resources, analysis.resources);
                expect(eq(planned.selected.resource_limits.max_snapshot_bytes_per_worker, budget));
                expect(planned.rejected.empty());
                expect(eq(info.schedule_inputs.size(), size_t{1u}));
                expect(eq(info.policy.observed_work.size(), size_t{1u}));
            } else {
                expect(planned.candidates.empty());
                expect(eq(planned.rejected.size(), size_t{1u}));
                for (const auto &rejection : planned.rejected) {
                    expect(eq(rejection.candidate.block_size, 64u));
                    expect_same_resources(rejection.candidate.resources, analysis.resources);
                    expect(eq(rejection.candidate.resource_limits.max_snapshot_bytes_per_worker, budget));
                    expect(!rejection.reason.empty());
                }
                expect(info.schedule_inputs.empty());
                expect(info.policy.observed_work.empty());
            }
            auto lowered = bx::lower(kernel.function(), {.max_local_bytes = static_cast<uint32_t>(budget)});
            expect(eq(lowered.ok(), budget >= bytes)) << lowered.error;
            if (lowered) {
                expect_same_resources(lowered.resources, analysis.resources);
            } else {
                expect(lowered.module == nullptr && lowered.function == nullptr);
                expect(lowered.error.find("snapshot storage budget") != string::npos) << lowered.error;
            }
        }
        auto small = copy_fixture<float>(31u);
        MockGpuTargetInfo zero_budget;
        zero_budget.snapshot_budget = 0u;
        auto expanded = bx::plan(small.function(), zero_budget, {.block_size = 64u});
        expect(expanded.ok()) << expanded.error;
        if (expanded) { expect_same_resources(expanded.selected.resources, {}); }
        auto no_storage = check_resources(small, {.max_local_bytes = 0u});
        if (no_storage) { expect_same_resources(no_storage.resources, {}); }

        MockGpuTargetInfo info;
        info.snapshot_budget = bytes;
        info.constrained_block = 128u;
        info.constrained_block_budget = bytes - 1u;
        RecordingCostPolicy policy;
        policy.prefer_large_blocks = true;
        auto planned = bx::plan(kernel.function(), info, {.cost_policy = &policy});
        expect(planned.ok()) << planned.error;
        if (!planned) { return; }
        expect(eq(planned.selected.block_size, 64u));// 128 would win, but is not resource-admissible.
        expect(eq(planned.candidates.size(), size_t{2u}));
        expect(eq(planned.rejected.size(), size_t{1u}));
        expect(eq(info.schedule_inputs.size(), size_t{2u}));
        expect(eq(policy.observed_candidates.size(), size_t{2u}));
        expect(info.policy.observed_work.empty());
        for (const auto &candidate : policy.observed_candidates) {
            expect(candidate.block_size != 128u);
            expect_same_resources(candidate.resources, analysis.resources);
        }
    };

    "tile_xir_resource_admission_filters_auto_but_never_relaxes_fixed_distribution"_test = [] {
        auto kernel = copy_fixture<float>(129u);
        for (auto packet : {32u, 64u}) {
            auto packet_bytes = static_cast<uint64_t>(ceil_div(129u, packet)) * sizeof(float);
            MockGpuTargetInfo info;
            info.physical_width = packet;
            info.proposed_blocks = {packet, 2u * packet};
            info.snapshot_budget = packet_bytes;
            info.policy.prefer_complete_programs = true;
            auto automatic = bx::plan(kernel.function(), info, {.local_lanes = 0u});
            expect(automatic.ok()) << automatic.error;
            if (automatic) {
                expect(eq(automatic.selected.local_lanes, packet));
                expect_same_resources(automatic.selected.resources, {packet_bytes, 1u});
                expect(eq(automatic.candidates.size(), size_t{2u}));
                expect(eq(automatic.rejected.size(), size_t{2u}));
                for (const auto &rejection : automatic.rejected) {
                    expect(eq(rejection.candidate.local_lanes, 1u));
                    expect_same_resources(rejection.candidate.resources, {129u * sizeof(float), 1u});
                    expect(!rejection.reason.empty());
                }
            }
            for (const auto &candidate : info.policy.observed_candidates) { expect(eq(candidate.local_lanes, packet)); }
            info.schedule_inputs.clear();
            info.policy.observed_work.clear();
            info.policy.observed_candidates.clear();
            auto forced = bx::plan(kernel.function(), info, {.local_lanes = 1u});
            expect(!forced.ok());
            expect(forced.candidates.empty());
            expect(eq(forced.rejected.size(), size_t{2u}));
            expect(!forced.error.empty());
            expect(info.schedule_inputs.empty());
            expect(info.policy.observed_work.empty());
            for (const auto &rejection : forced.rejected) { expect(eq(rejection.candidate.local_lanes, 1u)); }
            info.snapshot_budget = 0u;
            auto exhausted = bx::plan(kernel.function(), info, {.local_lanes = 0u});
            expect(!exhausted.ok());
            expect(exhausted.candidates.empty());
            expect(eq(exhausted.rejected.size(), size_t{4u}));
            expect(info.schedule_inputs.empty());
            expect(info.policy.observed_work.empty());
            for (const auto &rejection : exhausted.rejected) { expect(!rejection.reason.empty()); }
        }
    };

    "tile_xir_static_resources_match_actual_allocations_across_types_and_packets"_test = [] {
        test_scalar_snapshot_sizes<uint8_t>();
        test_scalar_snapshot_sizes<half>();
        test_scalar_snapshot_sizes<tile::bfloat16>();
        test_scalar_snapshot_sizes<float>();
        test_scalar_snapshot_sizes<double>();// Generic XIR storage accounting, not a Metal FP64 capability claim.
        for (auto rows : {1u, 17u, 257u}) {
            auto kernel = copy_fixture<float>(129u, rows);
            auto lowered = check_resources(kernel);
            if (lowered) { expect_same_resources(lowered.resources, {129u * sizeof(float), 1u}); }
        }
        // Resource demand is 64-bit even though a logical extent is uint32.
        auto huge = copy_fixture<float>(UINT32_MAX, 1u);
        auto analysis = bx::analyze_resources(huge.function());
        expect(analysis.ok()) << analysis.error;
        if (analysis) { expect_same_resources(analysis.resources, {uint64_t{UINT32_MAX} * sizeof(float), 1u}); }
    };

    "tile_xir_static_resources_count_carry_storage_not_runtime_iterations"_test = [] {
        using namespace tile;
        for (auto pipelined : {false, true}) {
            for (auto iterations : {0u, 1u, 3u, 129u}) {
                auto kernel = tile_kernel("resource_carry", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                                  auto m = axis("m", 1), n = axis("n", 65);
                                  for (auto &nest : parallel(shape(17))) {
                                      auto acc = input[coord(nest.index(), 0), shape(m, n)];
                                      if (pipelined) {
                                          for (auto &step : nest.pipeline(shape(iterations))) {
                                              step.stage("update");
                                              acc = acc + 1.0f;
                                          }
                                      } else {
                                          for (auto &step : nest.serial(shape(iterations))) {
                                              static_cast<void>(step);
                                              acc = acc + 1.0f;
                                          }
                                      }
                                      output(coord(nest.index(), 0), shape(m, n)).store(acc);
                                  }
                              }).capture(tensor_shape(17, 65), tensor_shape(17, 65));
                auto lowered = check_resources(kernel);
                // One input plus current/next carry. The emitted loop body is
                // static even for zero trips; this is not peak live storage.
                if (lowered) { expect_same_resources(lowered.resources, {3u * 65u * sizeof(float), 3u}); }
            }
        }
    };

    "tile_xir_static_resources_follow_map_and_fusion_materialization"_test = [] {
        using namespace tile;
        auto map_kernel = tile_kernel("resource_map", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
                              auto m = axis("m", 1), n = axis("n", 65);
                              for (auto &nest : parallel(shape(17))) {
                                  auto x = input[coord(nest.index(), 0), shape(m, n)];
                                  auto reversed = reindex(x, shape(m, n), [&](const Nest &element) {
                                      return coord(element.index(m), 64 - element.index(n));
                                  });
                                  output(coord(nest.index(), 0), shape(m, n)).store(reversed);
                              }
                          }).capture(tensor_shape(17, 65), tensor_shape(17, 65));
        auto map_off = check_resources(map_kernel);
        auto map_on = check_resources(map_kernel, {.enable_map_fusion = true});
        if (map_off && map_on) {
            expect(eq(map_on.deferred_maps, 1u));
            expect(eq(map_off.resources.snapshot_bytes_per_worker - map_on.resources.snapshot_bytes_per_worker, uint64_t{65u * sizeof(float)}));
            expect(eq(map_off.resources.snapshot_allocations - map_on.resources.snapshot_allocations, uint64_t{1u}));
        }
        for (auto expression : {false, true}) {
            for (auto retained : {false, true}) {
                for (auto lanes : {1u, 32u, 64u}) {
                    auto kernel = tile_kernel("resource_fused_reduce", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                                      auto m = axis("m", 1), n = axis("n", 65);
                                      for (auto &nest : parallel(shape(17))) {
                                          auto x = input[coord(nest.index(), 0), shape(m, n)];
                                          auto y = expression ? exp(x) : x;
                                          auto sum = Scalar<float>{0.0f};
                                          for (auto &element : nest.reduce(shape(n))) {
                                              sum += y.at(coord(0, element.index())) * y.at(coord(0, element.index()));
                                          }
                                          output(coord(nest.index(), 0), shape(m, n)).store(retained ? y + sum : full<float>(shape(m, n), sum));
                                      }
                                  }).capture(tensor_shape(17, 65), tensor_shape(17, 65));
                    bx::LowerOptions options{.block_size = 128u, .local_lanes = lanes};
                    auto off = check_resources(kernel, options);
                    options.enable_load_reduction_fusion = !expression;
                    options.enable_expression_reduction_fusion = expression;
                    auto on = check_resources(kernel, options);
                    if (!off || !on) { continue; }
                    expect(eq(expression ? on.fused_reduction_expressions : on.fused_reduction_loads, 1u));
                    expect(eq(expression ? on.elided_expression_snapshots : on.elided_load_snapshots, retained ? 0u : 1u));
                    auto bytes = static_cast<uint64_t>(lanes == 1u ? 65u : ceil_div(65u, lanes)) * sizeof(float);
                    expect(eq(off.resources.snapshot_bytes_per_worker - on.resources.snapshot_bytes_per_worker, retained ? uint64_t{0u} : bytes));
                }
            }
        }
        auto pointwise_kernel = row_fixture(65u, false);
        auto pointwise_off = check_resources(pointwise_kernel);
        auto pointwise_on = check_resources(pointwise_kernel, {.enable_pointwise_fusion = true});
        if (pointwise_off && pointwise_on) {
            expect(gt(pointwise_on.fused_pointwise_regions, 0u));
            // Runtime-disjoint streaming does not erase the compiled aliasing
            // fallback's allocation sites from this static resource contract.
            expect_same_resources(pointwise_on.resources, pointwise_off.resources);
        }
    };

    "tile_xir_static_resources_count_each_partial_reduction_body_emission"_test = [] {
        using namespace tile;
        auto kernel = tile_kernel("resource_partial_reduction", [](TensorView<float, 1> output) {
                          for (auto &nest : parallel(shape(1))) {
                              auto sum = Scalar<float>{0.0f};
                              for (auto &element : nest.reduce(shape(65))) {
                                  auto constants = full<float>(shape(17), 2.0f);
                                  sum += constants.at(coord(element.index() % 17));
                              }
                              output(nest.index()).store(sum);
                          }
                      }).capture(tensor_shape(1));
        for (auto partitions : {1u, 4u}) {
            auto lowered = check_resources(kernel, {.reduction_partitions = partitions});
            if (!lowered) { continue; }
            auto allocations = partitions == 1u ? uint64_t{1u} : uint64_t{9u};
            // p4 emits 4 seeds + 4 recurrence contributions + 1 tail. The
            // constant Tile is dynamically indexed, hence one 68-byte snapshot
            // per emitted definition: 9 * 68 = 612, not 65 * 68 or one * 68.
            expect_same_resources(lowered.resources, {allocations * 17u * sizeof(float), allocations});
        }
    };
    return 0;
}

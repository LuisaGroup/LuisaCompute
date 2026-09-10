// Test backend-supplied XIR execution capabilities and scheduling.
// Covers physical GPU packet widths, legality before costs, and CPU parity.

#include "ut/ut.hpp"
#include <luisa/ast/type.h>
#include <luisa/core/mathematics.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/dsl.h>
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

struct RecordingCostPolicy final : bx::ExecutionCostPolicy {
    bool prefer_large_blocks{false};
    mutable vector<bx::ExecutionWork> observed_work;
    [[nodiscard]] bx::ExecutionCost evaluate(bx::ExecutionTarget, const bx::ExecutionPlan &candidate,
                                             const bx::ExecutionWork &work, const bx::ExecutionCostModel &) const noexcept override {
        observed_work.emplace_back(work);
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
    return 0;
}

// Test backend-supplied XIR execution capabilities and scheduling.
// Covers physical GPU packet widths, resource admission before costs, CPU
// parity, and static resource analysis against independently inspected XIR.

#include "ut/ut.hpp"
#include <luisa/ast/type.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/stl/format.h>
#include <luisa/tile/algorithms.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/dsl.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/metadata/strided_mma.h>
#include <luisa/xir/metadata/contiguous_copy.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/verifier.h>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
namespace bx = tile::bridge::xir;

template<typename Target>
concept PlannerTarget = requires(const tile::Function &function, const Target &target, const bx::PlannerOptions &options) {
    { bx::plan(function, target, options) } noexcept -> std::same_as<bx::PlanningResult>;
    { bx::plan(function, target) } noexcept -> std::same_as<bx::PlanningResult>;
};

using PlannerEntry = bx::PlanningResult (*)(const tile::Function &, const bx::ExecutionTargetInfo &, const bx::PlannerOptions &) noexcept;
static_assert(std::is_same_v<decltype(&bx::plan), PlannerEntry>);
static_assert(PlannerTarget<bx::ExecutionTargetInfo>);
static_assert(PlannerTarget<bx::ThreadPoolExecutionTargetInfo>);
static_assert(!PlannerTarget<bx::ExecutionTarget>);

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
        for (auto width : {1u, 7u, 31u, 65u, 129u}) {
            auto kernel = copy_fixture<T>(width);
            auto lowered = check_resources(kernel, {.block_size = 128u, .local_lanes = lanes});
            if (!lowered) { continue; }
            auto materialized = width > 64u || (lanes > 1u && width > 1u);
            auto count = lanes > 1u && width > 1u ? ceil_div(width, lanes) : width;
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
    mutable vector<bx::ExecutionPlan> admission_candidates;
    mutable vector<bx::ExecutionWork> schedule_inputs;
    RecordingCostPolicy policy;

    // Zero CPU counts are intentional: a GPU must not inherit thread-pool
    // validation, home chunks, or caller-thread scheduling.
    [[nodiscard]] bx::ExecutionTarget target() const noexcept override { return {physical_width, 0u, 0u}; }
    [[nodiscard]] vector<uint32_t> block_sizes() const noexcept override { return proposed_blocks; }
    [[nodiscard]] bool supports_local_distribution() const noexcept override { return local_distribution; }
    [[nodiscard]] bool accepts(const bx::ExecutionPlan &candidate) const noexcept override {
        admission_checks.emplace_back(candidate.block_size);
        admission_candidates.emplace_back(candidate);
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

struct NativeCpuTargetInfo final : bx::ThreadPoolExecutionTargetInfo {
    NativeCpuTargetInfo() noexcept : ThreadPoolExecutionTargetInfo{{8u, 1u}} {}
    [[nodiscard]] bool supports_native_mma_vector_width(uint32_t width) const noexcept override {
        return width == 2u || width == 4u || width == 8u;
    }
    [[nodiscard]] bool supports_native_copy_vector_width(uint32_t width) const noexcept override {
        return width == 2u || width == 4u || width == 8u;
    }
};

void expect_same_plan(const bx::ExecutionPlan &a, const bx::ExecutionPlan &b) {
    expect(eq(a.block_size, b.block_size));
    expect(a.root_axis_order == b.root_axis_order);
    expect(a.root_axis_tiles == b.root_axis_tiles);
    expect(eq(a.dispatch_size, b.dispatch_size));
    expect(eq(a.local_lanes, b.local_lanes));
    expect(eq(a.blocks_per_task, b.blocks_per_task));
    expect(eq(a.mma_output_block, b.mma_output_block));
    expect(eq(a.max_unrolled_mma_terms, b.max_unrolled_mma_terms));
    expect(eq(a.native_mma_vector_width, b.native_mma_vector_width));
    expect(eq(a.native_copy_vector_width, b.native_copy_vector_width));
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

    "tile_xir_native_copy_geometry_resources_and_policy_work"_test = [] {
        using namespace tile;
        NativeCpuTargetInfo info;
        for (auto rows : {1u, 2u}) {
            for (auto columns : {1u, 8u, 9u, 11u, 13u}) {
                auto kernel = tile_kernel("native_copy_geometry", [=](TensorView<const float, 3> input,
                                                                      TensorView<float, 3> output) {
                                  auto p = axis("p", 1), m = axis("m", rows), n = axis("n", columns);
                                  for (auto &nest : parallel(shape(17))) {
                                      // Dynamic logical row bounds cannot be replaced by
                                      // a flat buffer-range check, even for dense strides.
                                      auto offset = nest.index() % 5 * 3 - 3;
                                      auto value = input.tile(coord(nest.index(), 0, offset), shape(p, m, n), bounds::zero).load();
                                      output(coord(nest.index(), 0, 0), shape(p, m, n)).store(value);
                                  }
                              }).capture(tensor_shape(17, 2, 11), tensor_shape(17, rows, columns));
                auto count = static_cast<uint64_t>(rows) * columns;
                auto options = bx::LowerOptions{.block_size = 32u, .max_unrolled_tile_elements = 4u};
                auto baseline = check_resources(kernel, options);
                if (!baseline) { continue; }
                expect(eq(baseline.native_copies, 0u));
                expect_same_resources(baseline.resources, {count > 4u ? count * sizeof(float) : 0u, count > 4u ? 1u : 0u});
                RecordingCostPolicy baseline_policy;
                auto planner = bx::PlannerOptions{.block_size = 32u, .max_unrolled_tile_elements = 4u, .cost_policy = &baseline_policy};
                auto baseline_plan = bx::plan(kernel.function(), info, planner);
                expect(baseline_plan.ok()) << baseline_plan.error;
                expect(!baseline_policy.observed_work.empty());
                if (!baseline_plan || baseline_policy.observed_work.empty()) { continue; }
                expect(eq(baseline_plan.selected.native_copy_vector_width, 0u));
                expect(eq(baseline_policy.observed_work.front().native_copy_per_program.invocations, 0.0));
                for (auto width : {2u, 4u, 8u}) {
                    options.native_copy_vector_width = width;
                    auto candidate = check_resources(kernel, options);
                    if (!candidate) { continue; }
                    // The two-row subview is contiguous only when it spans
                    // whole physical rows. Oversize domains keep the fallback.
                    auto admitted = count > 4u && count >= width && columns <= 11u && (rows == 1u || columns == 11u);
                    expect(eq(candidate.native_copies, admitted ? 1u : 0u)) << "rows=" << rows << " columns=" << columns << " vector=" << width;
                    expect_same_resources(candidate.resources, baseline.resources);
                    expect(eq(candidate.required_packet_width, 0u));
                    expect(eq(candidate.dispatch_size, baseline.dispatch_size));
                    expect(static_cast<bool>(candidate.argument_usages == baseline.argument_usages));
                    auto calls = 0u;
                    candidate.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                        if (!instruction->isa<xir::CallInst>()) { return; }
                        auto call = static_cast<xir::CallInst *>(instruction);
                        auto md = call->callee()->find_metadata<xir::ContiguousCopyMD>();
                        expect(md != nullptr);
                        if (!md) { return; }
                        calls++;
                        expect(eq(md->descriptor.element_count, count));
                        expect(eq(md->descriptor.vector_width, width));
                        expect(eq(call->argument_count(), size_t{3u}));
                        if (call->argument_count() != 3u) { return; }
                        expect(call->argument(0u)->type()->is_buffer());
                        expect(call->argument(1u)->type()->is_uint64());
                        expect(call->argument(2u)->isa<xir::AllocaInst>());
                        expect(call->argument(2u)->type()->is_array());
                        expect(eq(call->argument(2u)->type()->size(), count * sizeof(float)));
                    });
                    expect(eq(calls, candidate.native_copies));
                    RecordingCostPolicy policy;
                    planner.native_copy_vector_width = width;
                    planner.cost_policy = &policy;
                    auto planned = bx::plan(kernel.function(), info, planner);
                    expect(planned.ok()) << planned.error;
                    expect(!policy.observed_work.empty());
                    if (!planned || policy.observed_work.empty()) { continue; }
                    expect(eq(planned.selected.native_copy_vector_width, width));
                    expect_same_resources(planned.selected.resources, candidate.resources);
                    auto &work = policy.observed_work.front();
                    auto &copy = work.native_copy_per_program;
                    // Conditional alternatives per logical program, not W8
                    // replicated bytes, additive cycles or a measured speedup.
                    expect(eq(copy.invocations, admitted ? 1.0 : 0.0));
                    expect(eq(copy.fastpath_vector_groups, admitted ? static_cast<double>(count / width) : 0.0));
                    expect(eq(copy.fastpath_tail_elements, admitted ? static_cast<double>(count % width) : 0.0));
                    expect(eq(copy.fallback_elements, admitted ? static_cast<double>(count) : 0.0));
                    expect(eq(work.arithmetic_per_packet, baseline_policy.observed_work.front().arithmetic_per_packet));
                    expect(eq(work.memory_per_packet, baseline_policy.observed_work.front().memory_per_packet));
                }
            }
        }
    };

    "tile_xir_native_copy_capability_and_materialization_exclusions"_test = [] {
        using namespace tile;
        auto kernel = copy_fixture<float>(9u);
        NativeCpuTargetInfo cpu;
        MockGpuTargetInfo gpu;
        auto rejected = bx::plan(kernel.function(), gpu, {.native_copy_vector_width = 4u});
        expect(!rejected.ok());
        expect(rejected.error.find("native copy vector width") != string::npos) << rejected.error;
        expect(gpu.admission_checks.empty());
        expect(gpu.policy.observed_work.empty());
        for (auto width : {1u, 3u, 16u, UINT32_MAX}) {
            RecordingCostPolicy policy;
            expect(!bx::plan(kernel.function(), cpu, {.native_copy_vector_width = width, .cost_policy = &policy}).ok());
            expect(policy.observed_work.empty());
        }
        // A width request cannot create an otherwise-unneeded allocation.
        auto expanded = check_resources(kernel, {.max_unrolled_tile_elements = 0u, .native_copy_vector_width = 4u});
        if (expanded) {
            expect(eq(expanded.native_copies, 0u));
            expect_same_resources(expanded.resources, {});
        }
        auto local = check_resources(copy_fixture<float>(65u), {.block_size = 32u, .native_copy_vector_width = 4u, .local_lanes = 8u});
        if (local) {
            expect(eq(local.native_copies, 0u));
            expect(eq(local.required_packet_width, 8u));
            expect_same_resources(local.resources, {ceil_div(uint64_t{65u}, uint64_t{8u}) * sizeof(float), 1u});
        }
        auto half_copy = check_resources(copy_fixture<half>(9u), {.max_unrolled_tile_elements = 4u, .native_copy_vector_width = 4u});
        if (half_copy) {
            expect(eq(half_copy.native_copies, 0u));
            expect_same_resources(half_copy.resources, {9u * sizeof(half), 1u});
        }
        auto reduced = tile_kernel("native_copy_producer_fusion", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
                           auto m = axis("m", 1), n = axis("n", 65);
                           for (auto &nest : parallel(shape(17))) {
                               auto value = input.tile(coord(nest.index(), 0), shape(m, n)).load();
                               auto sum = Scalar<float>{0.0f};
                               for (auto &element : nest.reduce(shape(n))) { sum += value.at(coord(0, element.index())); }
                               output(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), sum));
                           }
                       }).capture(tensor_shape(17, 65), tensor_shape(17, 65));
        auto fused_baseline = check_resources(reduced, {.enable_load_reduction_fusion = true});
        auto fused = check_resources(reduced, {.native_copy_vector_width = 4u, .enable_load_reduction_fusion = true});
        if (fused) {
            expect(eq(fused.fused_reduction_loads, 1u));
            expect(eq(fused.elided_load_snapshots, 1u));
            expect(eq(fused.native_copies, 0u));
            // Dynamic full(shape, sum) is a materialized output map, not a
            // constant splat. The load snapshot alone has been eliminated.
            expect_same_resources(fused.resources, {65u * sizeof(float), 1u});
            if (fused_baseline) { expect_same_resources(fused.resources, fused_baseline.resources); }
        }
    };

    "tile_xir_native_mma_capabilities_snapshots_and_work"_test = [] {
        using namespace tile;
        NativeCpuTargetInfo info;
        for (auto width : {2u, 4u, 8u}) {
            for (auto terms : {0u, 1u, 5u, 9u}) {
                for (auto columns : {1u, 5u}) {
                    for (auto transpose : {false, true}) {
                        for (auto strict : {false, true}) {
                            auto kernel = tile_kernel("native_mma_geometry", [=](TensorView<float, 3> output) {
                                              auto m = axis("m", 2), n = axis("n", columns), k = axis("k", terms);
                                              for (auto &nest : parallel(shape(17))) {
                                                  auto b = axis("b", 1);
                                                  auto lhs = full<float>(shape(b, m, k), 2.0f);
                                                  auto rhs = full<float>(transpose ? shape(b, n, k) : shape(b, k, n), 3.0f);
                                                  auto seed = full<float>(shape(b, m, n), 1.0f);
                                                  auto value = mma(lhs, rhs, seed, {.allow_reassociation = !strict});
                                                  output(coord(nest.index(), 0, 0), shape(b, m, n)).store(value);
                                              }
                                          }).capture(tensor_shape(17, 2, columns));
                            auto lowered = check_resources(kernel, {.block_size = 32u, .native_mma_vector_width = width});
                            if (!lowered) { continue; }
                            // Degenerate dimensions still use physical stride:
                            // K=1 makes either RHS order unit-stride; N=1 makes
                            // M the innermost output axis (with RHS broadcast).
                            auto inner_extent = columns > 1u ? columns : 2u;
                            auto output_mode = terms != 0u && inner_extent >= width &&
                                               (columns > 1u ? !transpose || terms == 1u : terms == 1u);
                            auto contraction_mode = !output_mode && !strict && terms >= width && (transpose || columns == 1u);
                            auto admitted = output_mode || contraction_mode;
                            expect(eq(lowered.native_mmas, admitted ? 1u : 0u));
                            expect(eq(lowered.native_output_mmas, output_mode ? 1u : 0u));
                            expect(eq(lowered.native_contraction_mmas, contraction_mode ? 1u : 0u));
                            auto calls = 0u;
                            lowered.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                                if (instruction->isa<xir::CallInst>()) {
                                    auto call = static_cast<xir::CallInst *>(instruction);
                                    auto md = call->callee()->find_metadata<xir::StridedMmaMD>();
                                    expect(md != nullptr);
                                    if (md) {
                                        calls++;
                                        expect(eq(md->descriptor.vector_width, width));
                                        expect(eq(md->descriptor.allow_reassociation, !strict));
                                        expect(md->descriptor.output_extents == vector<uint64_t>{1u, 2u, columns});
                                    }
                                }
                            });
                            expect(eq(calls, lowered.native_mmas));
                            RecordingCostPolicy policy;
                            auto options = bx::PlannerOptions{.block_size = 32u, .native_mma_vector_width = width, .cost_policy = &policy};
                            auto planned = bx::plan(kernel.function(), info, options);
                            expect(planned.ok()) << planned.error;
                            if (!planned || policy.observed_work.empty()) { continue; }
                            expect_same_resources(planned.selected.resources, lowered.resources);
                            expect(eq(planned.selected.native_mma_vector_width, width));
                            auto &work = policy.observed_work.front().mma_per_packet;
                            expect(eq(work.native_calls, admitted ? 1.0 : 0.0));
                            expect(eq(work.native_output_vector_updates, output_mode ? static_cast<double>(2u * columns / inner_extent * (inner_extent / width) * terms) : 0.0));
                            expect(eq(work.native_contraction_vector_updates, contraction_mode ? static_cast<double>(2u * columns * (terms / width)) : 0.0));
                            expect(eq(work.multiply_adds, static_cast<double>(2u * columns * terms)));
                            if (output_mode) {
                                auto groups = 2u * columns / inner_extent * (inner_extent / width + inner_extent % width);
                                expect(eq(work.loop_invocations, static_cast<double>(groups)));
                                expect(eq(work.loop_iterations, static_cast<double>(groups * terms)));
                            } else if (contraction_mode) {
                                auto chunks = terms / width, tail = terms % width;
                                expect(eq(work.loop_invocations, static_cast<double>(2u * columns * ((chunks > 1u) + (tail != 0u)))));
                                expect(eq(work.loop_iterations, static_cast<double>(2u * columns * (chunks - 1u + tail))));
                            }
                            if (admitted) {
                                expect(eq(lowered.resources.snapshot_allocations, uint64_t{4u}));
                                expect(eq(lowered.resources.snapshot_bytes_per_worker, static_cast<uint64_t>(2u * terms + terms * columns + 4u * columns) * sizeof(float)));
                            }
                        }
                    }
                }
            }
        }
        auto kernel = row_fixture(5u, false);
        MockGpuTargetInfo gpu;
        auto rejected = bx::plan(kernel.function(), gpu, {.native_mma_vector_width = 4u});
        expect(!rejected.ok());
        expect(rejected.error.find("native MMA vector width") != string::npos);
        expect(gpu.admission_checks.empty());
        expect(!bx::plan(kernel.function(), info, {.native_mma_vector_width = 3u}).ok());
    };

    "tile_xir_gpu_physical_packet_abi_and_ragged_rows"_test = [] {
        for (auto packet : {32u, 64u}) {
            for (auto width : {1u, 2u, 7u, 16u, 31u, 32u, 33u, 65u, 129u}) {
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

    "tile_xir_packet_mapping_preserves_unsupported_reduction_fallbacks"_test = [] {
        using namespace tile;
        // Short domains are supported now, but arbitrary cross-owner reads,
        // ordered folds and zero domains are not part of this executable slice.
        for (auto variant : {0u, 1u, 2u}) {
            auto kernel = tile_kernel("packet_reduction_fallback", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                              auto m = axis("m", 1), n = axis("n", 33);
                              for (auto &nest : parallel(shape(17))) {
                                  auto origin = coord(nest.index(), 0);
                                  auto x = input.tile(origin, shape(m, n)).load();
                                  auto sum = Scalar<float>{2.5f};
                                  auto domain = variant == 2u ? shape(0) : shape(n);
                                  auto policy = variant == 0u ? reduction::fold_left : reduction::unordered_tree;
                                  for (auto &step : nest.reduce(domain, policy)) {
                                      auto index = variant == 1u ? (step.index() + 1) % 33 : step.index();
                                      sum += x.at(coord(0, index));
                                  }
                                  output(origin, shape(m, n)).store(x + sum);
                              }
                          }).capture(tensor_shape(17, 33), tensor_shape(17, 33));
            expect(kernel.valid());
            if (!kernel.valid()) { continue; }
            for (auto packet : {32u, 64u}) {
                MockGpuTargetInfo info;
                info.physical_width = packet;
                info.proposed_blocks = {packet, 2u * packet};
                auto forced = bx::plan(kernel.function(), info, {.local_lanes = packet});
                expect(!forced.ok()) << "variant=" << variant << " packet=" << packet;
                expect(forced.error.find("local-axis distribution") != string::npos) << forced.error;
                auto automatic = bx::plan(kernel.function(), info, {.local_lanes = 0u});
                expect(automatic.ok()) << automatic.error;
                if (automatic) {
                    for (const auto &candidate : automatic.candidates) { expect(eq(candidate.local_lanes, 1u)); }
                    auto fallback = check_resources(kernel, {.block_size = automatic.selected.block_size});
                    expect(fallback.ok()) << fallback.error;
                }
            }
        }
    };

    "tile_xir_short_packet_reduction_validity_and_storage_are_explicit"_test = [] {
        for (auto packet : {32u, 64u}) {
            auto kernel = row_fixture(7u);
            auto lowered = check_resources(kernel, {.block_size = 2u * packet, .local_lanes = packet});
            if (!lowered) { continue; }
            uint32_t validity_shuffles = 0u;
            lowered.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                if (!instruction->isa<xir::ThreadGroupInst>()) { return; }
                auto op = static_cast<xir::ThreadGroupInst *>(instruction);
                if (op->op() == xir::ThreadGroupOp::WARP_READ_LANE && op->type()->is_uint32()) { validity_shuffles++; }
            });
            auto tree_levels = 0u;
            for (auto distance = 1u; distance < packet; distance *= 2u) { tree_levels++; }
            // One reduction, with payload plus integer has_value exchanged
            // at every tree level. A payload-only zero-padded tree fails here.
            expect(validity_shuffles >= tree_levels) << "packet=" << packet << " validity_shuffles=" << validity_shuffles;
            expect(lowered.resources.snapshot_allocations > 0u);
            // check_resources independently compares every actual Alloca type
            // against both the static estimator and lowerer's accounting.
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

    "tile_xir_thread_pool_target_info_explicit_and_polymorphic"_test = [] {
        auto kernel = row_fixture(65u, true, 257u);
        auto default_target = bx::plan(kernel.function(), bx::ThreadPoolExecutionTargetInfo{bx::ExecutionTarget{}});
        expect(default_target.ok()) << default_target.error;
        bx::ExecutionTarget target{8u, 8u, 32u};
        bx::ThreadPoolExecutionTargetInfo info{target};
        const bx::ExecutionTargetInfo &abstract_info = info;
        expect(info.supports_task_grain());
        for (auto options : {bx::PlannerOptions{},
                             bx::PlannerOptions{.block_size = 64u, .local_lanes = 0u},
                             bx::PlannerOptions{.block_size = 32u, .local_lanes = 0u, .search_task_grain = true},
                             bx::PlannerOptions{.block_size = 32u, .local_lanes = 8u, .blocks_per_task = 3u}}) {
            auto polymorphic = bx::plan(kernel.function(), abstract_info, options);
            auto explicit_info = bx::plan(kernel.function(), info, options);
            expect(polymorphic.ok() && explicit_info.ok()) << polymorphic.error << explicit_info.error;
            if (!polymorphic || !explicit_info) { continue; }
            expect_same_plan(polymorphic.selected, explicit_info.selected);
            expect(eq(polymorphic.candidates.size(), explicit_info.candidates.size()));
            for (size_t i = 0u; i < polymorphic.candidates.size() && i < explicit_info.candidates.size(); i++) {
                expect_same_plan(polymorphic.candidates[i], explicit_info.candidates[i]);
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

    "tile_xir_mma_output_blocking_preserves_storage_and_cost_admission"_test = [] {
        using namespace tile;
        auto fixture = [](uint32_t rows, uint32_t columns, uint32_t terms, uint32_t variant) {
            auto transposed = variant == 2u || variant == 4u;
            return tile_kernel("mma_output_blocks", [=](TensorView<const float, 2> a, TensorView<const float, 2> b, TensorView<float, 2> c) {
                       auto m = axis("m", rows), n = axis("n", columns), k = axis("k", terms);
                       for (auto &nest : parallel(shape(1))) {
                           static_cast<void>(nest);
                           auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                           auto rhs = b.tile(coord(0, 0), transposed ? shape(n, k) : shape(k, n)).load();
                           auto seed = c.tile(coord(0, 0), shape(m, n)).load();
                           auto value = variant == 1u || variant == 4u ? mma(rhs, lhs, seed, {.allow_reassociation = false}) :
                                                                         mma(lhs, rhs, seed, {.allow_reassociation = false});
                           if (variant == 3u) {
                               for (auto &element : nest.serial(shape(rows * columns))) {
                                   auto row = element.index() / columns, column = element.index() % columns;
                                   c(coord(row, column), shape(1, 1)).store(full<float>(shape(1, 1), value.at(coord(row, column))));
                               }
                           } else {
                               c(coord(0, 0), shape(m, n)).store(value);
                           }
                       }
                   })
                .capture(tensor_shape(rows, std::max(terms, 1u)), transposed ? tensor_shape(columns, std::max(terms, 1u)) : tensor_shape(std::max(terms, 1u), columns), tensor_shape(rows, columns));
        };
        auto text = [](const bx::NativeFunction &value) {
            string result;
            xir::XIRDebugPrinter{}.emit_function(result, value.function);
            return result;
        };
        for (auto variant : {0u, 1u, 2u, 3u, 4u}) {
            for (auto size : {std::array<uint32_t, 3>{2u, 5u, 3u}, {2u, 35u, 17u}, {1u, 7u, 65u}}) {
                auto kernel = fixture(size[0], size[1], size[2], variant);
                auto baseline = check_resources(kernel);
                auto baseline_plan = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 64u});
                if (!baseline || !baseline_plan) { continue; }
                for (auto width : {1u, 2u, 4u}) {
                    auto candidate = check_resources(kernel, {.mma_output_block = width});
                    auto plan = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 64u, .mma_output_block = width});
                    expect(plan.ok()) << plan.error;
                    if (!candidate || !plan) { continue; }
                    expect(eq(plan.selected.mma_output_block, width));
                    expect_same_resources(candidate.resources, baseline.resources);
                    expect_same_resources(plan.selected.resources, candidate.resources);
                    expect(eq(candidate.blocked_mmas, width == 1u || variant == 4u ? 0u : 1u));
                    // Grouping removes broadcast projections, not MUL/ADDs.
                    // The prior reflects that work without an assumed native
                    // speedup (code size and mask realization remain unpriced).
                    expect(plan.selected.cost.arithmetic_work <= baseline_plan.selected.cost.arithmetic_work);
                    expect(plan.selected.cost.memory_work <= baseline_plan.selected.cost.memory_work);
                    if (width == 1u || variant == 4u) {
                        expect(text(candidate) == text(baseline));
                        expect(eq(plan.selected.cost.score, baseline_plan.selected.cost.score));
                    }
                }
                auto budget = check_resources(kernel, {.max_unrolled_region_work = 1u, .mma_output_block = 4u});
                if (budget) {
                    expect(eq(budget.blocked_mmas, 0u));
                    expect(text(budget) == text(baseline));
                }
            }
        }
        // MMA K uses an additional unroll cap without changing the global
        // Tile threshold. Newly dynamic reads require explicit resource-plan
        // snapshots, even when output-block admission falls back.
        for (auto variant : {0u, 2u, 3u, 4u}) {
            for (auto terms : {0u, 1u, 8u, 9u, 16u, 64u, 65u}) {
                auto kernel = fixture(1u, 5u, terms, variant);
                auto reference_plan = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 64u});
                expect(reference_plan.ok()) << reference_plan.error;
                for (auto width : {1u, 2u, 4u}) {
                    auto baseline = check_resources(kernel, {.mma_output_block = width});
                    if (!baseline) { continue; }
                    for (auto cap : {0u, 1u, 8u, 64u, 128u}) {
                        auto candidate = check_resources(kernel, {.mma_output_block = width, .max_unrolled_mma_terms = cap});
                        auto options = bx::PlannerOptions{.block_size = 64u, .mma_output_block = width, .max_unrolled_mma_terms = cap};
                        auto planned = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, options);
                        expect(planned.ok()) << planned.error;
                        if (!candidate || !planned) { continue; }
                        auto rolled = terms > 64u || (cap != 0u && terms > cap);
                        expect(eq(candidate.rolled_mmas, rolled ? 1u : 0u));
                        // Only the broadcast-LHS candidate accepts a strided
                        // counterpart. The old symmetric candidate is unchanged.
                        auto strided_lhs = variant == 4u && terms != 1u;
                        expect(eq(candidate.blocked_mmas, width == 1u || strided_lhs ? 0u : 1u));
                        auto expected_resources = baseline.resources;
                        auto additional_roll = rolled && terms <= 64u;
                        if (additional_roll) {
                            // A is [1,K], B is [K,5] or [5,K]. Each newly
                            // indexable small input adds one array, never C.
                            for (auto elements : {terms, 5u * terms}) {
                                if (elements > 1u && elements <= 64u) {
                                    expected_resources.snapshot_bytes_per_worker += elements * sizeof(float);
                                    expected_resources.snapshot_allocations++;
                                }
                            }
                        }
                        expect_same_resources(candidate.resources, expected_resources);
                        expect_same_resources(planned.selected.resources, candidate.resources);
                        expect(eq(planned.selected.max_unrolled_mma_terms, cap));
                        if (reference_plan && width == 1u && !additional_roll) {
                            expect(eq(planned.selected.cost.score, reference_plan.selected.cost.score));
                        }
                        if (rolled == (terms > 64u)) { expect(text(candidate) == text(baseline)); }
                        if (terms == 9u && cap == 8u) {
                            auto phis = 0u, indices = 0u, selects = 0u;
                            candidate.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                                phis += instruction->isa<xir::PhiInst>();
                                indices += instruction->isa<xir::PhiInst>() && instruction->type()->is_int64();
                                if (instruction->isa<xir::ArithmeticInst>()) {
                                    selects += static_cast<xir::ArithmeticInst *>(instruction)->op() == xir::ArithmeticOp::SELECT;
                                }
                            });
                            expect(phis >= 2u);     // Runtime K index and accumulator, not counter-only metadata.
                            expect(eq(selects, 0u));// Dynamic operands use indexed loads, not an SSA SELECT chain.
                            if (variant == 2u || variant == 4u) {
                                auto groups = variant == 4u ? 5u : ceil_div(5u, width);
                                expect(eq(indices, groups));// Actual K loop grouping, including the partial output block.
                            }
                            expect(text(candidate) != text(baseline));
                        }
                    }
                }
            }
        }
        // The register-block work guard must price the same rolled K body as
        // emission. R4 * K16 * 8 exceeds 100, whereas R4 * 1 * 8 does not.
        auto work_kernel = fixture(1u, 5u, 16u, 0u);
        auto expanded_work = check_resources(work_kernel, {.max_unrolled_region_work = 100u, .mma_output_block = 4u});
        auto rolled_work = check_resources(work_kernel, {.max_unrolled_region_work = 100u, .mma_output_block = 4u, .max_unrolled_mma_terms = 8u});
        if (expanded_work && rolled_work) {
            expect(eq(expanded_work.blocked_mmas, 0u));
            expect(eq(rolled_work.blocked_mmas, 1u));
            expect(eq(rolled_work.rolled_mmas, 1u));
            auto expected_resources = expanded_work.resources;
            expected_resources.snapshot_bytes_per_worker += 16u * sizeof(float);// Only A was below the global threshold.
            expected_resources.snapshot_allocations++;
            expect_same_resources(rolled_work.resources, expected_resources);
        }
        auto shared_fixture = [](uint32_t variant) {
            return tile_kernel("mma_indexable_operands", [=](TensorView<const float, 2> a, TensorView<const float, 2> b, TensorView<float, 2> c) {
                       auto m = axis("m", 1), n = axis("n", 5), k = axis("k", 9);
                       for (auto &nest : parallel(shape(1))) {
                           auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                           auto rhs = b.tile(coord(0, 0), shape(k, n)).load();
                           auto seed = c.tile(coord(0, 0), shape(m, n)).load();
                           if (variant == 1u) {
                               // Constant operands retain their current representation.
                               lhs = full<float>(shape(m, k), 2.0f);
                               rhs = full<float>(shape(k, n), 3.0f);
                           }
                           if (variant == 2u) {
                               for (auto &step : nest.serial(shape(2))) {
                                   static_cast<void>(step);
                                   seed = mma(lhs, rhs, seed, {.allow_reassociation = false});
                                   lhs = lhs + 1.0f;
                               }
                           } else {
                               if (variant == 3u) {
                                   // Existing dynamic extraction already requires A's
                                   // snapshot; the newly rolled MMA must not duplicate it.
                                   for (auto &step : nest.serial(shape(1))) {
                                       c(coord(0, 0), shape(1, 1)).store(full<float>(shape(1, 1), lhs.at(coord(0, step.index()))));
                                   }
                               }
                               seed = mma(lhs, rhs, seed, {.allow_reassociation = false});
                               seed = mma(lhs, rhs, seed, {.allow_reassociation = false});
                           }
                           c(coord(0, 0), shape(m, n)).store(seed);
                       }
                   })
                .capture(tensor_shape(1, 9), tensor_shape(9, 5), tensor_shape(1, 5));
        };
        for (auto variant : {0u, 1u, 2u, 3u}) {
            auto shared = shared_fixture(variant);
            auto baseline = check_resources(shared);
            auto candidate = check_resources(shared, {.max_unrolled_mma_terms = 8u});
            if (!baseline || !candidate) { continue; }
            auto baseline_bytes = variant == 3u ? 9u * sizeof(float) : 0u;
            expect_same_resources(baseline.resources, {baseline_bytes, variant == 3u ? 1u : 0u});
            // Direct inputs or a small carried argument each get one array;
            // repeated uses, intermediate seeds and final outputs add none.
            auto expected = variant == 1u ? bx::ExecutionResources{} : bx::ExecutionResources{54u * sizeof(float), 2u};
            expect_same_resources(candidate.resources, expected);
            expect(eq(candidate.rolled_mmas, variant == 2u ? 1u : 2u));
        }
        for (auto budget : {215u, 216u}) {
            MockGpuTargetInfo limited;
            limited.snapshot_budget = budget;
            auto candidate = bx::plan(shared_fixture(0u).function(), limited, {.block_size = 64u, .max_unrolled_mma_terms = 8u});
            expect(eq(candidate.ok(), budget == 216u));
            if (candidate) {
                expect_same_resources(candidate.selected.resources, {216u, 2u});
            } else {
                expect(limited.policy.observed_candidates.empty());// Capacity rejection precedes cost evaluation.
            }
        }
        // A large deferred expression remains lazy. Only the emitted small A
        // gets new storage; B's pre-existing array is not replicated through
        // its recipe. This policy intentionally does not recurse into recipes.
        auto lazy = tile_kernel("mma_lazy_operand", [](TensorView<const float, 2> a, TensorView<const float, 2> b, TensorView<float, 2> c) {
                        auto m = axis("m", 1), n = axis("n", 9), k = axis("k", 9);
                        for (auto &nest : parallel(shape(1))) {
                            static_cast<void>(nest);
                            auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                            auto rhs = b.tile(coord(0, 0), shape(k, n)).load() + 1.0f;
                            auto seed = c.tile(coord(0, 0), shape(m, n)).load();
                            c(coord(0, 0), shape(m, n)).store(mma(lhs, rhs, seed));
                        }
                    }).capture(tensor_shape(1, 9), tensor_shape(9, 9), tensor_shape(1, 9));
        auto lazy_baseline = check_resources(lazy);
        auto lazy_candidate = check_resources(lazy, {.max_unrolled_mma_terms = 8u});
        if (lazy_baseline && lazy_candidate) {
            expect_same_resources(lazy_baseline.resources, {81u * sizeof(float), 1u});
            expect_same_resources(lazy_candidate.resources, {90u * sizeof(float), 2u});
        }
        MockGpuTargetInfo target_info;
        auto policy_plan = bx::plan(work_kernel.function(), target_info, {.mma_output_block = 4u, .max_unrolled_mma_terms = 8u});
        expect(policy_plan.ok()) << policy_plan.error;
        expect(!target_info.admission_candidates.empty());
        expect(!target_info.policy.observed_candidates.empty());
        for (const auto &candidate : target_info.admission_candidates) {
            expect(eq(candidate.mma_output_block, 4u));
            expect(eq(candidate.max_unrolled_mma_terms, 8u));
        }
        for (const auto &candidate : target_info.policy.observed_candidates) {
            expect(eq(candidate.mma_output_block, 4u));
            expect(eq(candidate.max_unrolled_mma_terms, 8u));
        }
        // Explicit fully expanded diagnostics override both MMA controls.
        auto diagnostic = check_resources(work_kernel, {.max_unrolled_tile_elements = 0u});
        for (auto width : {1u, 2u, 4u}) {
            auto candidate = check_resources(work_kernel, {.max_unrolled_tile_elements = 0u, .mma_output_block = width, .max_unrolled_mma_terms = 1u});
            if (diagnostic && candidate) {
                expect(eq(candidate.rolled_mmas, 0u));
                expect(eq(candidate.blocked_mmas, 0u));
                expect_same_resources(candidate.resources, diagnostic.resources);
                expect(text(candidate) == text(diagnostic));
            }
        }
        // No change to ordinary reductions, including their partition choice.
        auto ordinary = row_fixture(65u);
        auto ordinary_default = check_resources(ordinary);
        auto ordinary_cap = check_resources(ordinary, {.max_unrolled_mma_terms = 1u});
        if (ordinary_default && ordinary_cap) { expect(text(ordinary_default) == text(ordinary_cap)); }
        auto kernel = fixture(2u, 5u, 3u, 0u);
        for (auto width : {0u, 3u, 8u}) {
            expect(!bx::analyze_resources(kernel.function(), {.mma_output_block = width}));
            expect(!bx::lower(kernel.function(), {.mma_output_block = width}));
            expect(!bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.mma_output_block = width}));
        }
        auto bf16_fixture = [](bool strict) {
            return tile_kernel("bf16_mma_accumulator_policy", [=](TensorView<const float, 2> a, TensorView<const float, 2> b, TensorView<float, 2> c) {
                       auto m = axis("m", 1), n = axis("n", 2), k = axis("k", 3);
                       for (auto &nest : parallel(shape(1))) {
                           static_cast<void>(nest);
                           auto seed = cast<tile::bfloat16>(c.tile(coord(0, 0), shape(m, n)).load());
                           auto value = mma(a.tile(coord(0, 0), shape(m, k)).load(), b.tile(coord(0, 0), shape(k, n)).load(), seed, {.allow_reassociation = !strict});
                           c(coord(0, 0), shape(m, n)).store(cast<float>(value));
                       }
                   })
                .capture(tensor_shape(1, 3), tensor_shape(3, 2), tensor_shape(1, 2));
        };
        auto bf16 = bf16_fixture(true);
        auto wide_bf16 = bf16_fixture(false);
        expect(bf16.valid());
        expect(wide_bf16.valid());
        for (auto width : {1u, 4u}) {
            auto rejected = bx::lower(bf16.function(), {.mma_output_block = width});
            expect(!rejected && rejected.error.find("BF16 MMA accumulation") != string::npos);
            expect(!bx::analyze_resources(bf16.function(), {.mma_output_block = width}));
            expect(!bx::plan(bf16.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.mma_output_block = width}));
            auto admitted = check_resources(wide_bf16, {.mma_output_block = width});
            expect(admitted.ok()) << admitted.error;
            auto planned = bx::plan(wide_bf16.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.mma_output_block = width});
            expect(planned.ok()) << planned.error;
            if (admitted && planned) { expect_same_resources(admitted.resources, planned.selected.resources); }
        }
    };

    "tile_xir_mma_work_tracks_grouped_reads_snapshots_and_loops"_test = [] {
        using namespace tile;
        auto fixture = [](uint32_t terms, uint32_t columns, bool transpose, bool swap, uint32_t repeats) {
            return tile_kernel("mma_work", [=](TensorView<const float, 2> a, TensorView<const float, 2> b, TensorView<float, 2> c) {
                       auto m = axis("m", 1), n = axis("n", columns), k = axis("k", terms);
                       for (auto &nest : parallel(shape(8))) {
                           auto lhs = a.tile(coord(0, 0), shape(m, k)).load();
                           auto rhs = b.tile(coord(0, 0), transpose ? shape(n, k) : shape(k, n)).load();
                           auto seed = c.tile(coord(nest.index(), 0), shape(m, n)).load();
                           for (auto &step : nest.serial(shape(repeats))) {
                               static_cast<void>(step);
                               seed = swap ? mma(rhs, lhs, seed, {.allow_reassociation = false}) :
                                             mma(lhs, rhs, seed, {.allow_reassociation = false});
                           }
                           c(coord(nest.index(), 0), shape(m, n)).store(seed);
                       }
                   })
                .capture(tensor_shape(1, std::max(terms, 1u)), transpose ? tensor_shape(std::max(columns, 1u), std::max(terms, 1u)) : tensor_shape(std::max(terms, 1u), std::max(columns, 1u)), tensor_shape(8, std::max(columns, 1u)));
        };
        for (auto terms : {0u, 1u, 9u}) {
            for (auto columns : {0u, 1u, 5u}) {
                for (auto transpose : {false, true}) {
                    for (auto swap : {false, true}) {
                        for (auto repeats : {1u, 3u}) {
                            auto kernel = fixture(terms, columns, transpose, swap, repeats);
                            for (auto width : {1u, 2u, 4u}) {
                                for (auto cap : {0u, 8u}) {
                                    RecordingCostPolicy policy;
                                    auto options = bx::PlannerOptions{.block_size = 32u, .mma_output_block = width, .max_unrolled_mma_terms = cap, .cost_policy = &policy};
                                    auto planned = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, options);
                                    expect(planned.ok()) << planned.error;
                                    if (!planned || policy.observed_work.empty()) { continue; }
                                    const auto &work = policy.observed_work.front();
                                    const auto &mma = work.mma_per_packet;
                                    auto admitted = columns > 1u && !(swap && transpose && terms != 1u);
                                    auto groups = ceil_div(columns, admitted ? width : 1u);
                                    auto updates = static_cast<double>(columns * terms * repeats);
                                    auto common = static_cast<double>(groups * terms * repeats);
                                    auto rolled = cap != 0u && terms > cap;
                                    auto blocked = admitted && width != 1u;
                                    auto loop_terms = blocked ? terms : terms / 8u;
                                    expect(eq(mma.multiply_adds, updates));
                                    expect(eq(mma.lhs_reads, swap ? updates : common));
                                    expect(eq(mma.rhs_reads, swap ? common : updates));
                                    expect(eq(mma.seed_reads, static_cast<double>(columns * repeats)));
                                    expect(eq(mma.loop_invocations, rolled ? static_cast<double>(groups * repeats) : 0.0));
                                    expect(eq(mma.loop_iterations, rolled ? static_cast<double>(groups * loop_terms * repeats) : 0.0));
                                    expect(eq(work.arithmetic_per_packet, 2.0 * updates));
                                    // A/B view loads are uniform across programs;
                                    // C loads/stores have stride N. Cap8
                                    // adds exactly A+B snapshot stores, once outside
                                    // the repeated MMA; each dynamic read then uses
                                    // those arrays. Expanded reads keep their SSA
                                    // elements even when the same definition has storage.
                                    auto external = static_cast<double>(terms + terms * columns) +
                                                    2.0 * static_cast<double>(columns) * (columns == 1u ? 2.0 : 16.0);
                                    auto snapshots = rolled && columns != 0u ? static_cast<double>(terms + terms * columns) * 16.0 : 0.0;
                                    // Scalar k_pack tails use constant SSA
                                    // indices; only the complete chunks read
                                    // the small operand snapshots dynamically.
                                    auto read_terms = blocked ? terms : terms / 8u * 8u;
                                    auto reads = rolled && columns != 0u ? static_cast<double>((columns + groups) * read_terms * repeats) * 16.0 : 0.0;
                                    expect(eq(work.memory_per_packet, external + snapshots + reads));
                                }
                            }
                        }
                    }
                }
            }
        }
        // A [1,K] is still read through constant SSA elements when only the
        // output traversal is dynamic. Its definition snapshot is required
        // by the large-output consumer, but R4 must not claim saved array
        // reads until K itself is dynamic.
        auto large = fixture(9u, 65u, false, false, 1u);
        for (auto cap : {0u, 8u}) {
            RecordingCostPolicy r1, r4;
            auto baseline = bx::plan(large.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 32u, .max_unrolled_mma_terms = cap, .cost_policy = &r1});
            auto candidate = bx::plan(large.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 32u, .mma_output_block = 4u, .max_unrolled_mma_terms = cap, .cost_policy = &r4});
            expect(baseline.ok() && candidate.ok());
            if (r1.observed_work.empty() || r4.observed_work.empty()) { continue; }
            auto a = r1.observed_work.front(), b = r4.observed_work.front();
            expect(eq(a.arithmetic_per_packet, b.arithmetic_per_packet));
            expect(eq(a.mma_per_packet.lhs_reads, 585.0));
            expect(eq(b.mma_per_packet.lhs_reads, 153.0));
            expect(eq(a.memory_per_packet - b.memory_per_packet, cap ? (520.0 - 153.0) * 16.0 : 0.0));
        }
        // A requested width is not an admitted width: the expansion budget
        // can keep the reference contraction, and costs must do the same.
        RecordingCostPolicy budget_policy;
        auto budget = bx::plan(large.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 32u, .max_unrolled_region_work = 1u, .mma_output_block = 4u, .max_unrolled_mma_terms = 8u, .cost_policy = &budget_policy});
        expect(budget.ok());
        if (!budget_policy.observed_work.empty()) {
            const auto &mma = budget_policy.observed_work.front().mma_per_packet;
            expect(eq(mma.lhs_reads, 585.0));
            expect(eq(mma.rhs_reads, 585.0));
            expect(eq(mma.loop_invocations, 65.0));
        }
        // Small constants remain SSA lists, not the large-Tile SPLAT path.
        // Rolled reads currently emit SELECT/CMP chains. Verify that fact in
        // pre-cleanup XIR as well as its dynamic arithmetic work estimate.
        auto constants = tile_kernel("mma_constant_work", [](TensorView<float, 2> c) {
                             auto m = axis("m", 1), n = axis("n", 5), k = axis("k", 9);
                             for (auto &nest : parallel(shape(8))) {
                                 auto seed = c.tile(coord(nest.index(), 0), shape(m, n)).load();
                                 auto value = mma(full<float>(shape(m, k), 2.0f), full<float>(shape(k, n), 3.0f), seed);
                                 c(coord(nest.index(), 0), shape(m, n)).store(value);
                             }
                         }).capture(tensor_shape(8, 5));
        for (auto width : {1u, 4u}) {
            for (auto cap : {0u, 8u}) {
                RecordingCostPolicy policy;
                auto planned = bx::plan(constants.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 32u, .mma_output_block = width, .max_unrolled_mma_terms = cap, .cost_policy = &policy});
                auto lowered = check_resources(constants, {.mma_output_block = width, .max_unrolled_mma_terms = cap});
                expect(planned.ok());
                if (!lowered || policy.observed_work.empty()) { continue; }
                auto selects = 0u;
                lowered.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                    if (instruction->isa<xir::ArithmeticInst>()) {
                        selects += static_cast<xir::ArithmeticInst *>(instruction)->op() == xir::ArithmeticOp::SELECT;
                    }
                });
                auto per_term_selects = ceil_div(5u, width) * 9u + 5u * 45u;
                auto scalar = width == 1u;
                auto expected_selects = cap ? per_term_selects * (scalar ? 8u : 1u) : 0u;
                expect(eq(selects, expected_selects));
                expect_same_resources(lowered.resources, {});
                const auto &work = policy.observed_work.front();
                auto dynamic_selects = cap ? per_term_selects * (scalar ? 8u : 9u) : 0u;
                expect(eq(work.arithmetic_per_packet, 90.0 + 2.0 * dynamic_selects));
                expect(eq(work.memory_per_packet, 160.0));
            }
        }
        // CPU scheduling must preserve unweighted features in both caller-
        // thread and multi-worker schedules; a GPU uses its own schedule.
        for (auto workers : {1u, 4u}) {
            bx::ThreadPoolExecutionTargetInfo info{{8u, workers}};
            bx::ExecutionWork input{.packet_count = 32u, .block_count = 8u};
            input.mma_per_packet = {45.0, 18.0, 45.0, 5.0, 2.0, 18.0};
            auto output = info.schedule({.block_size = 32u}, input);
            expect(eq(output.mma_per_packet.multiply_adds, 45.0));
            expect(eq(output.mma_per_packet.lhs_reads, 18.0));
            expect(eq(output.mma_per_packet.rhs_reads, 45.0));
            expect(eq(output.mma_per_packet.seed_reads, 5.0));
            expect(eq(output.mma_per_packet.loop_invocations, 2.0));
            expect(eq(output.mma_per_packet.loop_iterations, 18.0));
        }
    };

    "tile_xir_mma_2d_blocking_preserves_work_storage_and_ssa_order"_test = [] {
        using namespace tile;
        auto fixture = [](uint32_t batches, uint32_t rows, uint32_t columns, uint32_t terms, bool transpose, bool swap, bool trailing_unit) {
            return tile_kernel("mma_2d_work", [=](TensorView<const float, 5> a, TensorView<const float, 5> b, TensorView<float, 5> c) {
                       auto r = axis("program", 1), batch = axis("batch", batches), m = axis("m", rows);
                       auto n = axis("n", columns), k = axis("k", terms), unit = axis("unit", 1);
                       auto lhs_space = trailing_unit ? shape(r, batch, m, k, unit) : shape(r, batch, unit, m, k);
                       auto rhs_space = trailing_unit ? (transpose ? shape(r, batch, n, k, unit) : shape(r, batch, k, n, unit)) :
                                                        (transpose ? shape(r, batch, unit, n, k) : shape(r, batch, unit, k, n));
                       auto output_space = trailing_unit ? shape(r, batch, m, n, unit) : shape(r, batch, m, unit, n);
                       for (auto &nest : parallel(shape(8))) {
                           auto lhs = a.tile(coord(0, 0, 0, 0, 0), lhs_space).load();
                           auto rhs = b.tile(coord(0, 0, 0, 0, 0), rhs_space).load();
                           auto origin = coord(nest.index(), 0, 0, 0, 0);
                           auto seed = c.tile(origin, output_space).load();
                           auto value = swap ? mma(rhs, lhs, seed, {.allow_reassociation = false}) :
                                               mma(lhs, rhs, seed, {.allow_reassociation = false});
                           c(origin, output_space).store(value);
                       }
                   })
                .capture(trailing_unit ? tensor_shape(1, batches, rows, std::max(terms, 1u), 1) : tensor_shape(1, batches, 1, rows, std::max(terms, 1u)), trailing_unit ? (transpose ? tensor_shape(1, batches, columns, std::max(terms, 1u), 1) : tensor_shape(1, batches, std::max(terms, 1u), columns, 1)) : (transpose ? tensor_shape(1, batches, 1, columns, std::max(terms, 1u)) : tensor_shape(1, batches, 1, std::max(terms, 1u), columns)), trailing_unit ? tensor_shape(8, batches, rows, columns, 1) : tensor_shape(8, batches, rows, 1, columns));
        };
        auto text = [](const bx::NativeFunction &value) {
            string result;
            xir::XIRDebugPrinter{}.emit_function(result, value.function);
            return result;
        };
        // All outputs fit the small SSA representation. A common batch and
        // singleton axes must not be mistaken for either blocking direction.
        // Ninety-six configurations include both row/column tails and the
        // unchanged one-dimensional M1/N1 fallback, without using its planner
        // implementation to compute the independent expected work.
        for (auto size : {std::array<uint32_t, 4>{2u, 1u, 5u, 0u}, {1u, 3u, 1u, 1u}, {1u, 2u, 2u, 1u}, {1u, 3u, 5u, 0u}, {2u, 3u, 5u, 1u}, {2u, 2u, 3u, 0u}}) {
            auto batches = size[0], rows = size[1], columns = size[2];
            constexpr uint32_t terms = 9u;
            auto outputs = batches * rows * columns;
            auto a_elements = batches * rows * terms, b_elements = batches * columns * terms;
            for (auto transpose : {false, true}) {
                for (auto swap : {false, true}) {
                    auto kernel = fixture(batches, rows, columns, terms, transpose, swap, size[3] != 0u);
                    for (auto cap : {0u, 8u}) {
                        auto baseline = check_resources(kernel, {.mma_output_block = 4u, .max_unrolled_mma_terms = cap});
                        for (auto enabled : {false, true}) {
                            auto context = format("B={} M={} N={} transpose={} swap={} cap={} 2d={}", batches, rows, columns, transpose, swap, cap, enabled);
                            RecordingCostPolicy policy;
                            auto planned = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 32u, .mma_output_block = 4u, .enable_mma_2d_blocking = enabled, .max_unrolled_mma_terms = cap, .cost_policy = &policy});
                            auto lowered = check_resources(kernel, {.mma_output_block = 4u, .enable_mma_2d_blocking = enabled, .max_unrolled_mma_terms = cap});
                            expect(planned.ok()) << context << planned.error;
                            expect(static_cast<bool>(baseline) && static_cast<bool>(lowered)) << context;
                            if (!planned || !baseline || !lowered || policy.observed_work.empty()) { continue; }
                            expect(eq(planned.selected.enable_mma_2d_blocking, enabled)) << context;
                            auto two_dimensional = enabled && rows > 1u && columns > 1u;
                            auto column_group = !two_dimensional && columns > 1u && !(swap && transpose);
                            auto row_group = !two_dimensional && columns == 1u && rows > 1u && swap;
                            auto blocked = two_dimensional || column_group || row_group;
                            auto groups = outputs;
                            double a_reads = static_cast<double>(outputs * terms), b_reads = a_reads;
                            if (two_dimensional) {
                                groups = batches * ceil_div(rows, 2u) * ceil_div(columns, 2u);
                                a_reads = static_cast<double>(batches * rows * ceil_div(columns, 2u) * terms);
                                b_reads = static_cast<double>(batches * columns * ceil_div(rows, 2u) * terms);
                            } else if (column_group) {
                                groups = batches * rows * ceil_div(columns, 4u);
                                a_reads = static_cast<double>(groups * terms);
                            } else if (row_group) {
                                groups = batches * columns * ceil_div(rows, 4u);
                                b_reads = static_cast<double>(groups * terms);
                            }
                            expect(eq(lowered.two_dimensional_mmas, two_dimensional ? 1u : 0u)) << context;
                            expect(eq(lowered.blocked_mmas, blocked ? 1u : 0u)) << context;
                            expect_same_resources(lowered.resources, baseline.resources);
                            expect_same_resources(planned.selected.resources, lowered.resources);
                            auto a_snapshot = cap != 0u || a_elements > 64u;
                            auto b_snapshot = cap != 0u || b_elements > 64u;
                            auto stored = (a_snapshot ? a_elements : 0u) + (b_snapshot ? b_elements : 0u);
                            expect_same_resources(lowered.resources, {stored * sizeof(float), static_cast<uint64_t>(a_snapshot) + b_snapshot});
                            if (!two_dimensional) { expect(text(lowered) == text(baseline)) << context; }
                            const auto &work = policy.observed_work.front();
                            expect(eq(work.mma_per_packet.multiply_adds, static_cast<double>(outputs * terms))) << context;
                            expect(eq(work.mma_per_packet.lhs_reads, swap ? b_reads : a_reads)) << context;
                            expect(eq(work.mma_per_packet.rhs_reads, swap ? a_reads : b_reads)) << context;
                            expect(eq(work.mma_per_packet.seed_reads, static_cast<double>(outputs))) << context;
                            expect(eq(work.mma_per_packet.loop_invocations, cap ? static_cast<double>(groups) : 0.0)) << context;
                            auto loop_terms = blocked ? terms : terms / 8u;
                            expect(eq(work.mma_per_packet.loop_iterations, cap ? static_cast<double>(groups * loop_terms) : 0.0)) << context;
                            expect(eq(work.arithmetic_per_packet, static_cast<double>(2u * outputs * terms))) << context;
                            auto external = static_cast<double>(a_elements + b_elements) + 2.0 * static_cast<double>(outputs) * (outputs == 1u ? 2.0 : 16.0);
                            auto snapshot_reads = [&](bool snapshot, uint32_t elements, double count) {
                                if (!snapshot) { return 0.0; }
                                // A small SSA-backed operand's constant tail
                                // bypasses its snapshot; a large array cannot.
                                auto read_terms = cap && !blocked && elements <= 64u ? terms / 8u * 8u : terms;
                                return count / terms * read_terms;
                            };
                            auto reads = snapshot_reads(a_snapshot, a_elements, a_reads) + snapshot_reads(b_snapshot, b_elements, b_reads);
                            expect(eq(work.memory_per_packet, external + (static_cast<double>(stored) + reads) * 16.0)) << context;
                            uint32_t indices = 0u, accumulators = 0u;
                            lowered.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                                if (instruction->isa<xir::PhiInst>()) {
                                    indices += instruction->type()->is_int64();
                                    accumulators += instruction->type()->is_float32();
                                }
                            });
                            // Large input traversals each have one additional
                            // induction PHI. Snapshot stores alone add none.
                            expect(eq(indices, (cap ? groups : 0u) + static_cast<uint32_t>(a_elements > 64u) + static_cast<uint32_t>(b_elements > 64u))) << context;
                            expect(eq(accumulators, cap ? outputs : 0u)) << context;
                        }
                    }
                }
            }
        }
        auto kernel = fixture(1u, 3u, 5u, 9u, false, false, true);
        // The flag is not permission to exceed the requested accumulator
        // budget. R1/R2 retain their byte-identical one-dimensional lowering.
        for (auto width : {1u, 2u}) {
            auto baseline = check_resources(kernel, {.mma_output_block = width, .max_unrolled_mma_terms = 8u});
            auto candidate = check_resources(kernel, {.mma_output_block = width, .enable_mma_2d_blocking = true, .max_unrolled_mma_terms = 8u});
            if (!baseline || !candidate) { continue; }
            expect(eq(candidate.two_dimensional_mmas, 0u));
            expect(text(candidate) == text(baseline));
            expect_same_resources(candidate.resources, baseline.resources);
        }
        RecordingCostPolicy budget_policy;
        auto budget = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 32u, .max_unrolled_region_work = 1u, .mma_output_block = 4u, .enable_mma_2d_blocking = true, .max_unrolled_mma_terms = 8u, .cost_policy = &budget_policy});
        auto fallback = check_resources(kernel, {.max_unrolled_region_work = 1u, .mma_output_block = 4u, .max_unrolled_mma_terms = 8u});
        auto bounded = check_resources(kernel, {.max_unrolled_region_work = 1u, .mma_output_block = 4u, .enable_mma_2d_blocking = true, .max_unrolled_mma_terms = 8u});
        expect(budget.ok()) << budget.error;
        if (fallback && bounded) {
            expect(eq(bounded.two_dimensional_mmas, 0u));
            expect(eq(bounded.blocked_mmas, 0u));
            expect(text(bounded) == text(fallback));
            expect_same_resources(bounded.resources, fallback.resources);
        }
        if (!budget_policy.observed_work.empty()) {
            auto mma = budget_policy.observed_work.front().mma_per_packet;
            expect(eq(mma.lhs_reads, 135.0));
            expect(eq(mma.rhs_reads, 135.0));
            expect(eq(mma.loop_invocations, 15.0));
        }
        // A zero contraction is an especially direct layout oracle: each
        // whole-Tile store must use the same ordered seed read. This catches
        // microtile-order appends masquerading as row-major SSA elements,
        // without executing generated code or using the planner as an oracle.
        for (auto trailing_unit : {false, true}) {
            auto empty = fixture(2u, 3u, 5u, 0u, false, false, trailing_unit);
            auto lowered = check_resources(empty, {.mma_output_block = 4u, .enable_mma_2d_blocking = true, .max_unrolled_mma_terms = 8u});
            if (!lowered) { continue; }
            expect(eq(lowered.two_dimensional_mmas, 1u));
            expect(eq(lowered.rolled_mmas, 0u));
            expect_same_resources(lowered.resources, {});
            vector<const xir::Value *> seeds, stored;
            lowered.function->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                if (instruction->isa<xir::ResourceReadInst>() && static_cast<xir::ResourceReadInst *>(instruction)->op() == xir::ResourceReadOp::BUFFER_READ) { seeds.emplace_back(instruction); }
                if (instruction->isa<xir::ResourceWriteInst>() && static_cast<xir::ResourceWriteInst *>(instruction)->op() == xir::ResourceWriteOp::BUFFER_WRITE) { stored.emplace_back(instruction->operand(2u)); }
            });
            expect(eq(seeds.size(), size_t{30u}));
            expect(eq(stored.size(), seeds.size()));
            if (stored.size() == seeds.size()) {
                for (size_t i = 0u; i < seeds.size(); i++) { expect(stored[i] == seeds[i]) << "zero-K SSA row-major seed index=" << i; }
            }
        }
    };

    "tile_xir_structured_map_budget_shares_coordinates_storage_and_emission"_test = [] {
        using namespace tile;
        constexpr uint32_t width = 33u;
        auto fixture = [](uint32_t variant, uint64_t iterations = 33u) {
            return tile_kernel("structured_map_budget", [=](TensorView<const float, 1> input, TensorView<float, 1> output) {
                       auto n = axis("n", width);
                       for (auto &root : parallel(shape(1))) {
                           static_cast<void>(root);
                           auto x = input.tile(coord(0), shape(n)).load();
                           auto y = map<float>(shape(n), [&](const Nest &element) {
                               // This is the input's only use. For an expanded
                               // map it projects SSA without a snapshot; forcing
                               // a runtime map must change that classification.
                               auto base = x.at(coord(element.index()));
                               if (variant == 0u) { return base * 2.0f + 1.0f; }
                               if (variant == 3u) {
                                   auto inner = map<float>(shape(65), [&](const Nest &item) {
                                       return base + cast<float>(item.index());
                                   });
                                   return inner.at(coord(0));
                               }
                               if (variant == 4u) {
                                   auto m = axis("m", 1), k = axis("k", 65), p = axis("p", 1);
                                   auto product = mma(full<float>(shape(m, k), base), full<float>(shape(k, p), 2.0f), zeros<float>(shape(m, p)));
                                   return product.at(coord(0, 0));
                               }
                               auto sum = Scalar<float>{0.0f};
                               if (variant == 1u) {
                                   for (auto &step : element.serial(shape(iterations))) { sum += base * cast<float>(step.index() + 1); }
                               } else {
                                   for (auto &step : element.reduce(shape(iterations))) { sum += base * cast<float>(step.index() + 1); }
                               }
                               return sum;
                           });
                           output(coord(0), shape(n)).store(y);
                       }
                   })
                .capture(tensor_shape(width), tensor_shape(width));
        };
        auto instruction_count = [](const bx::NativeFunction &lowered) {
            auto count = size_t{0u};
            lowered.function->traverse_instructions([&](xir::Instruction *) noexcept { count++; });
            return count;
        };
        for (auto variant : {0u, 1u, 2u, 3u, 4u}) {
            auto kernel = fixture(variant);
            auto bounded = check_resources(kernel);
            auto expanded = check_resources(kernel, {.max_unrolled_region_work = 0u});
            if (!bounded || !expanded) { continue; }
            auto planned = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 64u});
            auto unbounded_plan = bx::plan(kernel.function(), tile::bridge::xir::ThreadPoolExecutionTargetInfo{tile::bridge::xir::ExecutionTarget{8u, 1u}}, {.block_size = 64u, .max_unrolled_region_work = 0u});
            expect(planned.ok()) << planned.error;
            expect(unbounded_plan.ok()) << unbounded_plan.error;
            if (planned) { expect_same_resources(planned.selected.resources, bounded.resources); }
            if (unbounded_plan) { expect_same_resources(unbounded_plan.selected.resources, expanded.resources); }
            if (variant == 0u) {
                auto tiny_budget = check_resources(kernel, {.max_unrolled_region_work = 1u});
                if (tiny_budget) { expect(eq(instruction_count(tiny_budget), instruction_count(expanded))); }
                expect_same_resources(bounded.resources, {});
                expect(eq(instruction_count(bounded), instruction_count(expanded)));
            } else {
                expect(lt(instruction_count(bounded), instruction_count(expanded)));
                // full(shape, Scalar) is itself a map, so the MMA variant
                // also materializes one 65-element input, unlike a constant.
                auto nested_snapshot = variant == 3u || variant == 4u;
                auto nested_bytes = nested_snapshot ? uint64_t{65u * sizeof(float)} : uint64_t{0u};
                expect_same_resources(bounded.resources, {2u * width * sizeof(float) + nested_bytes, nested_snapshot ? 3u : 2u});
                // Reject a one-byte deficit using the very same allocation plan.
                expect(!bx::lower(kernel.function(), {.max_local_bytes = bounded.resources.snapshot_bytes_per_worker - 1u}));
                expect(bx::lower(kernel.function(), {.max_expanded_values = 2048u,
                                                     .max_local_bytes = bounded.resources.snapshot_bytes_per_worker})
                           .ok());
            }
        }
        // A saturating work product must select a loop without overflowing or
        // enumerating billions of contributions during compilation.
        auto huge = check_resources(fixture(1u, UINT32_MAX));
        if (huge) { expect_same_resources(huge.resources, {2u * width * sizeof(float), 2u}); }
        // The original element-limit-zero spelling remains an explicit fully
        // expanded diagnostic, even when a nonzero region budget is supplied.
        auto kernel = fixture(1u);
        auto legacy = check_resources(kernel, {.max_unrolled_tile_elements = 0u, .max_unrolled_region_work = 1u});
        auto disabled = check_resources(kernel, {.max_unrolled_region_work = 0u});
        if (legacy && disabled) {
            expect_same_resources(legacy.resources, disabled.resources);
            expect(eq(instruction_count(legacy), instruction_count(disabled)));
        }
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

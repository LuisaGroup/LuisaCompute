#include "ut/ut.hpp"
#include "tile_xir_test_utils.h"
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/verifier.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main() {
    "tile_xir_task_distribution_matches_static_home_chunks"_test = [] {
        using namespace tile;
        namespace bx = bridge::xir;
        struct AuditPolicy final : bx::AnalyticExecutionCostPolicy {
            [[nodiscard]] bx::ExecutionCost evaluate(bx::ExecutionTarget target, const bx::ExecutionPlan &candidate,
                                                     const bx::ExecutionWork &work, const bx::ExecutionCostModel &model) const noexcept override {
                auto blocks = ceil_div(candidate.dispatch_size, candidate.block_size);
                auto grain = candidate.blocks_per_task ? candidate.blocks_per_task : ceil_div(blocks, target.worker_count * target.task_chunks_per_worker);
                auto tasks = ceil_div(blocks, grain);
                auto workers = std::min(tasks, target.worker_count);
                if (workers == 1u) {
                    grain = blocks;
                    tasks = 1u;
                }
                vector<uint64_t> packets_per_worker(workers), blocks_per_worker(workers), tasks_per_worker(workers);
                for (auto task = 0u; task < tasks; task++) {
                    auto begin = static_cast<uint64_t>(task) * grain;
                    auto end = std::min<uint64_t>(begin + grain, blocks);
                    auto owner = task % workers;
                    tasks_per_worker[owner]++;
                    for (auto block = begin; block < end; block++) {
                        auto threads = std::min<uint64_t>(candidate.block_size, candidate.dispatch_size - block * candidate.block_size);
                        packets_per_worker[owner] += ceil_div(threads, static_cast<uint64_t>(target.packet_width));
                        blocks_per_worker[owner]++;
                    }
                }
                expect(eq(work.block_count, static_cast<uint64_t>(blocks)));
                expect(eq(work.packet_count, static_cast<uint64_t>(ceil_div(candidate.dispatch_size, target.packet_width))));
                expect(eq(work.task_count, static_cast<uint64_t>(tasks)));
                expect(eq(work.active_workers, workers));
                expect(eq(work.blocks_per_task, std::min(grain, blocks)));
                expect(eq(work.critical_packets, *std::max_element(packets_per_worker.begin(), packets_per_worker.end())));
                expect(eq(work.critical_blocks, *std::max_element(blocks_per_worker.begin(), blocks_per_worker.end())));
                expect(eq(work.critical_tasks, *std::max_element(tasks_per_worker.begin(), tasks_per_worker.end())));
                return AnalyticExecutionCostPolicy::evaluate(target, candidate, work, model);
            }
        } policy;
        for (auto count : {1, 7, 8, 9, 31, 32, 33, 65, 127, 257, 4097}) {
            auto kernel = tile_kernel("task_home", [=](TensorView<float, 1> out) {
                              for (auto &nest : parallel(shape(count))) { out(coord(nest.index()), shape(1)).store(full<float>(shape(1), 1.0f)); }
                          }).capture(tensor_shape(count));
            for (auto width : {1u, 2u, 4u, 8u, 16u}) {
                for (auto workers : {1u, 3u, 8u}) {
                    for (auto grain : {0u, 1u, 2u, 3u, 16u, UINT32_MAX}) {
                        auto options = bx::PlannerOptions{.block_size = 32u, .blocks_per_task = grain, .cost_policy = &policy};
                        auto result = bx::plan(kernel.function(), {width, workers}, options);
                        expect(result.ok()) << result.error;
                        if (result) { expect(eq(result.selected.blocks_per_task, grain)); }
                    }
                }
            }
        }
    };
    "tile_xir_task_search_and_policy_are_independent_of_legality"_test = [] {
        using namespace tile;
        namespace bx = bridge::xir;
        auto definition = tile_kernel("task_search", [](TensorView<float, 1> out) {
            for (auto &nest : parallel(shape(257))) { out(coord(nest.index()), shape(1)).store(full<float>(shape(1), 1.0f)); }
        });
        auto kernel = definition.capture(tensor_shape(257));
        auto options = bx::PlannerOptions{.block_size = 32u, .search_task_grain = true};
        auto result = bx::plan(kernel.function(), {8u, 8u}, options);
        expect(result.ok() && result.candidates.size() == 5u);
        if (result) {
            vector<uint32_t> grains;
            for (auto &candidate : result.candidates) {
                grains.emplace_back(candidate.blocks_per_task);
                expect(result.selected.cost.score <= candidate.cost.score);
            }
            expect(grains == vector<uint32_t>{1u, 2u, 4u, 8u, 9u});
            expect(result.selected.blocks_per_task < 9u);
        }
        options.cost.worker_activation = 1e12;
        options.cost.task_dispatch = 17.0;
        result = bx::plan(kernel.function(), {8u, 8u}, options);
        expect(result.ok());
        if (result) {
            expect(eq(result.selected.blocks_per_task, 9u));
            expect(eq(result.selected.cost.activation_work, 0.0));
            expect(eq(result.selected.cost.task_dispatch_work, 17.0));
        }
        options.max_candidates = 4u;
        expect(!bx::plan(kernel.function(), {8u, 8u}, options));
        options.blocks_per_task = 3u;
        expect(bx::plan(kernel.function(), {8u, 8u}, options).candidates.size() == 1u);
        expect(!bx::plan(kernel.function(), {8u, 8u, 0u}, options));
        struct OverridePolicy final : bx::AnalyticExecutionCostPolicy {
            double coefficient{2.0};
            double objective{5.0};
            mutable uint32_t evaluations{0u};
            [[nodiscard]] bx::ExecutionCostModel coefficients(bx::ExecutionTarget, const bx::ExecutionCostModel &prior) const noexcept override {
                auto model = prior;
                model.worker_activation = coefficient;
                return model;
            }
            [[nodiscard]] bx::ExecutionCost evaluate(bx::ExecutionTarget, const bx::ExecutionPlan &,
                                                     const bx::ExecutionWork &, const bx::ExecutionCostModel &) const noexcept override {
                evaluations++;
                return {.score = objective};
            }
        } policy;
        options.cost_policy = &policy;
        result = bx::plan(kernel.function(), {8u, 8u}, options);
        expect(result.ok() && result.selected.cost.score == 5.0 && policy.evaluations == 1u);
        for (auto invalid : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            policy.objective = invalid;
            expect(!bx::plan(kernel.function(), {8u, 8u}, options));
            policy.objective = 5.0;
            policy.coefficient = invalid;
            expect(!bx::plan(kernel.function(), {8u, 8u}, options));
            policy.coefficient = 2.0;
        }
        auto evaluations = policy.evaluations;
        options.block_size = 33u;
        expect(!bx::plan(kernel.function(), {8u, 8u}, options));
        expect(eq(policy.evaluations, evaluations));
    };
    "tile_xir_load_reduction_fusion_contract_and_cost"_test = [] {
        using namespace tile;
        for (auto lanes : {1u, 2u, 4u, 8u, 16u}) {
            for (auto variant = 0u; variant < 6u; variant++) {
                // 0: single consumer, 1: retain x, 2: intervening possibly
                // aliasing write, 3: strict fold, 4: reordered coordinates,
                // 5: source outside a nonunit execution scope.
                auto definition = tile_kernel("fusion_contract", [=](TensorView<const float, 2> input,
                                                                     TensorView<float, 2> other, TensorView<float, 2> output) {
                    auto m = axis("m", 1), n = axis("n", 65);
                    for (auto &nest : parallel(shape(17))) {
                        auto x = input[coord(nest.index(), 0), shape(m, n)];
                        auto y = other[coord(nest.index(), 0), shape(m, n)];
                        if (variant == 2u) { other(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), 9.0f)); }
                        if (variant == 4u || variant == 5u) {
                            auto sum = Scalar<float>{2.5f};
                            if (variant == 4u) {
                                for (auto &step : nest.reduce(shape(n))) { sum += x.at(coord(0, 64 - step.index())) + y.at(coord(0, step.index())); }
                            } else {
                                for (auto &step : nest.serial(shape(2))) {
                                    for (auto &element : step.reduce(shape(n))) { sum += x.at(coord(0, element.index())) + y.at(coord(0, element.index())); }
                                }
                            }
                            output(coord(nest.index(), 0), shape(m, n)).store(full<float>(shape(m, n), sum));
                        } else {
                            auto policy = variant == 3u ? reduction::fold_left : reduction::unordered_tree;
                            auto sum = reduce(x * y, n, add, policy);
                            output(coord(nest.index(), 0), shape(m, n)).store(variant == 1u ? x + sum : full<float>(shape(m, n), sum.at(coord(0))));
                        }
                    }
                });
                auto kernel = definition.capture(tensor_shape(17, 65), tensor_shape(17, 65), tensor_shape(17, 65));
                expect(kernel.valid());
                if (lanes > 1u && variant >= 3u) { continue; }
                auto off = bridge::xir::lower(kernel.function(), {.local_lanes = lanes});
                auto on = bridge::xir::lower(kernel.function(), {.local_lanes = lanes, .enable_load_reduction_fusion = true});
                expect(off.ok()) << off.error;
                expect(on.ok()) << on.error;
                if (!off || !on) { continue; }
                auto admitted = variant < 2u || variant == 4u;
                // Reversing x does not prevent independently fusing y.
                expect(eq(on.fused_reduction_loads, variant < 2u ? 2u : variant == 4u ? 1u :
                                                                                        0u))
                    << "variant=" << variant << " lanes=" << lanes;
                expect(eq(on.elided_load_snapshots, variant == 0u ? 2u : variant == 1u || variant == 4u ? 1u :
                                                                                                          0u));
                expect(eq(off.fused_reduction_loads, 0u));
                expect(xir::xir_verify_module(on.module.get(), {.require_reachable_blocks = true}).succeeded());
                auto count_allocations = [](const auto &lowered) {
                    size_t count = 0u;
                    lowered.function->traverse_instructions([&](xir::Instruction *inst) noexcept { count += inst->isa<xir::AllocaInst>(); });
                    return count;
                };
                expect(eq(count_allocations(off) - count_allocations(on), static_cast<size_t>(on.elided_load_snapshots)));
                auto a = bridge::xir::plan(kernel.function(), {8u, 8u}, {.block_size = 32u, .local_lanes = lanes, .enable_load_reduction_fusion = false});
                auto b = bridge::xir::plan(kernel.function(), {8u, 8u}, {.block_size = 32u, .local_lanes = lanes, .enable_load_reduction_fusion = true});
                if (lanes != 1u && lanes != 8u) { continue; }
                expect(a.ok() && b.ok());
                if (a && b) {
                    expect(admitted ? b.selected.cost.memory_work < a.selected.cost.memory_work :
                                      b.selected.cost.memory_work == a.selected.cost.memory_work);
                    expect(eq(a.selected.cost.arithmetic_work, b.selected.cost.arithmetic_work));
                }
            }
        }
    };
    "tile_xir_packet_local_admission_and_abi"_test = [] {
        using namespace tile;
        for (auto lanes : {2u, 4u, 8u, 16u}) {
            for (auto width : {lanes, lanes + 1u, 65u, 4096u}) {
                auto definition = tile_kernel("local_axis", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                    auto m = axis("m", 1), n = axis("n", width);
                    for (auto &nest : parallel(shape(17))) {
                        auto x = input[coord(nest.index(), 0), shape(m, n)];
                        output(coord(nest.index(), 0), shape(m, n)).store(x + reduce(x, n, add));
                    }
                });
                auto kernel = definition.capture(tensor_shape(17, width), tensor_shape(17, width));
                auto options = bridge::xir::LowerOptions{.max_local_bytes = ceil_div(width, lanes) * 4u, .local_lanes = lanes};
                auto lowered = bridge::xir::lower(kernel.function(), options);
                expect(lowered.ok()) << lowered.error;
                if (!lowered) { continue; }
                expect(eq(lowered.dispatch_size, 17u * lanes));
                expect(eq(lowered.required_packet_width, lanes));
                expect(xir::xir_verify_module(lowered.module.get(), {.require_reachable_blocks = true}).succeeded());
                options.max_local_bytes--;
                expect(!bridge::xir::lower(kernel.function(), options));
                auto automatic = bridge::xir::plan(kernel.function(), {lanes, 8u}, {.local_lanes = 0u});
                expect(automatic.ok() && automatic.candidates.size() == 12u);
                auto stable = bridge::xir::plan(kernel.function(), {lanes, 8u});
                expect(stable.ok() && stable.selected.local_lanes == 1u);
                for (auto local : {1u, lanes}) {
                    auto plan = bridge::xir::plan(kernel.function(), {lanes, 8u}, {.block_size = 32u, .max_candidates = 1u, .local_lanes = local});
                    expect(plan.ok()) << plan.error;
                    if (plan) {
                        expect(eq(plan.selected.local_lanes, local));
                        expect(eq(plan.selected.dispatch_size, 17u * local));
                    }
                }
                expect(!bridge::xir::plan(kernel.function(), {lanes, 8u}, {.max_candidates = 11u, .local_lanes = 0u}));
                expect(!bridge::xir::plan(kernel.function(), {lanes, 8u}, {.local_lanes = 3u}));
            }
        }
    };
    "tile_xir_packet_local_rejects_unrealized_redistribution_and_folds"_test = [] {
        using namespace tile;
        for (auto variant = 0; variant < 7; variant++) {
            auto definition = tile_kernel("local_contract", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                auto height = variant == 6 ? 2 : 1;
                auto m = axis("m", height), n = axis("n", 65), other = axis("other", 65);
                auto binding = variant == 4 ? exec::Scope::WORKER : exec::Scope::AUTOMATIC;
                for (auto &nest : parallel(shape(17), binding)) {
                    auto x = input[coord(nest.index() * height, 0), shape(m, n)];
                    if (variant == 0 || variant == 5) {
                        auto y = map<float>(shape(m, n), [&](const Nest &element) {
                            if (variant == 5) { return reduce(x, n, add).at(coord(0)); }
                            return x.at(coord(0, 64 - element.index(n)));
                        });
                        output(coord(nest.index() * height, 0), shape(m, n)).store(y);
                    } else {
                        auto policy = variant == 1 ? reduction::fold_left : variant == 2 ? reduction::fold_right :
                                                                                           reduction::unordered_tree;
                        auto sum = Scalar<float>{0.0f};
                        for (auto &step : nest.reduce(shape(variant == 3 ? other : n), policy)) { sum += x.at(coord(0, step.index())); }
                        output(coord(nest.index() * height, 0), shape(m, n)).store(x + sum);
                    }
                }
            });
            auto rows = variant == 6 ? 34 : 17;
            auto kernel = definition.capture(tensor_shape(rows, 65), tensor_shape(rows, 65));
            expect(kernel.valid());
            expect(bridge::xir::lower(kernel.function()).ok());
            expect(!bridge::xir::lower(kernel.function(), {.local_lanes = 8u}));
            expect(!bridge::xir::plan(kernel.function(), {8u, 8u}, {.local_lanes = 8u}));
            auto fallback = bridge::xir::plan(kernel.function(), {8u, 8u});
            expect(fallback.ok() && fallback.candidates.size() == 6u);
            if (fallback) { expect(eq(fallback.selected.local_lanes, 1u)); }
        }
    };
    "tile_xir_dynamic_extract_has_linear_snapshot_storage"_test = [] {
        using namespace tile;
        double previous_work = 0.0;
        for (auto width : {16, 256}) {
            auto definition = tile_kernel("indexed_sum", [=](TensorView<const float, 2> input, TensorView<float, 1> output) {
                auto m = axis("m", 1), n = axis("n", width);
                for (auto &nest : parallel(shape(17))) {
                    auto x = input[coord(nest.index(), 0), shape(m, n)];
                    auto sum = reduce(x * x, n, add);
                    output(coord(nest.index()), shape(1)).store(full<float>(shape(1), sum.at(coord(0))));
                }
            });
            auto kernel = definition.capture(tensor_shape(17, width), tensor_shape(17));
            expect(kernel.valid());
            auto result = bridge::xir::lower(kernel.function(), {.max_unrolled_tile_elements = 0u});
            expect(result.ok()) << result.error;
            if (!result) { continue; }
            size_t allocations = 0u, stores = 0u, loads = 0u, selects = 0u;
            result.function->traverse_instructions([&](xir::Instruction *inst) noexcept {
                allocations += inst->isa<xir::AllocaInst>();
                stores += inst->isa<xir::StoreInst>();
                loads += inst->isa<xir::LoadInst>();
                if (inst->isa<xir::ArithmeticInst>()) { selects += static_cast<xir::ArithmeticInst *>(inst)->op() == xir::ArithmeticOp::SELECT; }
            });
            expect(eq(allocations, size_t{1}));
            expect(eq(stores, static_cast<size_t>(width)));
            expect(eq(loads, size_t{1}));
            expect(eq(selects, size_t{0}));
            auto limited = bridge::xir::lower(kernel.function(), {.max_local_bytes = static_cast<uint32_t>(width * 4 - 1), .max_unrolled_tile_elements = 0u});
            expect(!limited && limited.module == nullptr);
            expect(limited.error.find("snapshot storage budget") != string::npos);
            expect(bridge::xir::lower(kernel.function(), {.max_local_bytes = static_cast<uint32_t>(width * 4), .max_unrolled_tile_elements = 0u}).ok());
            auto plan = bridge::xir::plan(kernel.function(), {8u, 8u}, {.max_unrolled_tile_elements = 0u});
            expect(plan.ok()) << plan.error;
            if (plan) {
                auto work = plan.selected.cost.arithmetic_work + plan.selected.cost.memory_work;
                if (previous_work > 0.0) { expect(work <= previous_work * 16.0); }
                previous_work = work;
            }
        }
    };
    "tile_xir_large_tiles_have_bounded_code_and_eager_load_snapshots"_test = [] {
        using namespace tile;
        size_t previous_instructions = 0u;
        for (auto width : {65, 256, 1537, 4096, 16384}) {
            auto definition = tile_kernel("bounded_sum", [=](TensorView<const float, 2> input, TensorView<float, 1> output) {
                auto m = axis("m", 1), n = axis("n", width);
                for (auto &nest : parallel(shape(17))) {
                    auto x = input[coord(nest.index(), 0), shape(m, n)];
                    auto sum = reduce(x * x, n, add);
                    output(coord(nest.index()), shape(1)).store(full<float>(shape(1), sum.at(coord(0))));
                }
            });
            auto kernel = definition.capture(tensor_shape(17, width), tensor_shape(17));
            auto result = bridge::xir::lower(kernel.function(), {.max_expanded_values = 256u, .max_local_bytes = static_cast<uint32_t>(width * 4), .reduction_partitions = 1u});
            expect(result.ok()) << result.error;
            if (!result) { continue; }
            size_t instructions = 0u, allocations = 0u, stores = 0u, loads = 0u, selects = 0u;
            result.function->traverse_instructions([&](xir::Instruction *inst) noexcept {
                instructions++;
                allocations += inst->isa<xir::AllocaInst>();
                stores += inst->isa<xir::StoreInst>();
                loads += inst->isa<xir::LoadInst>();
                if (inst->isa<xir::ArithmeticInst>()) { selects += static_cast<xir::ArithmeticInst *>(inst)->op() == xir::ArithmeticOp::SELECT; }
            });
            expect(eq(allocations, size_t{1}));
            expect(eq(stores, size_t{1}));
            expect(eq(loads, size_t{2}));
            expect(eq(selects, size_t{0}));
            expect(instructions < 256u);
            if (previous_instructions) { expect(eq(instructions, previous_instructions)); }
            previous_instructions = instructions;
            auto limited = bridge::xir::lower(kernel.function(), {.max_local_bytes = static_cast<uint32_t>(width * 4 - 1), .enable_load_reduction_fusion = false});
            expect(!limited && limited.error.find("snapshot storage budget") != string::npos);
            expect(!bridge::xir::lower(kernel.function(), {.max_expanded_values = 256u, .max_unrolled_tile_elements = 0u}));
            auto plan = bridge::xir::plan(kernel.function(), {8u, 8u});
            expect(plan.ok()) << plan.error;
        }
    };
    "tile_xir_expanded_map_extracts_need_no_snapshot"_test = [] {
        using namespace tile;
        auto definition = tile_kernel("map_projection", [](TensorView<const float, 1> input, TensorView<float, 1> output) {
            for (auto &nest : parallel(shape(1))) {
                auto x = input[coord(0), shape(17)];
                auto reversed = map<float>(shape(17), [&](const Nest &element) { return x.at(coord(16 - element.index())); });
                output(coord(nest.index()), shape(17)).store(reversed);
            }
        });
        auto kernel = definition.capture(tensor_shape(17), tensor_shape(17));
        auto result = bridge::xir::lower(kernel.function(), {.max_local_bytes = 0u});
        expect(result.ok()) << result.error;
        if (!result) { return; }
        result.function->traverse_instructions([&](xir::Instruction *inst) noexcept {
            expect(!inst->isa<xir::AllocaInst>());
            if (inst->isa<xir::ArithmeticInst>()) { expect(static_cast<xir::ArithmeticInst *>(inst)->op() != xir::ArithmeticOp::SELECT); }
        });
    };
    "tile_xir_verified_ssa_and_abi"_test = [] {
        auto kernel = test::tile_xir::gemm({17, 19, 13, 2, 3, 4});
        expect(kernel.valid());
        auto result = tile::bridge::xir::lower(kernel.function());
        expect(result.ok()) << result.error;
        if (!result) { return; }
        expect(eq(result.dispatch_size, 63u));
        expect(eq(result.argument_sizes_bytes[0], size_t{17 * 13 * 4}));
        expect(eq(result.argument_sizes_bytes[1], size_t{13 * 19 * 4}));
        expect(eq(result.argument_sizes_bytes[2], size_t{17 * 19 * 4}));
        expect(result.argument_usages[0] == Usage::READ);
        expect(result.argument_usages[1] == Usage::READ);
        expect(result.argument_usages[2] == Usage::WRITE);
        expect(xir::xir_verify_module(result.module.get(), {.require_reachable_blocks = true}).succeeded());
        // Re-lowering the same input remains legal: the bridge borrows TileIR.
        expect(tile::verify(kernel.module()).ok());
        expect(tile::bridge::xir::lower(kernel.function()).ok());
    };
    "tile_xir_expansion_budget_is_fail_closed"_test = [] {
        auto kernel = test::tile_xir::gemm({8, 8, 8, 4, 4, 8});
        auto result = tile::bridge::xir::lower(kernel.function(), {.max_expanded_values = 8u});
        expect(!result);
        expect(result.error.find("expansion budget") != string::npos);
        expect(result.module == nullptr);
        for (auto width : {0u, 1u, 33u, 2048u}) {
            expect(!tile::bridge::xir::lower(kernel.function(), {.block_size = width}));
        }
    };
    "tile_xir_rejects_unrealized_execution_binding"_test = [] {
        using namespace tile;
        auto definition = tile_kernel("cooperative", [](TensorView<float, 1> out) {
            for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
                out(coord(nest.index()), shape(1)).store(full<float>(shape(1), 1.0f));
            }
        });
        auto kernel = definition.capture(tensor_shape(1));
        expect(kernel.valid());
        auto result = bridge::xir::lower(kernel.function());
        expect(!result);
        expect(result.error.find("execution binding") != string::npos);
    };
    "tile_xir_planner_searches_execution_order_not_memory_layout"_test = [] {
        using namespace tile;
        for (auto transpose : {false, true}) {
            auto definition = tile_kernel("order", [=](TensorView<float, 2> out) {
                auto m = axis("m", 16), n = axis("n", 32);
                for (auto &nest : parallel(shape(n, m))) {
                    auto origin = transpose ? coord(nest.index(n), nest.index(m)) : coord(nest.index(m), nest.index(n));
                    out(origin, shape(1, 1)).store(full<float>(shape(1, 1), 1.0f));
                }
            });
            auto kernel = definition.capture(transpose ? tensor_shape(32, 16) : tensor_shape(16, 32));
            auto result = bridge::xir::plan(kernel.function(), {8u, 8u});
            expect(result.ok()) << result.error;
            if (!result) { continue; }
            expect(eq(result.candidates.size(), size_t{12}));
            expect(result.selected.root_axis_order == (transpose ? vector<uint32_t>{0u, 1u} : vector<uint32_t>{1u, 0u}));
            for (auto &candidate : result.candidates) { expect(result.selected.cost.score <= candidate.cost.score); }
            auto fixed = bridge::xir::plan(kernel.function(), {8u, 8u}, {.block_size = 64u, .root_axis_order = {0u, 1u}});
            expect(fixed.ok() && fixed.candidates.size() == 1u);
            expect(!bridge::xir::plan(kernel.function(), {8u, 8u}, {.root_axis_order = {0u, 0u}}));
            expect(!bridge::xir::plan(kernel.function(), {8u, 8u}, {.max_candidates = 11u}));
            expect(!bridge::xir::plan(kernel.function(), {8u, 8u}, {.block_size = 33u}));
            expect(!bridge::xir::plan(kernel.function(), {3u, 8u}));
            auto options = bridge::xir::PlannerOptions{};
            options.cost.arithmetic = -1.0;
            expect(!bridge::xir::plan(kernel.function(), {8u, 8u}, options));
            expect(bridge::xir::lower(kernel.function(), {.root_axis_order = result.selected.root_axis_order}).ok());
            expect(!bridge::xir::lower(kernel.function(), {.root_axis_order = {0u, 0u}}));
        }
    };
}

#include "ut/ut.hpp"
#include "tile_xir_test_utils.h"
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/verifier.h>
#include <luisa/tile/algorithms.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/debug_printer.h>
#include <array>
#include <numeric>
#include <tuple>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

// Evaluate only the integer address DAG of a one-element root fixture. This
// tests emitted XIR, not a second invocation of the planner's mapping helper.
[[nodiscard]] optional<uint64_t> root_address(const xir::Value *value, uint64_t dispatch, uint32_t depth = 0u) {
    if (depth > 256u) { return {}; }
    auto constant = uint64_t{0u};
    if (xir::try_decode_constant_nonnegative_integer(value, constant)) { return constant; }
    if (value->isa<xir::CastInst>()) { return root_address(static_cast<const xir::CastInst *>(value)->value(), dispatch, depth + 1u); }
    if (!value->isa<xir::ArithmeticInst>()) { return {}; }
    auto inst = static_cast<const xir::ArithmeticInst *>(value);
    if (inst->op() == xir::ArithmeticOp::EXTRACT && inst->operand(0u)->isa<xir::SpecialRegister>() &&
        static_cast<const xir::SpecialRegister *>(inst->operand(0u))->derived_special_register_tag() == xir::DerivedSpecialRegisterTag::DISPATCH_ID &&
        xir::try_decode_constant_nonnegative_integer(inst->operand(1u), constant) && constant == 0u) { return dispatch; }
    if (inst->operand_count() != 2u) { return {}; }
    auto a = root_address(inst->operand(0u), dispatch, depth + 1u);
    auto b = root_address(inst->operand(1u), dispatch, depth + 1u);
    if (!a || !b) { return {}; }
    switch (inst->op()) {
        case xir::ArithmeticOp::BINARY_ADD: return *a + *b;
        case xir::ArithmeticOp::BINARY_MUL: return *a * *b;
        case xir::ArithmeticOp::BINARY_DIV: return *b ? optional<uint64_t>{*a / *b} : optional<uint64_t>{};
        case xir::ArithmeticOp::BINARY_MOD: return *b ? optional<uint64_t>{*a % *b} : optional<uint64_t>{};
        default: return {};
    }
}

template<size_t Rank>
[[nodiscard]] tile::Kernel root_fixture(std::array<uint64_t, Rank> extents) {
    using namespace tile;
    return tile_kernel("root_traversal", [=](TensorView<float, 1> output) {
               std::array<Axis, Rank> axes;
               constexpr string_view names[]{"a", "b", "c"};
               for (size_t i = 0u; i < Rank; i++) { axes[i] = axis(names[i], extents[i]); }
               auto domain = std::apply([](auto... axes) { return shape(axes...); }, axes);
               for (auto &nest : parallel(domain)) {
                   auto linear = Scalar<int64_t>{0};
                   for (size_t i = 0u; i < Rank; i++) { linear = linear * static_cast<int64_t>(extents[i]) + nest.index(axes[i]); }
                   output(coord(linear), shape(1)).store(full<float>(shape(1), 1.0f));
               }
           })
        .capture(tensor_shape(std::accumulate(extents.begin(), extents.end(), uint64_t{1u}, std::multiplies{})));
}

// Independent outer-block / inner-element reference, in original-axis space.
template<size_t Rank>
[[nodiscard]] vector<uint64_t> root_sequence(std::array<uint64_t, Rank> extents,
                                             span<const uint32_t> order, span<const uint32_t> tiles) {
    auto inner_count = std::accumulate(tiles.begin(), tiles.end(), uint64_t{1u}, std::multiplies{});
    auto total = std::accumulate(extents.begin(), extents.end(), uint64_t{1u}, std::multiplies{});
    vector<uint64_t> result;
    for (auto outer = uint64_t{0u}; outer < total / inner_count; outer++) {
        for (auto inner = uint64_t{0u}; inner < inner_count; inner++) {
            auto outer_remaining = outer, inner_remaining = inner;
            std::array<uint64_t, Rank> coordinate{};
            for (size_t position = Rank; position-- > 0u;) {
                auto axis = order[position];
                auto blocks = extents[axis] / tiles[axis];
                coordinate[axis] = outer_remaining % blocks * tiles[axis] + inner_remaining % tiles[axis];
                outer_remaining /= blocks;
                inner_remaining /= tiles[axis];
            }
            auto linear = uint64_t{0u};
            for (size_t axis = 0u; axis < Rank; axis++) { linear = linear * extents[axis] + coordinate[axis]; }
            result.emplace_back(linear);
        }
    }
    return result;
}

template<size_t Rank>
void check_root_traversal(std::array<uint64_t, Rank> extents, span<const vector<uint32_t>> factors) {
    auto kernel = root_fixture(extents);
    auto order = vector<uint32_t>(Rank);
    std::iota(order.begin(), order.end(), 0u);
    do {
        auto baseline = tile::bridge::xir::lower(kernel.function(), {.root_axis_order = order});
        expect(baseline.ok()) << baseline.error;
        if (!baseline) { return; }
        string baseline_text;
        xir::XIRDebugPrinter{}.emit_function(baseline_text, baseline.function);
        auto flat = root_sequence(extents, order, vector<uint32_t>(Rank, 1u));
        for (auto &tiles : factors) {
            auto lowered = tile::bridge::xir::lower(kernel.function(), {.root_axis_order = order, .root_axis_tiles = tiles});
            auto planned = tile::bridge::xir::plan(kernel.function(), {8u, 1u}, {.block_size = 32u, .root_axis_order = order, .root_axis_tiles = tiles});
            expect(lowered.ok() && planned.ok()) << lowered.error << planned.error;
            if (!lowered || !planned) { continue; }
            expect(xir::xir_verify_module(lowered.module.get(), {.require_reachable_blocks = true}).succeeded());
            expect(planned.selected.root_axis_tiles == tiles);
            const xir::Value *address = nullptr;
            auto stores = 0u;
            lowered.function->traverse_instructions([&](xir::Instruction *inst) noexcept {
                if (inst->isa<xir::ResourceWriteInst>() && static_cast<xir::ResourceWriteInst *>(inst)->op() == xir::ResourceWriteOp::BUFFER_WRITE) {
                    address = inst->operand(1u);
                    stores++;
                }
            });
            expect(eq(stores, 1u));
            if (!address || stores != 1u) { continue; }
            auto expected = root_sequence(extents, order, tiles);
            vector<uint64_t> actual;
            vector<uint32_t> visits(expected.size(), 0u);
            for (auto ordinal = 0u; ordinal < lowered.dispatch_size; ordinal++) {
                auto decoded = root_address(address, ordinal);
                expect(decoded && *decoded < visits.size());
                if (!decoded || *decoded >= visits.size()) { break; }
                actual.emplace_back(*decoded);
                visits[*decoded]++;
            }
            expect(actual == expected);
            expect(std::all_of(visits.begin(), visits.end(), [](auto count) { return count == 1u; }));
            string text;
            xir::XIRDebugPrinter{}.emit_function(text, lowered.function);
            if (expected == flat) {
                expect(text == baseline_text);// identity must preserve exact XIR, not only outputs
            } else {
                expect(actual != flat);// ensure the realization really changed traversal
                expect(text != baseline_text);
            }
        }
    } while (std::next_permutation(order.begin(), order.end()));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_xir_root_traversal_is_bijective_and_identity_is_exact"_test = [] {
        check_root_traversal<1u>({12u}, vector<vector<uint32_t>>{{1u}, {3u}, {12u}});
        check_root_traversal<2u>({4u, 6u}, vector<vector<uint32_t>>{{1u, 1u}, {1u, 3u}, {4u, 6u}, {2u, 3u}, {2u, 1u}});
        check_root_traversal<3u>({4u, 1u, 6u}, vector<vector<uint32_t>>{{1u, 1u, 1u}, {2u, 1u, 3u}, {4u, 1u, 6u}});
        check_root_traversal<3u>({4u, 6u, 3u}, vector<vector<uint32_t>>{{1u, 1u, 1u}, {2u, 3u, 1u}, {4u, 6u, 3u}});
        auto expected = vector<uint64_t>{0u, 1u, 2u, 6u, 7u, 8u, 3u, 4u, 5u, 9u, 10u, 11u,
                                         12u, 13u, 14u, 18u, 19u, 20u, 15u, 16u, 17u, 21u, 22u, 23u};
        expect(root_sequence<2u>({4u, 6u}, vector<uint32_t>{0u, 1u}, vector<uint32_t>{2u, 3u}) == expected);
    };
    "tile_xir_root_traversal_rejects_invalid_constraints"_test = [] {
        auto kernel = root_fixture<2u>({4u, 6u});
        for (auto tiles : vector<vector<uint32_t>>{{2u}, {2u, 3u, 1u}, {0u, 3u}, {3u, 3u}, {8u, 3u}, {2u, UINT32_MAX}}) {
            expect(!tile::bridge::xir::lower(kernel.function(), {.root_axis_tiles = tiles}));
            expect(!tile::bridge::xir::plan(kernel.function(), {8u, 1u}, {.root_axis_tiles = tiles}));
        }
        for (auto order : vector<vector<uint32_t>>{{0u}, {0u, 0u}, {0u, 2u}}) {
            expect(!tile::bridge::xir::lower(kernel.function(), {.root_axis_order = order, .root_axis_tiles = {2u, 3u}}));
            expect(!tile::bridge::xir::plan(kernel.function(), {8u, 1u}, {.root_axis_order = order, .root_axis_tiles = {2u, 3u}}));
        }
        auto overflow = root_fixture<2u>({65536u, 65536u});
        expect(!tile::bridge::xir::lower(overflow.function(), {.root_axis_tiles = {256u, 256u}}));
        expect(!tile::bridge::xir::plan(overflow.function(), {8u, 1u}, {.root_axis_tiles = {256u, 256u}}));
    };
    "tile_xir_root_traversal_cost_uses_physical_fast_digit"_test = [] {
        using namespace tile;
        // Memory is column-major relative to the root's (a,b) order. Factoring
        // only a makes a the physical fast digit, even though b is last in order.
        auto kernel = tile_kernel("root_stride", [](TensorView<const float, 2> input, TensorView<float, 2> output) {
                          auto a = axis("a", 32), b = axis("b", 24);
                          for (auto &nest : parallel(shape(a, b))) {
                              auto origin = coord(nest.index(b), nest.index(a));
                              output(origin, shape(1, 1)).store(input[origin, shape(1, 1)]);
                          }
                      }).capture(tensor_shape(24, 32), tensor_shape(24, 32));
        auto score = [&](vector<uint32_t> tiles) {
            return bridge::xir::plan(kernel.function(), {8u, 1u}, {.block_size = 32u, .root_axis_order = {0u, 1u}, .root_axis_tiles = std::move(tiles)});
        };
        auto flat = score({}), identity = score({1u, 8u}), coherent = score({8u, 1u}), crossing = score({2u, 1u});
        expect(flat.ok() && identity.ok() && coherent.ok() && crossing.ok());
        if (!flat || !identity || !coherent || !crossing) { return; }
        expect(eq(flat.selected.cost.score, identity.selected.cost.score));
        expect(lt(coherent.selected.cost.memory_work, flat.selected.cost.memory_work));
        expect(eq(crossing.selected.cost.memory_work, flat.selected.cost.memory_work));
        expect(gt(coherent.selected.cost.arithmetic_work, flat.selected.cost.arithmetic_work));
    };
    "tile_xir_deferred_map_depth_budget"_test = [] {
        using namespace tile;
        for (auto depth_limit : {63u, 64u, 65u, 70u}) {
            auto kernel = tile_kernel("map_depth", [=](TensorView<const float, 1> input, TensorView<float, 1> output) {
                              auto n = axis("n", 65);
                              for (auto &nest : parallel(shape(1))) {
                                  auto x = input[coord(0), shape(n)];
                                  for (auto depth = 0u; depth < depth_limit; depth++) {
                                      x = reindex(x, shape(n), [&](const Nest &element) { return coord(64 - element.index()); });
                                  }
                                  output(coord(0), shape(n)).store(x);
                              }
                          }).capture(tensor_shape(65), tensor_shape(65));
            auto baseline = bridge::xir::lower(kernel.function());
            auto candidate = bridge::xir::lower(kernel.function(), {.enable_map_fusion = true});
            auto planning = bridge::xir::plan(kernel.function(), {8u, 1u}, {.enable_map_fusion = true});
            expect(baseline.ok()) << baseline.error;
            if (depth_limit <= 64u) {
                expect(candidate.ok()) << candidate.error;
                expect(planning.ok()) << planning.error;
                if (candidate) { expect(xir::xir_verify_module(candidate.module.get(), {.require_reachable_blocks = true}).succeeded()); }
            } else {
                expect(!candidate.ok() && candidate.error.find("depth budget") != string::npos);
                expect(!planning.ok() && planning.error.find("depth budget") != string::npos);
            }
        }
    };
    "tile_xir_pure_map_fusion_admission_and_cost"_test = [] {
        using namespace tile;
        for (auto variant = 0u; variant < 3u; variant++) {
            auto kernel = tile_kernel("map_contract", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                              auto m = axis("m", 1), n = axis("n", 65);
                              for (auto &nest : parallel(shape(17))) {
                                  auto x = input[coord(nest.index(), 0), shape(m, n)];
                                  auto y = reindex(x, shape(m, n), [&](const Nest &element) {
                                      return coord(element.index(m), 64 - element.index(n));
                                  });
                                  if (variant == 1u) { output(coord(nest.index(), 0), shape(m, n)).store(y); }
                                  if (variant == 2u) {
                                      for (auto &step : nest.serial(shape(2))) {
                                          output(coord(nest.index(), step.index()), shape(1, 1)).store(full<float>(shape(1, 1), 2.0f));
                                      }
                                  }
                                  output(coord(nest.index(), 0), shape(m, n)).store(y);
                              }
                          }).capture(tensor_shape(17, 65), tensor_shape(17, 65));
            auto off = bridge::xir::lower(kernel.function());
            auto on = bridge::xir::lower(kernel.function(), {.enable_map_fusion = true});
            expect(off.ok() && on.ok()) << off.error << on.error;
            if (!off || !on) { continue; }
            expect(eq(off.deferred_maps, 0u));
            expect(eq(on.deferred_maps, variant == 0u ? 1u : 0u));
            expect(xir::xir_verify_module(on.module.get(), {.require_reachable_blocks = true}).succeeded());
            auto allocations = [](const auto &lowered) {
                size_t count = 0u;
                lowered.function->traverse_instructions([&](xir::Instruction *inst) noexcept { count += inst->isa<xir::AllocaInst>(); });
                return count;
            };
            expect(eq(allocations(off) - allocations(on), static_cast<size_t>(on.deferred_maps)));
            auto a = bridge::xir::plan(kernel.function(), {8u, 1u}, {.block_size = 32u});
            auto b = bridge::xir::plan(kernel.function(), {8u, 1u}, {.block_size = 32u, .enable_map_fusion = true});
            expect(a.ok() && b.ok());
            if (a && b) {
                expect(variant == 0u ? b.selected.cost.memory_work < a.selected.cost.memory_work :
                                       b.selected.cost.memory_work == a.selected.cost.memory_work);
            }
        }
    };
    "tile_xir_expression_reduction_fusion_contract_and_cost"_test = [] {
        using namespace tile;
        for (auto lanes : {1u, 2u, 4u, 8u, 16u}) {
            for (auto variant = 0u; variant < 6u; variant++) {
                // Repeated first-consumer reads, retained snapshot, write
                // boundary, strict fold, reversed coordinates, first store.
                if (lanes > 1u && (variant == 3u || variant == 4u)) { continue; }
                auto kernel = tile_kernel("expression_consumer", [=](TensorView<const float, 2> input,
                                                                     TensorView<float, 2> alias, TensorView<float, 2> output) {
                                  auto m = axis("m", 1), n = axis("n", 65);
                                  for (auto &nest : parallel(shape(17))) {
                                      auto x = input[coord(nest.index(), 0), shape(m, n)];
                                      auto y = exp(x);
                                      if (variant == 2u || variant == 5u) {
                                          alias(coord(nest.index(), 0), shape(m, n)).store(variant == 5u ? y : full<float>(shape(m, n), 9.0f));
                                      }
                                      auto sum = Scalar<float>{2.5f};
                                      auto policy = variant == 3u ? reduction::fold_left : reduction::unordered_tree;
                                      for (auto &element : nest.reduce(shape(n), policy)) {
                                          auto i = variant == 4u ? 64 - element.index() : element.index();
                                          sum += y.at(coord(0, i)) * y.at(coord(0, i));
                                      }
                                      output(coord(nest.index(), 0), shape(m, n)).store(variant == 1u ? y + sum : full<float>(shape(m, n), sum));
                                  }
                              }).capture(tensor_shape(17, 65), tensor_shape(17, 65), tensor_shape(17, 65));
                auto off = bridge::xir::lower(kernel.function(), {.local_lanes = lanes});
                auto on = bridge::xir::lower(kernel.function(), {.local_lanes = lanes, .enable_expression_reduction_fusion = true});
                expect(off.ok() && on.ok()) << off.error << on.error;
                if (!off || !on) { continue; }
                auto admitted = variant < 2u;
                expect(eq(on.fused_reduction_expressions, admitted ? 1u : 0u)) << variant << lanes;
                expect(eq(on.elided_expression_snapshots, variant == 0u ? 1u : 0u));
                expect(eq(off.fused_reduction_expressions, 0u));
                expect(eq(on.fused_reduction_loads, 0u));
                expect(xir::xir_verify_module(on.module.get(), {.require_reachable_blocks = true}).succeeded());
                auto allocations = [](const auto &lowered) {
                    size_t count = 0u;
                    lowered.function->traverse_instructions([&](xir::Instruction *inst) noexcept { count += inst->isa<xir::AllocaInst>(); });
                    return count;
                };
                expect(eq(allocations(off) - allocations(on), static_cast<size_t>(on.elided_expression_snapshots)));
                // One CPU worker makes the prior's packet multiplier explicit.
                auto a = bridge::xir::plan(kernel.function(), {lanes, 1u}, {.block_size = 32u, .local_lanes = lanes});
                auto b = bridge::xir::plan(kernel.function(), {lanes, 1u}, {.block_size = 32u, .local_lanes = lanes, .enable_expression_reduction_fusion = true});
                expect(a.ok() && b.ok());
                if (a && b) {
                    auto packets = ceil_div(17u * lanes, lanes);
                    auto saved = admitted ? static_cast<double>((2u + on.elided_expression_snapshots) * ceil_div(65u, lanes) * lanes * 2u * packets) : 0.0;
                    expect(eq(a.selected.cost.memory_work - b.selected.cost.memory_work, saved)) << variant << lanes;
                    expect(eq(a.selected.cost.arithmetic_work, b.selected.cost.arithmetic_work));
                }
            }
        }
    };
    "tile_xir_pointwise_effect_intervals_and_partitioned_outputs"_test = [] {
        using namespace tile;
        for (auto lanes : {1u, 2u, 4u, 8u, 16u}) {
            for (auto variant = 0u; variant < 6u; variant++) {
                // Split output, distinct outputs, overlapping output, same
                // point output, escaping snapshot, read-after-write boundary.
                auto kernel = tile_kernel("shared_dag", [=](TensorView<const float, 2> input,
                                                            TensorView<float, 2> output, TensorView<float, 2> other) {
                                  auto m = axis("m", 1), n = axis("n", 65);
                                  for (auto &nest : parallel(shape(17))) {
                                      auto x = input[coord(nest.index(), 0), shape(m, n)];
                                      auto square = x * x;
                                      output(coord(nest.index(), 0), shape(m, n)).store(square + x);
                                      if (variant == 1u) {
                                          other(coord(nest.index(), 0), shape(m, n)).store(square - x);
                                      } else if (variant == 5u) {
                                          auto later = output[coord(nest.index(), 0), shape(m, n)];
                                          other(coord(nest.index(), 0), shape(m, n)).store(later * 2.0f);
                                      } else {
                                          auto offset = variant == 2u ? 1 : variant == 3u ? 0 :
                                                                                            65;
                                          output(coord(nest.index(), offset), shape(m, n)).store(square - x);
                                      }
                                      if (variant == 4u) {
                                          other(coord(nest.index(), 0), shape(m, n)).store(x + reduce(x, n, add));
                                      }
                                  }
                              }).capture(tensor_shape(17, 130), tensor_shape(17, 130), tensor_shape(17, 130));
                auto off = bridge::xir::lower(kernel.function(), {.block_size = 32u, .local_lanes = lanes});
                auto on = bridge::xir::lower(kernel.function(), {.block_size = 32u, .local_lanes = lanes, .enable_pointwise_fusion = true});
                expect(off.ok() && on.ok()) << off.error << on.error;
                if (!off || !on) { continue; }
                expect(eq(off.fused_pointwise_regions, 0u));
                expect(eq(on.fused_pointwise_loads, variant == 2u || variant == 4u ? 0u : variant == 5u ? 2u :
                                                                                                          1u))
                    << variant << lanes;
                if (variant < 4u && variant != 2u) {
                    expect(eq(on.fused_pointwise_regions, 1u));
                    expect(eq(on.fused_pointwise_stores, 2u));
                    expect(eq(on.pointwise_alias_checks, variant == 1u ? 3u : 1u));
                }
                expect(xir::xir_verify_module(on.module.get(), {.require_reachable_blocks = true}).succeeded());
            }
        }
    };
    "tile_xir_pointwise_does_not_cross_stages_or_reduction_state"_test = [] {
        using namespace tile;
        for (auto count : {0, 1, 3}) {
            auto kernel = tile_kernel("pointwise_stages", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
                              auto m = axis("m", 1), n = axis("n", 65);
                              for (auto &nest : parallel(shape(1))) {
                                  for (auto &step : nest.pipeline(shape(count))) {
                                      auto x = input[coord(step.index(), 0), shape(m, n)];
                                      step.stage("store");
                                      output(coord(step.index(), 0), shape(m, n)).store(x * 2.0f);
                                  }
                              }
                          }).capture(tensor_shape(3, 65), tensor_shape(3, 65));
            auto on = bridge::xir::lower(kernel.function(), {.enable_pointwise_fusion = true});
            expect(on.ok()) << on.error;
            if (on) { expect(eq(on.fused_pointwise_loads, 0u)); }
        }
    };
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

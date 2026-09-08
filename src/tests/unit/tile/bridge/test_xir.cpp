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

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main() {
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
            auto result = bridge::xir::lower(kernel.function());
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
            auto limited = bridge::xir::lower(kernel.function(), {.max_local_bytes = static_cast<uint32_t>(width * 4 - 1)});
            expect(!limited && limited.module == nullptr);
            expect(limited.error.find("snapshot storage budget") != string::npos);
            expect(bridge::xir::lower(kernel.function(), {.max_local_bytes = static_cast<uint32_t>(width * 4)}).ok());
            auto plan = bridge::xir::plan(kernel.function(), {8u, 8u});
            expect(plan.ok()) << plan.error;
            if (plan) {
                auto work = plan.selected.cost.arithmetic_work + plan.selected.cost.memory_work;
                if (previous_work > 0.0) { expect(work <= previous_work * 16.0); }
                previous_work = work;
            }
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

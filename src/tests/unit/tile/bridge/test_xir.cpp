#include "ut/ut.hpp"
#include "tile_xir_test_utils.h"
#include <luisa/tile/bridge/xir/lower.h>
#include <luisa/tile/bridge/xir/planner.h>
#include <luisa/tile/verifier.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main() {
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

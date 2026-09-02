// This exact source is compiled as both C++20 (comma adapter) and C++23
// (native multidimensional subscript). Both must capture identical TileIR.
#include "ut/ut.hpp"

#include <luisa/tile.h>
#include <type_traits>
#include <utility>

using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;

#if defined(__clang__) && (!defined(__cpp_multidimensional_subscript) || __cpp_multidimensional_subscript < 202110L)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-comma-subscript"
#endif
#include "tile_programming_poc.cpp"

namespace {

using Origin = decltype(coord(0, 0));
using View = TensorView<float, 2>;
using ConstView = TensorView<const float, 2>;

// Clang diagnoses the C++20 grammar even when the comma is deliberately
// overloaded. Keep the suppression local; C++23 never needs it.
static_assert(std::same_as<decltype(std::declval<View>()(std::declval<Origin>(), std::declval<IndexSpace>())), MemoryRef<float, 2>>);
static_assert(std::same_as<decltype(std::declval<View>().tile(std::declval<Origin>(), std::declval<IndexSpace>())), MemoryRef<float, 2>>);
static_assert(std::same_as<decltype(std::declval<View>()[std::declval<Origin>(), std::declval<IndexSpace>()]), Tile<float>>);
static_assert(std::same_as<decltype(std::declval<ConstView>()[std::declval<Origin>(), std::declval<IndexSpace>(), bounds::zero]), Tile<float>>);
static_assert(!std::is_assignable_v<Tile<float> &&, Tile<float>>);
static_assert(std::is_assignable_v<Tile<float> &, Tile<float>>);
static_assert(!std::is_assignable_v<MemoryRef<float, 2>, Tile<float>>);
static_assert(!std::is_copy_assignable_v<MemoryRef<float, 2>>);
static_assert(!std::is_convertible_v<MemoryRef<float, 2>, Tile<float>>);

template<typename Ref>
concept can_store = requires(Ref ref, Tile<float> value) { ref.store(value); };
static_assert(can_store<MemoryRef<float, 2>>);
static_assert(!can_store<MemoryRef<const float, 2>>);

void test_equivalent_read_syntax() {
    auto definition = tile_kernel("equivalent_reads", [](ConstView a, View c) {
        auto m = axis("m", 3);
        auto n = axis("n", 5);
        auto origin = coord(1, 2);
        auto space = shape(m, n);
        auto x = a[origin, space];
        auto y = a(origin, space).load();
        auto z = a.tile(origin, space).load();
        auto explicit_bounds = a[origin, space, bounds::assume];
        c(origin, space).store(x + y + z + explicit_bounds);
    });
    auto kernel = definition.capture(tensor_shape(4, 7), tensor_shape(4, 7));
    expect(kernel.valid());
    auto root = kernel.function().body().block(0);
    luisa::vector<Operation *> loads;
    size_t stores = 0;
    for (auto operation : root->operations()) {
        if (operation->kind() == OperationKind::VIEW_LOAD) { loads.emplace_back(operation); }
        if (operation->kind() == OperationKind::VIEW_STORE) { stores++; }
    }
    expect(eq(loads.size(), 4u));
    expect(eq(stores, 1u));
    if (loads.size() != 4u) { return; }
    for (auto load : loads) {
        expect(load->result(0)->type().is_tile());
        expect(eq(load->operand_count(), 3u));
        expect(load->domain() == loads.front()->domain());
        expect(load->result(0)->type() == loads.front()->result(0)->type());
        for (auto i = 0u; i < 3u; i++) { expect(load->operand(i) == loads.front()->operand(i)); }
    }
    for (auto i = 0u; i < 3u; i++) { expect(loads[i]->bounds_mode() == BoundsMode::ZERO); }
    expect(loads.back()->bounds_mode() == BoundsMode::ASSUME);
}

void test_tile_pipeline_and_mma() {
    auto definition = tile_kernel("tiled_gemm", [](ConstView a, ConstView b, View c) {
        auto gm = axis("gm", 2);
        auto gn = axis("gn", 3);
        auto kt = axis("kt", 4);
        auto m = axis("m", 8);
        auto n = axis("n", 8);
        auto k = axis("k", 4);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(kt), {.stages = 2, .initiation_interval = 1})) {
                step.stage("load");
                auto x = a[coord(nest[gm] * 8, step.index() * 4), shape(m, k)];
                auto y = b[coord(step.index() * 4, nest[gn] * 8), shape(k, n)];
                step.stage("compute");
                acc = mma(x, y, acc);
            }
            c(coord(nest[gm] * 8, nest[gn] * 8), shape(m, n)).store(acc);
        }
    });
    auto kernel = definition.capture(tensor_shape(13, 15), tensor_shape(15, 19), tensor_shape(13, 19));
    expect(kernel.valid());
    auto outer = kernel.function().body().block(0)->operations().front();
    Operation *pipeline = nullptr;
    for (auto op : outer->region(0)->block(0)->operations()) {
        if (op->kind() == OperationKind::PIPELINE) { pipeline = op; }
    }
    expect(pipeline != nullptr);
    if (pipeline == nullptr) { return; }
    expect(eq(pipeline->result_count(), 1u));
    expect(pipeline->result(0)->type().is_tile());
    auto body = pipeline->region(0)->block(0);
    size_t mmas = 0;
    size_t loads = 0;
    for (auto op : body->operations()) {
        if (op->kind() == OperationKind::MMA) {
            mmas++;
            expect(op->operand(2) == body->argument(1));
        }
        if (op->kind() == OperationKind::VIEW_LOAD) {
            loads++;
            expect(op->result(0)->type().is_tile());
        }
    }
    expect(eq(mmas, 1u));
    expect(eq(loads, 2u));
}

void test_tile_reduction_and_broadcast() {
    auto definition = tile_kernel("tile_softmax", [](ConstView x, View y) {
        auto row = axis("row", 2);
        auto col = axis("col", 7);
        auto value = x[coord(0, 0), shape(row, col)];
        auto shifted = exp(value - reduce(value, col, maximum));
        y(coord(0, 0), shape(row, col)).store(shifted / reduce(shifted, col, add));
    });
    auto kernel = definition.capture(tensor_shape(2, 7), tensor_shape(2, 7));
    expect(kernel.valid());
}

void test_positional_shapes() {
    auto definition = tile_kernel("positional_shapes", [](ConstView a, ConstView b, View c) {
        auto value = a[coord(0, 0), shape(3, 4)];
        auto row = b[coord(0, 0), shape(1, 4)];
        auto column = b[coord(0, 0), shape(3, 1)];
        c(coord(0, 0), shape(3, 4)).store(value + row + column);
    });
    auto kernel = definition.capture(tensor_shape(3, 4), tensor_shape(3, 4), tensor_shape(3, 4));
    expect(kernel.valid());

    auto invalid = tile_kernel("incompatible_positional_shapes", [](ConstView a) {
                       auto x = a[coord(0, 0), shape(3, 4)];
                       auto y = a[coord(0, 0), shape(3, 5)];
                       static_cast<void>(x + y);
                   }).capture(tensor_shape(3, 5));
    expect(!invalid.valid());
    expect(!invalid.diagnostics().empty());

    auto matrix = tile_kernel("positional_mma", [](ConstView a, ConstView b, View c) {
                      auto x = a[coord(0, 0), shape(3, 4)];
                      auto y = b[coord(0, 0), shape(4, 5)];
                      auto z = mma(x, y, zeros<float>(shape(3, 5)));
                      c(coord(0, 0), shape(3, 5)).store(z);
                  }).capture(tensor_shape(3, 4), tensor_shape(4, 5), tensor_shape(3, 5));
    expect(matrix.valid());
}

void test_pure_map_rejects_memory() {
    auto kernel = tile_kernel("impure_map", [](TensorView<const float, 1> a) {
                      auto i = axis("i", a.extent<0>());
                      static_cast<void>(map<float>(shape(i), [&](const Nest &nest) { return a(nest.index()).load(); }));
                  }).capture(tensor_shape(4));
    expect(!kernel.valid());
}

void test_documented_gemm() {
    auto definition = poc::make_gemm({.block_m = 3, .block_n = 5, .block_k = 7, .stages = 2});
    auto kernel = definition.capture(tensor_shape(7, 11), tensor_shape(11, 13), tensor_shape(7, 13));
    expect(kernel.valid());
}

void test_ancestor_coordinates() {
    auto kernel = tile_kernel("ancestor_coordinates", [](View out) {
                      for (auto &outer : parallel(shape(3))) {
                          for (auto &inner : outer.serial(shape(5))) {
                              out(coord(outer.index(), inner.index()), shape(1, 1)).store(zeros<float>(shape(1, 1)));
                          }
                      }
                  }).capture(tensor_shape(3, 5));
    expect(kernel.valid());
    auto outer = kernel.function().body().block(0)->operations().front();
    auto outer_body = outer->region(0)->block(0);
    auto inner = outer_body->operations().front();
    auto inner_body = inner->region(0)->block(0);
    Operation *store = nullptr;
    for (auto operation : inner_body->operations()) {
        if (operation->kind() == OperationKind::VIEW_STORE) { store = operation; }
    }
    expect(store != nullptr);
    if (store != nullptr) {
        expect(store->operand(1) == outer_body->argument(0));
        expect(store->operand(2) == inner_body->argument(0));
    }
}

template<bool tiled>
[[nodiscard]] auto make_state(float value) {
    if constexpr (tiled) {
        return full<float>(shape(1), value);
    } else {
        return Scalar<float>{value};
    }
}

template<bool tiled>
void test_loop_variable_identity() {
    auto definition = tile_kernel("loop_variable_identity", [](TensorView<float, 1> out) {
        auto a = make_state<tiled>(1.0f);
        auto b = a;
        auto snapshot = a;
        for (auto &step : serial(shape(3))) {
            static_cast<void>(step);
            a += snapshot;
            b += 2.0f;
        }
        if constexpr (tiled) {
            out(coord(0), shape(1)).store(a + b);
        } else {
            out(coord(0), shape(1)).store(full<float>(shape(1), a + b));
        }
    });
    auto kernel = definition.capture(tensor_shape(1));
    expect(kernel.valid());
    Operation *loop = nullptr;
    for (auto operation : kernel.function().body().block(0)->operations()) {
        if (operation->kind() == OperationKind::SERIAL) { loop = operation; }
    }
    expect(loop != nullptr);
    if (loop == nullptr) { return; }
    expect(eq(loop->result_count(), 2u));
    auto body = loop->region(0)->block(0);
    luisa::vector<Operation *> adds;
    for (auto operation : body->operations()) {
        if (operation->elementwise_op() == ElementwiseOp::ADD) { adds.emplace_back(operation); }
    }
    expect(eq(adds.size(), 2u));
    if (adds.size() != 2u || loop->result_count() != 2u) { return; }
    expect(loop->operand(0) == loop->operand(1));
    expect(adds[0]->operand(0) == body->argument(1));
    expect(adds[0]->operand(1) == loop->operand(0)); // immutable snapshot, not a's phi
    expect(adds[1]->operand(0) == body->argument(2));// b has a separate variable identity
}

template<bool tiled>
void test_direct_assignment_yield() {
    auto definition = tile_kernel("direct_assignment_yield", [](TensorView<float, 1> out) {
        auto a = make_state<tiled>(1.0f);
        auto b = make_state<tiled>(2.0f);
        for (auto &step : serial(shape(3))) {
            static_cast<void>(step);
            auto old_a = a;
            a += b;
            b = old_a;
        }
        if constexpr (tiled) {
            out(coord(0), shape(1)).store(a + b);
        } else {
            out(coord(0), shape(1)).store(full<float>(shape(1), a + b));
        }
    });
    auto kernel = definition.capture(tensor_shape(1));
    expect(kernel.valid());
    Operation *loop = nullptr;
    for (auto operation : kernel.function().body().block(0)->operations()) {
        if (operation->kind() == OperationKind::SERIAL) { loop = operation; }
    }
    expect(loop != nullptr);
    if (loop == nullptr) { return; }
    expect(eq(loop->result_count(), 2u));
    if (loop->result_count() != 2u) { return; }
    auto body = loop->region(0)->block(0);
    auto yield = body->operations().back();
    expect(yield->kind() == OperationKind::YIELD);
    expect(yield->operand(1) == body->argument(1));// previous iteration, not initial a
}

#if defined(__clang__) && (!defined(__cpp_multidimensional_subscript) || __cpp_multidimensional_subscript < 202110L)
#pragma clang diagnostic pop
#endif

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "tile_read_syntax_equivalence"_test = test_equivalent_read_syntax;
    "tile_pipeline_mma_and_implicit_carry"_test = test_tile_pipeline_and_mma;
    "tile_reduction_and_broadcast"_test = test_tile_reduction_and_broadcast;
    "tile_positional_shapes_and_mma"_test = test_positional_shapes;
    "tile_map_rejects_memory_effects"_test = test_pure_map_rejects_memory;
    "tile_documented_gemm"_test = test_documented_gemm;
    "tile_ancestor_coordinates"_test = test_ancestor_coordinates;
    "tile_loop_variable_identity"_test = test_loop_variable_identity<true>;
    "scalar_loop_variable_identity"_test = test_loop_variable_identity<false>;
    "tile_direct_assignment_yield"_test = test_direct_assignment_yield<true>;
    "scalar_direct_assignment_yield"_test = test_direct_assignment_yield<false>;
}

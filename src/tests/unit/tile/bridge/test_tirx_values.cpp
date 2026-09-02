#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <luisa/tile/algorithms.h>

using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;

namespace {

void test_bounds_and_snapshot(Runtime &runtime) {
    auto definition = tile_kernel("tile_snapshot", [](TensorView<float, 2> a, TensorView<float, 2> out) {
        auto program = axis("program", 1);
        auto m = axis("m", 3);
        auto n = axis("n", 5);
        for (auto &nest : parallel(shape(program))) {
            auto origin = coord(nest.index() - 1, 2);
            auto x = a[origin, shape(m, n)];
            auto y = a(origin, shape(m, n)).load();
            auto z = a.tile(origin, shape(m, n)).load();
            auto filled = a(origin, shape(m, n)).load(-3.0f);
            a(coord(0, 0), shape(m, n)).store(full<float>(shape(m, n), 17.0f));
            out(coord(0, 0), shape(m, n)).store(x + y + z + filled);
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 5), tensor_shape(3, 5));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    luisa::vector<float> input(15);
    for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i + 1); }
    auto a = runtime.upload<float>({3, 5}, input);
    auto out = runtime.allocate<float>({3, 5});
    (*executable.entry)(a, out);
    auto actual = runtime.download<float>(out, 15);
    auto overwritten = runtime.download<float>(a, 15);
    for (auto m = 0; m < 3; m++) {
        for (auto n = 0; n < 5; n++) {
            auto expected = m >= 1 && n < 3 ? 4.0f * input[(m - 1) * 5 + n + 2] : -3.0f;
            expect(eq(actual[m * 5 + n], expected));
            expect(eq(overwritten[m * 5 + n], 17.0f));
        }
    }
}

void test_singleton_execution_coordinates(Runtime &runtime) {
    auto definition = tile_kernel("singleton_coordinates", [](TensorView<float, 4> out) {
        auto a = axis("a", 2);
        auto b = axis("b", 1);
        auto c = axis("c", 3);
        auto d = axis("d", 1);
        for (auto &nest : parallel(shape(a, b, c, d))) {
            auto value = full<float>(shape(1, 1, 1, 1), cast<float>(1000 * nest[a] + 100 * nest[b] + 10 * nest[c] + nest[d]));
            out(coord(nest[a], nest[b], nest[c], nest[d]), shape(1, 1, 1, 1)).store(value);
        }
    });
    auto kernel = definition.capture(tensor_shape(2, 1, 3, 1));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto out = runtime.upload<float>({2, 1, 3, 1}, luisa::vector<float>(6, -1.0f));
    (*executable.entry)(out);
    auto actual = runtime.download<float>(out, 6);
    for (auto i = 0u; i < 6u; i++) { expect(eq(actual[i], static_cast<float>((i / 3) * 1000 + (i % 3) * 10))); }
}

void test_tile_yield_is_simultaneous(Runtime &runtime) {
    auto definition = tile_kernel("tile_simultaneous_yield", [](TensorView<const float, 1> a,
                                                                TensorView<const float, 1> b,
                                                                TensorView<float, 1> out_a,
                                                                TensorView<float, 1> out_b) {
        auto program = axis("program", 1);
        auto i = axis("i", 7);
        auto iterations = axis("iterations", 3);
        for (auto &nest : parallel(shape(program))) {
            auto x = a[coord(0), shape(i)];
            auto y = b[coord(0), shape(i)];
            for (auto &step : nest.serial(shape(iterations))) {
                static_cast<void>(step);
                auto old_x = x;
                x = y + 1.0f;
                y = old_x - 2.0f;
            }
            out_a(coord(0), shape(i)).store(x);
            out_b(coord(0), shape(i)).store(y);
        }
    });
    auto kernel = definition.capture(tensor_shape(7), tensor_shape(7), tensor_shape(7), tensor_shape(7));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    luisa::vector<float> a_values{0, 1, 2, 3, 4, 5, 6};
    luisa::vector<float> b_values{10, 11, 12, 13, 14, 15, 16};
    auto a = runtime.upload<float>({7}, a_values);
    auto b = runtime.upload<float>({7}, b_values);
    auto out_a = runtime.allocate<float>({7});
    auto out_b = runtime.allocate<float>({7});
    (*executable.entry)(a, b, out_a, out_b);
    auto x = runtime.download<float>(out_a, 7);
    auto y = runtime.download<float>(out_b, 7);
    for (auto i = 0u; i < 7u; i++) {
        expect(eq(x[i], b_values[i]));
        expect(eq(y[i], a_values[i] - 3.0f));
    }
}

void test_positional_broadcast(Runtime &runtime) {
    auto definition = tile_kernel("positional_broadcast", [](TensorView<const float, 2> a, TensorView<float, 2> out) {
        auto program = axis("program", 1);
        for (auto &nest : parallel(shape(program))) {
            auto origin = coord(nest.index(), 0);
            auto x = a[origin, shape(3, 5)];
            auto row = a[origin, shape(1, 5)];
            auto column = a[origin, shape(3, 1)];
            out(origin, shape(3, 5)).store(x + row + column);
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 5), tensor_shape(3, 5));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    luisa::vector<float> input(15);
    for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i); }
    auto a = runtime.upload<float>({3, 5}, input);
    auto out = runtime.allocate<float>({3, 5});
    (*executable.entry)(a, out);
    auto actual = runtime.download<float>(out, 15);
    for (auto m = 0; m < 3; m++) {
        for (auto n = 0; n < 5; n++) { expect(eq(actual[m * 5 + n], input[m * 5 + n] + input[n] + input[m * 5])); }
    }
}

void test_ancestor_coordinates(Runtime &runtime) {
    auto definition = tile_kernel("ancestor_coordinates", [](TensorView<float, 2> out) {
        for (auto &outer : parallel(shape(3))) {
            for (auto &inner : outer.serial(shape(5))) {
                auto value = cast<float>(outer.index() * 10 + inner.index());
                out(coord(outer.index(), inner.index()), shape(1, 1)).store(full<float>(shape(1, 1), value));
            }
        }
    });
    auto executable = runtime.build(definition.capture(tensor_shape(3, 5)));
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto out = runtime.upload<float>({3, 5}, luisa::vector<float>(15, -1.0f));
    (*executable.entry)(out);
    auto actual = runtime.download<float>(out, 15);
    for (auto m = 0; m < 3; m++) {
        for (auto n = 0; n < 5; n++) { expect(eq(actual[m * 5 + n], static_cast<float>(m * 10 + n))); }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_subtile_bounds_and_snapshot"_test = [&] { test_bounds_and_snapshot(runtime); };
    "tile_singleton_execution_coordinates"_test = [&] { test_singleton_execution_coordinates(runtime); };
    "tile_simultaneous_value_yield"_test = [&] { test_tile_yield_is_simultaneous(runtime); };
    "tile_positional_broadcast"_test = [&] { test_positional_broadcast(runtime); };
    "tile_ancestor_coordinates"_test = [&] { test_ancestor_coordinates(runtime); };
}

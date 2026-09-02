// Execution constraints must survive structural export and cannot be ignored
// by a target schedule that does not know how to realize them.

#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/tile/value.h>

using namespace luisa::compute::tile;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;

namespace {

[[nodiscard]] Kernel make_copy(exec::Scope scope, int64_t count) {
    auto definition = tile_kernel("execution_copy", [scope, count](TensorView<const float, 1> input,
                                                                   TensorView<float, 1> output) {
        for (auto &nest : parallel(shape(count), scope)) {
            auto origin = coord(nest.index());
            output(origin, shape(1)).store(input.tile(origin, shape(1)).load() * 2.0f + 1.0f);
        }
    });
    // An empty execution domain still has valid nonempty buffers. Its body
    // must neither write the sentinel nor attempt an empty GPU launch.
    auto size = std::max<int64_t>(count, 1);
    return definition.capture(tensor_shape(size), tensor_shape(size));
}

void check_copy(Runtime &runtime, exec::Scope scope, int64_t count) {
    auto kernel = make_copy(scope, count);
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto size = std::max<int64_t>(count, 1);
    luisa::vector<float> input(static_cast<size_t>(size));
    for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i % 29u) - 7.0f; }
    auto source = runtime.upload<float>({size}, input);
    auto output = runtime.upload<float>({size}, luisa::vector<float>(input.size(), -31.0f));
    (*executable.entry)(source, output);
    auto actual = runtime.download<float>(output, input.size());
    for (auto i = 0u; i < actual.size(); i++) {
        expect(eq(actual[i], count == 0 ? -31.0f : input[i] * 2.0f + 1.0f));
    }
}

void test_explicit_worker(Runtime &runtime) {
    for (auto count : {1, 37, 257, 1003}) { check_copy(runtime, exec::Scope::WORKER, count); }
}

void test_empty_parallel(Runtime &runtime) {
    check_copy(runtime, exec::Scope::AUTOMATIC, 0);
    check_copy(runtime, exec::Scope::WORKER, 0);
    if (runtime.target() == "metal") { check_copy(runtime, exec::Scope::GROUP, 0); }
}

void test_scope_survives_export() {
    auto kernel = make_copy(exec::Scope::WORKER, 7);
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto count = 0u;
    tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto loop = node.as<tvm::tirx::ForNode>()) {
            if (loop->annotations.count("luisa.tile.logical_parallel")) {
                count++;
                expect(loop->kind == tvm::tirx::ForKind::kSerial);
                auto scope = loop->annotations.Get("luisa.tile.execution_scope");
                expect(scope.has_value());
                if (scope) { expect(scope.value().cast<tvm::ffi::String>() == "worker"); }
            }
        }
    });
    expect(eq(count, 1u));
}

void test_unsupported_scopes(Runtime &runtime) {
    constexpr std::array cases{
        std::pair{exec::Scope::DEVICE, "device"},
        std::pair{exec::Scope::GROUP, "group"},
        std::pair{exec::Scope::SUBGROUP, "subgroup"}};
    for (auto [scope, name] : cases) {
        if (scope == exec::Scope::GROUP && runtime.target() == "metal") {
            check_copy(runtime, scope, 7);
            continue;
        }
        auto executable = runtime.build(make_copy(scope, 7));
        expect(!executable.ok()) << "unsupported scope must not silently compile: " << name;
        expect(executable.error.find(name) != luisa::string::npos) << executable.error;
        expect(executable.error.find("execution scope") != luisa::string::npos) << executable.error;
    }
}

void test_unknown_scope(Runtime &runtime) {
    auto kernel = make_copy(exec::Scope::WORKER, 7);
    for (auto operation : kernel.function().body().block(0)->operations()) {
        if (operation->kind() == OperationKind::PARALLEL) {
            operation->set_execution_scope_constraint("unavailable_accelerator");
        }
    }
    expect(kernel.valid());
    auto executable = runtime.build(kernel);
    expect(!executable.ok());
    expect(executable.error.find("unavailable_accelerator") != luisa::string::npos) << executable.error;
}

void test_nested_worker_rejected(Runtime &runtime) {
    for (auto outer_scope : {exec::Scope::AUTOMATIC, exec::Scope::WORKER}) {
        auto definition = tile_kernel("nested_worker", [outer_scope](TensorView<float, 2> output) {
            for (auto &outer : parallel(shape(3), outer_scope)) {
                // An unbound intermediate level must not hide a constraint
                // from the nearest ancestor already mapped to a worker.
                for (auto &middle : outer.serial(shape(1))) {
                    for (auto &inner : middle.parallel(shape(5), exec::Scope::WORKER)) {
                        output(coord(outer.index(), inner.index()), shape(1, 1))
                            .store(full<float>(shape(1, 1), 1.0f));
                    }
                }
            }
        });
        auto kernel = definition.capture(tensor_shape(3, 5));
        expect(kernel.valid());
        auto executable = runtime.build(kernel);
        expect(!executable.ok());
        expect(executable.error.find("worker") != luisa::string::npos) << executable.error;
        expect(executable.error.find("nested") != luisa::string::npos) << executable.error;
    }
}

[[nodiscard]] Kernel make_vector_copy(int64_t count) {
    auto definition = tile_kernel("worker_vector_copy", [count](TensorView<const float, 1> input,
                                                                TensorView<float, 1> output) {
        for (auto &worker : parallel(shape((count + 3) / 4), exec::Scope::WORKER)) {
            for (auto &lane : worker.parallel(shape(4), exec::Scope::VECTOR)) {
                auto origin = coord(worker.index() * 4 + lane.index());
                output(origin, shape(1)).store(input.tile(origin, shape(1)).load() * 2.0f + 1.0f);
            }
        }
    });
    return definition.capture(tensor_shape(count), tensor_shape(count));
}

void test_worker_vector(Runtime &runtime) {
    if (runtime.target() != "llvm") {
        auto executable = runtime.build(make_vector_copy(37));
        expect(!executable.ok());
        expect(executable.error.find("vector") != luisa::string::npos) << executable.error;
        return;
    }
    for (auto count : {4, 37, 256, 1003}) {
        auto kernel = make_vector_copy(count);
        auto executable = runtime.build(kernel, true);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        luisa::vector<float> input(count);
        for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i % 31u); }
        auto source = runtime.upload<float>({count}, input);
        auto output = runtime.allocate<float>({count});
        (*executable.entry)(source, output);
        auto actual = runtime.download<float>(output, input.size());
        auto mismatch = std::mismatch(actual.begin(), actual.end(), input.begin(),
                                      [](float actual_value, float input_value) { return actual_value == input_value * 2.0f + 1.0f; });
        expect(mismatch.first == actual.end()) << "count=" << count << " index=" << (mismatch.first - actual.begin());
        if (count == 256) {
            auto source_ir = executable.module.value()->InspectSource("ll");
            expect(std::string_view{source_ir.data(), source_ir.size()}.find("<4 x float>") != std::string_view::npos)
                << "explicit vector scope must reach LLVM vector instructions";
        }
    }
    auto kernel = make_vector_copy(37);
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.vectorize = false;
    auto compilation = compile(native.value, kernel.function().name(), options);
    expect(!compilation.ok());
    expect(compilation.error().find("vector") != luisa::string_view::npos) << compilation.error();
}

void test_vector_private_tiles_and_carries(Runtime &runtime) {
    auto definition = tile_kernel("vector_private_state", [](TensorView<const float, 2> input,
                                                             TensorView<float, 1> output) {
        auto rows = input.extent<0>();
        auto m = axis("m", 1);
        auto k = axis("k", input.extent<1>());
        for (auto &worker : parallel(shape((rows + 3) / 4), exec::Scope::WORKER)) {
            // An ancestor Tile is shared read-only by this worker's vector
            // lanes; it must not itself acquire another lane dimension.
            auto first = input.tile(coord(worker.index() * 4, 0), shape(4, 1)).load();
            for (auto &lane : worker.parallel(shape(4), exec::Scope::VECTOR)) {
                auto row = worker.index() * 4 + lane.index();
                auto x = input.tile(coord(row, 0), shape(m, k)).load();
                auto acc = reduce(x, k, add);
                auto previous = full<float>(shape(m), first.at(coord(lane.index(), 0)));
                for (auto &step : lane.serial(shape(3))) {
                    static_cast<void>(step);
                    auto old = acc;
                    acc += previous;
                    previous = old;
                }
                output(coord(row), shape(m)).store(acc);
            }
        }
    });
    for (auto rows : {1, 4, 17, 37}) {
        constexpr auto columns = 7;
        auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows));
        auto executable = runtime.build(kernel, true);
        if (runtime.target() != "llvm") {
            expect(!executable.ok());
            expect(executable.error.find("vector") != luisa::string::npos) << executable.error;
            continue;
        }
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        luisa::vector<float> input(static_cast<size_t>(rows * columns));
        for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(static_cast<int32_t>(i % 23u) - 11) * 0.125f; }
        auto source = runtime.upload<float>({rows, columns}, input);
        auto output = runtime.allocate<float>({rows});
        (*executable.entry)(source, output);
        auto actual = runtime.download<float>(output, rows);
        luisa::vector<float> expected(rows);
        for (auto row = 0; row < rows; row++) {
            auto sum = 0.0f;
            for (auto column = 0; column < columns; column++) { sum += input[row * columns + column]; }
            expected[row] = sum * 3.0f + input[row * columns] * 2.0f;
        }
        expect(actual == expected) << "vector private state rows=" << rows;
    }
}

void test_vector_roots_and_nesting(Runtime &runtime) {
    if (runtime.target() == "llvm") {
        for (auto count : {0, 1, 3, 4, 7, 16}) { check_copy(runtime, exec::Scope::VECTOR, count); }
    }
    auto definition = tile_kernel("nested_vector", [](TensorView<float, 2> output) {
        for (auto &outer : parallel(shape(3), exec::Scope::VECTOR)) {
            for (auto &middle : outer.serial(shape(1))) {
                for (auto &inner : middle.parallel(shape(4), exec::Scope::VECTOR)) {
                    output(coord(outer.index(), inner.index()), shape(1, 1))
                        .store(full<float>(shape(1, 1), 1.0f));
                }
            }
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 4));
    expect(kernel.valid());
    auto executable = runtime.build(kernel);
    expect(!executable.ok());
    expect(executable.error.find("vector") != luisa::string::npos) << executable.error;
    if (runtime.target() == "llvm") {
        expect(executable.error.find("nested") != luisa::string::npos) << executable.error;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_execution_explicit_worker"_test = [&] { test_explicit_worker(runtime); };
    "tile_execution_empty_domain"_test = [&] { test_empty_parallel(runtime); };
    "tile_execution_scope_preserved"_test = test_scope_survives_export;
    "tile_execution_unsupported_scopes"_test = [&] { test_unsupported_scopes(runtime); };
    "tile_execution_unknown_scope"_test = [&] { test_unknown_scope(runtime); };
    "tile_execution_nested_worker"_test = [&] { test_nested_worker_rejected(runtime); };
    "tile_execution_worker_vector"_test = [&] { test_worker_vector(runtime); };
    "tile_execution_vector_private_tiles_and_carries"_test = [&] { test_vector_private_tiles_and_carries(runtime); };
    "tile_execution_vector_roots_and_nesting"_test = [&] { test_vector_roots_and_nesting(runtime); };
}

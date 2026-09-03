// Test native Tile software-pipeline planning on CPU and physical Metal.
// Covers preserved cuts, real buffer versions, short/ragged/multiaxis loops,
// simultaneous carries, aliases, explicit Memory, and shared-capacity limits.
#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <luisa/tile/memory.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <tvm/tirx/stmt_functor.h>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;

namespace {

[[nodiscard]] tvm::ffi::String metal_source(const tvm::ffi::Module &module) {
    if (std::string_view{module->kind()} == "metal") { return module->InspectSource("metal"); }
    for (auto &&child : module->imports()) {
        auto source = metal_source(child.cast<tvm::ffi::Module>());
        if (!source.empty()) { return source; }
    }
    return {};
}

class PipelineIR final : public tvm::tirx::StmtVisitor {
public:
    size_t pipelines{0u};
    size_t cuts{0u};
    std::string first_storage;

protected:
    void VisitStmt_(const tvm::tirx::ForNode *loop) final {
        if (loop->annotations.count("luisa.tile.pipeline") != 0u) {
            pipelines++;
            if (auto sequence = loop->body.as<tvm::tirx::SeqStmtNode>()) {
                for (auto &&statement : sequence->seq) {
                    if (auto allocation = statement.as<tvm::tirx::AllocBufferNode>()) {
                        if (first_storage.empty()) { first_storage = std::string{allocation->buffer.name()}; }
                    }
                }
            }
        }
        StmtVisitor::VisitStmt_(loop);
    }
    void VisitStmt_(const tvm::tirx::AttrStmtNode *attribute) final {
        if (attribute->attr_key == "luisa.tile.pipeline_stage") { cuts++; }
        StmtVisitor::VisitStmt_(attribute);
    }
};

void expect_near(const luisa::vector<float> &actual, const luisa::vector<float> &expected) {
    expect(eq(actual.size(), expected.size()));
    if (actual.size() != expected.size()) { return; }
    auto correct = true;
    for (auto i = 0u; i < actual.size(); i++) {
        correct &= std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= 1e-5f;
    }
    expect(correct) << "pipeline execution must preserve sequential value/effect semantics";
}

void test_prefetch(Runtime &runtime, int32_t iterations, int32_t columns,
                   exec::Scope scope, PipelinePolicy policy = {.stages = 2}, bool multi_axis = false) {
    constexpr auto rows = 3;
    auto height = std::max(1, rows * iterations);
    auto definition = tile_kernel("pipeline_prefetch", [=](TensorView<const float, 2> input,
                                                           TensorView<float, 2> output) {
        auto m = axis("m", 1);
        auto n = axis("n", columns);
        auto u = axis("u", multi_axis ? 2 : 1);
        auto v = axis("v", multi_axis ? iterations / 2 : iterations);
        for (auto &nest : parallel(shape(rows), scope)) {
            auto acc = full<float>(shape(m, n), 0.75f);
            auto other = full<float>(shape(m, n), 1.25f);
            for (auto &step : nest.pipeline(shape(u, v), policy)) {
                auto index = step.index(u) * (multi_axis ? iterations / 2 : iterations) + step.index(v);
                step.stage("load");
                auto x = input.tile(coord(nest.index() * iterations + index, 0), shape(m, n)).load();
                step.stage("compute");
                auto previous = acc;
                acc = acc * 0.5f + x;
                other = other + previous * 0.25f;
                // Three source phases with a two-iteration scheduling window.
                // The final yield snapshots both recurrences simultaneously.
                step.stage("finish");
            }
            output(coord(nest.index(), 0), shape(m, n)).store(acc + other);
        }
    });
    auto kernel = definition.capture(tensor_shape(height, columns), tensor_shape(rows, columns));
    auto native = bridge::tirx::lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    // The installed TVMx package exposes this checker through its native
    // registry, but hides the direct C++ function symbol.
    static auto verify_native = tvm::ffi::Function::GetGlobalRequired("tirx.analysis.VerifyWellFormed");
    expect(verify_native(native.value, false).cast<bool>());
    PipelineIR structure;
    structure(native.value->body);
    expect(eq(structure.pipelines, 1u));
    expect(eq(structure.cuts, 3u));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    if (runtime.target() == "metal" && iterations == 5 && columns == 37 && policy.stages == 2 && policy.initiation_interval == 1) {
        auto source = metal_source(executable.module.value());
        auto code = std::string_view{source.data(), source.size()};
        // This is a structural acceptance check, not just equivalent serial
        // results: the cross-stage load must have two actual storage versions.
        expect(!structure.first_storage.empty());
        auto storage_name = structure.first_storage + (scope == exec::Scope::GROUP ? "_shared" : "");
        expect(code.find(storage_name + "[74]") != std::string_view::npos) << source;
    }
    luisa::vector<float> input_values(static_cast<size_t>(height * columns));
    for (auto i = 0u; i < input_values.size(); i++) {
        input_values[i] = static_cast<float>(static_cast<int32_t>(i % 31u) - 15) * 0.0625f;
    }
    auto input = runtime.upload<float>({height, columns}, input_values);
    auto output = runtime.allocate<float>({rows, columns});
    for (auto repeat = 0; repeat < 2; repeat++) {
        (*executable.entry)(input, output);
        luisa::vector<float> expected(rows * columns);
        for (auto row = 0; row < rows; row++) {
            for (auto col = 0; col < columns; col++) {
                auto acc = 0.75f;
                auto other = 1.25f;
                for (auto iteration = 0; iteration < iterations; iteration++) {
                    auto previous = acc;
                    acc = acc * 0.5f + input_values[(row * iterations + iteration) * columns + col];
                    other += previous * 0.25f;
                }
                expected[row * columns + col] = acc + other;
            }
        }
        expect_near(runtime.download<float>(output, expected.size()), expected);
    }
}

void test_aliases(Runtime &runtime, bool noalias) {
    constexpr auto iterations = 7;
    constexpr auto columns = 37;
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto definition = tile_kernel("pipeline_alias", [=](TensorView<const float, 2> input,
                                                        TensorView<float, 2> output) {
        auto m = axis("m", 1);
        auto n = axis("n", columns);
        for (auto &nest : parallel(shape(1), scope)) {
            for (auto &step : nest.pipeline(shape(iterations))) {
                step.stage("read");
                auto x = input.tile(coord(step.index(), 0), shape(m, n)).load();
                step.stage("write");
                output(coord(step.index() + 1, 0), shape(m, n)).store(x + 1.0f);
            }
        }
    });
    auto kernel = definition.capture(tensor_shape(iterations + 1, columns), tensor_shape(iterations + 1, columns));
    auto executable = runtime.build(kernel, noalias);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    luisa::vector<float> values((iterations + 1) * columns);
    for (auto i = 0u; i < values.size(); i++) { values[i] = static_cast<float>(i) * 0.0625f; }
    auto input = runtime.upload<float>({iterations + 1, columns}, values);
    auto output = noalias ? runtime.upload<float>({iterations + 1, columns}, values) : input;
    (*executable.entry)(input, output);
    auto expected = values;
    for (auto row = 1; row <= iterations; row++) {
        for (auto col = 0; col < columns; col++) {
            expected[row * columns + col] = noalias ? values[(row - 1) * columns + col] + 1.0f : values[col] + static_cast<float>(row);
        }
    }
    expect_near(runtime.download<float>(output, expected.size()), expected);
}

void test_memory_and_carries(Runtime &runtime, bool iteration_local) {
    constexpr auto iterations = 7;
    constexpr auto columns = 37;
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto definition = tile_kernel("pipeline_memory", [=](TensorView<const float, 2> input,
                                                         TensorView<float, 2> output) {
        auto m = axis("m", 1);
        auto n = axis("n", columns);
        for (auto &nest : parallel(shape(1), scope)) {
            auto state = memory<float>(shape(m, n));
            state.store(full<float>(shape(m, n), 0.5f));
            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(iterations))) {
                step.stage("load");
                auto x = input.tile(coord(step.index(), 0), shape(m, n)).load();
                if (iteration_local) {
                    auto temporary = memory<float>(shape(m, n));
                    temporary.store(x);
                    auto snapshot = temporary.load();
                    step.stage("compute");
                    acc = acc + snapshot + temporary.load();
                    step.stage("late write");
                    temporary.store(full<float>(shape(m, n), -1024.0f));
                } else {
                    auto previous = state.load();
                    step.stage("compute");
                    state.store(previous * 0.5f + x);
                    acc = acc + previous;
                }
            }
            output(coord(0, 0), shape(m, n)).store(acc);
        }
    });
    auto executable = runtime.build(definition.capture(tensor_shape(iterations, columns), tensor_shape(1, columns)));
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    luisa::vector<float> values(iterations * columns);
    for (auto i = 0u; i < values.size(); i++) { values[i] = static_cast<float>(i % 17u) * 0.125f; }
    auto input = runtime.upload<float>({iterations, columns}, values);
    auto output = runtime.allocate<float>({1, columns});
    (*executable.entry)(input, output);
    luisa::vector<float> expected(columns, 0.0f);
    for (auto col = 0; col < columns; col++) {
        auto state = 0.5f;
        for (auto iteration = 0; iteration < iterations; iteration++) {
            auto x = values[iteration * columns + col];
            expected[col] += iteration_local ? 2.0f * x : state;
            state = state * 0.5f + x;
        }
    }
    expect_near(runtime.download<float>(output, columns), expected);
}

void test_stable_yield(Runtime &runtime, bool pipelined, uint32_t window) {
    constexpr auto columns = 37;
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    for (auto iterations : {0, 1, 5}) {
        auto definition = tile_kernel("stable_yield", [=](TensorView<const float, 2> input,
                                                          TensorView<float, 2> output) {
            auto m = axis("m", 1);
            auto n = axis("n", columns);
            auto space = shape(m, n);
            for (auto &nest : parallel(shape(1), scope)) {
                auto current = zeros<float>(space);
                auto history = full<float>(space, 1.0f);
                auto range = pipelined ? nest.pipeline(shape(iterations), {.stages = window}) : nest.serial(shape(iterations));
                for (auto &step : range) {
                    if (pipelined) { step.stage("load"); }
                    auto next = input[coord(step.index(), 0), space];
                    if (pipelined) { step.stage("update"); }
                    auto previous = current;
                    // A stable first update must not overwrite current until
                    // the later, dependent history expression is snapshotted.
                    current = next;
                    history += previous;
                }
                output(coord(0, 0), space).store(current);
                output(coord(1, 0), space).store(history);
            }
        });
        auto height = std::max(iterations, 1);
        auto kernel = definition.capture(tensor_shape(height, columns), tensor_shape(2, columns));
        auto native = bridge::tirx::lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { continue; }
        auto allocations = size_t{0u};
        tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
            allocations += node.as<tvm::tirx::AllocBufferNode>() != nullptr;
        });
        // Two carries, one immutable input load, and only ONE yield snapshot.
        // The stable input load must not get copied into a fifth allocation.
        expect(eq(allocations, 4u));
        auto executable = runtime.build(kernel);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        luisa::vector<float> values(height * columns);
        for (auto i = 0u; i < values.size(); i++) { values[i] = static_cast<float>(i % 29u) * 0.125f; }
        auto input = runtime.upload<float>({height, columns}, values);
        auto output = runtime.allocate<float>({2, columns});
        (*executable.entry)(input, output);
        luisa::vector<float> expected(2 * columns, 0.0f);
        for (auto col = 0; col < columns; col++) {
            expected[col] = iterations == 0 ? 0.0f : values[(iterations - 1) * columns + col];
            expected[columns + col] = 1.0f;
            for (auto row = 0; row + 1 < iterations; row++) { expected[columns + col] += values[row * columns + col]; }
        }
        expect_near(runtime.download<float>(output, expected.size()), expected);
    }
}

void test_shared_capacity(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto columns = 2200;
    auto definition = tile_kernel("pipeline_capacity", [](TensorView<const float, 2> input,
                                                          TensorView<float, 2> output) {
        auto m = axis("m", 1);
        auto n = axis("n", columns);
        for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(3))) {
                step.stage("load");
                auto x = input.tile(coord(step.index(), 0), shape(m, n)).load();
                step.stage("compute");
                acc += x;
            }
            output(coord(0, 0), shape(m, n)).store(acc);
        }
    });
    // Three 8,800-byte arrays fit in 32 KiB. A speculative fourth version
    // would not; the compiler must retain the legal ordered implementation.
    auto executable = runtime.build(definition.capture(tensor_shape(3, columns), tensor_shape(1, columns)));
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto input = runtime.upload<float>({3, columns}, luisa::vector<float>(3 * columns, 0.25f));
    auto output = runtime.allocate<float>({1, columns});
    (*executable.entry)(input, output);
    expect_near(runtime.download<float>(output, columns), luisa::vector<float>(columns, 0.75f));
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_native_pipeline_prefetch"_test = [&] {
        for (auto iterations : {0, 1, 2, 3, 5, 9}) {
            for (auto columns : {1, 7, 37, 129}) {
                test_prefetch(runtime, iterations, columns, exec::Scope::WORKER);
                if (runtime.target() == "metal") { test_prefetch(runtime, iterations, columns, exec::Scope::GROUP); }
            }
        }
    };
    "tile_native_pipeline_policies"_test = [&] {
        for (auto window : {0u, 1u, 2u, 4u}) {
            for (auto interval : {1u, 3u}) { test_prefetch(runtime, 6, 7, exec::Scope::WORKER, {window, interval}, true); }
        }
    };
    "tile_native_pipeline_aliases"_test = [&] {
        test_aliases(runtime, false);
        test_aliases(runtime, true);
    };
    "tile_native_pipeline_memory"_test = [&] {
        test_memory_and_carries(runtime, false);
        test_memory_and_carries(runtime, true);
    };
    "tile_native_pipeline_shared_capacity"_test = [&] { test_shared_capacity(runtime); };
    "tile_native_stable_yield_snapshot_elision"_test = [&] {
        test_stable_yield(runtime, false, 1u);
        for (auto window : {1u, 2u}) { test_stable_yield(runtime, true, window); }
    };
}

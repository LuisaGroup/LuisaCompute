// Execution constraints must survive structural export and cannot be ignored
// by a target schedule that does not know how to realize them.

#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string_view>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <luisa/core/mathematics.h>
#include <luisa/tile/value.h>

using namespace luisa::compute::tile;
using luisa::ceil_div;
using namespace luisa::compute::tile::bridge::tirx;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;

namespace {

[[nodiscard]] tvm::ffi::String metal_source(
    const tvm::ffi::Module &module) {
    if (std::string_view{module->kind()} == "metal") {
        return module->InspectSource("metal");
    }
    for (auto &&child : module->imports()) {
        auto source = metal_source(child.cast<tvm::ffi::Module>());
        if (!source.empty()) { return source; }
    }
    return {};
}

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

[[nodiscard]] bool has_cpu_parallel_launch(const luisa::test::tile_tirx::Executable &executable) {
    if (!executable.module) { return false; }
    auto source = executable.module.value()->InspectSource("ll");
    auto code = std::string_view{source.data(), source.size()};
    return code.find("load ptr, ptr @__TVMBackendParallelLaunch") != std::string_view::npos;
}

[[nodiscard]] Kernel make_exp_copy(exec::Scope scope, int64_t count) {
    auto definition = tile_kernel("execution_exp_copy", [scope, count](TensorView<const float, 1> input,
                                                                       TensorView<float, 1> output) {
        for (auto &nest : parallel(shape(count), scope)) {
            auto origin = coord(nest.index());
            output(origin, shape(1)).store(exp(input[origin, shape(1)]));
        }
    });
    return definition.capture(tensor_shape(count), tensor_shape(count));
}

void test_cpu_parallel_launch_cost(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    auto small = runtime.build(make_copy(exec::Scope::AUTOMATIC, 7));
    auto expensive = runtime.build(make_exp_copy(exec::Scope::AUTOMATIC, 7));
    auto explicit_worker = runtime.build(make_copy(exec::Scope::WORKER, 7));
    auto boundary = runtime.build(make_copy(exec::Scope::AUTOMATIC, 64));
    auto disabled_options = PlannerOptions{};
    disabled_options.enabled = false;
    auto disabled = runtime.build(make_copy(exec::Scope::AUTOMATIC, 7), false, false, true, false, disabled_options);
    for (auto executable : {&small, &expensive, &explicit_worker, &boundary, &disabled}) {
        expect(executable->ok()) << executable->error;
    }
    if (!small.ok() || !expensive.ok() || !explicit_worker.ok() || !boundary.ok() || !disabled.ok()) { return; }
    expect(!has_cpu_parallel_launch(small));
    expect(has_cpu_parallel_launch(expensive));
    expect(has_cpu_parallel_launch(explicit_worker));
    expect(has_cpu_parallel_launch(boundary));
    expect(has_cpu_parallel_launch(disabled));
}

void test_shared_exp_is_materialized_once() {
    auto definition = tile_kernel("shared_exp", [](TensorView<const float, 2> input,
                                                   TensorView<float, 2> output) {
        auto row = axis("row", input.extent<0>());
        auto column = axis("column", input.extent<1>());
        auto one = axis("one", 1);
        for (auto &nest : parallel(shape(row))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, column)];
            auto exponential = exp(value);
            output(origin, shape(one, column)).store(exponential + reduce(exponential, column, add));
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 37), tensor_shape(3, 37));
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto exp_op = tvm::Op::Get("tirx.exp");
    auto calls = 0u;
    tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto call = node.as<tvm::CallNode>()) { calls += call->op.same_as(exp_op); }
    });
    expect(eq(calls, 1u));
}

[[nodiscard]] Kernel make_row_sum(int64_t rows, int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_sum", [](TensorView<const float, 2> input,
                                                           TensorView<float, 1> output) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", input.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto value = input[coord(nest.index(), 0), shape(one, columns)];
            output(coord(nest.index()), shape(one)).store(reduce(value, columns, add));
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows));
}

[[nodiscard]] Kernel make_row_softmax(int64_t rows, int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_softmax", [](TensorView<const float, 2> input,
                                                               TensorView<float, 2> output) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", input.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, columns)];
            auto shifted = exp(value - reduce(value, columns, maximum));
            output(origin, shape(one, columns))
                .store(shifted / reduce(shifted, columns, add));
        }
    });
    return definition.capture(tensor_shape(rows, columns),
                              tensor_shape(rows, columns));
}

[[nodiscard]] Kernel make_row_extrema(int64_t rows, int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_extrema", [](TensorView<const float, 2> input,
                                                               TensorView<float, 1> minima,
                                                               TensorView<float, 1> maxima) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", input.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto value = input[coord(nest.index(), 0), shape(one, columns)];
            minima(coord(nest.index()), shape(one)).store(reduce(value, columns, minimum));
            maxima(coord(nest.index()), shape(one)).store(reduce(value, columns, maximum));
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows),
                              tensor_shape(rows));
}

[[nodiscard]] PlannerOptions subgroup_reduction_options() {
    auto planner = PlannerOptions{};
    planner.metal_subgroup_reductions = true;
    return planner;
}

void test_metal_subgroup_reduction_contract(Runtime &runtime) {
    auto planner = subgroup_reduction_options();
    auto kernel = make_row_sum(3, 37);
    if (runtime.target() != "metal") {
        auto unavailable = runtime.build(
            kernel, true, false, true, false, planner, false, true);
        expect(!unavailable.ok());
        expect(unavailable.error.find("Metal") != luisa::string::npos)
            << unavailable.error;
        return;
    }
    auto missing_noalias = runtime.build(
        kernel, false, false, true, false, planner, false, true);
    expect(!missing_noalias.ok());
    expect(missing_noalias.error.find("noalias") != luisa::string::npos)
        << missing_noalias.error;

    auto reference = runtime.build(kernel, true, false, true, false,
                                   PlannerOptions{}, false, false);
    expect(reference.ok()) << reference.error;
    if (reference.ok()) {
        auto source = metal_source(reference.module.value());
        expect(std::string_view{source.data(), source.size()}.find("simd_sum(") ==
               std::string_view::npos)
            << source;
    }
}

void test_metal_subgroup_sum(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr std::array cases{
        std::pair{int64_t{127}, uint32_t{1}},
        std::pair{int64_t{257}, uint32_t{1}},
        std::pair{int64_t{1024}, uint32_t{4}},
        std::pair{int64_t{4096}, uint32_t{8}}};
    for (auto [columns, expected_subgroups] : cases) {
        constexpr auto rows = int64_t{5};
        auto kernel = make_row_sum(rows, columns);
        expect(kernel.valid()) << "columns=" << columns;
        auto executable = runtime.build(
            kernel, true, false, true, false,
            subgroup_reduction_options(), false, true);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        expect(eq(executable.plans.size(), 1u));
        if (executable.plans.empty()) { continue; }
        auto &plan = executable.plans.front();
        expect(eq(plan.reduction_subgroups_per_program,
                  expected_subgroups));
        expect(eq(plan.reduction_operations, 1u));
        expect(eq(plan.reduction_elements,
                  static_cast<uint64_t>(columns)));
        expect(eq(plan.shared_memory_bytes,
                  expected_subgroups == 1u ? 0u :
                                             expected_subgroups * sizeof(float)));
        expect(eq(plan.group_barrier_sites_after,
                  expected_subgroups == 1u ? 0u : 1u));
        expect(eq(plan.independent_subgroups,
                  expected_subgroups == 1u));
        auto source = metal_source(executable.module.value());
        auto code = std::string_view{source.data(), source.size()};
        expect(code.find("simd_sum(") != std::string_view::npos) << source;
        luisa::vector<float> values(static_cast<size_t>(rows * columns));
        for (auto i = 0u; i < values.size(); i++) {
            values[i] = static_cast<float>(static_cast<int64_t>(i % 127u) - 63) /
                        64.0f;
        }
        auto input = runtime.upload<float>({rows, columns}, values);
        auto output = runtime.allocate<float>({rows});
        (*executable.entry)(input, output);
        auto actual = runtime.download<float>(output, rows);
        for (auto row = int64_t{0}; row < rows; row++) {
            auto expected = 0.0;
            for (auto column = int64_t{0}; column < columns; column++) {
                expected += values[static_cast<size_t>(row * columns + column)];
            }
            expect(std::abs(static_cast<double>(actual[row]) - expected) <=
                   2e-5 * std::max(1.0, std::abs(expected)))
                << "columns=" << columns << " row=" << row;
        }
    }
}

void test_metal_subgroup_striped_softmax(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto rows = int64_t{3};
    constexpr auto columns = int64_t{4096};
    auto kernel = make_row_softmax(rows, columns);
    expect(kernel.valid());
    auto executable = runtime.build(
        kernel, true, false, true, false,
        subgroup_reduction_options(), false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(eq(executable.plans.size(), 1u));
    if (executable.plans.empty()) { return; }
    auto &plan = executable.plans.front();
    expect(eq(plan.reduction_subgroups_per_program, 8u));
    expect(eq(plan.reduction_operations, 2u));
    expect(eq(plan.reduction_elements, 8192u));
    expect(eq(plan.striped_storage_scalars_per_worker, 16u));
    expect(!plan.independent_subgroups);
    auto source = metal_source(executable.module.value());
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("_worker_stripe[16]") != std::string_view::npos) << source;
    expect(code.find("thread float tile_storage_7[4096]") ==
           std::string_view::npos)
        << source;
    expect(code.find("simd_max(") != std::string_view::npos) << source;
    expect(code.find("simd_sum(") != std::string_view::npos) << source;

    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int64_t>((i * 5u + 17u) %
                                                            127u) -
                                       63) /
                    64.0f;
    }
    auto input = runtime.upload<float>({rows, columns}, values);
    auto output = runtime.allocate<float>({rows, columns});
    (*executable.entry)(input, output);
    auto actual = runtime.download<float>(output, values.size());
    for (auto row = int64_t{0}; row < rows; row++) {
        auto maximum_value = -std::numeric_limits<double>::infinity();
        for (auto column = int64_t{0}; column < columns; column++) {
            maximum_value = std::max(
                maximum_value,
                static_cast<double>(values[static_cast<size_t>(
                    row * columns + column)]));
        }
        auto denominator = 0.0;
        for (auto column = int64_t{0}; column < columns; column++) {
            denominator += std::exp(
                static_cast<double>(values[static_cast<size_t>(
                    row * columns + column)]) -
                maximum_value);
        }
        for (auto column = int64_t{0}; column < columns; column++) {
            auto index = static_cast<size_t>(row * columns + column);
            auto expected = std::exp(static_cast<double>(values[index]) -
                                     maximum_value) /
                            denominator;
            expect(std::abs(static_cast<double>(actual[index]) - expected) <=
                   2e-6 + 2e-5 * std::abs(expected))
                << "row=" << row << " column=" << column;
        }
    }
}

void test_metal_subgroup_extrema(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto rows = int64_t{7};
    constexpr auto columns = int64_t{1024};
    auto executable = runtime.build(
        make_row_extrema(rows, columns), true, false, true, false,
        subgroup_reduction_options(), false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(eq(executable.plans.size(), 1u));
    if (executable.plans.empty()) { return; }
    expect(eq(executable.plans.front().reduction_operations, 2u));
    expect(eq(executable.plans.front().reduction_elements, 2048u));
    auto source = metal_source(executable.module.value());
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("simd_min(") != std::string_view::npos) << source;
    expect(code.find("simd_max(") != std::string_view::npos) << source;

    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int64_t>((i * 13u + 5u) %
                                                            509u) -
                                       254) /
                    32.0f;
    }
    auto input = runtime.upload<float>({rows, columns}, values);
    auto minima = runtime.allocate<float>({rows});
    auto maxima = runtime.allocate<float>({rows});
    (*executable.entry)(input, minima, maxima);
    auto actual_minima = runtime.download<float>(minima, rows);
    auto actual_maxima = runtime.download<float>(maxima, rows);
    for (auto row = int64_t{0}; row < rows; row++) {
        auto first = values.begin() + row * columns;
        auto last = first + columns;
        expect(eq(actual_minima[row], *std::min_element(first, last)));
        expect(eq(actual_maxima[row], *std::max_element(first, last)));
    }
}

void test_cpu_accelerate_math(Runtime &runtime) {
    auto definition = tile_kernel("accelerate_exp", [](TensorView<const float, 2> input,
                                                       TensorView<float, 2> output) {
        auto row = axis("row", input.extent<0>());
        auto column = axis("column", input.extent<1>());
        auto one = axis("one", 1);
        for (auto &nest : parallel(shape(row))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, column)];
            auto exponential = exp(value);
            output(origin, shape(one, column)).store(exponential + reduce(exponential, column, add));
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 37), tensor_shape(3, 37));
    auto planner = PlannerOptions{};
    planner.max_cpu_stack_bytes = 4096u;
    auto executable = runtime.build(
        kernel, true, false, true, true, planner, false, true,
        CpuMatrixBackend::REFERENCE, CpuMathBackend::ACCELERATE);
    if (runtime.target() != "llvm") {
        expect(!executable.ok());
        expect(executable.error.find("array-math") != luisa::string::npos) << executable.error;
        return;
    }
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto source = executable.module.value()->InspectSource("ll");
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("call void @luisa_tile_accelerate_expf(") != std::string_view::npos) << source;
    expect(code.find("call void @luisa_tile_accelerate_reduce_add_f32(") != std::string_view::npos) << source;
    auto provider_calls = 0u;
    constexpr auto prefix = std::string_view{"call void @luisa_tile_accelerate_"};
    for (auto position = code.find(prefix); position != std::string_view::npos;
         position = code.find(prefix, position + prefix.size())) { provider_calls++; }
    expect(eq(provider_calls, 2u)) << source;
    luisa::vector<float> values(3u * 37u);
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int>(i % 23u) - 11) * 0.0625f;
    }
    auto input = runtime.upload<float>({3, 37}, values);
    auto output = runtime.allocate<float>({3, 37});
    (*executable.entry)(input, output);
    auto actual = runtime.download<float>(output, values.size());
    for (auto row = 0u; row < 3u; row++) {
        auto sum = 0.0f;
        for (auto column = 0u; column < 37u; column++) {
            sum += std::exp(values[row * 37u + column]);
        }
        for (auto column = 0u; column < 37u; column++) {
            auto expected = std::exp(values[row * 37u + column]) + sum;
            expect(std::abs(actual[row * 37u + column] - expected) <= 2e-5f * std::max(1.0f, std::abs(expected)));
        }
    }
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
        for (auto &worker : parallel(shape(ceil_div(count, 4)), exec::Scope::WORKER)) {
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
        for (auto &worker : parallel(shape(ceil_div(rows, 4)), exec::Scope::WORKER)) {
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

enum class VectorGuardCase { BOUNDS,
                             HOLE,
                             TEMPORAL,
                             CONDITIONAL,
                             NESTED_LAZY,
                             DYNAMIC_DIVISOR,
                             DYNAMIC_TEMPORAL };

void test_auto_vector_guards(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    auto i64 = [](int64_t value) { return tvm::IntImm::Int64(value); };
    auto f32 = [](float value) { return tvm::FloatImm{tvm::PrimType::Float(32), value}; };
    for (auto mode : {VectorGuardCase::BOUNDS, VectorGuardCase::HOLE, VectorGuardCase::TEMPORAL,
                      VectorGuardCase::CONDITIONAL, VectorGuardCase::NESTED_LAZY, VectorGuardCase::DYNAMIC_DIVISOR, VectorGuardCase::DYNAMIC_TEMPORAL}) {
        auto a = tvm::tirx::decl_buffer({i64(37)}, tvm::PrimType::Float(32), "input");
        auto d = tvm::tirx::decl_buffer({i64(4), i64(35)}, tvm::PrimType::Float(32), "output");
        auto origin = tvm::tirx::PrimVar{"origin", tvm::PrimType::Int(64)};
        auto trips = tvm::tirx::PrimVar{"trips", tvm::PrimType::Int(64)};
        auto row = tvm::tirx::PrimVar{"row", tvm::PrimType::Int(64)};
        auto column = tvm::tirx::PrimVar{"column", tvm::PrimType::Int(64)};
        auto time = tvm::tirx::PrimVar{"time", tvm::PrimType::Int(64)};
        auto col = column - i64(7);
        auto temporal = mode == VectorGuardCase::TEMPORAL || mode == VectorGuardCase::DYNAMIC_TEMPORAL;
        tvm::PrimExpr address = origin + col;
        if (temporal) { address += time; }
        if (mode == VectorGuardCase::DYNAMIC_DIVISOR) { address = tvm::floordiv(col, origin); }
        tvm::PrimExpr guard = address >= i64(0) && address < i64(37);
        if (mode == VectorGuardCase::HOLE) {
            // The endpoints can both be valid while interior lanes are not.
            guard = guard && tvm::floormod(col, i64(7)) != i64(3);
        }
        auto value = tvm::if_then_else(guard, tvm::tirx::BufferLoad{a, {address}}, f32(-2.25f));
        if (mode == VectorGuardCase::NESTED_LAZY) { value = tvm::if_then_else(origin < i64(100), value, f32(-2.25f)); }
        if (mode == VectorGuardCase::DYNAMIC_DIVISOR) { value = tvm::if_then_else(origin > i64(0), value, f32(-2.25f)); }
        tvm::ffi::Array<tvm::PrimExpr> output_index{row, col};
        tvm::tirx::Stmt body = tvm::tirx::BufferStore{d, tvm::tirx::BufferLoad{d, output_index} + value * f32(2.0f), output_index};
        if (temporal) {
            body = tvm::tirx::For{time, i64(0), mode == VectorGuardCase::DYNAMIC_TEMPORAL ? tvm::PrimExpr{trips} : tvm::PrimExpr{i64(3)}, tvm::tirx::ForKind::kSerial, std::move(body)};
        }
        if (mode == VectorGuardCase::CONDITIONAL) { body = tvm::tirx::IfThenElse{origin < i64(100), std::move(body)}; }
        body = tvm::tirx::SeqStmt{{tvm::tirx::BufferStore{d, f32(0.5f), output_index}, std::move(body)}};
        body = tvm::tirx::For{column, i64(7), i64(35), tvm::tirx::ForKind::kSerial, std::move(body)};
        body = tvm::tirx::For{row, i64(0), i64(4), tvm::tirx::ForKind::kSerial, std::move(body), {}, {{"luisa.tile.independent_elements", tvm::IntImm::Int32(2)}}};
        auto function = tvm::tirx::PrimFunc{{a, d, origin, trips}, std::move(body)};
        for (auto lanes : {0u, 16u, 64u}) {
            CompileOptions options;
            options.target = "llvm";
            options.auto_vectorize = lanes != 0u;
            options.planner.max_cpu_vector_lanes = lanes == 0u ? 16u : lanes;
            auto compiled = compile(function, "auto_vector_guards", options);
            expect(compiled.ok()) << compiled.error();
            if (!compiled) { continue; }
            auto entry = compiled.module().value()->GetFunction("auto_vector_guards", true);
            expect(entry.has_value());
            if (!entry) { continue; }
            luisa::vector<float> input(37);
            for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i) * 0.125f - 1.0f; }
            auto source = runtime.upload<float>({37}, input);
            for (auto start : {int64_t{-9}, int64_t{0}, int64_t{16}, int64_t{34}, int64_t{37}, std::numeric_limits<int64_t>::max() - 16}) {
                auto huge = start > 100;
                if (huge && mode != VectorGuardCase::CONDITIONAL && mode != VectorGuardCase::NESTED_LAZY && mode != VectorGuardCase::DYNAMIC_TEMPORAL) { continue; }
                for (auto count : {int64_t{0}, int64_t{3}}) {
                    // A zero-trip/untaken region must not evaluate its address
                    // arithmetic (which would overflow for this sentinel).
                    if (huge && mode == VectorGuardCase::DYNAMIC_TEMPORAL && count != 0) { continue; }
                    auto output = runtime.upload<float>({4, 35}, luisa::vector<float>(140, -19.0f));
                    (*entry)(source, output, start, count);
                    auto actual = runtime.download<float>(output, 140u);
                    for (auto i = 0u; i < actual.size(); i++) {
                        auto c = static_cast<int64_t>(i % 35u);
                        auto expected = 0.5f;
                        auto iterations = mode == VectorGuardCase::DYNAMIC_TEMPORAL ? count : temporal ? 3 :
                                                                                                         1;
                        if (mode == VectorGuardCase::CONDITIONAL && huge) { iterations = 0; }
                        for (auto t = int64_t{0}; t < iterations; t++) {
                            auto v = -2.25f;
                            if (!(mode == VectorGuardCase::NESTED_LAZY && huge) && !(mode == VectorGuardCase::DYNAMIC_DIVISOR && start <= 0)) {
                                auto index = mode == VectorGuardCase::DYNAMIC_DIVISOR ? c / start : start + c + (temporal ? t : 0);
                                if (index >= 0 && index < 37 && !(mode == VectorGuardCase::HOLE && c % 7 == 3)) { v = input[index]; }
                            }
                            expected += v * 2.0f;
                        }
                        expect(eq(actual[i], expected)) << "mode=" << static_cast<uint32_t>(mode) << " lanes=" << lanes << " start=" << start << " count=" << count << " i=" << i;
                    }
                }
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_execution_explicit_worker"_test = [&] { test_explicit_worker(runtime); };
    "tile_execution_empty_domain"_test = [&] { test_empty_parallel(runtime); };
    "tile_execution_cpu_parallel_launch_cost"_test = [&] { test_cpu_parallel_launch_cost(runtime); };
    "tile_execution_shared_exp_materialization"_test = test_shared_exp_is_materialized_once;
    "tile_execution_metal_subgroup_reduction_contract"_test = [&] { test_metal_subgroup_reduction_contract(runtime); };
    "tile_execution_metal_subgroup_sum"_test = [&] { test_metal_subgroup_sum(runtime); };
    "tile_execution_metal_subgroup_striped_softmax"_test = [&] { test_metal_subgroup_striped_softmax(runtime); };
    "tile_execution_metal_subgroup_extrema"_test = [&] { test_metal_subgroup_extrema(runtime); };
    "tile_execution_cpu_accelerate_math"_test = [&] { test_cpu_accelerate_math(runtime); };
    "tile_execution_scope_preserved"_test = test_scope_survives_export;
    "tile_execution_unsupported_scopes"_test = [&] { test_unsupported_scopes(runtime); };
    "tile_execution_unknown_scope"_test = [&] { test_unknown_scope(runtime); };
    "tile_execution_nested_worker"_test = [&] { test_nested_worker_rejected(runtime); };
    "tile_execution_worker_vector"_test = [&] { test_worker_vector(runtime); };
    "tile_execution_vector_private_tiles_and_carries"_test = [&] { test_vector_private_tiles_and_carries(runtime); };
    "tile_execution_vector_roots_and_nesting"_test = [&] { test_vector_roots_and_nesting(runtime); };
    "tile_execution_auto_vector_guards"_test = [&] { test_auto_vector_guards(runtime); };
}

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
#include <luisa/tile/algorithms.h>
#include <luisa/tile/memory.h>
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

void test_fused_element_grid(Runtime &runtime) {
    for (auto dims : {std::array<int64_t, 4>{1, 127, 8, 8}, {17, 257, 3, 7}, {35, 63, 8, 8}}) {
        auto [rows, columns, bm, bn] = dims;
        for (auto mode = 0; mode != 3; mode++) {
            auto scope = mode == 2 ? exec::Scope::WORKER : exec::Scope::AUTOMATIC;
            auto definition = tile_kernel("element_grid", [=](TensorView<const float, 2> x, TensorView<float, 2> y) {
                auto gr = axis("gr", ceil_div(rows, bm)), gc = axis("gc", ceil_div(columns, bn));
                auto r = axis("r", bm), c = axis("c", bn);
                for (auto &nest : parallel(shape(gr, gc), scope)) {
                    auto r0 = nest.index(gr) * bm, c0 = nest.index(gc) * bn;
                    auto value = x[coord(r0 - 2, c0 - 3), shape(r, c)];
                    y(coord(r0, c0), shape(r, c)).store(tanh(value * 0.3f) + value * value);
                }
            });
            auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
            PlannerOptions planner;
            planner.fuse_gpu_elementwise = mode != 1;
            auto executable = runtime.build(kernel, true, false, true, false, planner);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            auto fused = runtime.target() == "metal" && mode == 0;
            expect(eq(executable.plans.size(), fused ? size_t{1} : size_t{0}));
            if (fused && !executable.plans.empty()) {
                expect(eq(executable.plans.front().elementwise_elements_per_program, static_cast<uint64_t>(bm * bn)));
                auto source = metal_source(executable.module.value());
                expect(std::string_view{source.data(), source.size()}.find("tile_storage_") == std::string_view::npos);
            }
            luisa::vector<float> input(static_cast<size_t>(rows * columns));
            for (size_t i = 0; i < input.size(); i++) { input[i] = static_cast<float>(static_cast<int64_t>(i % 43u) - 21) / 17.0f; }
            auto source = runtime.upload<float>({rows, columns}, input);
            auto output = runtime.upload<float>({rows, columns}, luisa::vector<float>(input.size(), -19.0f));
            (*executable.entry)(source, output);
            auto actual = runtime.download<float>(output, input.size());
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < columns; c++) {
                    auto x = r >= 2 && c >= 3 ? input[(r - 2) * columns + c - 3] : 0.0f;
                    auto expected = std::tanh(x * 0.3f) + x * x;
                    expect(std::abs(actual[r * columns + c] - expected) < 5e-6f);
                }
            }
        }
    }
}

void test_element_grid_retains_snapshot(Runtime &runtime) {
    auto kernel = tile_kernel("overlapping_snapshot", [](TensorView<float, 1> x) {
                      for (auto &nest : parallel(shape(1))) {
                          auto old = x[coord(0), shape(8)];
                          x(coord(1), shape(8)).store(old);
                      }
                  }).capture(tensor_shape(9));
    auto executable = runtime.build(kernel, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(executable.plans.empty());
    auto buffer = runtime.upload<float>({9}, luisa::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    (*executable.entry)(buffer);
    auto actual = runtime.download<float>(buffer, 9);
    for (size_t i = 0; i < actual.size(); i++) { expect(eq(actual[i], static_cast<float>(i ? i - 1 : 0))); }
}

void test_element_grid_shared_producers(Runtime &runtime) {
    for (auto dims : {std::array<int64_t, 4>{1, 127, 8, 8}, {17, 257, 3, 7}, {35, 63, 8, 8}}) {
        auto [rows, columns, bm, bn] = dims;
        for (auto mode = 0u; mode != 4u; mode++) {
            auto definition = tile_kernel("element_grid_ssa", [=](TensorView<const float, 2> x, TensorView<float, 2> y) {
                auto gr = axis("gr", ceil_div(rows, bm)), gc = axis("gc", ceil_div(columns, bn));
                auto r = axis("r", bm), c = axis("c", bn);
                auto scope = mode == 2u ? exec::Scope::WORKER : exec::Scope::AUTOMATIC;
                for (auto &nest : parallel(shape(gr, gc), scope)) {
                    auto r0 = nest.index(gr) * bm, c0 = nest.index(gc) * bn;
                    auto value = x[coord(r0 - 2, c0 - 3), shape(r, c)];
                    auto activated = exp(value * 0.3f);
                    auto shared = activated * activated + activated;
                    y(coord(r0, c0), shape(r, c)).store(shared * shared + activated);
                }
            });
            auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
            PlannerOptions planner;
            planner.fuse_gpu_elementwise = mode != 1u;
            auto executable = runtime.build(kernel, mode != 3u, false, true, false, planner);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            auto fused = runtime.target() == "metal" && mode == 0u;
            expect(eq(executable.plans.size(), fused ? size_t{1u} : size_t{0u}));
            if (fused && !executable.plans.empty()) {
                expect(eq(executable.plans.front().elementwise_scalar_temporaries, 2u));
                auto source = metal_source(executable.module.value());
                auto code = std::string_view{source.data(), source.size()};
                expect(code.find("thread float tile_storage_") == std::string_view::npos);
                auto first = code.find("exp(");
                expect(first != std::string_view::npos) << code;
                if (first != std::string_view::npos) { expect(code.find("exp(", first + 4u) == std::string_view::npos); }
            }
            luisa::vector<float> input(static_cast<size_t>(rows * columns));
            for (size_t i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(static_cast<int64_t>(i % 43u) - 21) / 17.0f; }
            auto source = runtime.upload<float>({rows, columns}, input);
            auto output = runtime.upload<float>({rows, columns}, luisa::vector<float>(input.size(), -19.0f));
            (*executable.entry)(source, output);
            auto actual = runtime.download<float>(output, input.size());
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < columns; c++) {
                    auto value = r >= 2 && c >= 3 ? input[(r - 2) * columns + c - 3] : 0.0f;
                    auto activated = std::exp(value * 0.3f);
                    auto shared = activated * activated + activated;
                    expect(std::abs(actual[r * columns + c] - (shared * shared + activated)) < 1e-5f);
                }
            }
        }
    }
}

void test_element_grid_respects_exact_reduction(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    auto kernel = make_copy(exec::Scope::AUTOMATIC, 37);
    for (auto request = 0u; request != 4u; request++) {
        PlannerOptions planner;
        planner.metal_subgroup_reductions = true;
        if (request == 0u) { planner.threads_per_group = 64u; }
        if (request == 1u) { planner.reduction_programs_per_group = 2u; }
        if (request == 2u) { planner.reduction_unroll_factor = 4u; }
        if (request == 3u) { planner.reduction_lane_elements = 4u; }
        auto executable = runtime.build(kernel, true, false, true, false, planner);
        expect(!executable.ok());
        expect(executable.error.find("exact reduction mapping") != luisa::string::npos) << executable.error;
    }
}

enum class ElementChainCase { POINTWISE,
                              NEIGHBOR,
                              TRANSPOSE,
                              REWRITTEN,
                              UNMARKED,
                              MANUAL,
                              CONDITIONAL,
                              DIFFERENT_DOMAIN,
                              INPUT_WRITE };

void test_element_grid_producer_contract(Runtime &runtime) {
    auto i64 = [](int64_t value) { return tvm::IntImm::Int64(value); };
    auto f32 = [](float value) { return tvm::FloatImm{tvm::PrimType::Float(32), value}; };
    for (auto mode : {ElementChainCase::POINTWISE, ElementChainCase::NEIGHBOR, ElementChainCase::TRANSPOSE,
                      ElementChainCase::REWRITTEN, ElementChainCase::UNMARKED, ElementChainCase::MANUAL,
                      ElementChainCase::CONDITIONAL, ElementChainCase::DIFFERENT_DOMAIN, ElementChainCase::INPUT_WRITE}) {
        auto a = tvm::tirx::decl_buffer({i64(3), i64(8), i64(8)}, tvm::PrimType::Float(32), "input");
        auto d = tvm::tirx::decl_buffer({i64(3), i64(8), i64(8)}, tvm::PrimType::Float(32), "output");
        auto columns = mode == ElementChainCase::DIFFERENT_DOMAIN ? 7 : 8;
        auto temporary = tvm::tirx::decl_buffer({i64(8), i64(columns)}, tvm::PrimType::Float(32), "temporary", "local");
        auto p = tvm::tirx::PrimVar{"program", tvm::PrimType::Int(64)};
        auto r = tvm::tirx::PrimVar{"producer_row", tvm::PrimType::Int(64)};
        auto c = tvm::tirx::PrimVar{"producer_column", tvm::PrimType::Int(64)};
        auto y = tvm::tirx::PrimVar{"consumer_row", tvm::PrimType::Int(64)};
        auto x = tvm::tirx::PrimVar{"consumer_column", tvm::PrimType::Int(64)};
        auto row = r - i64(3), column = c - i64(7);
        auto output_row = y - i64(5), output_column = x - i64(11);
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> annotations{{"luisa.tile.independent_elements", i64(2)}};
        if (mode != ElementChainCase::UNMARKED) { annotations.Set("luisa.tile.contract.materialized_pure_tile", i64(1)); }
        auto producer = [&](float increment) {
            auto value = tvm::tirx::BufferLoad{a, {p - i64(2), row, column}} + f32(increment);
            tvm::tirx::Stmt result = tvm::tirx::BufferStore{temporary, value, {row, column}};
            if (mode == ElementChainCase::CONDITIONAL) { result = tvm::tirx::IfThenElse{row < i64(7), std::move(result)}; }
            result = tvm::tirx::For{c, i64(7), i64(columns), tvm::tirx::ForKind::kSerial, std::move(result)};
            return tvm::tirx::For{r, i64(3), i64(8), tvm::tirx::ForKind::kSerial, std::move(result), {}, annotations};
        };
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> allocation_annotations;
        if (mode == ElementChainCase::MANUAL) { allocation_annotations.Set("luisa.tile.manual_memory", i64(1)); }
        tvm::ffi::Array<tvm::tirx::Stmt> parts{tvm::tirx::AllocBuffer{temporary, allocation_annotations}, producer(0.25f)};
        if (mode == ElementChainCase::REWRITTEN) { parts.push_back(producer(0.5f)); }
        tvm::PrimExpr read_row = output_row, read_column = output_column;
        if (mode == ElementChainCase::NEIGHBOR) { read_row = tvm::floormod(output_row + i64(1), i64(8)); }
        if (mode == ElementChainCase::TRANSPOSE) { std::swap(read_row, read_column); }
        if (mode == ElementChainCase::DIFFERENT_DOMAIN) { read_column = tvm::floormod(read_column, i64(7)); }
        auto destination = mode == ElementChainCase::INPUT_WRITE ? a : d;
        auto value = tvm::tirx::BufferLoad{temporary, {read_row, read_column}};
        tvm::tirx::Stmt consumer = tvm::tirx::BufferStore{destination, value * value + value, {p - i64(2), output_row, output_column}};
        if (mode == ElementChainCase::CONDITIONAL) { consumer = tvm::tirx::IfThenElse{output_row < i64(7), std::move(consumer)}; }
        consumer = tvm::tirx::For{x, i64(11), i64(8), tvm::tirx::ForKind::kSerial, std::move(consumer)};
        consumer = tvm::tirx::For{y, i64(5), i64(8), tvm::tirx::ForKind::kSerial, std::move(consumer), {}, {{"luisa.tile.independent_elements", i64(2)}}};
        parts.push_back(std::move(consumer));
        auto body = tvm::tirx::For{p, i64(2), i64(3), tvm::tirx::ForKind::kSerial, tvm::tirx::SeqStmt::Flatten(parts), {}, {{"luisa.tile.logical_parallel", i64(1)}}};
        CompileOptions options;
        options.target = runtime.target();
        options.noalias = true;
        auto compiled = compile(tvm::tirx::PrimFunc{{a, d}, std::move(body)}, "element_chain_contract", options);
        expect(compiled.ok()) << compiled.error();
        if (!compiled) { continue; }
        auto fused = runtime.target() == "metal" && mode == ElementChainCase::POINTWISE;
        expect(eq(compiled.plans().size(), fused ? size_t{1u} : size_t{0u}));
        auto entry = compiled.module().value()->GetFunction("element_chain_contract", true);
        expect(entry.has_value());
        if (!entry) { continue; }
        luisa::vector<float> input(192u);
        for (auto i = 0u; i < input.size(); i++) { input[i] = static_cast<float>(i) / 37.0f - 2.0f; }
        auto source = runtime.upload<float>({3, 8, 8}, input);
        auto output = runtime.upload<float>({3, 8, 8}, luisa::vector<float>(192u, -19.0f));
        (*entry)(source, output);
        auto actual = runtime.download<float>(mode == ElementChainCase::INPUT_WRITE ? source : output, 192u);
        for (auto i = 0u; i < actual.size(); i++) {
            auto r0 = i / 8u % 8u, c0 = i % 8u;
            if (mode == ElementChainCase::CONDITIONAL && r0 == 7u) {
                expect(eq(actual[i], -19.0f));
                continue;
            }
            if (mode == ElementChainCase::NEIGHBOR) { r0 = (r0 + 1u) % 8u; }
            if (mode == ElementChainCase::TRANSPOSE) { std::swap(r0, c0); }
            if (mode == ElementChainCase::DIFFERENT_DOMAIN) { c0 %= 7u; }
            auto v = input[i / 64u * 64u + r0 * 8u + c0] + (mode == ElementChainCase::REWRITTEN ? 0.5f : 0.25f);
            expect(std::abs(actual[i] - (v * v + v)) < 5e-6f) << "mode=" << static_cast<uint32_t>(mode) << " i=" << i;
        }
    }
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

void test_shared_tanh_is_materialized_once() {
    auto definition = tile_kernel("shared_tanh", [](TensorView<const float, 2> input,
                                                    TensorView<float, 2> output) {
        auto row = axis("row", input.extent<0>());
        auto column = axis("column", input.extent<1>());
        auto one = axis("one", 1);
        for (auto &nest : parallel(shape(row))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, column)];
            auto activated = tanh(value);
            output(origin, shape(one, column))
                .store(activated + reduce(activated, column, add));
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 37),
                                     tensor_shape(3, 37));
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto tanh_op = tvm::Op::Get("tirx.tanh");
    auto calls = 0u;
    tvm::tirx::PostOrderVisit(
        native.value->body, [&](const tvm::ffi::ObjectRef &node) {
            if (auto call = node.as<tvm::CallNode>()) {
                calls += call->op.same_as(tanh_op);
            }
        });
    expect(eq(calls, 1u));
}

void test_shared_arithmetic_preserves_ssa() {
    auto definition = tile_kernel("shared_arithmetic", [](TensorView<const float, 2> input,
                                                          TensorView<float, 2> output) {
        auto row = axis("row", input.extent<0>());
        auto column = axis("column", input.extent<1>());
        auto one = axis("one", 1);
        for (auto &nest : parallel(shape(row))) {
            auto origin = coord(nest.index(), 0);
            auto combined = input[origin, shape(one, column)] + 1.0f;
            output(origin, shape(one, column))
                .store(combined + reduce(combined, column, add));
        }
    });
    auto kernel = definition.capture(tensor_shape(3, 37),
                                     tensor_shape(3, 37));
    auto count_materializations = [](const NativeFunction &native) {
        auto materializations = 0u;
        tvm::tirx::PostOrderVisit(
            native.value->body, [&](const tvm::ffi::ObjectRef &node) {
                if (auto loop = node.as<tvm::tirx::ForNode>()) {
                    materializations += loop->annotations.count(
                        "luisa.tile.contract.materialized_pure_tile");
                }
            });
        return materializations;
    };
    auto preserved = lower(kernel.function());
    expect(preserved.ok()) << preserved.error;
    if (preserved) { expect(eq(count_materializations(preserved), 1u)); }
    auto recomputed = lower(
        kernel.function(),
        {.shared_tiles = SharedTileMaterialization::EXPENSIVE_ONLY});
    expect(recomputed.ok()) << recomputed.error;
    if (recomputed) { expect(eq(count_materializations(recomputed), 0u)); }
    auto invalid = lower(
        kernel.function(),
        {.shared_tiles = static_cast<SharedTileMaterialization>(255u)});
    expect(!invalid.ok());
}

[[nodiscard]] Kernel make_row_sum(int64_t rows, int64_t columns, exec::Scope scope = exec::Scope::AUTOMATIC) {
    auto definition = tile_kernel("metal_subgroup_sum", [scope](TensorView<const float, 2> input,
                                                                TensorView<float, 1> output) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", input.extent<1>());
        for (auto &nest : parallel(shape(rows), scope)) {
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

[[nodiscard]] Kernel make_row_layernorm(int64_t rows, int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_layernorm", [=](TensorView<const float, 2> input,
                                                                  TensorView<const float, 2> parameters,
                                                                  TensorView<float, 2> output) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns_axis = axis("columns", input.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, columns_axis)];
            auto denominator = static_cast<float>(columns);
            auto mean = reduce(value, columns_axis, add) / denominator;
            auto centered = value - mean;
            auto variance = reduce(centered * centered, columns_axis, add) / denominator;
            auto gamma = parameters[coord(0, 0), shape(one, columns_axis)];
            auto beta = parameters[coord(1, 0), shape(one, columns_axis)];
            output(origin, shape(one, columns_axis))
                .store(centered / sqrt(variance + 1e-5f) * gamma + beta);
        }
    });
    return definition.capture(tensor_shape(rows, columns),
                              tensor_shape(2, columns),
                              tensor_shape(rows, columns));
}

[[nodiscard]] Kernel make_row_tanh_statistic(int64_t rows,
                                             int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_tanh_statistic", [](TensorView<const float, 2> input,
                                                                      TensorView<float, 2> output) {
        auto rows = axis("rows", input.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", input.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            auto value = input[origin, shape(one, columns)];
            auto activated = tanh(value);
            output(origin, shape(one, columns))
                .store(activated + reduce(activated, columns, add));
        }
    });
    return definition.capture(tensor_shape(rows, columns),
                              tensor_shape(rows, columns));
}

[[nodiscard]] Kernel make_row_cross_entropy(int64_t rows, int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_cross_entropy", [](TensorView<const float, 2> logits,
                                                                     TensorView<const int64_t, 1> labels,
                                                                     TensorView<float, 1> losses) {
        auto rows = axis("rows", logits.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", logits.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            auto value = logits[origin, shape(one, columns)];
            auto label = labels[coord(nest.index()), shape(one)];
            auto peak = reduce(value, columns, maximum);
            auto total = reduce(exp(value - peak), columns, add);
            auto selected = gather(value, label, columns);
            losses(coord(nest.index()), shape(one))
                .store(luisa::compute::tile::log(total) + peak - selected);
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows),
                              tensor_shape(rows));
}

[[nodiscard]] Kernel make_row_derived_cross_entropy(int64_t rows,
                                                    int64_t columns) {
    auto definition = tile_kernel("metal_subgroup_derived_cross_entropy", [](TensorView<const float, 2> logits,
                                                                             TensorView<const int64_t, 1> labels,
                                                                             TensorView<float, 1> losses) {
        auto rows = axis("rows", logits.extent<0>());
        auto one = axis("one", 1);
        auto columns = axis("columns", logits.extent<1>());
        for (auto &nest : parallel(shape(rows))) {
            auto origin = coord(nest.index(), 0);
            // This explicitly materialized derived Tile cannot be forwarded
            // to an immutable input. Its dynamic gather therefore crosses the
            // distributed private ownership partition and must make the
            // subgroup mapper decline.
            auto storage = memory<float>(shape(one, columns), mem::private_);
            storage.store(logits[origin, shape(one, columns)] * 1.25f + 0.125f);
            auto value = storage.load();
            auto label = labels[coord(nest.index()), shape(one)];
            auto peak = reduce(value, columns, maximum);
            auto total = reduce(exp(value - peak), columns, add);
            auto selected = gather(value, label, columns);
            losses(coord(nest.index()), shape(one))
                .store(luisa::compute::tile::log(total) + peak - selected);
        }
    });
    return definition.capture(tensor_shape(rows, columns), tensor_shape(rows),
                              tensor_shape(rows));
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

// Independently enumerate the current scalar-round objective, using the
// runtime's target limit rather than baking the old eight-subgroup cap into
// source/stripe assertions. All domains in these V=1 fixtures have width N.
[[nodiscard]] uint32_t expected_reduction_subgroups(Runtime &runtime, uint64_t rows,
                                                    uint64_t columns, uint32_t domains, uint32_t reductions) {
    auto best = std::numeric_limits<double>::infinity();
    auto chosen = uint32_t{0u};
    for (auto groups = 1u; groups <= std::min(32u, runtime.metal_max_threads() / 32u); groups++) {
        auto packing = groups == 1u ? std::min<uint64_t>(rows, 8u) : 1u;
        auto score = static_cast<double>(domains * ceil_div(columns, uint64_t{groups * 32u})) +
                     reductions * groups * 2.0 + 16.0 / static_cast<double>(packing);
        if (score < best) {
            best = score;
            chosen = groups;
        }
    }
    return chosen;
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

void test_metal_reduction_packing_and_policy(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    class Policy final : public AnalyticExecutionCostPolicy {
    public:
        mutable uint32_t candidates{0u};
        bool invalid{false};
        double reduction_score(const ReductionCandidate &candidate, const ExecutionCostModel &) const noexcept override {
            candidates++;
            if (invalid) { return std::numeric_limits<double>::quiet_NaN(); }
            return candidate.subgroups_per_program == 1u && candidate.programs_per_group == 2u ? 0.0 : 1.0;
        }
    } policy;
    constexpr auto rows = int64_t{5}, columns = int64_t{257};
    auto kernel = make_row_sum(rows, columns);
    for (auto [packing, unroll] : {std::pair{0u, 1u}, {1u, 3u}, {3u, 4u}, {8u, 16u}}) {
        auto planner = subgroup_reduction_options();
        planner.reduction_programs_per_group = packing;
        planner.reduction_unroll_factor = unroll;
        planner.cost_policy = &policy;
        if (packing == 1u) { planner.threads_per_group = 32u; }
        auto executable = runtime.build(kernel, true, false, true, false, planner, false, true);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto expected_packing = packing ? packing : 2u;
        expect(eq(executable.plans.size(), 1u));
        if (executable.plans.empty()) { continue; }
        auto &plan = executable.plans.front();
        expect(eq(plan.reduction_programs_per_group, expected_packing));
        expect(eq(plan.reduction_unroll_factor, unroll));
        expect(eq(plan.reduction_subgroups_per_program, 1u));
        expect(eq(plan.threads, expected_packing * 32u));
        expect(eq(plan.group_barrier_sites_after, 0u));
        luisa::vector<float> values(static_cast<size_t>(rows * columns));
        for (size_t i = 0u; i < values.size(); i++) { values[i] = static_cast<float>(static_cast<int>(i % 53u) - 26) / 19.0f; }
        auto input = runtime.upload<float>({rows, columns}, values);
        auto output = runtime.allocate<float>({rows});
        (*executable.entry)(input, output);
        auto actual = runtime.download<float>(output, rows);
        for (int64_t row = 0; row < rows; row++) {
            auto expected = 0.0;
            for (int64_t column = 0; column < columns; column++) { expected += values[row * columns + column]; }
            expect(std::abs(actual[row] - expected) < 2e-5 * std::max(1.0, std::abs(expected)));
        }
    }
    expect(policy.candidates > 4u);
    auto planner = subgroup_reduction_options();
    planner.reduction_programs_per_group = 3u;
    planner.threads_per_group = 64u;
    auto conflict = runtime.build(kernel, true, false, true, false, planner, false, true);
    expect(!conflict.ok());
    expect(conflict.error.find("packing") != luisa::string::npos) << conflict.error;
    planner.threads_per_group = 0u;
    auto over_budget = runtime.build(make_row_softmax(rows, 4096), true, false, true, false, planner, false, true);
    expect(!over_budget.ok());
    expect(over_budget.error.find("packing") != luisa::string::npos) << over_budget.error;
    auto bound_worker = runtime.build(make_row_sum(rows, columns, exec::Scope::WORKER), true, false, true, false, planner, false, true);
    expect(!bound_worker.ok());
    expect(bound_worker.error.find("conflicts") != luisa::string::npos) << bound_worker.error;
    planner.reduction_programs_per_group = 0u;
    planner.threads_per_group = 32u;
    auto exact_over_budget = runtime.build(make_row_softmax(rows, 4096), true, false, true, false, planner, false, true);
    expect(!exact_over_budget.ok());
    expect(exact_over_budget.error.find("exact reduction mapping") != luisa::string::npos) << exact_over_budget.error;
    planner.threads_per_group = 0u;
    planner.cost_policy = &policy;
    policy.invalid = true;
    auto bad_cost = runtime.build(kernel, true, false, true, false, planner, false, true);
    expect(!bad_cost.ok());
    expect(bad_cost.error.find("nonfinite") != luisa::string::npos) << bad_cost.error;
}

void test_reduction_lane_elements(Runtime &runtime) {
    constexpr auto rows = int64_t{5};
    for (auto width : {0u, 3u, 16u, 4u}) {
        auto planner = subgroup_reduction_options();
        planner.reduction_lane_elements = width;
        if (width == 4u) { planner.metal_subgroup_reductions = false; }
        auto rejected = runtime.build(make_row_softmax(rows, 257), true, false, true, false, planner);
        expect(!rejected.ok());
        expect(rejected.error.find("lane elements") != luisa::string::npos) << rejected.error;
    }
    if (runtime.target() != "metal") { return; }
    class Policy final : public AnalyticExecutionCostPolicy {
    public:
        mutable uint32_t observed_width{0u};
        double reduction_score(const ReductionCandidate &candidate, const ExecutionCostModel &model) const noexcept override {
            observed_width = candidate.lane_elements;
            return AnalyticExecutionCostPolicy::reduction_score(candidate, model);
        }
    } policy;
    for (auto width : {2u, 4u, 8u}) {
        for (auto columns : {int64_t{1}, int64_t{31}, int64_t{33}, int64_t{127}, int64_t{128}, int64_t{129}, int64_t{257}, int64_t{4103}}) {
            auto planner = subgroup_reduction_options();
            planner.reduction_lane_elements = width;
            planner.reduction_unroll_factor = 3u;
            planner.cost_policy = &policy;
            if (columns <= 257) {
                planner.reduction_programs_per_group = 3u;
            } else {
                planner.threads_per_group = 256u;
            }
            auto executable = runtime.build(make_row_softmax(rows, columns), true, false, true, false, planner, false, true);
            expect(executable.ok()) << "width=" << width << " columns=" << columns << " " << executable.error;
            if (!executable.ok()) { continue; }
            expect(eq(policy.observed_width, width));
            expect(eq(executable.plans.size(), 1u));
            if (executable.plans.empty()) { continue; }
            auto &plan = executable.plans.front();
            expect(eq(plan.reduction_lane_elements, width));
            expect(eq(plan.reduction_unroll_factor, 3u));
            auto workers = columns <= 257 ? 32u : 256u;
            auto stride = static_cast<int64_t>(workers * width);
            auto slots = static_cast<uint64_t>(columns / stride * width + std::min<int64_t>(columns % stride, width));
            expect(eq(plan.striped_storage_scalars_per_worker, slots));
            expect(eq(plan.reduction_programs_per_group, columns <= 257 ? 3u : 1u));
            luisa::vector<float> values(static_cast<size_t>(rows * columns));
            for (auto i = size_t{0u}; i < values.size(); i++) {
                values[i] = static_cast<float>(static_cast<int32_t>(i % 127u) - 63) / 64.0f;
            }
            auto input = runtime.upload<float>({rows, columns}, values);
            auto output = runtime.allocate<float>({rows, columns});
            (*executable.entry)(input, output);
            auto actual = runtime.download<float>(output, values.size());
            for (auto row = int64_t{0}; row < rows; row++) {
                auto denominator = 0.0;
                for (auto column = int64_t{0}; column < columns; column++) {
                    denominator += std::exp(static_cast<double>(values[row * columns + column]));
                }
                for (auto column = int64_t{0}; column < columns; column++) {
                    auto index = static_cast<size_t>(row * columns + column);
                    auto expected = std::exp(static_cast<double>(values[index])) / denominator;
                    expect(std::abs(static_cast<double>(actual[index]) - expected) <= 2e-6 + 2e-5 * std::abs(expected))
                        << "width=" << width << " columns=" << columns << " index=" << index;
                }
            }
        }
    }
    auto planner = subgroup_reduction_options();
    planner.reduction_lane_elements = 8u;
    planner.threads_per_group = 32u;
    planner.max_reduction_striped_scalars_per_worker = 1u;
    auto over_budget = runtime.build(make_row_softmax(rows, 33), true, false, true, false, planner, false, true);
    expect(!over_budget.ok());
    expect(over_budget.error.find("exact reduction mapping") != luisa::string::npos) << over_budget.error;
}

void test_reduction_complete_width_search(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    class Policy final : public AnalyticExecutionCostPolicy {
    public:
        mutable std::array<bool, 32u> widths{};
        mutable bool features_valid{true};
        double reduction_score(const ReductionCandidate &candidate, const ExecutionCostModel &model) const noexcept override {
            widths[candidate.subgroups_per_program - 1u] = true;
            features_valid &= candidate.threadgroups == ceil_div(candidate.programs, uint64_t{candidate.programs_per_group});
            features_valid &= candidate.lane_utilization > 0.0 && candidate.lane_utilization <= 1.0;
            auto workers = candidate.subgroups_per_program * 32u;
            features_valid &= std::abs(candidate.lane_utilization * candidate.scalar_rounds * workers - candidate.scalar_elements) < 1e-8;
            // Deliberately choose a non-power-of-two width from the automatic
            // family, independently of the default coefficient ranking.
            return candidate.subgroups_per_program == 3u ? 0.0 :
                                                           1.0 + AnalyticExecutionCostPolicy::reduction_score(candidate, model);
        }
    } policy;
    constexpr auto rows = int64_t{5};
    for (auto threads : {0u, 96u, 160u, 224u, 288u, 512u, 1024u}) {
        if (threads > runtime.metal_max_threads()) { continue; }
        for (auto width : {1u, 4u}) {
            auto planner = subgroup_reduction_options();
            planner.threads_per_group = threads;
            planner.reduction_lane_elements = width;
            planner.reduction_unroll_factor = 3u;
            planner.cost_policy = &policy;
            auto columns = int64_t{threads ? threads * 4u + 3u : 1031u};
            auto executable = runtime.build(make_row_softmax(rows, columns), true, false, true, false, planner, false, true);
            expect(executable.ok()) << "threads=" << threads << " width=" << width << " " << executable.error;
            if (!executable.ok()) { continue; }
            expect(eq(executable.plans.size(), 1u));
            if (executable.plans.empty()) { continue; }
            auto &plan = executable.plans.front();
            auto workers = threads ? threads : 96u;
            expect(eq(plan.threads, workers));
            expect(eq(plan.reduction_subgroups_per_program, workers / 32u));
            expect(eq(plan.reduction_programs_per_group, 1u));
            expect(eq(plan.reduction_threadgroups, static_cast<uint64_t>(rows)));
            // Enumerate ownership independently, including the ragged final
            // worker and the 32-partial boundary of the second collective.
            luisa::vector<uint64_t> counts(workers, 0u);
            for (auto i = int64_t{0}; i < columns; i++) { counts[(i / width) % workers]++; }
            auto slots = *std::max_element(counts.begin(), counts.end());
            expect(eq(plan.striped_storage_scalars_per_worker, slots));
            expect(std::abs(plan.reduction_scalar_rounds - 4.0 * slots) < 1e-8);
            expect(std::abs(plan.reduction_lane_utilization - static_cast<double>(columns) / static_cast<double>(slots * workers)) < 1e-8);
            expect(eq(plan.shared_memory_bytes, 2u * workers / 32u * sizeof(float)));
            auto source = metal_source(executable.module.value());
            auto code = std::string_view{source.data(), source.size()};
            expect(code.find("subgroup_partials_0[" + std::to_string(workers / 32u) + "]") != std::string_view::npos) << source;
            luisa::vector<float> values(static_cast<size_t>(rows * columns));
            for (size_t i = 0u; i < values.size(); i++) { values[i] = static_cast<float>(static_cast<int32_t>(i % 127u) - 63) / 31.0f; }
            auto input = runtime.upload<float>({rows, columns}, values);
            auto output = runtime.allocate<float>({rows, columns});
            (*executable.entry)(input, output);
            auto actual = runtime.download<float>(output, values.size());
            for (int64_t row = 0; row < rows; row++) {
                auto denominator = 0.0;
                for (int64_t column = 0; column < columns; column++) { denominator += std::exp(static_cast<double>(values[row * columns + column])); }
                for (int64_t column = 0; column < columns; column++) {
                    auto i = row * columns + column;
                    auto expected = std::exp(static_cast<double>(values[i])) / denominator;
                    expect(std::abs(actual[i] - expected) < 2e-6 + 2e-5 * expected) << "threads=" << threads << " width=" << width << " index=" << i;
                }
            }
        }
    }
    expect(policy.features_valid);
    auto legal_widths = std::min(32u, runtime.metal_max_threads() / 32u);
    for (auto i = 0u; i < legal_widths; i++) { expect(policy.widths[i]) << "missing candidate=" << (i + 1u) * 32u; }
    auto planner = subgroup_reduction_options();
    planner.max_thread_candidates = legal_widths - 1u;
    auto rejected = runtime.build(make_row_sum(rows, 257), true, false, true, false, planner, false, true);
    expect(!rejected.ok());
    expect(rejected.error.find("candidate budget") != luisa::string::npos) << rejected.error;
    planner.threads_per_group = 96u;
    auto exact = runtime.build(make_row_sum(rows, 257), true, false, true, false, planner, false, true);
    expect(exact.ok()) << exact.error;
    planner.threads_per_group = runtime.metal_max_threads() + 32u;
    auto over_limit = runtime.build(make_row_sum(rows, 257), true, false, true, false, planner, false, true);
    expect(!over_limit.ok());
    expect(over_limit.error.find("exact reduction mapping") != luisa::string::npos) << over_limit.error;
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
    auto expected_subgroups = expected_reduction_subgroups(runtime, rows, columns, 4u, 2u);
    auto expected_slots = ceil_div(static_cast<uint64_t>(columns), uint64_t{expected_subgroups * 32u});
    expect(eq(plan.reduction_subgroups_per_program, expected_subgroups));
    expect(eq(plan.reduction_operations, 2u));
    expect(eq(plan.reduction_elements, 8192u));
    expect(eq(plan.striped_storage_scalars_per_worker, expected_slots));
    expect(!plan.independent_subgroups);
    auto source = metal_source(executable.module.value());
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("_worker_stripe[" + std::to_string(expected_slots) + "]") != std::string_view::npos) << source;
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

void test_metal_subgroup_layernorm(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto rows = int64_t{3};
    constexpr auto columns = int64_t{4096};
    auto executable = runtime.build(
        make_row_layernorm(rows, columns), true, false, true, false,
        subgroup_reduction_options(), false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(eq(executable.plans.size(), 1u));
    if (executable.plans.empty()) { return; }
    auto &plan = executable.plans.front();
    auto expected_subgroups = expected_reduction_subgroups(runtime, rows, columns, 4u, 2u);
    auto expected_slots = ceil_div(static_cast<uint64_t>(columns), uint64_t{expected_subgroups * 32u});
    expect(eq(plan.reduction_subgroups_per_program, expected_subgroups));
    expect(eq(plan.reduction_operations, 2u));
    expect(eq(plan.reduction_elements, 8192u));
    expect(eq(plan.shared_memory_bytes, 2u * expected_subgroups * sizeof(float)));
    expect(eq(plan.striped_storage_scalars_per_worker, expected_slots));
    expect(!plan.independent_subgroups);
    auto source = metal_source(executable.module.value());
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("simd_sum(") != std::string_view::npos) << source;
    expect(code.find("_worker_stripe[" + std::to_string(expected_slots) + "]") != std::string_view::npos)
        << source;
    expect(code.find("[4096]") == std::string_view::npos) << source;

    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    luisa::vector<float> parameters(static_cast<size_t>(2 * columns));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int64_t>((i * 5u + 17u) %
                                                            127u) -
                                       63) /
                    64.0f;
    }
    for (auto column = int64_t{0}; column < columns; column++) {
        parameters[static_cast<size_t>(column)] =
            0.75f + static_cast<float>(column % 17) / 32.0f;
        parameters[static_cast<size_t>(columns + column)] =
            static_cast<float>(column % 13 - 6) / 64.0f;
    }
    auto input = runtime.upload<float>({rows, columns}, values);
    auto parameter_tensor = runtime.upload<float>({2, columns}, parameters);
    auto output = runtime.allocate<float>({rows, columns});
    (*executable.entry)(input, parameter_tensor, output);
    auto actual = runtime.download<float>(output, values.size());
    for (auto row = int64_t{0}; row < rows; row++) {
        auto mean = 0.0;
        for (auto column = int64_t{0}; column < columns; column++) {
            mean += values[static_cast<size_t>(row * columns + column)];
        }
        mean /= static_cast<double>(columns);
        auto variance = 0.0;
        for (auto column = int64_t{0}; column < columns; column++) {
            auto centered =
                static_cast<double>(values[static_cast<size_t>(
                    row * columns + column)]) -
                mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(columns);
        auto inverse_stddev = 1.0 / std::sqrt(variance + 1e-5);
        for (auto column = int64_t{0}; column < columns; column++) {
            auto index = static_cast<size_t>(row * columns + column);
            auto expected =
                (static_cast<double>(values[index]) - mean) * inverse_stddev *
                    parameters[static_cast<size_t>(column)] +
                parameters[static_cast<size_t>(columns + column)];
            expect(std::abs(static_cast<double>(actual[index]) - expected) <=
                   2e-6 + 2e-5 * std::abs(expected))
                << "row=" << row << " column=" << column;
        }
    }
}

void test_metal_subgroup_generic_striped_tile(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto rows = int64_t{3};
    constexpr auto columns = int64_t{4096};
    auto executable = runtime.build(
        make_row_tanh_statistic(rows, columns), true, false, true, false,
        subgroup_reduction_options(), false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(eq(executable.plans.size(), 1u));
    if (executable.plans.empty()) { return; }
    auto &plan = executable.plans.front();
    auto expected_subgroups = expected_reduction_subgroups(runtime, rows, columns, 3u, 1u);
    auto expected_slots = ceil_div(static_cast<uint64_t>(columns), uint64_t{expected_subgroups * 32u});
    expect(eq(plan.reduction_subgroups_per_program, expected_subgroups));
    expect(eq(plan.reduction_operations, 1u));
    expect(eq(plan.reduction_elements, 4096u));
    expect(eq(plan.shared_memory_bytes, expected_subgroups * sizeof(float)));
    expect(eq(plan.striped_storage_scalars_per_worker, expected_slots));
    auto source = metal_source(executable.module.value());
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("simd_sum(") != std::string_view::npos) << source;
    expect(code.find("_worker_stripe[" + std::to_string(expected_slots) + "]") != std::string_view::npos)
        << source;
    expect(code.find("[4096]") == std::string_view::npos) << source;

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
        auto sum = 0.0;
        for (auto column = int64_t{0}; column < columns; column++) {
            sum += std::tanh(static_cast<double>(
                values[static_cast<size_t>(row * columns + column)]));
        }
        for (auto column = int64_t{0}; column < columns; column++) {
            auto index = static_cast<size_t>(row * columns + column);
            auto expected = std::tanh(static_cast<double>(values[index])) + sum;
            expect(std::abs(static_cast<double>(actual[index]) - expected) <=
                   2e-5 + 2e-5 * std::abs(expected))
                << "row=" << row << " column=" << column;
        }
    }
}

void test_metal_subgroup_cross_entropy(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto rows = int64_t{7};
    constexpr auto columns = int64_t{4096};
    auto executable = runtime.build(
        make_row_cross_entropy(rows, columns), true, false, true, false,
        subgroup_reduction_options(), false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(eq(executable.plans.size(), 1u));
    if (executable.plans.empty()) { return; }
    auto &plan = executable.plans.front();
    expect(eq(plan.reduction_subgroups_per_program, 8u));
    expect(eq(plan.reduction_operations, 2u));
    expect(eq(plan.reduction_elements, 8192u));
    expect(eq(plan.shared_memory_bytes, 64u));
    expect(eq(plan.striped_storage_scalars_per_worker, 0u));
    auto source = metal_source(executable.module.value());
    auto code = std::string_view{source.data(), source.size()};
    expect(code.find("simd_max(") != std::string_view::npos) << source;
    expect(code.find("simd_sum(") != std::string_view::npos) << source;
    expect(code.find("thread float tile_storage_0[4096]") ==
           std::string_view::npos)
        << source;

    auto fallback = runtime.build(
        make_row_derived_cross_entropy(3, 257), true, false, true, false,
        subgroup_reduction_options(), false, true);
    expect(fallback.ok()) << fallback.error;
    if (fallback.ok()) {
        expect(fallback.plans.empty());
        auto fallback_source = metal_source(fallback.module.value());
        auto fallback_code = std::string_view{fallback_source.data(),
                                              fallback_source.size()};
        expect(fallback_code.find("simd_sum(") == std::string_view::npos)
            << fallback_source;
    }

    luisa::vector<float> values(static_cast<size_t>(rows * columns));
    luisa::vector<int64_t> labels(static_cast<size_t>(rows));
    for (auto i = 0u; i < values.size(); i++) {
        values[i] = static_cast<float>(static_cast<int64_t>((i * 5u + 17u) %
                                                            127u) -
                                       63) /
                    64.0f;
    }
    for (auto row = int64_t{0}; row < rows; row++) {
        labels[static_cast<size_t>(row)] = (row * 13 + 7) % columns;
    }
    auto logits = runtime.upload<float>({rows, columns}, values);
    auto label_tensor = runtime.upload<int64_t>({rows}, labels);
    auto losses = runtime.allocate<float>({rows});
    (*executable.entry)(logits, label_tensor, losses);
    auto actual = runtime.download<float>(losses, rows);
    for (auto row = int64_t{0}; row < rows; row++) {
        auto begin = static_cast<size_t>(row * columns);
        auto peak = -std::numeric_limits<double>::infinity();
        for (auto column = int64_t{0}; column < columns; column++) {
            peak = std::max(peak,
                            static_cast<double>(values[begin +
                                                       static_cast<size_t>(column)]));
        }
        auto total = 0.0;
        for (auto column = int64_t{0}; column < columns; column++) {
            total += std::exp(static_cast<double>(
                                  values[begin + static_cast<size_t>(column)]) -
                              peak);
        }
        auto expected = std::log(total) + peak -
                        values[begin + static_cast<size_t>(
                                           labels[static_cast<size_t>(row)])];
        expect(std::abs(static_cast<double>(actual[static_cast<size_t>(row)]) -
                        expected) <=
               2e-6 + 2e-5 * std::abs(expected))
            << "row=" << row;
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
    "tile_execution_fused_element_grid"_test = [&] { test_fused_element_grid(runtime); };
    "tile_execution_element_grid_snapshot"_test = [&] { test_element_grid_retains_snapshot(runtime); };
    "tile_execution_element_grid_shared_producers"_test = [&] { test_element_grid_shared_producers(runtime); };
    "tile_execution_element_grid_exact_reduction"_test = [&] { test_element_grid_respects_exact_reduction(runtime); };
    "tile_execution_element_grid_producer_contract"_test = [&] { test_element_grid_producer_contract(runtime); };
    "tile_execution_reduction_packing_and_policy"_test = [&] { test_metal_reduction_packing_and_policy(runtime); };
    "tile_execution_reduction_lane_elements"_test = [&] { test_reduction_lane_elements(runtime); };
    "tile_execution_reduction_complete_width_search"_test = [&] { test_reduction_complete_width_search(runtime); };
    "tile_execution_cpu_parallel_launch_cost"_test = [&] { test_cpu_parallel_launch_cost(runtime); };
    "tile_execution_shared_exp_materialization"_test = test_shared_exp_is_materialized_once;
    "tile_execution_shared_tanh_materialization"_test = test_shared_tanh_is_materialized_once;
    "tile_execution_shared_arithmetic_materialization"_test = test_shared_arithmetic_preserves_ssa;
    "tile_execution_metal_subgroup_reduction_contract"_test = [&] { test_metal_subgroup_reduction_contract(runtime); };
    "tile_execution_metal_subgroup_sum"_test = [&] { test_metal_subgroup_sum(runtime); };
    "tile_execution_metal_subgroup_striped_softmax"_test = [&] { test_metal_subgroup_striped_softmax(runtime); };
    "tile_execution_metal_subgroup_layernorm"_test = [&] { test_metal_subgroup_layernorm(runtime); };
    "tile_execution_metal_subgroup_generic_striped_tile"_test = [&] { test_metal_subgroup_generic_striped_tile(runtime); };
    "tile_execution_metal_subgroup_cross_entropy"_test = [&] { test_metal_subgroup_cross_entropy(runtime); };
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

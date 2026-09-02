// Run manual Memory POCs through native TVMx on CPU and physical Metal.
// Covers multiple resources, explicit placement, immutable load snapshots,
// mixed Tile/MemoryState carries, ancestor reads, ragged GEMM, local address
// maps, empty domains, and resource/owner rejection before device compilation.
#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <luisa/tile/memory.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <string_view>
#include <tvm/tirx/stmt_functor.h>

using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;

namespace {

[[nodiscard]] exec::Scope root_scope(const Runtime &runtime) {
    return runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
}

[[nodiscard]] mem::Resource root_resource(const Runtime &runtime) {
    return runtime.target() == "metal" ? mem::shared : mem::private_;
}

[[nodiscard]] luisa::vector<float> values(size_t count) {
    luisa::vector<float> result(count);
    for (auto i = 0u; i < count; i++) { result[i] = static_cast<float>(static_cast<int32_t>(i % 29u) - 14) * 0.125f; }
    return result;
}

void expect_near(const luisa::vector<float> &actual, const luisa::vector<float> &expected) {
    expect(eq(actual.size(), expected.size()));
    if (actual.size() != expected.size()) { return; }
    auto error = 0.0f;
    auto finite = true;
    for (auto i = 0u; i < actual.size(); i++) {
        finite &= std::isfinite(actual[i]);
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    expect(finite && error <= 1e-5f) << "max error=" << error;
}

[[nodiscard]] tvm::ffi::String metal_source(const tvm::ffi::Module &module) {
    if (std::string_view{module->kind()} == "metal") { return module->InspectSource("metal"); }
    for (auto &&child : module->imports()) {
        auto source = metal_source(child.cast<tvm::ffi::Module>());
        if (!source.empty()) { return source; }
    }
    return {};
}

void test_snapshots_and_carries(Runtime &runtime) {
    for (auto columns : {1, 7, 37, 257, 513}) {
        for (auto iterations : {0, 1, 3}) {
            constexpr auto rows = 3;
            auto scope = root_scope(runtime);
            auto resource = root_resource(runtime);
            auto definition = tile_kernel("manual_memory_state", [=](TensorView<const float, 2> input,
                                                                     TensorView<float, 2> output) {
                for (auto &nest : parallel(shape(rows), scope)) {
                    auto space = shape(1, columns);
                    auto origin = coord(nest.index(), 0);
                    auto a = memory<float>(space, resource);
                    auto b = memory<float>(space);// Same owner; independent placement choice.
                    a.store(input.tile(origin, space).load());
                    b.store(full<float>(space, 2.0f));
                    auto snapshot = a.load();
                    auto sum = zeros<float>(space);
                    for (auto &step : nest.pipeline(shape(iterations))) {
                        step.stage("read");
                        auto old_a = a.load();
                        auto old_b = b.load();
                        step.stage("write");
                        a.store(old_b + 1.0f);
                        b.store(old_a * 2.0f - 1.0f);
                        for (auto &inner : step.serial(shape(2))) { b.store(b.load() + 1.0f); }
                        sum += old_a + old_b;
                    }
                    output(origin, space).store(snapshot + a.load() + b.load() + sum);
                }
            });
            auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
            auto executable = runtime.build(kernel);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            auto output = runtime.allocate<float>({rows, columns});
            auto input = values(rows * columns);
            for (auto repeat = 0; repeat < 3; repeat++) {
                for (auto &value : input) { value += 0.25f; }
                auto source = runtime.upload<float>({rows, columns}, input);
                (*executable.entry)(source, output);
                auto expected = input;
                for (auto i = 0u; i < expected.size(); i++) {
                    auto a = input[i];
                    auto b = 2.0f;
                    auto sum = 0.0f;
                    for (auto step = 0; step < iterations; step++) {
                        auto old_a = a;
                        auto old_b = b;
                        a = old_b + 1.0f;
                        b = old_a * 2.0f + 1.0f;
                        sum += old_a + old_b;
                    }
                    expected[i] += a + b + sum;
                }
                expect_near(runtime.download<float>(output, expected.size()), expected);
            }
        }
    }
}

void test_ancestor_resources(Runtime &runtime, bool mapped) {
    constexpr auto rows = 3;
    constexpr auto columns = 7;
    auto scope = root_scope(runtime);
    auto resource = root_resource(runtime);
    auto child_scope = runtime.target() == "metal" ? exec::Scope::WORKER : exec::Scope::AUTOMATIC;
    auto definition = tile_kernel("manual_memory_ancestor", [=](TensorView<const float, 2> input,
                                                                TensorView<float, 2> output) {
        for (auto &nest : parallel(shape(rows), scope)) {
            auto shared = mapped ? memory<float>(layout(shape(1, columns), stride(columns * 2, 2)), resource) :
                                   memory<float>(shape(1, columns), resource);
            shared.store(input.tile(coord(nest.index(), 0), shape(1, columns)).load());
            for (auto &worker : nest.parallel(shape(columns), child_scope)) {
                IndexExpr address[]{IndexExpr::constant(2)};
                auto private_memory = mapped ? memory<float>(IndexMap{shape(1, 1), shape(3), address}, mem::private_) :
                                               memory<float>(shape(1, 1), mem::private_);
                auto value = shared.load().at(coord(0, columns - 1 - worker.index()));
                private_memory.store(full<float>(shape(1, 1), value));
                for (auto &phase : worker.pipeline(shape(3))) {
                    // A temporal declaration has lifetime in this iteration,
                    // but does not introduce a new spatial owner.
                    auto temporary = memory<float>(shape(1, 1));
                    temporary.store(private_memory.load() + cast<float>(phase.index()));
                    private_memory.store(temporary.load());
                }
                output(coord(nest.index(), worker.index()), shape(1, 1)).store(private_memory.load());
            }
        }
    });
    auto kernel = definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns));
    auto native = bridge::tirx::lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native.ok()) { return; }
    auto constraints = 0u;
    tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) {
            constraints += allocation->annotations.count("luisa.tile.memory_resource");
        }
    });
    expect(eq(constraints, 2u));
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    if (runtime.target() == "metal") {
        auto source = metal_source(executable.module.value());
        auto code = std::string_view{source.data(), source.size()};
        expect(code.find("threadgroup float") != std::string_view::npos);
        expect(code.find("metal::threadgroup_barrier(metal::mem_flags(3))") != std::string_view::npos);
        expect(code.find("luisa.tile.memory_resource") == std::string_view::npos);
    }
    auto input = values(rows * columns);
    auto source = runtime.upload<float>({rows, columns}, input);
    auto output = runtime.allocate<float>({rows, columns});
    (*executable.entry)(source, output);
    auto expected = input;
    for (auto row = 0; row < rows; row++) {
        for (auto column = 0; column < columns; column++) {
            expected[row * columns + column] = input[row * columns + columns - 1 - column] + 3.0f;
        }
    }
    expect_near(runtime.download<float>(output, expected.size()), expected);
}

void test_manual_gemm(Runtime &runtime) {
    for (auto [rows, inner, columns] : {std::array{1, 3, 5}, std::array{7, 13, 19}, std::array{17, 21, 23}, std::array{37, 41, 29}}) {
        auto scope = root_scope(runtime);
        auto resource = root_resource(runtime);
        auto definition = tile_kernel("manual_memory_gemm", [=](TensorView<const float, 2> a,
                                                                TensorView<const float, 2> b,
                                                                TensorView<float, 2> output) {
            auto gm = axis("gm", (rows + 7) / 8);
            auto gn = axis("gn", (columns + 7) / 8);
            auto m = axis("m", 8);
            auto n = axis("n", 8);
            auto k = axis("k", 8);
            for (auto &nest : parallel(shape(gm, gn), scope)) {
                auto m0 = nest[gm] * 8;
                auto n0 = nest[gn] * 8;
                auto as = memory<float>(layout(shape(m, k), stride(9, 1)), resource);
                auto bs = memory<float>(layout(shape(k, n), stride(1, 10)), resource);
                auto acc = zeros<float>(shape(m, n));
                for (auto &step : nest.pipeline(shape((inner + 7) / 8))) {
                    step.stage("load");
                    as.store(a.tile(coord(m0, step.index() * 8), shape(m, k)).load());
                    bs.store(b.tile(coord(step.index() * 8, n0), shape(k, n)).load());
                    step.stage("compute");
                    acc = mma(as.load(), bs.load(), acc);
                }
                output(coord(m0, n0), shape(m, n)).store(acc);
            }
        });
        auto executable = runtime.build(definition.capture(tensor_shape(rows, inner), tensor_shape(inner, columns), tensor_shape(rows, columns)));
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto a = values(rows * inner);
        auto b = values(inner * columns);
        auto lhs = runtime.upload<float>({rows, inner}, a);
        auto rhs = runtime.upload<float>({inner, columns}, b);
        auto output = runtime.allocate<float>({rows, columns});
        (*executable.entry)(lhs, rhs, output);
        luisa::vector<float> expected(rows * columns, 0.0f);
        for (auto row = 0; row < rows; row++) {
            for (auto column = 0; column < columns; column++) {
                for (auto k = 0; k < inner; k++) { expected[row * columns + column] += a[row * inner + k] * b[k * columns + column]; }
            }
        }
        expect_near(runtime.download<float>(output, expected.size()), expected);
    }
}

void test_worker_private_memory(Runtime &runtime) {
    for (auto columns : {1, 7, 37}) {
        auto definition = tile_kernel("manual_worker_memory", [columns](TensorView<const float, 2> input,
                                                                        TensorView<float, 2> output) {
            for (auto &worker : parallel(shape(37), exec::Scope::WORKER)) {
                auto space = shape(1, columns);
                auto origin = coord(worker.index(), 0);
                auto scratch = memory<float>(space, mem::private_);
                scratch.store(input.tile(origin, space).load());
                auto old = scratch.load();
                for (auto &step : worker.reduce(shape(3))) { scratch.store(scratch.load() + 1.0f); }
                output(origin, space).store(scratch.load() + old * 2.0f);
            }
        });
        auto executable = runtime.build(definition.capture(tensor_shape(37, columns), tensor_shape(37, columns)));
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto input = values(37 * columns);
        auto source = runtime.upload<float>({37, columns}, input);
        auto output = runtime.allocate<float>({37, columns});
        (*executable.entry)(source, output);
        for (auto &value : input) { value = value * 3.0f + 3.0f; }
        expect_near(runtime.download<float>(output, input.size()), input);
    }
}

void test_vector_private_memory(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    for (auto mapped : {false, true}) {
        auto definition = tile_kernel("manual_vector_memory", [mapped](TensorView<const float, 2> input, TensorView<float, 2> output) {
            for (auto &worker : parallel(shape(3), exec::Scope::WORKER)) {
                for (auto &lane : worker.parallel(shape(7), exec::Scope::VECTOR)) {
                    auto origin = coord(worker.index(), lane.index());
                    IndexExpr address[]{IndexExpr::constant(2)};
                    auto scratch = mapped ? memory<float>(IndexMap{shape(1, 1), shape(3), address}, mem::private_) :
                                            memory<float>(shape(1, 1), mem::private_);
                    scratch.store(input.tile(origin, shape(1, 1)).load());
                    auto old = scratch.load();
                    for (auto &step : lane.serial(shape(3))) { scratch.store(scratch.load() + 1.0f); }
                    output(origin, shape(1, 1)).store(scratch.load() + old * 2.0f);
                }
            }
        });
        auto executable = runtime.build(definition.capture(tensor_shape(3, 7), tensor_shape(3, 7)));
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { return; }
        auto input = values(21);
        auto source = runtime.upload<float>({3, 7}, input);
        auto output = runtime.allocate<float>({3, 7});
        (*executable.entry)(source, output);
        for (auto &value : input) { value = value * 3.0f + 3.0f; }
        expect_near(runtime.download<float>(output, input.size()), input);
    }
}

void test_empty_memory_layout(Runtime &runtime) {
    auto scope = root_scope(runtime);
    auto kernel = tile_kernel("manual_empty_memory_layout", [scope] {
                      for (auto &nest : parallel(shape(1), scope)) {
                          auto space = shape(0);
                          // Totality is vacuous on an empty domain. No address
                          // arithmetic or physical load/store may be evaluated.
                          IndexExpr address[]{floor_div(IndexExpr::coordinate(space.axis(0).dimension), IndexExpr::constant(0))};
                          auto scratch = memory<float>(IndexMap{space, space, address});
                          scratch.store(zeros<float>(space));
                          static_cast<void>(scratch.load());
                      }
                  }).capture();
    expect(kernel.valid());
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (executable.ok()) { (*executable.entry)(); }
}

void test_live_empty_storage(Runtime &runtime) {
    // A transform must never erase an allocation while leaving an actual
    // load/store dangling. Feed malformed native IR directly to this guard.
    for (auto read : {false, true}) {
        auto zero = tvm::IntImm::Int64(0);
        auto buffer = tvm::tirx::decl_buffer({zero}, tvm::PrimType::Float(32), "empty_live");
        tvm::tirx::Stmt use = read ? tvm::tirx::Stmt{tvm::tirx::Return{tvm::tirx::BufferLoad{buffer, {zero}}}} :
                                     tvm::tirx::Stmt{tvm::tirx::BufferStore{buffer, tvm::FloatImm{tvm::PrimType::Float(32), 1.0}, {zero}}};
        tvm::tirx::PrimFunc function{
            {}, tvm::tirx::SeqStmt{{tvm::tirx::AllocBuffer{buffer}, use}}, read ? tvm::Type{tvm::PrimType::Float(32)} : tvm::VoidType()};
        bridge::tirx::CompileOptions options;
        options.target = runtime.target();
        auto compilation = bridge::tirx::compile(std::move(function), "live_empty_storage", options);
        expect(!compilation.ok());
        expect(compilation.error().find("zero-sized Tile storage still has live buffer uses") != luisa::string_view::npos)
            << compilation.error();
    }
}

void test_memory_layouts(Runtime &runtime) {
    constexpr auto programs = 3;
    constexpr auto rows = 3;
    for (auto columns : {1, 7, 37, 65, 257}) {
        for (auto kind : {0, 1, 2, 3}) {
            auto scope = root_scope(runtime);
            auto resource = root_resource(runtime);
            auto padded = std::bit_ceil(static_cast<uint64_t>(columns));
            auto definition = tile_kernel("manual_memory_layout", [=](TensorView<const float, 2> input,
                                                                      TensorView<float, 2> output) {
                auto m = axis("m", rows);
                auto n = axis("n", columns);
                auto space = shape(m, n);
                IndexMap address;
                if (kind == 0) {
                    address = layout(space, stride(columns + 3, 1));
                } else if (kind == 1) {
                    Dim order[]{n.dimension(), m.dimension()};
                    auto transposed = IndexMap::permute(space, order);
                    expect(transposed.has_value());
                    if (!transposed) { return; }
                    address = *transposed;
                } else {
                    auto physical = shape(rows, padded);
                    auto row = IndexExpr::coordinate(m.dimension());
                    auto column = IndexExpr::coordinate(n.dimension());
                    auto width = std::bit_width(padded - 1u);
                    auto shift = IndexExpr::constant(width == 0u ? 0 : static_cast<int64_t>(64u - width));
                    auto transformed = kind == 2 ?
                                           bit_xor(column, bit_and(row, IndexExpr::constant(static_cast<int64_t>(padded - 1u)))) :
                                           shift_right(shift_left(column, shift), shift);
                    IndexExpr outputs[]{row, std::move(transformed)};
                    auto composed = IndexMap::compose(layout(physical, stride(padded + 2u, 1u)), IndexMap{space, physical, outputs});
                    expect(composed.has_value());
                    if (!composed) { return; }
                    address = *composed;
                }
                for (auto &nest : parallel(shape(programs), scope)) {
                    auto origin = coord(nest.index() * rows, 0);
                    auto scratch = memory<float>(address, resource);
                    scratch.store(input.tile(origin, space).load());
                    auto old = scratch.load();
                    for (auto &step : nest.pipeline(shape(3))) {
                        scratch.store(scratch.load() + cast<float>(step.index()));
                    }
                    output(origin, space).store(old + scratch.load() * 2.0f);
                }
            });
            auto kernel = definition.capture(tensor_shape(programs * rows, columns), tensor_shape(programs * rows, columns));
            expect(kernel.valid());
            auto native = bridge::tirx::lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            luisa::vector<int64_t> expected_shape;
            if (kind == 0) {
                expected_shape = {rows * columns + (rows - 1) * 3};
            } else if (kind == 1) {
                expected_shape = {columns, rows};
            } else {
                expected_shape = {static_cast<int64_t>(rows * padded + (rows - 1) * 2u)};
            }
            auto allocations = 0u;
            tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
                if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) {
                    auto name = allocation->buffer.name();
                    if (!std::string_view{name.data(), name.size()}.starts_with("tile_memory_")) { return; }
                    allocations++;
                    luisa::vector<int64_t> actual_shape;
                    for (auto &&extent : allocation->buffer->shape) {
                        auto value = extent.as<tvm::IntImmNode>();
                        actual_shape.emplace_back(value == nullptr ? -1 : value->value);
                    }
                    expect(actual_shape == expected_shape) << "storage shape must follow the explicit map";
                }
            });
            expect(eq(allocations, 1u));
            auto executable = runtime.build(kernel);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            auto input = values(programs * rows * columns);
            auto source = runtime.upload<float>({programs * rows, columns}, input);
            auto output = runtime.allocate<float>({programs * rows, columns});
            (*executable.entry)(source, output);
            for (auto &value : input) { value = value * 3.0f + 6.0f; }
            expect_near(runtime.download<float>(output, input.size()), input);
        }
    }
}

void test_unproved_layout(Runtime &runtime) {
    auto kernel = tile_kernel("manual_unproved_layout", [] {
                      for (auto &nest : parallel(shape(1), exec::Scope::WORKER)) {
                          auto space = shape(1048577);
                          auto scratch = memory<float>(IndexMap::identity(space));
                          scratch.store(zeros<float>(space));
                      }
                  }).capture();
    expect(kernel.valid()) << "the representation is not limited to the current proof budget";
    auto executable = runtime.build(kernel);
    expect(!executable.ok());
    expect(executable.error.find("proof budget") != luisa::string::npos) << executable.error;
}

void test_manual_capacity(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    for (auto padded : {false, true}) {
        auto count = padded ? 3 : 37;
        auto kernel = tile_kernel("manual_capacity", [=](TensorView<float, 1> output) {
                          for (auto &group : parallel(shape(1), exec::Scope::GROUP)) {
                              auto space = shape(count);
                              auto scratch = padded ? memory<float>(layout(space, stride(20)), mem::shared) :
                                                      memory<float>(space, mem::shared);
                              scratch.store(zeros<float>(space));
                              output(coord(group.index()), space).store(scratch.load());
                          }
                      }).capture(tensor_shape(count));
        auto native = bridge::tirx::lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { return; }
        bridge::tirx::CompileOptions options;
        options.target = R"({"kind":"metal","max_shared_memory_per_block":128})";
        auto compilation = bridge::tirx::compile(native.value, kernel.function().name(), options);
        expect(!compilation.ok());
        expect(compilation.error().find("shared-memory capacity") != luisa::string_view::npos) << compilation.error();
    }
}

void test_unsupported_resources(Runtime &runtime) {
    for (auto count : {0, 1}) {
        for (auto elements : {0, 7}) {
            for (auto resource : {mem::global, mem::cluster, mem::tensor}) {
                auto scope = root_scope(runtime);
                auto kernel = tile_kernel("manual_unsupported_resource", [=](TensorView<float, 1> output) {
                                  for (auto &nest : parallel(shape(count), scope)) {
                                      auto scratch = memory<float>(shape(elements), resource);
                                      scratch.store(zeros<float>(shape(elements)));
                                      output(coord(nest.index()), shape(elements)).store(scratch.load());
                                  }
                              }).capture(tensor_shape(7));
                expect(kernel.valid());
                auto executable = runtime.build(kernel);
                expect(!executable.ok());
                expect(executable.error.find("Memory resource") != luisa::string::npos) << executable.error;
            }
        }
    }
    auto scope = root_scope(runtime);
    auto resource = runtime.target() == "metal" ? mem::private_ : mem::shared;
    auto kernel = tile_kernel("manual_incompatible_owner", [=](TensorView<float, 1> output) {
                      for (auto &nest : parallel(shape(1), scope)) {
                          auto a = memory<float>(shape(7), resource);
                          a.store(zeros<float>(shape(7)));
                          output(coord(nest.index()), shape(7)).store(a.load());
                      }
                  }).capture(tensor_shape(7));
    auto executable = runtime.build(kernel);
    expect(!executable.ok());
    expect(executable.error.find("Memory resource") != luisa::string::npos) << executable.error;
    if (runtime.target() == "metal") {
        auto descendant = tile_kernel("manual_worker_shared_without_slice", [](TensorView<float, 1> output) {
                              for (auto &group : parallel(shape(1), exec::Scope::GROUP)) {
                                  for (auto &worker : group.parallel(shape(7), exec::Scope::WORKER)) {
                                      auto scratch = memory<float>(shape(1), mem::shared);
                                      scratch.store(zeros<float>(shape(1)));
                                      output(coord(worker.index()), shape(1)).store(scratch.load());
                                  }
                              }
                          }).capture(tensor_shape(7));
        auto rejected = runtime.build(descendant);
        expect(!rejected.ok());
        expect(rejected.error.find("Memory resource") != luisa::string::npos) << rejected.error;
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_native_memory_snapshots_and_carries"_test = [&] { test_snapshots_and_carries(runtime); };
    "tile_native_memory_ancestor_resources"_test = [&] {
        for (auto mapped : {false, true}) { test_ancestor_resources(runtime, mapped); }
    };
    "tile_native_memory_gemm"_test = [&] { test_manual_gemm(runtime); };
    "tile_native_memory_worker_private"_test = [&] { test_worker_private_memory(runtime); };
    "tile_native_memory_vector_private"_test = [&] { test_vector_private_memory(runtime); };
    "tile_native_memory_empty_layout"_test = [&] { test_empty_memory_layout(runtime); };
    "tile_native_memory_live_empty_storage"_test = [&] { test_live_empty_storage(runtime); };
    "tile_native_memory_layouts"_test = [&] { test_memory_layouts(runtime); };
    "tile_native_memory_unproved_layout"_test = [&] { test_unproved_layout(runtime); };
    "tile_native_memory_capacity"_test = [&] { test_manual_capacity(runtime); };
    "tile_native_memory_resource_constraints"_test = [&] { test_unsupported_resources(runtime); };
}

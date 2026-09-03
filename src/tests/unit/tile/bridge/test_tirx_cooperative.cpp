// Test cooperative execution and resource mapping through native TVMx.
// Covers shared ancestor Tiles, multiple resources, explicit worker descent,
// global-memory phase ordering, reductions, MMA, and ragged dimensions.
#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <luisa/tile/algorithms.h>

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;

namespace {

[[nodiscard]] exec::Scope root_scope(const Runtime &runtime) {
    return runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
}

[[nodiscard]] luisa::vector<float> values(size_t count) {
    luisa::vector<float> result(count);
    for (auto i = 0u; i < count; i++) { result[i] = static_cast<float>(static_cast<int32_t>(i % 29u) - 14) * 0.125f; }
    return result;
}

void expect_near(const luisa::vector<float> &actual, const luisa::vector<float> &expected, float tolerance = 1e-5f) {
    expect(eq(actual.size(), expected.size()));
    if (actual.size() != expected.size()) { return; }
    auto max_error = 0.0f;
    auto finite = true;
    for (auto i = 0u; i < actual.size(); i++) {
        finite &= std::isfinite(actual[i]);
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    expect(finite && max_error <= tolerance) << "max error=" << max_error;
}

[[nodiscard]] tvm::ffi::String metal_source(const tvm::ffi::Module &module) {
    if (std::string_view{module->kind()} == "metal") { return module->InspectSource("metal"); }
    for (auto &&child : module->imports()) {
        auto source = metal_source(child.cast<tvm::ffi::Module>());
        if (!source.empty()) { return source; }
    }
    return {};
}

void test_shared_tiles_and_global_order(Runtime &runtime) {
    for (auto columns : {1, 7, 37, 129, 257, 1003}) {
        constexpr auto rows = 3;
        auto scope = root_scope(runtime);
        auto definition = tile_kernel("cooperative_resources", [=](TensorView<const float, 2> input,
                                                                   TensorView<float, 2> scratch,
                                                                   TensorView<float, 2> output) {
            auto m = axis("m", 1);
            auto n = axis("n", columns);
            for (auto &nest : parallel(shape(rows), scope)) {
                auto origin = coord(nest.index(), 0);
                auto x = input.tile(origin, shape(m, n)).load();
                // Two resources at the same execution level; y is produced
                // by a different lane than the one later consuming it.
                auto y = map<float>(shape(m, n), [&](const Nest &element) {
                    return x.at(coord(0, columns - 1 - element.index(n))) * 3.0f + 2.0f;
                });
                for (auto &phase : nest.pipeline(shape(3))) {
                    phase.stage("publish");
                    auto worker_scope = runtime.target() == "metal" ? exec::Scope::WORKER : exec::Scope::AUTOMATIC;
                    for (auto &worker : phase.parallel(shape(columns), worker_scope)) {
                        auto value = x.at(coord(0, worker.index())) + y.at(coord(0, worker.index())) + cast<float>(phase.index());
                        scratch(coord(nest.index(), worker.index()), shape(1, 1)).store(full<float>(shape(1, 1), value));
                    }
                    // The worker region must fence device memory as well as
                    // shared memory before another worker reads its neighbor.
                    phase.stage("consume");
                    auto published = scratch.tile(origin, shape(m, n)).load();
                    auto reversed = map<float>(shape(m, n), [&](const Nest &element) {
                        return published.at(coord(0, columns - 1 - element.index(n)));
                    });
                    output(origin, shape(m, n)).store(reversed);
                }
            }
        });
        auto executable = runtime.build(definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns), tensor_shape(rows, columns)));
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        if (runtime.target() == "metal" && columns == 37) {
            auto source = metal_source(executable.module.value());
            auto code = std::string_view{source.data(), source.size()};
            expect(code.find("threadgroup float") != std::string_view::npos);
            expect(code.find("thread_position_in_threadgroup") != std::string_view::npos);
            expect(code.find("metal::threadgroup_barrier(metal::mem_flags(3))") != std::string_view::npos);
        }
        auto input = values(rows * columns);
        auto scratch = runtime.allocate<float>({rows, columns});
        auto output = runtime.allocate<float>({rows, columns});
        for (auto repeat = 0; repeat < 4; repeat++) {
            for (auto &value : input) { value += 0.125f; }
            auto source = runtime.upload<float>({rows, columns}, input);
            (*executable.entry)(source, scratch, output);
            auto actual = runtime.download<float>(output, input.size());
            luisa::vector<float> expected(input.size());
            for (auto row = 0; row < rows; row++) {
                for (auto col = 0; col < columns; col++) {
                    expected[row * columns + col] = input[row * columns + col] * 3.0f + input[row * columns + columns - 1 - col] + 4.0f;
                }
            }
            expect_near(actual, expected);
        }
    }
}

void test_resource_capacity(Runtime &runtime) {
    using namespace luisa::compute::tile::bridge::tirx;
    auto definition = tile_kernel("cooperative_capacity", [](TensorView<const float, 1> input, TensorView<float, 1> output) {
        for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
            auto origin = coord(nest.index());
            output(origin, shape(37)).store(input.tile(origin, shape(37)).load());
        }
    });
    auto kernel = definition.capture(tensor_shape(37), tensor_shape(37));
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    CompileOptions options;
    options.target = runtime.target() == "metal" ? R"({"kind":"metal","max_shared_memory_per_block":128})" : "llvm";
    auto compilation = compile(native.value, kernel.function().name(), options);
    expect(!compilation.ok());
    expect(compilation.error().find(runtime.target() == "metal" ? "shared-memory capacity" : "execution scope") != luisa::string_view::npos)
        << compilation.error();
}

void test_host_local_capture_rejected(Runtime &runtime) {
    auto definition = tile_kernel("cooperative_host_capture", [](TensorView<const float, 1> input, TensorView<float, 1> output) {
        auto x = input.tile(coord(0), shape(37)).load();
        for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
            output(coord(nest.index()), shape(37)).store(x);
        }
    });
    auto executable = runtime.build(definition.capture(tensor_shape(37), tensor_shape(37)));
    expect(!executable.ok());
    expect(executable.error.find(runtime.target() == "metal" ? "device allocation plan" : "execution scope") != luisa::string::npos) << executable.error;
}

void test_nested_scope_rejected(Runtime &runtime) {
    for (auto count : {0, 1}) {
        auto definition = tile_kernel("cooperative_nested_scope", [count](TensorView<float, 1> output) {
            for (auto &group : parallel(shape(count), exec::Scope::GROUP)) {
                for (auto &serial : group.serial(shape(1))) {
                    for (auto &subgroup : serial.parallel(shape(1), exec::Scope::SUBGROUP)) {
                        output(coord(subgroup.index()), shape(1)).store(zeros<float>(shape(1)));
                    }
                }
            }
        });
        auto kernel = definition.capture(tensor_shape(1));
        expect(kernel.valid());
        auto executable = runtime.build(kernel);
        expect(!executable.ok());
        expect(executable.error.find(runtime.target() == "metal" ? "nested execution scope 'subgroup'" : "execution scope 'group'") != luisa::string::npos)
            << executable.error;
    }
}

void test_rectangular_element_domain(Runtime &runtime) {
    constexpr auto groups = 3;
    constexpr auto group_elements = 5 * 7 * 11;
    auto scope = root_scope(runtime);
    auto definition = tile_kernel("cooperative_rectangular_domain", [scope](TensorView<const float, 3> input,
                                                                            TensorView<float, 3> output) {
        auto i = axis("i", 5);
        auto j = axis("j", 7);
        auto k = axis("k", 11);
        for (auto &group : parallel(shape(groups), scope)) {
            auto origin = coord(group.index() * 5, 0, 0);
            auto x = input.tile(origin, shape(i, j, k)).load();
            auto y = map<float>(shape(i, j, k), [&](const Nest &element) {
                return x.at(coord(4 - element.index(i), 6 - element.index(j), 10 - element.index(k))) * 2.0f + 1.0f;
            });
            output(origin, shape(i, j, k)).store(y);
        }
    });
    auto kernel = definition.capture(tensor_shape(groups * 5, 7, 11), tensor_shape(groups * 5, 7, 11));
    auto native = luisa::compute::tile::bridge::tirx::lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto domains = 0u;
    tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
        if (auto loop = node.as<tvm::tirx::ForNode>()) {
            if (auto annotation = loop->annotations.Get("luisa.tile.independent_elements")) {
                expect(eq(annotation.value().cast<tvm::IntImm>()->value, 3));
                domains++;
                for (auto extent : {5, 7, 11}) {
                    expect(loop != nullptr);
                    if (loop == nullptr) { break; }
                    expect(eq(loop->extent.as<tvm::IntImmNode>()->value, extent));
                    loop = loop->body.as<tvm::tirx::ForNode>();
                }
            }
        }
    });
    expect(eq(domains, 3u));// load, pure map, and store keep their axes
    auto executable = runtime.build(kernel);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto input = values(groups * group_elements);
    auto source = runtime.upload<float>({groups * 5, 7, 11}, input);
    auto output = runtime.allocate<float>({groups * 5, 7, 11});
    (*executable.entry)(source, output);
    auto actual = runtime.download<float>(output, input.size());
    luisa::vector<float> expected(input.size());
    for (auto group = 0; group < groups; group++) {
        for (auto element = 0; element < group_elements; element++) {
            expected[group * group_elements + element] = input[group * group_elements + group_elements - 1 - element] * 2.0f + 1.0f;
        }
    }
    expect_near(actual, expected);
}

class NonUnitDomain final : public tvm::tirx::StmtMutator {
private:
    bool _elements;
    bool _inner;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto result = StmtMutator::VisitStmt_(loop).as_or_throw<tvm::tirx::For>();
        auto annotation = _elements ? "luisa.tile.independent_elements" : "luisa.tile.logical_parallel";
        if (loop->annotations.count(annotation)) {
            if (_inner) {
                auto child = result->body.as_or_throw<tvm::tirx::For>();
                child.CopyOnWrite()->step = tvm::IntImm::Int64(2);
                result.CopyOnWrite()->body = std::move(child);
            } else {
                result.CopyOnWrite()->step = tvm::IntImm::Int64(2);
            }
            replacements++;
        }
        return result;
    }

public:
    uint32_t replacements{0u};
    NonUnitDomain(bool elements, bool inner) : _elements{elements}, _inner{inner} {}
    using StmtMutator::operator();
};

void test_noncanonical_domain_rejected(Runtime &runtime) {
    using namespace luisa::compute::tile::bridge::tirx;
    auto kernel = tile_kernel("cooperative_nonunit_domain", [](TensorView<float, 2> output) {
                      for (auto &group : parallel(shape(7), exec::Scope::GROUP)) {
                          output(coord(group.index() * 3, 0), shape(3, 5)).store(zeros<float>(shape(3, 5)));
                      }
                  }).capture(tensor_shape(21, 5));
    for (auto [elements, inner] : {std::pair{false, false}, std::pair{true, false}, std::pair{true, true}}) {
        auto native = lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { continue; }
        NonUnitDomain transform{elements, inner};
        native.value.CopyOnWrite()->body = transform(native.value->body);
        expect(transform.replacements > 0u);
        CompileOptions options;
        options.target = runtime.target();
        auto compilation = compile(std::move(native.value), kernel.function().name(), options);
        expect(!compilation.ok());
        expect(compilation.error().find(runtime.target() == "metal" ? "unit-step domains" : "execution scope 'group'") != luisa::string_view::npos)
            << compilation.error();
    }
}

void test_softmax(Runtime &runtime) {
    for (auto columns : {1, 7, 37, 257, 1003}) {
        constexpr auto rows = 17;
        auto scope = root_scope(runtime);
        auto definition = tile_kernel("cooperative_softmax", [=](TensorView<const float, 2> input, TensorView<float, 2> output) {
            auto m = axis("m", 1);
            auto n = axis("n", columns);
            for (auto &nest : parallel(shape(rows), scope)) {
                auto origin = coord(nest.index(), 0);
                auto x = input.tile(origin, shape(m, n)).load();
                auto e = exp(x - reduce(x, n, maximum));
                output(origin, shape(m, n)).store(e / reduce(e, n, add));
            }
        });
        auto executable = runtime.build(definition.capture(tensor_shape(rows, columns), tensor_shape(rows, columns)));
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto input = values(rows * columns);
        auto source = runtime.upload<float>({rows, columns}, input);
        auto output = runtime.allocate<float>({rows, columns});
        (*executable.entry)(source, output);
        auto actual = runtime.download<float>(output, input.size());
        luisa::vector<float> expected(input.size());
        for (auto row = 0; row < rows; row++) {
            auto sum = 0.0;
            for (auto col = 0; col < columns; col++) { sum += std::exp(static_cast<double>(input[row * columns + col])); }
            for (auto col = 0; col < columns; col++) { expected[row * columns + col] = static_cast<float>(std::exp(static_cast<double>(input[row * columns + col])) / sum); }
        }
        expect_near(actual, expected);
    }
}

void test_batched_copies(Runtime &runtime) {
    for (auto columns : {1, 7, 37, 129, 257, 1003}) {
        constexpr auto groups = 3;
        constexpr auto input_rows = groups * 3 - 1;
        auto input_columns = std::max(1, columns - 4);
        auto scope = root_scope(runtime);
        auto definition = tile_kernel("cooperative_copy_batches", [=](TensorView<const float, 2> input,
                                                                      TensorView<float, 2> output) {
            auto m = axis("m", 3);
            auto n = axis("n", columns);
            for (auto &group : parallel(shape(groups), scope)) {
                // Both ends need bounded loads. A batched full worker chunk
                // is not permission to speculate invalid global accesses.
                auto x = input.tile(coord(group.index() * 3 - 1, -2), shape(m, n), bounds::zero).load();
                auto y = map<float>(shape(m, n), [&](const Nest &element) {
                    return x.at(coord(2 - element.index(m), columns - 1 - element.index(n))) * 2.0f + 1.0f;
                });
                output(coord(group.index() * 3, 0), shape(m, n)).store(y);
            }
        });
        auto kernel = definition.capture(tensor_shape(input_rows, input_columns), tensor_shape(groups * 3, columns));
        auto input = values(input_rows * input_columns);
        auto source = runtime.upload<float>({input_rows, input_columns}, input);
        auto output = runtime.allocate<float>({groups * 3, columns});
        luisa::vector<float> expected(groups * 3 * columns);
        for (auto group = 0; group < groups; group++) {
            for (auto row = 0; row < 3; row++) {
                for (auto column = 0; column < columns; column++) {
                    auto source_row = group * 3 + 1 - row;
                    auto source_column = columns - 3 - column;
                    auto value = source_row >= 0 && source_row < input_rows && source_column >= 0 && source_column < input_columns ?
                                     input[source_row * input_columns + source_column] :
                                     0.0f;
                    expected[(group * 3 + row) * columns + column] = value * 2.0f + 1.0f;
                }
            }
        }
        for (auto threads : {32u, 48u, 256u}) {
            for (auto batch : {1u, 4u, 16u}) {
                if (runtime.target() != "metal" && (threads != 32u || batch != 1u)) { continue; }
                luisa::compute::tile::bridge::tirx::PlannerOptions options;
                options.threads_per_group = runtime.target() == "metal" ? threads : 0u;
                options.max_copy_batch = batch;
                auto executable = runtime.build(kernel, true, false, true, false, options);
                expect(executable.ok()) << executable.error;
                if (!executable.ok()) { continue; }
                if (runtime.target() == "metal") {
                    expect(eq(executable.plans.size(), size_t{1u}));
                    if (!executable.plans.empty()) {
                        auto batches_expected = batch > 1u && static_cast<uint32_t>(columns * 3) >= threads * 2u;
                        expect(eq(executable.plans[0].max_copy_batch, batch));
                        expect((executable.plans[0].batched_copy_operations != 0u) == batches_expected);
                        if (batches_expected) {
                            auto native = metal_source(executable.module.value());
                            auto code = std::string_view{native.data(), native.size()};
                            expect(code.find("_copy_value_") != std::string_view::npos);
                        }
                    }
                }
                (*executable.entry)(source, output);
                expect_near(runtime.download<float>(output, expected.size()), expected);
            }
        }
    }
}

void test_barrier_coalescing_nonadjacent_dependencies(Runtime &runtime) {
    constexpr auto groups = 3;
    auto scope = root_scope(runtime);
    for (auto columns : {1, 37, 257, 1003}) {
        auto kernel = tile_kernel("cooperative_nonadjacent_dependency", [=](TensorView<const float, 1> X,
                                                                            TensorView<const float, 1> Y,
                                                                            TensorView<float, 1> output) {
                          auto n = axis("n", columns);
                          for (auto &group : parallel(shape(groups), scope)) {
                              auto origin = coord(group.index() * columns);
                              auto x = X[origin, shape(n)];
                              auto y = Y[origin, shape(n)];
                              // The y load is independent of both the x load
                              // and this read of x. Only an accumulated effect
                              // summary notices the nonadjacent x dependence.
                              auto z = map<float>(shape(n), [&](const Nest &element) {
                                  return x.at(coord(columns - 1 - element.index(n))) * 3.0f + 1.0f;
                              });
                              output(origin, shape(n)).store(z + y);
                          }
                      }).capture(tensor_shape(groups * columns), tensor_shape(groups * columns), tensor_shape(groups * columns));
        for (auto threads : {32u, 48u, 256u}) {
            if (runtime.target() != "metal" && threads != 32u) { continue; }
            for (auto enabled : {false, true}) {
                bridge::tirx::PlannerOptions options;
                options.threads_per_group = runtime.target() == "metal" ? threads : 0u;
                options.coalesce_group_barriers = enabled;
                auto executable = runtime.build(kernel, false, false, true, false, options);
                expect(executable.ok()) << executable.error;
                if (!executable.ok()) { continue; }
                if (runtime.target() == "metal") {
                    expect(eq(executable.plans.size(), size_t{1u}));
                    if (!executable.plans.empty()) {
                        auto &plan = executable.plans[0];
                        // Only x-load -> y-load is an independent cut. In
                        // particular y-load -> z must still publish x.
                        expect(plan.group_barrier_sites_before >= 4u);
                        expect(eq(plan.group_barrier_sites_after + static_cast<uint64_t>(enabled), plan.group_barrier_sites_before));
                    }
                }
                for (auto repeat = 0; repeat < 4; repeat++) {
                    auto x = values(groups * columns);
                    auto y = values(groups * columns);
                    for (auto &value : x) { value += 0.125f * static_cast<float>(repeat); }
                    for (auto &value : y) { value = value * 2.0f - 0.25f * static_cast<float>(repeat); }
                    auto a = runtime.upload<float>({groups * columns}, x);
                    auto b = runtime.upload<float>({groups * columns}, y);
                    auto output = runtime.allocate<float>({groups * columns});
                    (*executable.entry)(a, b, output);
                    luisa::vector<float> expected(groups * columns);
                    for (auto group = 0; group < groups; group++) {
                        for (auto column = 0; column < columns; column++) {
                            auto index = group * columns + column;
                            expected[index] = x[group * columns + columns - 1 - column] * 3.0f + 1.0f + y[index];
                        }
                    }
                    expect_near(runtime.download<float>(output, expected.size()), expected);
                }
            }
        }
    }
}

void test_barrier_coalescing_global_aliases_and_loop_backedge(Runtime &runtime) {
    constexpr auto groups = 3;
    auto scope = root_scope(runtime);
    for (auto columns : {37, 257, 1003}) {
        auto kernel = tile_kernel("cooperative_alias_and_backedge", [=](TensorView<float, 1> A,
                                                                        TensorView<const float, 1> B,
                                                                        TensorView<float, 1> output) {
                          auto n = axis("n", columns);
                          for (auto &group : parallel(shape(groups), scope)) {
                              auto origin = coord(group.index() * columns);
                              for (auto &iteration : group.pipeline(shape(3), {.stages = 1u})) {
                                  auto x = A[origin, shape(n)];
                                  auto reversed = map<float>(shape(n), [&](const Nest &element) {
                                      return x.at(coord(columns - 1 - element.index(n))) + 1.0f;
                                  });
                                  A(origin, shape(n)).store(reversed);
                                  auto observed = B[origin, shape(n)];
                                  auto next = map<float>(shape(n), [&](const Nest &element) {
                                      return observed.at(coord(element.index(n))) + 2.0f;
                                  });
                                  output(origin, shape(n)).store(next);
                              }
                          }
                      }).capture(tensor_shape(groups * columns), tensor_shape(groups * columns), tensor_shape(groups * columns));
        for (auto enabled : {false, true}) {
            bridge::tirx::PlannerOptions options;
            options.threads_per_group = runtime.target() == "metal" ? 48u : 0u;
            options.coalesce_group_barriers = enabled;
            auto executable = runtime.build(kernel, false, false, true, false, options);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            if (runtime.target() == "metal") {
                for (auto &plan : executable.plans) {
                    // Every cut has a dependence. A/B/output are different
                    // parameter identities, but no noalias promise was made.
                    expect(plan.group_barrier_sites_before >= 5u);
                    expect(eq(plan.group_barrier_sites_before, plan.group_barrier_sites_after));
                }
            }
            for (auto alias_output : {false, true}) {
                for (auto repeat = 0; repeat < 4; repeat++) {
                    auto input = values(groups * columns);
                    for (auto &value : input) { value += 0.125f * static_cast<float>(repeat); }
                    auto shared = runtime.upload<float>({groups * columns}, input);
                    auto output = alias_output ? shared : runtime.allocate<float>({groups * columns});
                    (*executable.entry)(shared, shared, output);
                    luisa::vector<float> expected(groups * columns);
                    for (auto group = 0; group < groups; group++) {
                        for (auto column = 0; column < columns; column++) {
                            expected[group * columns + column] = input[group * columns + columns - 1 - column] + (alias_output ? 9.0f : 5.0f);
                        }
                    }
                    expect_near(runtime.download<float>(output, expected.size()), expected);
                }
            }
        }
    }
}

class InsertExplicitBarrier final : public tvm::tirx::StmtMutator {
protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto result = StmtMutator::VisitStmt_(loop);
        if (insertions == 0u && loop->annotations.count("luisa.tile.independent_elements")) {
            auto flags = tvm::Call{tvm::PrimType::Int(32), tvm::tirx::builtin::call_extern(), {tvm::tirx::StringImm{"metal::mem_flags"}, tvm::IntImm::Int32(3)}};
            auto barrier = tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), tvm::tirx::builtin::call_extern(), {tvm::tirx::StringImm{"metal::threadgroup_barrier"}, flags}}};
            result = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{result, barrier});
            insertions++;
        }
        return result;
    }

public:
    uint32_t insertions{0u};
    using StmtMutator::operator();
};

void test_explicit_barrier_identity_is_preserved(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    constexpr auto groups = 3, columns = 257;
    auto kernel = tile_kernel("cooperative_explicit_barrier", [](TensorView<const float, 1> A,
                                                                 TensorView<const float, 1> B,
                                                                 TensorView<float, 1> output) {
                      for (auto &group : parallel(shape(groups), exec::Scope::GROUP)) {
                          auto origin = coord(group.index() * columns);
                          auto a = A[origin, shape(columns)];
                          auto b = B[origin, shape(columns)];
                          output(origin, shape(columns)).store(a + b);
                      }
                  }).capture(tensor_shape(groups * columns), tensor_shape(groups * columns), tensor_shape(groups * columns));
    for (auto enabled : {false, true}) {
        auto native = bridge::tirx::lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { continue; }
        // Same intrinsic and flags as the compiler fence, but a distinct
        // explicit IR operation. Name matching must not make it removable.
        InsertExplicitBarrier insert;
        native.value.CopyOnWrite()->body = insert(native.value->body);
        expect(eq(insert.insertions, 1u));
        bridge::tirx::CompileOptions options;
        options.target = "metal";
        options.planner.threads_per_group = 48u;
        options.planner.coalesce_group_barriers = enabled;
        auto compilation = bridge::tirx::compile(native.value, kernel.function().name(), options);
        expect(compilation.ok()) << compilation.error();
        if (!compilation) { continue; }
        auto source = metal_source(compilation.module().value());
        auto code = std::string_view{source.data(), source.size()};
        auto barriers = uint64_t{0u};
        for (auto offset = code.find("metal::threadgroup_barrier("); offset != std::string_view::npos;
             offset = code.find("metal::threadgroup_barrier(", offset + 1u)) { barriers++; }
        expect(eq(compilation.plans().size(), size_t{1u}));
        for (auto &plan : compilation.plans()) {
            expect(eq(plan.group_barrier_sites_before, plan.group_barrier_sites_after));
            expect(eq(barriers, plan.group_barrier_sites_after + 1u));
        }
        auto name = kernel.function().name();
        auto entry = compilation.module().value()->GetFunction(tvm::ffi::String{name.data(), name.size()}, true);
        expect(entry.has_value());
        if (!entry) { continue; }
        auto a = values(groups * columns);
        auto b = values(groups * columns);
        for (auto &value : b) { value = value * 2.0f + 1.0f; }
        auto left = runtime.upload<float>({groups * columns}, a);
        auto right = runtime.upload<float>({groups * columns}, b);
        auto output = runtime.allocate<float>({groups * columns});
        (*entry)(left, right, output);
        for (auto i = 0u; i < a.size(); i++) { a[i] += b[i]; }
        expect_near(runtime.download<float>(output, a.size()), a);
    }
}

void test_gemm(Runtime &runtime) {
    for (auto size : {1, 7, 17, 37}) {
        auto rows = size;
        auto columns = size + 2;
        auto inner = size + 4;
        auto scope = root_scope(runtime);
        auto definition = tile_kernel("cooperative_gemm", [=](TensorView<const float, 2> a,
                                                              TensorView<const float, 2> b,
                                                              TensorView<float, 2> output) {
            auto gm = axis("gm", (rows + 7) / 8);
            auto gn = axis("gn", (columns + 7) / 8);
            auto m = axis("m", 8);
            auto n = axis("n", 8);
            auto k = axis("k", 8);
            for (auto &nest : parallel(shape(gm, gn), scope)) {
                auto m0 = nest.index(gm) * 8;
                auto n0 = nest.index(gn) * 8;
                auto acc = zeros<float>(shape(m, n));
                for (auto &step : nest.pipeline(shape((inner + 7) / 8))) {
                    step.stage("load");
                    auto x = a.tile(coord(m0, step.index() * 8), shape(m, k)).load();
                    auto y = b.tile(coord(step.index() * 8, n0), shape(k, n)).load();
                    step.stage("compute");
                    acc = mma(x, y, acc);
                }
                output(coord(m0, n0), shape(m, n)).store(acc);
            }
        });
        auto executable = runtime.build(definition.capture(tensor_shape(rows, inner), tensor_shape(inner, columns), tensor_shape(rows, columns)));
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto left = values(rows * inner);
        auto right = values(inner * columns);
        auto a = runtime.upload<float>({rows, inner}, left);
        auto b = runtime.upload<float>({inner, columns}, right);
        auto output = runtime.allocate<float>({rows, columns});
        (*executable.entry)(a, b, output);
        auto actual = runtime.download<float>(output, rows * columns);
        luisa::vector<float> expected(rows * columns, 0.0f);
        for (auto i = 0; i < rows; i++) {
            for (auto j = 0; j < columns; j++) {
                for (auto k = 0; k < inner; k++) { expected[i * columns + j] += left[i * inner + k] * right[k * columns + j]; }
            }
        }
        expect_near(actual, expected);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto runtime = Runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_cooperative_shared_resources_and_global_order"_test = [&] { test_shared_tiles_and_global_order(runtime); };
    "tile_cooperative_softmax"_test = [&] { test_softmax(runtime); };
    "tile_cooperative_batched_copies_and_guarded_tails"_test = [&] { test_batched_copies(runtime); };
    "tile_cooperative_barrier_nonadjacent_dependencies"_test = [&] { test_barrier_coalescing_nonadjacent_dependencies(runtime); };
    "tile_cooperative_barrier_global_aliases_and_backedge"_test = [&] { test_barrier_coalescing_global_aliases_and_loop_backedge(runtime); };
    "tile_cooperative_explicit_barrier_identity"_test = [&] { test_explicit_barrier_identity_is_preserved(runtime); };
    "tile_cooperative_gemm"_test = [&] { test_gemm(runtime); };
    "tile_cooperative_resource_capacity"_test = [&] { test_resource_capacity(runtime); };
    "tile_cooperative_host_local_capture"_test = [&] { test_host_local_capture_rejected(runtime); };
    "tile_cooperative_nested_scope"_test = [&] { test_nested_scope_rejected(runtime); };
    "tile_cooperative_rectangular_element_domain"_test = [&] { test_rectangular_element_domain(runtime); };
    "tile_cooperative_noncanonical_domain"_test = [&] { test_noncanonical_domain_rejected(runtime); };
}

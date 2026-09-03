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
    "tile_cooperative_gemm"_test = [&] { test_gemm(runtime); };
    "tile_cooperative_resource_capacity"_test = [&] { test_resource_capacity(runtime); };
    "tile_cooperative_host_local_capture"_test = [&] { test_host_local_capture_rejected(runtime); };
    "tile_cooperative_nested_scope"_test = [&] { test_nested_scope_rejected(runtime); };
    "tile_cooperative_rectangular_element_domain"_test = [&] { test_rectangular_element_domain(runtime); };
    "tile_cooperative_noncanonical_domain"_test = [&] { test_noncanonical_domain_rejected(runtime); };
}

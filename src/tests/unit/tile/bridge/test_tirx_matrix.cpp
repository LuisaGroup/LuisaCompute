// Test semantic MMA selection through native TVMx, including real Metal.
// Covers matrix atoms, transposed operands, ragged global tiles, pipeline
// versions, ordered math, capability gates, stale tensorization markers, and
// CPU column SIMD with scalar tails and cancellation-sensitive K ordering.
#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <luisa/core/stl/format.h>

#include <algorithm>
#include <cmath>
#include <string_view>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

using namespace luisa;
using namespace luisa::compute::tile;
using namespace boost::ut;
using namespace boost::ut::literals;
using luisa::test::tile_tirx::Runtime;
using luisa::test::tile_tirx::Executable;

namespace {

struct Shape {
    int64_t m, n, k;
    int64_t bm, bn, bk;
    bool transpose_a{false};
    bool transpose_b{false};
    MmaPolicy math;
};

[[nodiscard]] tvm::ffi::String metal_source(const tvm::ffi::Module &module) {
    if (std::string_view{module->kind()} == "metal") { return module->InspectSource("metal"); }
    for (auto &&child : module->imports()) {
        auto source = metal_source(child.cast<tvm::ffi::Module>());
        if (!source.empty()) { return source; }
    }
    return {};
}

[[nodiscard]] Kernel gemm(const Runtime &runtime, Shape cfg, uint32_t window) {
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto definition = tile_kernel("matrix_gemm", [=](TensorView<const float, 2> A, TensorView<const float, 2> B,
                                                     TensorView<const float, 2> C, TensorView<float, 2> D) {
        auto gm = axis("gm", (cfg.m + cfg.bm - 1) / cfg.bm);
        auto gn = axis("gn", (cfg.n + cfg.bn - 1) / cfg.bn);
        auto m = axis("m", cfg.bm);
        auto n = axis("n", cfg.bn);
        auto k = axis("k", cfg.bk);
        for (auto &nest : parallel(shape(gm, gn), scope)) {
            auto m0 = nest.index(gm) * cfg.bm;
            auto n0 = nest.index(gn) * cfg.bn;
            auto acc = C.tile(coord(m0, n0), shape(m, n)).load();
            for (auto &step : nest.pipeline(shape((cfg.k + cfg.bk - 1) / cfg.bk), {.stages = window})) {
                step.stage("load");
                auto k0 = step.index() * cfg.bk;
                auto a = cfg.transpose_a ? A.tile(coord(k0, m0), shape(k, m)).load() : A.tile(coord(m0, k0), shape(m, k)).load();
                auto b = cfg.transpose_b ? B.tile(coord(n0, k0), shape(n, k)).load() : B.tile(coord(k0, n0), shape(k, n)).load();
                step.stage("compute");
                acc = mma(a, b, acc, cfg.math);
            }
            D(coord(m0, n0), shape(m, n)).store(acc);
        }
    });
    return definition.capture(tensor_shape(cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k),
                              tensor_shape(cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n),
                              tensor_shape(cfg.m, cfg.n), tensor_shape(cfg.m, cfg.n));
}

[[nodiscard]] vector<float> values(size_t count, float phase) {
    vector<float> result(count);
    for (auto i = 0u; i < count; i++) { result[i] = std::sin(static_cast<float>(i) * 0.371f + phase) * 1.375f; }
    return result;
}

void check_gemm(Runtime &runtime, const Executable &executable, Shape cfg, double product_sign = 1.0) {
    auto a = values(cfg.m * cfg.k, 0.13f);
    auto b = values(cfg.k * cfg.n, 0.47f);
    auto c = values(cfg.m * cfg.n, 0.93f);
    auto destination = runtime.allocate<float>({cfg.m, cfg.n});
    for (auto repeat = 0u; repeat < 2u; repeat++) {
        for (auto &value : a) { value += 0.03125f; }
        for (auto &value : b) { value -= 0.046875f; }
        for (auto &value : c) { value += 0.015625f; }
        auto left = runtime.upload<float>({cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k}, a);
        auto right = runtime.upload<float>({cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n}, b);
        auto initial = runtime.upload<float>({cfg.m, cfg.n}, c);
        (*executable.entry)(left, right, initial, destination);
        auto actual = runtime.download<float>(destination, cfg.m * cfg.n);
        auto valid = true;
        auto maximum_error = 0.0;
        for (auto row = int64_t{0}; row < cfg.m; row++) {
            for (auto column = int64_t{0}; column < cfg.n; column++) {
                auto expected = static_cast<double>(c[row * cfg.n + column]);
                for (auto k = int64_t{0}; k < cfg.k; k++) {
                    auto ai = cfg.transpose_a ? k * cfg.m + row : row * cfg.k + k;
                    auto bi = cfg.transpose_b ? column * cfg.k + k : k * cfg.n + column;
                    expected += product_sign * static_cast<double>(a[ai]) * b[bi];
                }
                auto result = actual[row * cfg.n + column];
                auto error = std::abs(static_cast<double>(result) - expected);
                valid &= std::isfinite(result) && error <= 1e-4 + 2e-5 * std::abs(expected);
                maximum_error = std::max(maximum_error, error);
            }
        }
        expect(valid) << cfg.m << "x" << cfg.n << "x" << cfg.k << " max error=" << maximum_error;
    }
}

void test_matrix_cases(Runtime &runtime) {
    Shape cases[]{
        {8, 8, 8, 8, 8, 8}, {16, 24, 32, 16, 24, 16}, {37, 29, 21, 16, 16, 8}, {19, 35, 25, 8, 24, 16}, {17, 23, 31, 16, 24, 16, true, false}, {17, 23, 31, 16, 24, 16, false, true}, {17, 23, 31, 16, 24, 16, true, true}, {7, 9, 5, 3, 5, 7}, {37, 71, 45, 32, 64, 16}, {37, 71, 45, 32, 64, 16, true, false}, {37, 71, 45, 32, 64, 16, false, true}};
    for (auto cfg : cases) {
        for (auto window : {1u, 2u}) {
            auto kernel = gemm(runtime, cfg, window);
            expect(kernel.valid());
            expect(eq(luisa::test::tile_tirx::count_operations(kernel.function().body(), OperationKind::MMA), 1u));
            for (auto enabled : {false, true}) {
                auto executable = runtime.build(kernel, false, enabled);
                expect(executable.ok()) << executable.error;
                if (!executable.ok()) { continue; }
                if (runtime.target() == "metal") {
                    auto source = metal_source(executable.module.value());
                    auto code = std::string_view{source.data(), source.size()};
                    auto has_matrix = code.find("simdgroup_multiply_accumulate") != std::string_view::npos;
                    auto expected = enabled && cfg.bm % 8 == 0 && cfg.bn % 8 == 0 && cfg.bk % 8 == 0;
                    expect(eq(has_matrix, expected)) << cfg.bm << "x" << cfg.bn << "x" << cfg.bk << " window=" << window << "\n"
                                                     << code;
                    if (expected) {
                        expect(code.find("simdgroup_float8x8") != std::string_view::npos);
                        expect(code.find("simdgroup_load") != std::string_view::npos);
                        expect(code.find("simdgroup_store") != std::string_view::npos);
                    }
                }
                check_gemm(runtime, executable, cfg);
            }
        }
    }
}

[[nodiscard]] Executable compile_native(Runtime &runtime, const Kernel &kernel, tvm::tirx::PrimFunc native,
                                        uint32_t width = 32u, uint32_t threads = 256u,
                                        const bridge::tirx::PlannerOptions &planner = {}) {
    using namespace luisa::compute::tile::bridge::tirx;
    CompileOptions options;
    options.target = runtime.target() == "metal" ?
                         luisa::format(R"({{"kind":"metal","thread_warp_size":{},"max_num_threads":{}}})", width, threads) :
                         luisa::string{runtime.target()};
    options.cooperative_matrix = true;
    options.planner = planner;
    auto compilation = compile(std::move(native), kernel.function().name(), options);
    Executable executable;
    if (!compilation) {
        executable.error = compilation.error();
    } else {
        executable.module = compilation.module();
        executable.plans.assign(compilation.plans().begin(), compilation.plans().end());
        auto name = kernel.function().name();
        executable.entry = executable.module.value()->GetFunction(tvm::ffi::String{name.data(), name.size()}, true);
    }
    return executable;
}

void test_explicit_threads_use_target_capacity(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    Shape cfg{17, 23, 29, 16, 24, 16};
    auto kernel = gemm(runtime, cfg, 1u);
    bridge::tirx::PlannerOptions planner;
    planner.threads_per_group = 512u;
    for (auto capacity : {256u, 512u}) {
        auto native = bridge::tirx::lower(kernel.function());
        expect(native.ok()) << native.error;
        if (!native) { continue; }
        // Cross-compile only: a target capability assertion is not inferred
        // from whichever physical GPU happens to run this unit test.
        auto executable = compile_native(runtime, kernel, std::move(native.value), 32u, capacity, planner);
        expect(eq(executable.ok(), capacity == 512u)) << executable.error;
        if (executable.ok()) {
            expect(eq(executable.plans.size(), size_t{1u}));
            if (!executable.plans.empty()) { expect(eq(executable.plans[0].threads, 512u)); }
        } else {
            expect(executable.error.find("capacity") != luisa::string::npos) << executable.error;
        }
    }
}

void test_matrix_policy_and_participants(Runtime &runtime) {
    Shape cfg{17, 23, 29, 16, 24, 16};
    for (auto ordered : {false, true}) {
        cfg.math.allow_reassociation = !ordered;
        auto kernel = gemm(runtime, cfg, 2u);
        for (auto [width, threads] : {std::pair{16u, 256u}, std::pair{32u, 31u}, std::pair{32u, 32u}, std::pair{32u, 48u}, std::pair{32u, 96u}}) {
            auto native = bridge::tirx::lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            auto executable = compile_native(runtime, kernel, std::move(native.value), width, threads);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            if (runtime.target() == "metal") {
                auto source = metal_source(executable.module.value());
                auto code = std::string_view{source.data(), source.size()};
                auto expected = !ordered && width == 32u && threads >= 32u;
                expect(eq(code.find("simdgroup_multiply_accumulate") != std::string_view::npos, expected)) << code;
            }
            check_gemm(runtime, executable, cfg);
        }
    }
}

class SubtractProducts final : public tvm::tirx::StmtMutator {
private:
    bool _in_mma{false};

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto previous = _in_mma;
        _in_mma |= loop->annotations.count("luisa.tile.mma") != 0u;
        auto result = StmtMutator::VisitStmt_(loop);
        _in_mma = previous;
        return result;
    }
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        auto result = StmtMutator::VisitStmt_(store).as_or_throw<tvm::tirx::BufferStore>();
        if (_in_mma) {
            if (auto add = result->value.as<tvm::tirx::AddNode>(); add && add->b.as<tvm::tirx::MulNode>()) {
                result.CopyOnWrite()->value = add->a - add->b;
                replacements++;
            }
        }
        return result;
    }

public:
    uint32_t replacements{0u};
    using StmtMutator::operator();
};

void test_stale_matrix_marker(Runtime &runtime) {
    Shape cfg{17, 23, 29, 16, 24, 16};
    auto kernel = gemm(runtime, cfg, 2u);
    auto native = bridge::tirx::lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    SubtractProducts mutate;
    native.value.CopyOnWrite()->body = mutate(native.value->body);
    expect(eq(mutate.replacements, 1u));
    auto executable = compile_native(runtime, kernel, std::move(native.value));
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto source = metal_source(executable.module.value());
    expect(std::string_view{source.data(), source.size()}.find("simdgroup_multiply_accumulate") == std::string_view::npos);
    check_gemm(runtime, executable, cfg, -1.0);
}

void test_literal_initial_and_zero_contraction(Runtime &runtime) {
    for (auto contracted : {0, 8, 24}) {
        auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
        auto kernel = tile_kernel("matrix_literal_initial", [=](TensorView<const float, 2> A, TensorView<const float, 2> B, TensorView<float, 2> D) {
                          auto m = axis("m", 8);
                          auto n = axis("n", 16);
                          auto k = axis("k", contracted);
                          for (auto &nest : parallel(shape(1), scope)) {
                              auto a = A.tile(coord(nest.index(), 0), shape(m, k)).load();
                              auto b = B.tile(coord(0, 0), shape(k, n)).load();
                              D(coord(0, 0), shape(m, n)).store(mma(a, b, full<float>(shape(m, n), 0.375f)));
                          }
                      }).capture(tensor_shape(8, 24), tensor_shape(24, 16), tensor_shape(8, 16));
        auto executable = runtime.build(kernel, false, true);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        if (runtime.target() == "metal") {
            auto source = metal_source(executable.module.value());
            expect(eq(std::string_view{source.data(), source.size()}.find("simdgroup_multiply_accumulate") != std::string_view::npos, contracted != 0)) << source;
        }
        auto a = runtime.upload<float>({8, 24}, vector<float>(8 * 24, 0.5f));
        auto b = runtime.upload<float>({24, 16}, vector<float>(24 * 16, -0.25f));
        auto d = runtime.allocate<float>({8, 16});
        (*executable.entry)(a, b, d);
        auto output = runtime.download<float>(d, 8 * 16);
        auto expected = 0.375f - static_cast<float>(contracted) * 0.125f;
        expect(std::all_of(output.begin(), output.end(), [expected](float value) { return std::isfinite(value) && std::abs(value - expected) < 1e-5f; }));
    }
}

void test_worker_local_matrix_fallback(Runtime &runtime) {
    auto group_scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto child_scope = runtime.target() == "metal" ? exec::Scope::WORKER : exec::Scope::AUTOMATIC;
    auto kernel = tile_kernel("worker_local_matrix", [=](TensorView<const float, 2> A, TensorView<const float, 2> B,
                                                         TensorView<const float, 2> C, TensorView<float, 2> D) {
                      auto m = axis("m", 8);
                      auto n = axis("n", 8);
                      auto k = axis("k", 16);
                      for (auto &nest : parallel(shape(1), group_scope)) {
                          for (auto &worker : nest.parallel(shape(3), child_scope)) {
                              auto origin = coord(worker.index() * 8, 0);
                              auto a = A(origin, shape(m, k)).load();
                              auto b = B(coord(0, 0), shape(k, n)).load();
                              auto c = C(origin, shape(m, n)).load();
                              D(origin, shape(m, n)).store(mma(a, b, c));
                          }
                      }
                  }).capture(tensor_shape(24, 16), tensor_shape(16, 8), tensor_shape(24, 8), tensor_shape(24, 8));
    auto executable = runtime.build(kernel, false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto source = metal_source(executable.module.value());
    expect(std::string_view{source.data(), source.size()}.find("simdgroup_multiply_accumulate") == std::string_view::npos);
    check_gemm(runtime, executable, {24, 8, 16, 8, 8, 16});
}

void test_mixed_input_matrix_fallback(Runtime &runtime) {
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto kernel = tile_kernel("mixed_input_matrix", [=](TensorView<const int32_t, 2> A, TensorView<const float, 2> B, TensorView<float, 2> D) {
                      for (auto &nest : parallel(shape(1), scope)) {
                          auto a = A(coord(nest.index(), 0), shape(8, 16)).load();
                          auto b = B(coord(0, 0), shape(16, 8)).load();
                          D(coord(0, 0), shape(8, 8)).store(mma(a, b, full<float>(shape(8, 8), 0.25f)));
                      }
                  }).capture(tensor_shape(8, 16), tensor_shape(16, 8), tensor_shape(8, 8));
    auto executable = runtime.build(kernel, false, true);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto source = metal_source(executable.module.value());
    expect(std::string_view{source.data(), source.size()}.find("simdgroup_multiply_accumulate") == std::string_view::npos);
    auto a = runtime.upload<int32_t>({8, 16}, vector<int32_t>(8 * 16, 131073));
    auto b = runtime.upload<float>({16, 8}, vector<float>(16 * 8, 0.5f));
    auto d = runtime.allocate<float>({8, 8});
    (*executable.entry)(a, b, d);
    auto output = runtime.download<float>(d, 8 * 8);
    expect(std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value) && std::abs(value - 1048584.25f) < 1e-5f; }));
}

void test_cpu_matrix_vectors_and_tails(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    Shape cases[]{
        {9, 13, 29, 3, 5, 7}, {17, 23, 31, 4, 7, 16}, {16, 32, 32, 4, 16, 16}, {17, 23, 31, 4, 7, 16, true, false}, {17, 23, 31, 4, 7, 16, false, true}};
    for (auto cfg : cases) {
        cfg.math.allow_reassociation = false;
        for (auto window : {1u, 2u}) {
            auto kernel = gemm(runtime, cfg, window);
            for (auto vectorize : {false, true}) {
                auto executable = runtime.build(kernel, true, false, vectorize, vectorize);
                expect(executable.ok()) << executable.error;
                if (!executable.ok()) { continue; }
                check_gemm(runtime, executable, cfg);
                if (vectorize && !cfg.transpose_b && cfg.bn == 16) {
                    auto source = executable.module.value()->InspectSource("ll");
                    auto code = std::string_view{source.data(), source.size()};
                    auto vector_product = false;
                    for (auto lanes : {4, 8, 16}) {
                        vector_product |= code.find(luisa::format("llvm.fmuladd.v{}f32", lanes)) != std::string_view::npos ||
                                          code.find(luisa::format("llvm.fma.v{}f32", lanes)) != std::string_view::npos;
                        for (auto start = code.find("fmul "); start != std::string_view::npos; start = code.find("fmul ", start + 5u)) {
                            auto end = code.find('\n', start);
                            vector_product |= code.substr(start, end - start).find(luisa::format("<{} x float>", lanes)) != std::string_view::npos;
                        }
                    }
                    expect(vector_product) << "ordered CPU MMA must contain vector products\n"
                                           << code;
                }
            }
        }
    }
}

void test_cpu_matrix_preserves_k_order(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    auto kernel = tile_kernel("matrix_ordered_cancellation", [](TensorView<const float, 2> A,
                                                                TensorView<const float, 2> B, TensorView<float, 2> D) {
                      auto m = axis("m", 3);
                      auto n = axis("n", 5);
                      auto k = axis("k", 4);
                      for (auto &nest : parallel(shape(1), exec::Scope::WORKER)) {
                          auto a = A[coord(nest.index(), 0), shape(m, k)];
                          auto b = B[coord(0, 0), shape(k, n)];
                          D(coord(0, 0), shape(m, n)).store(mma(a, b, zeros<float>(shape(m, n)), {.allow_reassociation = false}));
                      }
                  }).capture(tensor_shape(3, 4), tensor_shape(4, 5), tensor_shape(3, 5));
    expect(!bridge::tirx::CompileOptions{}.auto_vectorize);
    auto invalid = runtime.build(kernel, true, false, false, true);
    expect(!invalid.ok());
    expect(invalid.error.find("requires vectorization") != luisa::string::npos);
    vector<float> inputs;
    for (auto i = 0u; i < 3u; i++) { inputs.insert(inputs.end(), {16777216.0f, 1.0f, -16777216.0f, 0.5f}); }
    auto a = runtime.upload<float>({3, 4}, inputs);
    auto b = runtime.upload<float>({4, 5}, vector<float>(20, 1.0f));
    auto d = runtime.allocate<float>({3, 5});
    for (auto vectorize : {false, true}) {
        auto executable = runtime.build(kernel, true, false, vectorize, vectorize);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        (*executable.entry)(a, b, d);
        auto output = runtime.download<float>(d, 15);
        // Sequential FP32 is 0.5; regrouping cancellation can produce 1.5.
        expect(std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value) && std::abs(value - 0.5f) < 1e-7f; }));
    }
}

void test_planned_fragment_reuse(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    Shape cfg{37, 71, 45, 32, 64, 16};
    auto kernel = gemm(runtime, cfg, 1u);
    for (auto enabled : {false, true}) {
        bridge::tirx::PlannerOptions planner;
        planner.enabled = enabled;
        auto executable = runtime.build(kernel, false, true, true, false, planner);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        expect(eq(executable.plans.size(), size_t{1u}));
        if (executable.plans.size() != 1u) { continue; }
        auto &plan = executable.plans[0];
        expect(eq(plan.optimized, enabled));
        auto source = metal_source(executable.module.value());
        auto code = std::string_view{source.data(), source.size()};
        if (enabled) {
            expect(eq(plan.threads, 128u));
            expect(eq(plan.cost.fragment_scalars_per_lane, 28ull));
            expect(code.find("_mma_c[8]") != std::string_view::npos) << code;
            expect(code.find("_mma_wave") == std::string_view::npos) << code;
            expect(plan.matrices[0].persistent_accumulator);
            expect(code.find("_mma_c[8]") < code.find("for (int pipeline_")) << code;
            expect(code.rfind("simdgroup_store(") > code.find("for (int pipeline_")) << code;
        } else {
            expect(eq(plan.threads, 256u));
            expect(code.find("_mma_c[1]") != std::string_view::npos) << code;
            expect(code.find("_mma_wave") != std::string_view::npos) << code;
        }
        check_gemm(runtime, executable, cfg);
    }
}

void test_observed_accumulator_stays_visible(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    for (auto iterations : {0, 1, 5}) {
        for (auto window : {1u, 2u}) {
            auto kernel = tile_kernel("observed_accumulator", [=](TensorView<const float, 2> A, TensorView<const float, 2> B,
                                                                  TensorView<float, 2> D, TensorView<float, 2> H) {
                              auto m = axis("m", 8);
                              auto n = axis("n", 16);
                              auto k = axis("k", 8);
                              for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
                                  auto acc = full<float>(shape(m, n), 0.5f);
                                  auto history = zeros<float>(shape(m, n));
                                  for (auto &step : nest.pipeline(shape(iterations), {.stages = window})) {
                                      step.stage("load");
                                      auto a = A[coord(0, step.index() * 8), shape(m, k)];
                                      auto b = B[coord(step.index() * 8, 0), shape(k, n)];
                                      step.stage("compute");
                                      history = history + acc;
                                      acc = mma(a, b, acc);
                                  }
                                  D(coord(0, 0), shape(m, n)).store(acc);
                                  H(coord(0, 0), shape(m, n)).store(history);
                              }
                          }).capture(tensor_shape(8, 40), tensor_shape(40, 16), tensor_shape(8, 16), tensor_shape(8, 16));
            auto executable = runtime.build(kernel, true, true);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            for (auto &plan : executable.plans) {
                for (auto &matrix : plan.matrices) { expect(!matrix.persistent_accumulator); }
            }
            auto a = runtime.upload<float>({8, 40}, vector<float>(8u * 40u, 0.25f));
            auto b = runtime.upload<float>({40, 16}, vector<float>(40u * 16u, 0.5f));
            auto d = runtime.allocate<float>({8, 16});
            auto h = runtime.allocate<float>({8, 16});
            (*executable.entry)(a, b, d, h);
            auto actual = runtime.download<float>(d, 128u);
            auto history = runtime.download<float>(h, 128u);
            auto expected = 0.5f + static_cast<float>(iterations);
            auto expected_history = 0.5f * static_cast<float>(iterations * iterations);
            expect(std::all_of(actual.begin(), actual.end(), [=](float value) { return std::isfinite(value) && std::abs(value - expected) < 1e-6f; }));
            expect(std::all_of(history.begin(), history.end(), [=](float value) { return std::isfinite(value) && std::abs(value - expected_history) < 1e-6f; }));
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    Runtime runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_matrix_shapes_transposes_and_pipeline_versions"_test = [&] { test_matrix_cases(runtime); };
    "tile_matrix_math_policy_and_participant_gates"_test = [&] { test_matrix_policy_and_participants(runtime); };
    "tile_matrix_exact_threads_use_actual_target_limit"_test = [&] { test_explicit_threads_use_target_capacity(runtime); };
    "tile_matrix_stale_marker_does_not_tensorize"_test = [&] { test_stale_matrix_marker(runtime); };
    "tile_matrix_literal_initial_and_zero_contraction"_test = [&] { test_literal_initial_and_zero_contraction(runtime); };
    "tile_matrix_worker_local_is_not_a_collective"_test = [&] { test_worker_local_matrix_fallback(runtime); };
    "tile_matrix_mixed_conversion_keeps_reference_types"_test = [&] { test_mixed_input_matrix_fallback(runtime); };
    "tile_cpu_matrix_vectors_and_scalar_tails"_test = [&] { test_cpu_matrix_vectors_and_tails(runtime); };
    "tile_cpu_matrix_preserves_k_order"_test = [&] { test_cpu_matrix_preserves_k_order(runtime); };
    "tile_matrix_planner_emits_reused_fragments"_test = [&] { test_planned_fragment_reuse(runtime); };
    "tile_matrix_observed_carry_cannot_be_promoted"_test = [&] { test_observed_accumulator_stays_visible(runtime); };
}

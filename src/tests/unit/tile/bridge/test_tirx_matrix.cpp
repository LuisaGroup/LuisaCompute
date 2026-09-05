// Test semantic MMA selection through native TVMx, including real Metal.
// Covers matrix atoms, transposed operands, ragged global tiles, pipeline
// versions, ordered math, capability gates, stale tensorization markers, and
// CPU column SIMD with scalar tails and cancellation-sensitive K ordering.
#include "ut/ut.hpp"
#include "tile_tirx_test_utils.h"

#include <luisa/core/mathematics.h>
#include <luisa/core/stl/format.h>
#include <luisa/tile/memory.h>

#include <algorithm>
#include <cmath>
#include <string_view>
#include <tvm/tirx/builtin.h>
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

[[nodiscard]] Kernel gemm(const Runtime &runtime, Shape cfg, uint32_t window, uint32_t interval = 1u,
                          bool literal_initial = false, float literal_value = 0.5f) {
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto definition = tile_kernel("matrix_gemm", [=](TensorView<const float, 2> A, TensorView<const float, 2> B,
                                                     TensorView<const float, 2> C, TensorView<float, 2> D) {
        auto gm = axis("gm", ceil_div(cfg.m, cfg.bm));
        auto gn = axis("gn", ceil_div(cfg.n, cfg.bn));
        auto m = axis("m", cfg.bm);
        auto n = axis("n", cfg.bn);
        auto k = axis("k", cfg.bk);
        for (auto &nest : parallel(shape(gm, gn), scope)) {
            auto m0 = nest.index(gm) * cfg.bm;
            auto n0 = nest.index(gn) * cfg.bn;
            auto acc = literal_initial ? full<float>(shape(m, n), literal_value) : C.tile(coord(m0, n0), shape(m, n)).load();
            for (auto &step : nest.pipeline(shape(ceil_div(cfg.k, cfg.bk)), {.stages = window, .initiation_interval = interval})) {
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

void check_gemm(Runtime &runtime, const Executable &executable, Shape cfg, double product_sign = 1.0,
                bool literal_initial = false, bool column_major_c = false, bool column_major_d = false,
                float literal_value = 0.5f) {
    auto a = values(cfg.m * cfg.k, 0.13f);
    auto b = values(cfg.k * cfg.n, 0.47f);
    auto c = values(cfg.m * cfg.n, 0.93f);
    auto destination = runtime.allocate<float>({column_major_d ? cfg.n : cfg.m, column_major_d ? cfg.m : cfg.n});
    for (auto repeat = 0u; repeat < 2u; repeat++) {
        for (auto &value : a) { value += 0.03125f; }
        for (auto &value : b) { value -= 0.046875f; }
        for (auto &value : c) { value += 0.015625f; }
        auto left = runtime.upload<float>({cfg.transpose_a ? cfg.k : cfg.m, cfg.transpose_a ? cfg.m : cfg.k}, a);
        auto right = runtime.upload<float>({cfg.transpose_b ? cfg.n : cfg.k, cfg.transpose_b ? cfg.k : cfg.n}, b);
        auto initial = runtime.upload<float>({column_major_c ? cfg.n : cfg.m, column_major_c ? cfg.m : cfg.n}, c);
        try {
            (*executable.entry)(left, right, initial, destination);
        } catch (const tvm::ffi::Error &error) {
            expect(false) << error.what();
            return;
        }
        auto actual = runtime.download<float>(destination, cfg.m * cfg.n);
        auto valid = true;
        auto maximum_error = 0.0;
        for (auto row = int64_t{0}; row < cfg.m; row++) {
            for (auto column = int64_t{0}; column < cfg.n; column++) {
                auto expected = literal_initial ? static_cast<double>(literal_value) :
                                                  static_cast<double>(c[column_major_c ? column * cfg.m + row : row * cfg.n + column]);
                for (auto k = int64_t{0}; k < cfg.k; k++) {
                    auto ai = cfg.transpose_a ? k * cfg.m + row : row * cfg.k + k;
                    auto bi = cfg.transpose_b ? column * cfg.k + k : k * cfg.n + column;
                    expected += product_sign * static_cast<double>(a[ai]) * b[bi];
                }
                auto result = actual[column_major_d ? column * cfg.m + row : row * cfg.n + column];
                auto error = std::abs(static_cast<double>(result) - expected);
                valid &= std::isfinite(result) && error <= 1e-4 + 2e-5 * std::abs(expected);
                maximum_error = std::max(maximum_error, error);
            }
        }
        expect(valid) << cfg.m << "x" << cfg.n << "x" << cfg.k
                      << " transpose A/B=" << cfg.transpose_a << "/" << cfg.transpose_b
                      << " column-major C/D=" << column_major_c << "/" << column_major_d
                      << " literal=" << literal_initial << " max error=" << maximum_error;
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
                                        const bridge::tirx::PlannerOptions &planner = {}, bool auto_vectorize = false) {
    using namespace luisa::compute::tile::bridge::tirx;
    CompileOptions options;
    options.target = runtime.target() == "metal" ?
                         luisa::format(R"({{"kind":"metal","thread_warp_size":{},"max_num_threads":{}}})", width, threads) :
                         luisa::string{runtime.target()};
    options.cooperative_matrix = true;
    options.auto_vectorize = auto_vectorize;
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

void test_mpp_memory_realization(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    auto capability = tvm::ffi::Function::GetGlobal("target.metal.mpp_memory_contract_version");
    if (!capability) {
        auto kernel = gemm(runtime, {32, 32, 32, 32, 32, 32}, 1u);
        auto unsupported = runtime.build(kernel, false, true, true, false, {}, true);
        expect(!unsupported.ok());
        expect(unsupported.error.find("MPP memory contract") != luisa::string::npos) << unsupported.error;
        return;
    }
    expect(eq((*capability)().cast<int64_t>(), int64_t{2}));
    auto small = gemm(runtime, {8, 8, 8, 8, 8, 8}, 1u);
    auto unsupported = runtime.build(small, false, true, true, false, {}, true);
    expect(!unsupported.ok());
    expect(unsupported.error.find("multiple of 16") != luisa::string::npos) << unsupported.error;
    Shape cases[]{
        {32, 64, 8, 32, 64, 8}, {32, 64, 64, 32, 64, 32}, {64, 64, 129, 64, 64, 32}, {37, 71, 45, 32, 64, 16}, {37, 71, 45, 32, 64, 16, true, false}, {37, 71, 45, 32, 64, 16, false, true}, {37, 71, 45, 32, 64, 16, true, true}};
    for (auto cfg : cases) {
        for (auto window : {1u, 2u}) {
            for (auto literal : {false, true}) {
                auto kernel = gemm(runtime, cfg, window, 1u, literal);
                auto executable = runtime.build(kernel, false, true, true, false, {}, true);
                if (cfg.bm == 64 && !literal) {
                    // Loading a separate full C tile in this fixture exceeds
                    // the existing planner's shared-memory budget. MPP must
                    // preserve that rejection, not bypass resource validation.
                    expect(!executable.ok());
                    expect(executable.error.find("shared-memory capacity") != luisa::string::npos) << executable.error;
                    continue;
                }
                expect(executable.ok()) << executable.error << " block=" << cfg.bm << "x" << cfg.bn << "x" << cfg.bk
                                        << " window=" << window << " literal=" << literal;
                if (!executable.ok()) { continue; }
                auto source = metal_source(executable.module.value());
                auto code = std::string_view{source.data(), source.size()};
                expect(code.find("mpp::tensor_ops::matmul2d<") != std::string_view::npos) << code;
                expect(code.find("::mode::multiply_accumulate") != std::string_view::npos);
                expect(code.find("simdgroup_multiply_accumulate(") == std::string_view::npos);
                expect(code.find("{}.run(") != std::string_view::npos);
                expect(!executable.plans.empty());
                for (auto &plan : executable.plans) { expect(plan.metal_mpp); }
                check_gemm(runtime, executable, cfg, 1.0, literal);
            }
        }
    }
    // A proved one-shot +0 recurrence maps to MPP's overwriting multiply.
    // A second K iteration must retain multiply-accumulate and its explicit C.
    for (auto [cfg, pure] : {std::pair{Shape{32, 64, 32, 32, 64, 32}, true},
                             std::pair{Shape{32, 64, 64, 32, 64, 32}, false}}) {
        auto kernel = gemm(runtime, cfg, 1u, 1u, true, 0.0f);
        auto executable = runtime.build(kernel, false, true, true, false, {}, true);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto source = metal_source(executable.module.value());
        auto code = std::string_view{source.data(), source.size()};
        expect(eq(code.find("::mode::multiply);") != std::string_view::npos, pure)) << code;
        expect(eq(code.find("::mode::multiply_accumulate") != std::string_view::npos, !pure)) << code;
        expect(eq(code.find("get_capacity()") != std::string_view::npos, !pure)) << code;
        check_gemm(runtime, executable, cfg, 1.0, true, false, false, 0.0f);
    }
}

class EscapeSnapshotTest final : public tvm::tirx::StmtMutator {
private:
    const tvm::tirx::ForNode *_copy{nullptr};
    tvm::tirx::BufferVar _snapshot;
    bool _observable;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto body = StmtMutator::VisitStmt_(loop);
        if (loop == _copy) {
            auto address = tvm::Call{_snapshot.DataPointerType(), tvm::tirx::builtin::address_of(), {tvm::tirx::BufferLoad{_snapshot, {tvm::IntImm::Int64(0), tvm::IntImm::Int64(0)}}}};
            if (_observable) {
                address = tvm::Call{tvm::PrimType::Int(32), tvm::tirx::builtin::call_extern(), {tvm::tirx::StringImm{"memcmp"}, address, address, tvm::IntImm{tvm::PrimType::UInt(64), 4}}};
            }
            return tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{body, tvm::tirx::Evaluate{address}});
        }
        return body;
    }

public:
    explicit EscapeSnapshotTest(const tvm::tirx::PrimFunc &function, bool observable = false) : _observable{observable} {
        tvm::tirx::PostOrderVisit(function->body, [&](const tvm::ffi::ObjectRef &node) {
            auto loop = node.as<tvm::tirx::ForNode>();
            if (loop == nullptr || !loop->annotations.count("luisa.tile.independent_elements")) { return; }
            auto inner = loop->body.as<tvm::tirx::ForNode>();
            auto store = inner == nullptr ? nullptr : inner->body.as<tvm::tirx::BufferStoreNode>();
            if (store == nullptr || store->buffer.scope() != "local") { return; }
            tvm::tirx::PostOrderVisit(store->value, [&](const tvm::ffi::ObjectRef &value) {
                if (auto load = value.as<tvm::tirx::BufferLoadNode>(); load != nullptr && load->buffer.same_as(function->params[0])) {
                    _copy = loop;
                    _snapshot = store->buffer;
                }
            });
        });
        expect(_copy != nullptr);
    }
};

class IndirectSnapshotTest final : public tvm::tirx::StmtExprMutator {
private:
    tvm::tirx::BufferVar _input;
    tvm::tirx::BufferVar _offset;

protected:
    [[nodiscard]] tvm::Expr VisitExpr_(const tvm::tirx::BufferLoadNode *load) final {
        if (load->buffer.same_as(_input)) {
            auto offset = tvm::cast(tvm::PrimType::Int(64), tvm::tirx::BufferLoad{_offset, {tvm::IntImm::Int64(0), tvm::IntImm::Int64(0)}});
            auto indices = load->indices;
            indices.Set(0, indices[0] + tvm::floormod(offset, tvm::IntImm::Int64(2)) * tvm::IntImm::Int64(32));
            replacements++;
            return tvm::tirx::BufferLoad{load->buffer, std::move(indices), load->predicate};
        }
        return StmtExprMutator::VisitExpr_(load);
    }

public:
    uint32_t replacements{0u};
    explicit IndirectSnapshotTest(const tvm::tirx::PrimFunc &function)
        : _input{function->params[0]}, _offset{function->params[2]} {}
};

void test_mpp_readonly_views(Runtime &runtime) {
    if (runtime.target() != "metal" || !tvm::ffi::Function::GetGlobal("target.metal.mpp_memory_contract_version")) { return; }
    bridge::tirx::PlannerOptions planner;
    planner.threads_per_group = 128u;
    // Full K would require 640 KiB of A/B staging for the large case. Proved
    // immutable inputs remove that resource requirement before planning.
    for (auto [base, window] : {std::pair{Shape{32, 64, 32, 32, 64, 16}, 1u},
                                std::pair{Shape{32, 64, 32, 32, 64, 16}, 2u},
                                std::pair{Shape{128, 64, 1024, 128, 32, 1024}, 1u}}) {
        for (auto transpose_a : {false, true}) {
            for (auto transpose_b : {false, true}) {
                for (auto literal : {false, true}) {
                    auto cfg = base;
                    cfg.transpose_a = transpose_a;
                    cfg.transpose_b = transpose_b;
                    auto kernel = gemm(runtime, cfg, window, 1u, literal);
                    auto reference = runtime.build(kernel, true, true, true, false, planner, true);
                    auto executable = runtime.build(kernel, true, true, true, false, planner, true, true);
                    expect(executable.ok()) << executable.error;
                    if (!executable.ok()) { continue; }
                    expect(!executable.plans.empty());
                    for (auto &plan : executable.plans) {
                        if (reference.ok()) {
                            expect(plan.shared_memory_bytes + static_cast<uint64_t>((cfg.bm + cfg.bn) * cfg.bk * sizeof(float)) <= reference.plans[0].shared_memory_bytes);
                        } else {
                            expect(reference.error.find("shared-memory capacity") != luisa::string::npos) << reference.error;
                            expect(plan.shared_memory_bytes <= 32768u);
                        }
                        expect(!plan.matrices.empty());
                    }
                    auto source = metal_source(executable.module.value());
                    expect(std::string_view{source.data(), source.size()}.find("{}.run(") != std::string_view::npos);
                    check_gemm(runtime, executable, cfg, 1.0, literal);
                }
            }
        }
    }
    // Absence of the noalias contract, or unproved padded bounds, must keep
    // the snapshot path. These are semantic gates, not performance choices.
    for (auto noalias : {false, true}) {
        auto cfg = noalias ? Shape{37, 71, 45, 32, 64, 16} : Shape{32, 64, 32, 32, 64, 16};
        auto kernel = gemm(runtime, cfg, 1u, 1u, true);
        auto reference = runtime.build(kernel, noalias, true, true, false, planner, true);
        auto forwarded = runtime.build(kernel, noalias, true, true, false, planner, true, true);
        expect(reference.ok()) << reference.error;
        expect(forwarded.ok()) << forwarded.error;
        if (reference.ok() && forwarded.ok()) {
            expect(eq(reference.plans[0].shared_memory_bytes, forwarded.plans[0].shared_memory_bytes));
            expect(metal_source(reference.module.value()) == metal_source(forwarded.module.value()));
            check_gemm(runtime, forwarded, cfg, 1.0, true);
        }
    }
    for (auto manual : {0u, 1u, 2u}) {
        auto kernel = tile_kernel("mpp_snapshot_before_input_write", [=](TensorView<float, 2> A, TensorView<const float, 2> B, TensorView<float, 2> D) {
                          for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
                              auto space = shape(16, 16);
                              auto a = A[coord(0, 0), space];
                              if (manual) {
                                  auto storage = manual == 1u ? memory<float>(space) : memory<float>(space, mem::shared);
                                  storage.store(a);
                                  a = storage.load();
                              }
                              A(coord(0, 0), space).store(full<float>(space, 0.0f));
                              auto b = B[coord(0, 0), space];
                              D(coord(0, 0), space).store(mma(a, b, full<float>(space, 0.5f)));
                          }
                      }).capture(tensor_shape(16, 16), tensor_shape(16, 16), tensor_shape(16, 16));
        auto native = bridge::tirx::lower(kernel.function());
        expect(static_cast<bool>(native));
        if (native) {
            auto materializations = 0u;
            tvm::tirx::PostOrderVisit(native.value->body, [&](const tvm::ffi::ObjectRef &node) {
                if (auto allocation = node.as<tvm::tirx::AllocBufferNode>()) {
                    materializations += allocation->annotations.count("luisa.tile.manual_memory");
                }
            });
            expect(eq(materializations, manual != 0u ? 1u : 0u));
        }
        planner.threads_per_group = 32u;
        auto executable = runtime.build(kernel, true, true, true, false, planner, true, true);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto a = runtime.upload<float>({16, 16}, vector<float>(256u, 0.25f));
        auto b = runtime.upload<float>({16, 16}, vector<float>(256u, 0.5f));
        auto d = runtime.allocate<float>({16, 16});
        (*executable.entry)(a, b, d);
        auto actual = runtime.download<float>(d, 256u);
        auto overwritten = runtime.download<float>(a, 256u);
        expect(std::all_of(actual.begin(), actual.end(), [](auto value) { return value == 2.5f; })) << "snapshot lost, manual=" << manual;
        expect(std::all_of(overwritten.begin(), overwritten.end(), [](auto value) { return value == 0.0f; }));
    }
    // The input itself is immutable, but its address depends on memory that
    // changes after the snapshot. A delayed address read would select the
    // second half of A and double the product. Both accesses are in bounds.
    auto indirect_kernel = tile_kernel("mpp_snapshot_before_index_write", [](TensorView<const float, 2> A, TensorView<const float, 2> B, TensorView<float, 2> D) {
                               for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
                                   auto space = shape(32, 32);
                                   auto a = A[coord(0, 0), space];
                                   D(coord(0, 0), space).store(full<float>(space, 1.0f));
                                   auto b = B[coord(0, 0), space];
                                   D(coord(0, 0), space).store(mma(a, b, full<float>(space, 0.5f)));
                               }
                           }).capture(tensor_shape(64, 32), tensor_shape(32, 32), tensor_shape(32, 32));
    for (auto forward : {false, true}) {
        auto indirect = bridge::tirx::lower(indirect_kernel.function());
        expect(static_cast<bool>(indirect)) << indirect.error;
        if (!indirect) { continue; }
        IndirectSnapshotTest mutation{indirect.value};
        auto body = mutation(indirect.value->body);
        expect(eq(mutation.replacements, 1u));
        indirect.value.CopyOnWrite()->body = std::move(body);
        bridge::tirx::CompileOptions options;
        options.target = R"({"kind":"metal","thread_warp_size":32})";
        options.noalias = true;
        options.cooperative_matrix = true;
        options.metal_mpp = true;
        options.forward_readonly_tile_loads = forward;
        options.planner.threads_per_group = 32u;
        auto compiled = bridge::tirx::compile(indirect.value, indirect_kernel.function().name(), options);
        expect(static_cast<bool>(compiled)) << compiled.error();
        if (!compiled) { continue; }
        auto entry = compiled.module().value()->GetFunction("mpp_snapshot_before_index_write", true);
        expect(entry.has_value());
        if (!entry) { continue; }
        vector<float> input(2048u, 0.25f);
        std::fill(input.begin() + 1024u, input.end(), 0.5f);
        auto a = runtime.upload<float>({64, 32}, input);
        auto b = runtime.upload<float>({32, 32}, vector<float>(1024u, 0.5f));
        auto d = runtime.upload<float>({32, 32}, vector<float>(1024u, 0.0f));
        (*entry)(a, b, d);
        auto actual = runtime.download<float>(d, 1024u);
        expect(std::all_of(actual.begin(), actual.end(), [](auto value) { return value == 4.5f; })) << "mutable snapshot index, forward=" << forward;
    }
    auto kernel = gemm(runtime, {32, 32, 32, 32, 32, 32}, 1u, 1u, true);
    auto native = bridge::tirx::lower(kernel.function());
    expect(static_cast<bool>(native));
    if (!native) { return; }
    auto escaped = EscapeSnapshotTest{native.value}(native.value->body);
    native.value.CopyOnWrite()->body = std::move(escaped);
    bridge::tirx::CompileOptions options;
    options.target = R"({"kind":"metal","thread_warp_size":32})";
    options.noalias = true;
    options.cooperative_matrix = true;
    options.metal_mpp = true;
    auto reference = bridge::tirx::compile_device(native.value, "escaped_snapshot", options);
    options.forward_readonly_tile_loads = true;
    auto forwarded = bridge::tirx::compile_device(native.value, "escaped_snapshot", options);
    expect(static_cast<bool>(reference)) << reference.error;
    expect(static_cast<bool>(forwarded)) << forwarded.error;
    if (reference && forwarded) {
        // Even a pure address_of observes storage identity. Do not turn its
        // private snapshot pointer into an external-input pointer. Later DCE
        // may remove this unused address, but this pass must remain closed.
        expect(eq(reference.plans[0].shared_memory_bytes, forwarded.plans[0].shared_memory_bytes));
        expect(reference.artifact.source == forwarded.artifact.source);
    }
}

void test_mpp_subgroup_isolation(Runtime &runtime) {
    if (runtime.target() != "metal" || !tvm::ffi::Function::GetGlobal("target.metal.mpp_memory_contract_version")) { return; }
    for (auto [cfg, window] : {std::pair{Shape{32, 64, 32, 32, 64, 16}, 1u},
                               std::pair{Shape{32, 64, 32, 32, 64, 16}, 2u},
                               std::pair{Shape{128, 64, 1024, 128, 32, 1024}, 1u}}) {
        for (auto [enabled, elide] : {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
            auto kernel = gemm(runtime, cfg, window, 1u, true);
            bridge::tirx::PlannerOptions planner;
            planner.threads_per_group = 128u;
            planner.coalesce_group_barriers = enabled;
            planner.elide_independent_subgroup_barriers = elide;
            auto executable = runtime.build(kernel, true, true, true, false, planner, true, true);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            for (auto &&plan : executable.plans) {
                expect(eq(plan.shared_memory_bytes, uint64_t{0u}));
                expect(plan.group_barrier_sites_before > 0u);
                expect(plan.independent_subgroups);
                expect(eq(plan.group_barrier_sites_after, enabled && elide ? uint64_t{0u} : plan.group_barrier_sites_before));
            }
            auto source = metal_source(executable.module.value());
            auto code = std::string_view{source.data(), source.size()};
            expect(eq(code.find("threadgroup_barrier(") == std::string_view::npos, enabled && elide));
            check_gemm(runtime, executable, cfg, 1.0, true);
        }
    }
    // Anonymous shape axes introduce a second value-copy layer. Forwarding
    // must reach a fixed point before proving the otherwise isolated case.
    // The same private accumulator does not authorize removing fences around
    // other global effects, a post-store consumer, or a store on a backedge.
    for (auto kind : {0u, 1u, 2u, 3u}) {
        auto kernel = tile_kernel("mpp_subgroup_isolation_boundaries", [=](TensorView<const float, 2> A,
                                                                           TensorView<const float, 2> B,
                                                                           TensorView<float, 2> D,
                                                                           TensorView<float, 2> E) {
                          for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
                              auto space = shape(32, 64);
                              auto emit = [&](auto &owner) {
                                  auto acc = full<float>(space, 0.5f);
                                  for (auto &step : owner.pipeline(shape(2), {.stages = 1})) {
                                      step.stage("load");
                                      auto k0 = step.index() * 16;
                                      auto a = A.tile(coord(0, k0), shape(32, 16)).load();
                                      auto b = B.tile(coord(k0, 0), shape(16, 64)).load();
                                      step.stage("compute");
                                      acc = mma(a, b, acc);
                                  }
                                  D(coord(0, 0), space).store(acc);
                              };
                              if (kind == 3u) {
                                  for (auto &iteration : nest.serial(shape(2))) { emit(iteration); }
                              } else {
                                  if (kind == 1u) { D(coord(0, 0), space).store(full<float>(space, -9.0f)); }
                                  emit(nest);
                                  if (kind == 2u) { E(coord(0, 0), space).store(D[coord(0, 0), space]); }
                              }
                          }
                      }).capture(tensor_shape(32, 32), tensor_shape(32, 64), tensor_shape(32, 64), tensor_shape(32, 64));
        bridge::tirx::PlannerOptions planner;
        planner.threads_per_group = 128u;
        planner.elide_independent_subgroup_barriers = true;
        auto executable = runtime.build(kernel, true, true, true, false, planner, true, true);
        expect(executable.ok()) << executable.error << " boundary=" << kind;
        if (!executable.ok()) { continue; }
        for (auto &&plan : executable.plans) {
            expect(eq(plan.independent_subgroups, kind == 0u)) << " boundary=" << kind;
            expect(eq(plan.group_barrier_sites_after > 0u, kind != 0u)) << " boundary=" << kind;
            expect(std::any_of(plan.matrices.begin(), plan.matrices.end(), [](auto &&matrix) { return matrix.direct_accumulator_store; }));
            if (kind != 2u) { expect(eq(plan.shared_memory_bytes, uint64_t{0u})) << " boundary=" << kind; }
        }
        auto source = metal_source(executable.module.value());
        expect(eq(std::string_view{source.data(), source.size()}.find("threadgroup_barrier(") != std::string_view::npos, kind != 0u));
        auto a = runtime.upload<float>({32, 32}, vector<float>(1024u, 0.25f));
        auto b = runtime.upload<float>({32, 64}, vector<float>(2048u, 0.5f));
        auto d = runtime.upload<float>({32, 64}, vector<float>(2048u, -7.0f));
        auto e = runtime.upload<float>({32, 64}, vector<float>(2048u, -7.0f));
        (*executable.entry)(a, b, d, e);
        auto actual = runtime.download<float>(d, 2048u);
        auto observed = runtime.download<float>(e, 2048u);
        expect(std::all_of(actual.begin(), actual.end(), [](auto value) { return std::abs(value - 4.5f) < 1e-5f; })) << " boundary=" << kind;
        auto expected = kind == 2u ? 4.5f : -7.0f;
        expect(std::all_of(observed.begin(), observed.end(), [=](auto value) { return std::abs(value - expected) < 1e-5f; })) << " boundary=" << kind;
    }
}

void test_mpp_typed_contract(Runtime &runtime) {
    if (runtime.target() != "metal" || !tvm::ffi::Function::GetGlobal("target.metal.mpp_memory_contract_version")) { return; }
    // Exercise the TVM extension directly: global strided A/B, independent
    // C/D tensors, a nonzero fragment ordinal, and malformed contracts. This
    // is a low-level ABI test, not another high-level kernel lowering path.
    for (auto transpose_a : {false, true}) {
        for (auto transpose_b : {false, true}) {
            for (auto cd_layout : {0, 1, 2, 3}) {
                auto column_major_c = (cd_layout & 1) != 0;
                auto column_major_d = (cd_layout & 2) != 0;
                Shape cfg{16, 8, 24, 16, 8, 24, transpose_a, transpose_b};
                auto kernel = gemm(runtime, cfg, 1u);
                auto i64 = [](int64_t value) { return tvm::IntImm::Int64(value); };
                auto buffer = [&](int64_t rows, int64_t columns, const char *name, const char *scope = "global") {
                    return tvm::tirx::decl_buffer({i64(rows), i64(columns)}, tvm::PrimType::Float(32), name, scope);
                };
                auto a = buffer(transpose_a ? 24 : 16, transpose_a ? 16 : 24, "mpp_left");
                auto b = buffer(transpose_b ? 8 : 24, transpose_b ? 24 : 8, "mpp_right");
                auto c = buffer(column_major_c ? 8 : 16, column_major_c ? 16 : 8, "mpp_element");
                auto d = buffer(column_major_d ? 8 : 16, column_major_d ? 16 : 8, "result");
                auto cf = buffer(2, 128, "initial_fragment", "metal.cooperative_tensor");
                auto df = buffer(1, 128, "result_fragment", "metal.cooperative_tensor");
                auto address = [&](const tvm::tirx::BufferVar &value) {
                    return tvm::Call{value.DataPointerType(), tvm::tirx::builtin::address_of(), {tvm::tirx::BufferLoad{value, {i64(0), i64(0)}}}};
                };
                auto call = [](const char *name, tvm::ffi::Array<tvm::Expr> arguments) {
                    return tvm::tirx::Evaluate{tvm::Call{tvm::PrimType::Void(), tvm::Op::Get(name), std::move(arguments)}};
                };
                // Bad cases are compile-only and must not reach a Metal launch.
                for (auto bad : {0, 1, 2, 3}) {
                    if (bad != 0 && cd_layout != 0) { continue; }
                    tvm::ffi::Array<tvm::tirx::Stmt> statements{tvm::tirx::AllocBuffer{cf}, tvm::tirx::AllocBuffer{df}};
                    statements.push_back(call("tirx.cooperative_tensor_load",
                                              {cf, i64(1), address(c), i64(column_major_c ? 16 : 8), i64(16), i64(8), tvm::IntImm::Bool(column_major_c),
                                               i64(16), i64(8), i64(24), i64(bad == 3 ? 0 : 2)}));
                    statements.push_back(call("tirx.cooperative_tensor_multiply_accumulate_from_memory",
                                              {df, i64(bad == 1 ? 1 : 0), address(a), i64(bad == 2 ? 1 : transpose_a ? 16 :
                                                                                                                       24),
                                               address(b), i64(transpose_b ? 24 : 8), cf, i64(1), i64(16), i64(8), i64(24),
                                               tvm::IntImm::Bool(transpose_a), tvm::IntImm::Bool(transpose_b)}));
                    statements.push_back(call("tirx.cooperative_tensor_store",
                                              {df, i64(0), address(d), i64(column_major_d ? 16 : 8), i64(16), i64(8), tvm::IntImm::Bool(column_major_d),
                                               i64(16), i64(8), i64(24), i64(2)}));
                    auto thread = tvm::tirx::PrimVar{"worker", tvm::PrimType::Int(64)};
                    auto group = tvm::tirx::PrimVar{"program", tvm::PrimType::Int(64)};
                    auto bind = [&](const tvm::tirx::PrimVar &index, int64_t extent, const char *tag, tvm::tirx::Stmt body) {
                        auto axis = tvm::tirx::IterVar{tvm::Range::FromMinExtent(i64(0), i64(extent)), index,
                                                       tvm::tirx::IterVarType::kThreadIndex, tag};
                        return tvm::tirx::For{index, i64(0), i64(extent), tvm::tirx::ForKind::kThreadBinding, std::move(body), axis};
                    };
                    auto body = bind(group, 1, "blockIdx.x", bind(thread, 32, "threadIdx.x", tvm::tirx::SeqStmt::Flatten(statements)));
                    auto function = tvm::tirx::PrimFunc{{a, b, c, d}, std::move(body)};
                    auto executable = compile_native(runtime, kernel, std::move(function));
                    expect(eq(executable.ok(), bad == 0)) << executable.error;
                    if (bad == 0 && executable.ok()) {
                        check_gemm(runtime, executable, cfg, 1.0, false, column_major_c, column_major_d);
                    } else if (bad != 0) {
                        auto expected = bad == 1 ? "fragment index out of range" : bad == 2 ? "leading stride" :
                                                                                              "destination tensors";
                        expect(executable.error.find(expected) != luisa::string::npos) << executable.error;
                    }
                }
            }
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

void test_cpu_stack_storage(Runtime &runtime) {
    using namespace bridge::tirx;
    if (runtime.target() != "llvm") { return; }
    expect(eq(PlannerOptions{}.max_cpu_stack_bytes, 0u));
    // 17 floats require 68 bytes, charged as 80 with alignment padding.
    // Snapshot semantics must survive a write to the original tensor.
    for (auto manual : {0u, 1u, 2u}) {
        auto kernel = tile_kernel("cpu_stack_snapshot", [=](TensorView<float, 2> A, TensorView<float, 2> D) {
                          for (auto &nest : parallel(shape(2), exec::Scope::WORKER)) {
                              auto origin = coord(nest.index(), 0);
                              auto space = shape(1, 17);
                              auto snapshot = A[origin, space];
                              if (manual) {
                                  auto storage = manual == 1u ? memory<float>(space) : memory<float>(space, mem::private_);
                                  storage.store(snapshot);
                                  snapshot = storage.load();
                              }
                              A(origin, space).store(zeros<float>(space));
                              D(origin, space).store(snapshot);
                          }
                      }).capture(tensor_shape(2, 17), tensor_shape(2, 17));
        for (auto budget : {0u, 64u, 80u, 8192u}) {
            PlannerOptions planner;
            planner.max_cpu_stack_bytes = budget;
            auto executable = runtime.build(kernel, false, false, true, true, planner);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            auto source = executable.module.value()->InspectSource("ll");
            auto workspace = std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") != std::string_view::npos;
            expect(eq(workspace, manual != 0u || budget < 80u)) << "budget=" << budget << " manual=" << manual << "\n"
                                                                << source;
            for (auto phase : {0.25f, 0.5f}) {
                auto input = values(34u, phase);
                auto a = runtime.upload<float>({2, 17}, input);
                auto d = runtime.allocate<float>({2, 17});
                (*executable.entry)(a, d);
                expect(runtime.download<float>(d, 34u) == input);
                expect(runtime.download<float>(a, 34u) == vector<float>(34u, 0.0f));
                auto aliased = runtime.upload<float>({2, 17}, input);
                (*executable.entry)(aliased, aliased);
                expect(runtime.download<float>(aliased, 34u) == input);
            }
        }
    }
    // The limit is a cumulative budget, including alignment between objects.
    auto two_snapshots = tile_kernel("cpu_stack_two_snapshots", [](TensorView<float, 2> A, TensorView<float, 2> B, TensorView<float, 2> D) {
                             for (auto &nest : parallel(shape(2), exec::Scope::WORKER)) {
                                 auto origin = coord(nest.index(), 0);
                                 auto space = shape(1, 17);
                                 auto a = A[origin, space];
                                 auto b = B[origin, space];
                                 A(origin, space).store(zeros<float>(space));
                                 B(origin, space).store(zeros<float>(space));
                                 D(origin, space).store(a + b);
                             }
                         }).capture(tensor_shape(2, 17), tensor_shape(2, 17), tensor_shape(2, 17));
    for (auto budget : {80u, 144u, 160u}) {
        PlannerOptions planner;
        planner.max_cpu_stack_bytes = budget;
        auto executable = runtime.build(two_snapshots, false, false, true, true, planner);
        expect(executable.ok()) << executable.error;
        if (!executable.ok()) { continue; }
        auto source = executable.module.value()->InspectSource("ll");
        auto workspace = std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") != std::string_view::npos;
        expect(eq(workspace, budget < 160u)) << "the budget is cumulative, not per buffer: " << budget;
        auto input = values(34u, 0.25f);
        auto a = runtime.upload<float>({2, 17}, input);
        auto b = runtime.upload<float>({2, 17}, input);
        auto d = runtime.allocate<float>({2, 17});
        (*executable.entry)(a, b, d);
        for (auto &value : input) { value *= 2.0f; }
        expect(runtime.download<float>(d, 34u) == input);
    }
    // Transposed/ragged MMA still follows the reference recurrence; only
    // the allocation mechanism changes.
    for (auto cfg : {Shape{17, 23, 31, 4, 7, 16}, Shape{16, 32, 32, 4, 16, 16},
                     Shape{17, 23, 31, 4, 7, 16, true, false}, Shape{17, 23, 31, 4, 7, 16, false, true}}) {
        cfg.math.allow_reassociation = false;
        for (auto window : {1u, 2u}) {
            auto kernel = gemm(runtime, cfg, window);
            PlannerOptions planner;
            planner.max_cpu_stack_bytes = 8192u;
            auto executable = runtime.build(kernel, true, false, true, true, planner);
            expect(executable.ok()) << executable.error;
            if (executable.ok()) {
                auto source = executable.module.value()->InspectSource("ll");
                expect(std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") == std::string_view::npos);
                check_gemm(runtime, executable, cfg);
            }
        }
    }
    auto cfg = Shape{16, 32, 32, 4, 16, 16};
    auto kernel = gemm(runtime, cfg, 1u);
    auto native = lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    native.value.CopyOnWrite()->body = EscapeSnapshotTest{native.value, true}(native.value->body);
    PlannerOptions planner;
    planner.max_cpu_stack_bytes = 8192u;
    auto escaped = compile_native(runtime, kernel, std::move(native.value), 32u, 256u, planner);
    expect(escaped.ok()) << escaped.error;
    if (escaped.ok()) {
        auto source = escaped.module.value()->InspectSource("ll");
        expect(std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") != std::string_view::npos);
        check_gemm(runtime, escaped, cfg);
    }
    planner.enabled = false;
    auto disabled = runtime.build(kernel, true, false, true, true, planner);
    expect(disabled.ok()) << disabled.error;
    if (disabled.ok()) {
        auto source = disabled.module.value()->InspectSource("ll");
        expect(std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") != std::string_view::npos);
        check_gemm(runtime, disabled, cfg);
    }
    planner.enabled = true;
    planner.max_cpu_stack_bytes = 65537u;
    auto invalid = runtime.build(kernel, true, false, true, true, planner);
    expect(!invalid.ok());
    expect(invalid.error.find("CPU stack planning") != luisa::string::npos);
}

void test_cpu_readonly_view_gemm(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    for (auto cfg : {Shape{16, 32, 32, 4, 16, 16}, Shape{19, 37, 31, 9, 19, 16},
                     Shape{17, 23, 31, 4, 7, 16, true, false},
                     Shape{17, 23, 31, 4, 7, 16, false, true},
                     Shape{17, 23, 31, 4, 7, 16, true, true}}) {
        cfg.math.allow_reassociation = false;
        for (auto window : {1u, 2u}) {
            auto kernel = gemm(runtime, cfg, window);
            for (auto lanes : {16u, 64u}) {
                for (auto stack : {0u, 8192u}) {
                    bridge::tirx::PlannerOptions planner;
                    planner.max_cpu_vector_lanes = lanes;
                    planner.max_cpu_stack_bytes = stack;
                    auto executable = runtime.build(kernel, true, false, true, true, planner, false, true);
                    expect(executable.ok()) << executable.error;
                    if (executable.ok()) { check_gemm(runtime, executable, cfg); }
                }
            }
        }
    }
    // Mixed-radix program coordinates previously defeated the native proof
    // of G => G. Correctness alone missed this no-op: verify that the input
    // staging arrays really disappear, independently of the requested flag.
    for (auto cfg : {Shape{127, 193, 61, 4, 16, 32}, Shape{513, 257, 129, 4, 16, 32}}) {
        for (auto window : {1u, 2u}) {
            auto kernel = gemm(runtime, cfg, window, 1u, true);
            for (auto forward : {false, true}) {
                bridge::tirx::PlannerOptions planner;
                planner.max_cpu_vector_lanes = 64u;
                planner.max_cpu_stack_bytes = 8192u;
                auto executable = runtime.build(kernel, true, false, true, true, planner, false, forward);
                expect(executable.ok()) << executable.error;
                if (!executable.ok()) { continue; }
                auto source = executable.module.value()->InspectSource("ll");
                auto code = std::string_view{source.data(), source.size()};
                for (auto elements : {cfg.bm * cfg.bk, cfg.bk * cfg.bn}) {
                    auto array = luisa::format("alloca [{} x float]", elements * window);
                    expect(eq(code.find(array) != std::string_view::npos, !forward)) << array << " forward=" << forward << "\n"
                                                                                     << code;
                }
                if (forward) {
                    expect(code.find(" x float> @llvm.fmuladd.v") != std::string_view::npos)
                        << "guarded input views must retain a vector arithmetic fast path\n"
                        << code;
                }
                check_gemm(runtime, executable, cfg, 1.0, true);
            }
        }
    }
}

enum class ReadonlyViewCase { PADDED,
                              MUTABLE_INDEX,
                              MUTABLE_GUARD,
                              MUTABLE_FILL,
                              NESTED_INDEX,
                              GUARDED_NESTED_INDEX,
                              MANUAL,
                              CONDITIONAL,
                              ESCAPED,
                              MUTABLE_SOURCE,
                              IMMUTABLE_GUARD,
                              WEAK_GUARD };

void test_cpu_readonly_view_proofs(Runtime &runtime) {
    using namespace bridge::tirx;
    if (runtime.target() != "llvm") { return; }
    // Native transform fixtures deliberately include nonzero padding, memory
    // in address/predicate/fill expressions, and a snapshot used as another
    // snapshot's index. They must not rely on incidental frontend copies.
    auto i64 = [](int64_t value) { return tvm::IntImm::Int64(value); };
    auto f32 = [](float value) { return tvm::FloatImm{tvm::PrimType::Float(32), value}; };
    for (auto mode : {ReadonlyViewCase::PADDED, ReadonlyViewCase::MUTABLE_INDEX, ReadonlyViewCase::MUTABLE_GUARD,
                      ReadonlyViewCase::MUTABLE_FILL, ReadonlyViewCase::NESTED_INDEX, ReadonlyViewCase::GUARDED_NESTED_INDEX,
                      ReadonlyViewCase::MANUAL,
                      ReadonlyViewCase::CONDITIONAL, ReadonlyViewCase::ESCAPED, ReadonlyViewCase::MUTABLE_SOURCE,
                      ReadonlyViewCase::IMMUTABLE_GUARD, ReadonlyViewCase::WEAK_GUARD}) {
        auto nested = mode == ReadonlyViewCase::NESTED_INDEX ||
                      mode == ReadonlyViewCase::GUARDED_NESTED_INDEX;
        auto size = nested ? 17 : 34;
        auto buffer = [&](int64_t length, const char *name, const char *scope = "global") {
            return tvm::tirx::decl_buffer({i64(length)}, tvm::PrimType::Float(32), name, scope);
        };
        auto a = buffer(17, "input");
        auto state = buffer(1, "state");
        auto d = buffer(size, "output");
        auto snapshot = buffer(size, "snapshot", "local");
        auto index_snapshot = buffer(1, "index_snapshot", "local");
        auto state_value = tvm::tirx::BufferLoad{state, {i64(0)}};
        auto index = tvm::tirx::PrimVar{"index", tvm::PrimType::Int(64)};
        auto elements = [&](const tvm::tirx::PrimVar &axis, int64_t count, tvm::tirx::Stmt body) {
            return tvm::tirx::For{axis, i64(0), i64(count), tvm::tirx::ForKind::kSerial, std::move(body), {}, {{"luisa.tile.independent_elements", tvm::IntImm::Int32(1)}}};
        };
        tvm::PrimExpr source_index = mode == ReadonlyViewCase::MUTABLE_INDEX ?
                                         tvm::floormod(tvm::cast(tvm::PrimType::Int(64), state_value), i64(17)) :
                                         tvm::PrimExpr{index};
        tvm::PrimExpr guard = index >= i64(0) && index < i64(17);
        if (mode == ReadonlyViewCase::MUTABLE_GUARD) { guard = guard && state_value < f32(2.0f); }
        if (mode == ReadonlyViewCase::IMMUTABLE_GUARD) {
            // Reordered/reassociated bounds plus a stable, non-affine mask.
            guard = (index < i64(17) && state_value < f32(2.0f)) && index >= i64(0);
        }
        if (mode == ReadonlyViewCase::WEAK_GUARD) {
            // One OR arm containing all bounds is not a bounds proof. Only
            // execute positive-state inputs below, where the original access
            // is defined; forwarding must not assume that input restriction.
            guard = guard || state_value < f32(0.0f);
        }
        tvm::PrimExpr fill = mode == ReadonlyViewCase::MUTABLE_FILL ? tvm::PrimExpr{state_value} : tvm::PrimExpr{f32(-2.25f)};
        auto value = tvm::if_then_else(guard, tvm::tirx::BufferLoad{a, {source_index}}, fill);
        auto copy = elements(index, size, tvm::tirx::BufferStore{snapshot, std::move(value), {index}});
        auto output_index = tvm::tirx::PrimVar{"output_index", tvm::PrimType::Int(64)};
        tvm::PrimExpr read_index = output_index;
        if (nested) {
            read_index = tvm::floormod(tvm::cast(tvm::PrimType::Int(64), tvm::tirx::BufferLoad{index_snapshot, {i64(0)}}), i64(17));
        }
        tvm::PrimExpr consumed = tvm::tirx::BufferLoad{snapshot, {read_index}};
        if (mode == ReadonlyViewCase::GUARDED_NESTED_INDEX) {
            consumed = tvm::if_then_else(
                read_index >= i64(0) && read_index < i64(size),
                std::move(consumed), f32(-2.25f));
        }
        auto consume = elements(
            output_index, size,
            tvm::tirx::BufferStore{d, std::move(consumed), {output_index}});
        tvm::ffi::Map<tvm::ffi::String, tvm::Any> annotations;
        if (mode == ReadonlyViewCase::MANUAL) { annotations.Set("luisa.tile.manual_memory", tvm::IntImm::Int32(1)); }
        tvm::ffi::Array<tvm::tirx::Stmt> body{tvm::tirx::AllocBuffer{snapshot, annotations}};
        if (nested) {
            auto axis = tvm::tirx::PrimVar{"index_copy", tvm::PrimType::Int(64)};
            body.push_back(tvm::tirx::AllocBuffer{index_snapshot});
            body.push_back(elements(axis, 1, tvm::tirx::BufferStore{index_snapshot, tvm::tirx::BufferLoad{state, {axis}}, {axis}}));
        }
        if (mode == ReadonlyViewCase::CONDITIONAL) {
            body.push_back(tvm::tirx::IfThenElse{state_value < f32(2.0f), tvm::tirx::SeqStmt{{copy, consume}}});
        } else {
            body.push_back(copy);
            if (mode == ReadonlyViewCase::MUTABLE_SOURCE) { body.push_back(tvm::tirx::BufferStore{a, f32(11.0f), {i64(0)}}); }
            if (mode == ReadonlyViewCase::MUTABLE_INDEX || mode == ReadonlyViewCase::MUTABLE_GUARD || mode == ReadonlyViewCase::MUTABLE_FILL) {
                body.push_back(tvm::tirx::BufferStore{state, f32(11.0f), {i64(0)}});
            }
            if (mode == ReadonlyViewCase::ESCAPED) {
                body.push_back(tvm::tirx::Evaluate{tvm::Call{snapshot.DataPointerType(), tvm::tirx::builtin::address_of(), {tvm::tirx::BufferLoad{snapshot, {i64(0)}}}}});
            }
            body.push_back(consume);
        }
        auto function = tvm::tirx::PrimFunc{{a, state, d}, tvm::tirx::SeqStmt::Flatten(body)};
        for (auto forward : {false, true}) {
            CompileOptions options;
            options.target = "llvm";
            options.noalias = true;
            options.forward_readonly_tile_loads = forward;
            options.auto_vectorize = true;
            // Budget zero makes actual snapshot removal observable separately
            // from stack placement/LLVM scalar replacement.
            auto compiled = compile(function, "cpu_view_proof", options);
            expect(compiled.ok()) << compiled.error();
            if (!compiled) { continue; }
            auto source = compiled.module().value()->InspectSource("ll");
            auto workspace = std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") != std::string_view::npos;
            // An unguarded memory-dependent consumer index remains unknown.
            // A lazy branch that syntactically proves the complete temporary
            // bounds can instead forward the immutable source expression.
            auto removable = mode == ReadonlyViewCase::PADDED ||
                             mode == ReadonlyViewCase::IMMUTABLE_GUARD ||
                             mode == ReadonlyViewCase::GUARDED_NESTED_INDEX;
            expect(eq(workspace, !forward || !removable)) << "mode=" << static_cast<uint32_t>(mode) << " forward=" << forward;
            auto entry = compiled.module().value()->GetFunction("cpu_view_proof", true);
            expect(entry.has_value());
            if (!entry) { continue; }
            for (auto phase : {1.0f, 3.0f}) {
                auto input = values(17u, phase);
                auto left = runtime.upload<float>({17}, input);
                auto selector = runtime.upload<float>({1}, vector<float>{phase});
                auto output = runtime.upload<float>({size}, vector<float>(size, -9.0f));
                (*entry)(left, selector, output);
                auto actual = runtime.download<float>(output, size);
                if (mode == ReadonlyViewCase::MUTABLE_SOURCE) { expect(eq(runtime.download<float>(left, 17u)[0], 11.0f)); }
                for (auto i = 0; i < size; i++) {
                    auto expected = i < 17 ? input[i] : -2.25f;
                    if (nested) { expected = input[static_cast<size_t>(phase)]; }
                    if (mode == ReadonlyViewCase::MUTABLE_INDEX && i < 17) { expected = input[static_cast<size_t>(phase)]; }
                    if ((mode == ReadonlyViewCase::MUTABLE_GUARD || mode == ReadonlyViewCase::IMMUTABLE_GUARD) && phase >= 2.0f) { expected = -2.25f; }
                    if (mode == ReadonlyViewCase::MUTABLE_FILL && i >= 17) { expected = phase; }
                    if (mode == ReadonlyViewCase::CONDITIONAL && phase >= 2.0f) { expected = -9.0f; }
                    expect(eq(actual[i], expected)) << "mode=" << static_cast<uint32_t>(mode) << " forward=" << forward << " index=" << i;
                }
            }
        }
    }
}

void test_cpu_readonly_view_aliases(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    // A const parameter may alias a writable output unless noalias is set.
    // Manual storage and a mutable source must also retain snapshot timing.
    for (auto manual : {0u, 1u, 2u}) {
        auto kernel = tile_kernel("cpu_view_alias", [=](TensorView<const float, 2> A, TensorView<float, 2> D) {
                          for (auto &nest : parallel(shape(2), exec::Scope::WORKER)) {
                              auto origin = coord(nest.index(), 0);
                              auto space = shape(1, 17);
                              auto snapshot = A[origin, space];
                              if (manual) {
                                  auto storage = manual == 1u ? memory<float>(space) : memory<float>(space, mem::private_);
                                  storage.store(snapshot);
                                  snapshot = storage.load();
                              }
                              D(origin, space).store(zeros<float>(space));
                              D(origin, space).store(snapshot);
                          }
                      }).capture(tensor_shape(2, 17), tensor_shape(2, 17));
        for (auto noalias : {false, true}) {
            auto executable = runtime.build(kernel, noalias, false, true, true, {}, false, true);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            auto source = executable.module.value()->InspectSource("ll");
            auto workspace = std::string_view{source.data(), source.size()}.find("@__TVMBackendAllocWorkspace") != std::string_view::npos;
            expect(eq(workspace, !noalias || manual != 0u));
            auto input = values(34u, 0.5f);
            auto a = runtime.upload<float>({2, 17}, input);
            auto d = noalias ? runtime.allocate<float>({2, 17}) : a;
            (*executable.entry)(a, d);
            expect(runtime.download<float>(d, 34u) == input);
        }
    }
}

void test_cpu_cartesian_packing(Runtime &runtime) {
    using bridge::tirx::PlannerOptions;
    expect(eq(PlannerOptions{}.max_cpu_vector_lanes, 16u));
    auto kernel = gemm(runtime, {16, 32, 32, 4, 16, 16}, 1u);
    if (runtime.target() != "llvm") {
        PlannerOptions planner;
        planner.max_cpu_vector_lanes = 64u;
        auto invalid = runtime.build(kernel, false, true, true, true, planner);
        expect(!invalid.ok());
        expect(invalid.error.find("CPU vector packing") != luisa::string::npos);
        return;
    }
    // Both row and column tails, multiple Cartesian packs, transposes, and
    // nonzero initial values. Test workspace and stack realization separately.
    for (auto cfg : {Shape{9, 13, 29, 3, 5, 7}, Shape{17, 23, 31, 4, 7, 16},
                     Shape{16, 32, 32, 4, 16, 16}, Shape{19, 37, 31, 9, 19, 16},
                     Shape{17, 23, 31, 4, 7, 16, true, false},
                     Shape{17, 23, 31, 4, 7, 16, false, true},
                     Shape{17, 23, 31, 4, 7, 16, true, true}}) {
        cfg.math.allow_reassociation = false;
        for (auto window : {1u, 2u}) {
            auto candidate = gemm(runtime, cfg, window);
            for (auto lanes : {32u, 64u, 128u}) {
                for (auto stack : {0u, 8192u}) {
                    PlannerOptions planner;
                    planner.max_cpu_vector_lanes = lanes;
                    planner.max_cpu_stack_bytes = stack;
                    auto executable = runtime.build(candidate, true, false, true, true, planner);
                    expect(executable.ok()) << executable.error;
                    if (!executable.ok()) { continue; }
                    check_gemm(runtime, executable, cfg);
                }
            }
        }
    }
    for (auto lanes : {0u, 8u, 17u, 48u, 256u}) {
        PlannerOptions planner;
        planner.max_cpu_vector_lanes = lanes;
        auto invalid = runtime.build(kernel, false, false, true, true, planner);
        expect(!invalid.ok());
        expect(invalid.error.find("CPU vector packing") != luisa::string::npos);
    }
    PlannerOptions planner;
    planner.max_cpu_vector_lanes = 64u;
    auto no_auto = runtime.build(kernel, false, false, true, false, planner);
    expect(!no_auto.ok());
    expect(no_auto.error.find("CPU vector packing") != luisa::string::npos);
    planner.enabled = false;
    auto disabled = runtime.build(kernel, true, false, true, true, planner);
    expect(disabled.ok()) << disabled.error;
    if (disabled.ok()) { check_gemm(runtime, disabled, {16, 32, 32, 4, 16, 16}); }
}

class CartesianDomainTest final : public tvm::tirx::StmtMutator {
private:
    uint32_t _mode;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        if (!loop->annotations.count("luisa.tile.mma")) { return StmtMutator::VisitStmt_(loop); }
        auto row = tvm::ffi::GetRef<tvm::tirx::For>(loop);
        auto column = row->body.as_or_throw<tvm::tirx::For>();
        auto sequence = column->body.as_or_throw<tvm::tirx::SeqStmt>();
        auto temporal = sequence->seq.back().as_or_throw<tvm::tirx::For>();
        if (_mode == 1u || _mode == 2u) {
            // Nonrectangular time domains must not be jammed across rows or
            // columns. The original per-element recurrence remains valid.
            temporal.CopyOnWrite()->extent -= _mode == 1u ? row->loop_var : column->loop_var;
        } else if (_mode == 3u) {
            temporal.CopyOnWrite()->body = tvm::tirx::IfThenElse{
                tvm::floormod(temporal->loop_var, tvm::IntImm::Int64(2)) == tvm::IntImm::Int64(0), temporal->body};
        } else if (_mode == 4u) {
            auto store = temporal->body.as_or_throw<tvm::tirx::BufferStore>();
            auto sum = store->value.as<tvm::tirx::AddNode>();
            expect(sum != nullptr);
            if (sum) { store.CopyOnWrite()->value = sum->a - sum->b; }
            temporal.CopyOnWrite()->body = std::move(store);
        }
        sequence.CopyOnWrite()->seq.Set(sequence->seq.size() - 1u, std::move(temporal));
        column.CopyOnWrite()->body = std::move(sequence);
        if (_mode == 0u) {
            column.CopyOnWrite()->body = tvm::tirx::Substitute(column->body,
                                                               tvm::ffi::Map<tvm::tirx::Var, tvm::Expr>{{row->loop_var, row->loop_var - tvm::IntImm::Int64(5)},
                                                                                                        {column->loop_var, column->loop_var - tvm::IntImm::Int64(7)}});
            row.CopyOnWrite()->min += tvm::IntImm::Int64(5);
            column.CopyOnWrite()->min += tvm::IntImm::Int64(7);
        }
        row.CopyOnWrite()->body = std::move(column);
        // Keep the independent-element contract, but remove the MMA marker.
        // Packing must operate on the actual region, including non-MMA math.
        row.CopyOnWrite()->annotations.erase("luisa.tile.mma");
        replacements++;
        return row;
    }

public:
    uint32_t replacements{0u};
    explicit CartesianDomainTest(uint32_t mode) noexcept : _mode{mode} {}
    using StmtMutator::operator();
};

void test_cpu_cartesian_transformed_domains(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    auto kernel = gemm(runtime, {5, 7, 11, 5, 7, 11}, 1u);
    for (auto mode = 0u; mode < 5u; mode++) {
        for (auto lanes : {16u, 64u}) {
            auto native = bridge::tirx::lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            CartesianDomainTest transform{mode};
            native.value.CopyOnWrite()->body = transform(native.value->body);
            expect(eq(transform.replacements, 1u));
            bridge::tirx::PlannerOptions planner;
            planner.max_cpu_stack_bytes = 8192u;
            planner.max_cpu_vector_lanes = lanes;
            auto executable = compile_native(runtime, kernel, std::move(native.value), 32u, 256u, planner, true);
            expect(executable.ok()) << executable.error << " mode=" << mode;
            if (!executable.ok()) { continue; }
            auto a = runtime.upload<float>({5, 11}, vector<float>(55u, 0.125f));
            auto b = runtime.upload<float>({11, 7}, vector<float>(77u, 0.25f));
            auto c = runtime.upload<float>({5, 7}, vector<float>(35u, 0.375f));
            // In-place C/D is allowed: no noalias assumption is passed.
            (*executable.entry)(a, b, c, c);
            auto actual = runtime.download<float>(c, 35u);
            for (auto m = 0u; m < 5u; m++) {
                for (auto n = 0u; n < 7u; n++) {
                    auto count = mode == 1u ? 11u - m : mode == 2u ? 11u - n :
                                                    mode == 3u     ? 6u :
                                                                     11u;
                    auto expected = 0.375f + (mode == 4u ? -1.0f : 1.0f) * 0.03125f * static_cast<float>(count);
                    expect(std::isfinite(actual[m * 7u + n]) && std::abs(actual[m * 7u + n] - expected) < 1e-6f)
                        << "mode=" << mode << " lanes=" << lanes << " at=" << m << "," << n;
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
    for (auto lanes : {0u, 16u, 32u, 64u, 128u}) {
        auto vectorize = lanes != 0u;
        bridge::tirx::PlannerOptions planner;
        planner.max_cpu_vector_lanes = std::max(lanes, 16u);
        for (auto forward : {false, true}) {
            auto executable = runtime.build(kernel, true, false, vectorize, vectorize, planner, false, forward);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            (*executable.entry)(a, b, d);
            auto output = runtime.download<float>(d, 15);
            // Sequential FP32 is 0.5; regrouping cancellation can produce 1.5.
            expect(std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value) && std::abs(value - 0.5f) < 1e-7f; }));
        }
    }
}

void test_planned_fragment_reuse(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    Shape cfg{37, 71, 45, 32, 64, 16};
    auto kernel = gemm(runtime, cfg, 1u);
    for (auto enabled : {false, true}) {
        bridge::tirx::PlannerOptions planner;
        planner.enabled = enabled;
        // Keep this reference fixture multi-wave even when the queried device
        // supports 1024 threads; its assertions exercise atom-by-atom waves.
        if (!enabled) { planner.threads_per_group = 256u; }
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

void test_late_matrix_prefetch(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    Shape cases[]{
        {64, 64, 7, 64, 64, 32}, {64, 64, 33, 64, 64, 32}, {67, 71, 95, 64, 64, 32}, {67, 71, 95, 64, 64, 32, true, false}, {67, 71, 95, 64, 64, 32, false, true}};
    struct Policy {
        uint32_t window, interval, budget, batch;
    };
    Policy policies[]{{1u, 1u, 32u, 4u}, {2u, 1u, 0u, 4u}, {2u, 1u, 8u, 4u}, {2u, 1u, 32u, 1u}, {2u, 1u, 32u, 4u}, {2u, 2u, 32u, 4u}};
    for (auto cfg : cases) {
        for (auto policy : policies) {
            auto kernel = gemm(runtime, cfg, policy.window, policy.interval, true);
            bridge::tirx::PlannerOptions planner;
            planner.threads_per_group = 256u;
            planner.max_copy_batch = policy.batch;
            planner.max_pipeline_prefetch_scalars_per_lane = policy.budget;
            auto executable = runtime.build(kernel, false, true, true, false, planner);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            expect(eq(executable.plans.size(), size_t{1u}));
            if (executable.plans.size() != 1u) { continue; }
            auto expected = cfg.k > cfg.bk && policy.window == 2u && policy.interval == 1u && policy.budget >= 16u;
            auto &plan = executable.plans[0];
            expect(eq(plan.prefetched_pipeline_loops, expected ? 1ull : 0ull));
            expect(eq(plan.prefetch_storage_scalars_per_lane, expected ? 16ull : 0ull));
            // Shared A/B remain single-buffered. A ragged final output keeps
            // C's backing for masked scalar stores; a full output elides it.
            auto full_output = cfg.m % cfg.bm == 0 && cfg.n % cfg.bn == 0;
            expect(eq(plan.shared_memory_bytes, full_output ? 16384ull : 32768ull));
            auto source = metal_source(executable.module.value());
            auto code = std::string_view{source.data(), source.size()};
            expect(eq(code.find("_prefetch[16]") != std::string_view::npos, expected)) << code;
            expect(code.find("luisa.tile.deferred_pipeline") == std::string_view::npos);
            check_gemm(runtime, executable, cfg, 1.0, true);
        }
    }
}

void test_matrix_prefetch_rejects_global_writes(Runtime &runtime) {
    if (runtime.target() != "metal") { return; }
    auto kernel = tile_kernel("matrix_prefetch_alias", [](TensorView<float, 2> A, TensorView<const float, 2> B,
                                                          TensorView<float, 2> D) {
                      auto m = axis("m", 64);
                      auto n = axis("n", 64);
                      auto k = axis("k", 32);
                      for (auto &nest : parallel(shape(1), exec::Scope::GROUP)) {
                          auto acc = zeros<float>(shape(m, n));
                          for (auto &step : nest.pipeline(shape(2), {.stages = 2u})) {
                              step.stage("read");
                              auto a = A[coord(0, step.index() * 32), shape(m, k)];
                              auto b = B[coord(step.index() * 32, 0), shape(k, n)];
                              step.stage("compute");
                              acc = mma(a, b, acc);
                              step.stage("mutate future input");
                              A(coord(0, (step.index() + 1) * 32), shape(m, k)).store(full<float>(shape(m, k), 0.5f));
                          }
                          D(coord(0, 0), shape(m, n)).store(acc);
                      }
                  }).capture(tensor_shape(64, 96), tensor_shape(64, 64), tensor_shape(64, 64));
    bridge::tirx::PlannerOptions planner;
    planner.threads_per_group = 256u;
    planner.max_copy_batch = 4u;
    auto executable = runtime.build(kernel, false, true, true, false, planner);
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    expect(eq(executable.plans.size(), size_t{1u}));
    for (auto &plan : executable.plans) {
        expect(eq(plan.prefetched_pipeline_loops, 0ull));
        expect(eq(plan.prefetch_storage_scalars_per_lane, 0ull));
    }
    auto a = runtime.upload<float>({64, 96}, vector<float>(64u * 96u, 0.25f));
    auto b = runtime.upload<float>({64, 64}, vector<float>(64u * 64u, 0.5f));
    auto d = runtime.allocate<float>({64, 64});
    (*executable.entry)(a, b, d);
    auto actual = runtime.download<float>(d, 64u * 64u);
    expect(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value) && std::abs(value - 12.0f) < 1e-5f; }))
        << "next-iteration reads must remain after the intervening global write";
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

class AdjustAccumulatorTest final : public tvm::tirx::StmtMutator {
private:
    tvm::tirx::BufferVar _output, _initial;
    bool _transpose, _pin;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::AllocBufferNode *allocation) final {
        auto result = StmtMutator::VisitStmt_(allocation).as_or_throw<tvm::tirx::AllocBuffer>();
        if (_pin && result->buffer.same_as(_initial)) {
            result.CopyOnWrite()->annotations.Set("luisa.tile.memory_resource", tvm::ffi::String{"shared"});
        }
        return result;
    }
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        auto result = StmtMutator::VisitStmt_(store).as_or_throw<tvm::tirx::BufferStore>();
        if (_transpose && result->buffer.same_as(_output)) {
            auto indices = result->indices;
            expect(eq(indices.size(), size_t{2u}));
            if (indices.size() == 2u) { result.CopyOnWrite()->indices = {indices[1], indices[0]}; }
        }
        return result;
    }

public:
    AdjustAccumulatorTest(const tvm::tirx::PrimFunc &function, bool transpose, bool pin)
        : _output{function->params[2].as_or_throw<tvm::tirx::BufferVar>()}, _transpose{transpose}, _pin{pin} {
        tvm::tirx::PostOrderVisit(function->body, [&](const tvm::ffi::ObjectRef &node) {
            if (auto store = node.as<tvm::tirx::BufferStoreNode>(); store != nullptr && store->value.as<tvm::FloatImmNode>() != nullptr) { _initial = store->buffer; }
        });
        expect(_initial.defined());
    }
    using StmtMutator::operator();
};

void test_direct_accumulator_output(Runtime &runtime) {
    struct Case {
        Shape shape;
        int64_t row_offset{0}, column_offset{0};
        int64_t row_padding{0}, column_padding{0};
        bool transpose_output{false};
        bool observe_accumulator{false};
        bool observe_old_output{false};
        bool pin_initial{false};
    };
    Case cases[]{
        {{8, 16, 8, 8, 16, 8}}, {{16, 32, 24, 8, 16, 8}}, {{32, 64, 29, 16, 32, 16}}, {{37, 71, 45, 32, 64, 16}}, {{16, 24, 0, 8, 24, 8}}, {{16, 16, 8, 8, 8, 8}, 1, 3, 2, 6}, {{16, 16, 8, 8, 8, 8}, -1, -2}, {{32, 32, 24, 16, 16, 8}, 0, 0, 0, 0, true}, {{32, 48, 24, 16, 24, 8}, 0, 0, 0, 0, false, true}, {{32, 48, 24, 16, 24, 8}, 0, 0, 0, 0, false, false, true}, {{32, 48, 24, 16, 24, 8}, 0, 0, 0, 0, false, false, false, true}};
    for (auto test : cases) {
        auto cfg = test.shape;
        auto rows = cfg.m + test.row_padding;
        auto columns = cfg.n + test.column_padding;
        auto input_k = std::max(int64_t{1}, cfg.k);
        auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
        auto kernel = tile_kernel("matrix_direct_output", [=](TensorView<const float, 2> A, TensorView<const float, 2> B,
                                                              TensorView<float, 2> D, TensorView<float, 2> H) {
                          auto gm = axis("gm", ceil_div(cfg.m, cfg.bm));
                          auto gn = axis("gn", ceil_div(cfg.n, cfg.bn));
                          auto m = axis("m", cfg.bm);
                          auto n = axis("n", cfg.bn);
                          auto k = axis("k", cfg.bk);
                          for (auto &nest : parallel(shape(gm, gn), scope)) {
                              auto m0 = nest.index(gm) * cfg.bm;
                              auto n0 = nest.index(gn) * cfg.bn;
                              auto acc = full<float>(shape(m, n), 0.375f);
                              for (auto &step : nest.pipeline(shape(ceil_div(cfg.k, cfg.bk)), {.stages = 1u})) {
                                  step.stage("load");
                                  auto a = A[coord(m0, step.index() * cfg.bk), shape(m, k)];
                                  auto b = B[coord(step.index() * cfg.bk, n0), shape(k, n)];
                                  step.stage("compute");
                                  acc = mma(a, b, acc);
                              }
                              auto origin = coord(m0 + test.row_offset, n0 + test.column_offset);
                              if (test.observe_old_output) { H(origin, shape(m, n)).store(D[origin, shape(m, n)]); }
                              D(origin, shape(m, n)).store(acc);
                              if (test.observe_accumulator) { H(origin, shape(m, n)).store(acc); }
                          }
                      }).capture(tensor_shape(cfg.m, input_k), tensor_shape(input_k, cfg.n), tensor_shape(rows, columns), tensor_shape(rows, columns));
        expect(kernel.valid());
        for (auto enabled : {false, true}) {
            auto native = bridge::tirx::lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            // A square global projection exercises a transposed *destination*,
            // not an extra materialized transpose tile. Pinning C simulates the
            // hard allocation annotation the manual-memory API preserves.
            AdjustAccumulatorTest adjust{native.value, test.transpose_output, test.pin_initial && runtime.target() == "metal"};
            native.value.CopyOnWrite()->body = adjust(native.value->body);
            bridge::tirx::PlannerOptions planner;
            planner.direct_accumulator_store = enabled;
            auto executable = compile_native(runtime, kernel, std::move(native.value), 32u, 256u, planner);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            if (runtime.target() == "metal") {
                auto expected = enabled && cfg.k > 0 && cfg.m % cfg.bm == 0 && cfg.n % cfg.bn == 0 &&
                                test.row_offset >= 0 && test.column_offset >= 0 && !test.observe_accumulator && !test.pin_initial;
                auto direct = false;
                for (auto &plan : executable.plans) {
                    for (auto &matrix : plan.matrices) { direct |= matrix.direct_accumulator_store; }
                }
                expect(eq(direct, expected)) << cfg.m << "x" << cfg.n << "x" << cfg.k << " offset=" << test.row_offset << "," << test.column_offset;
                if (direct) {
                    auto source = metal_source(executable.module.value());
                    auto code = std::string_view{source.data(), source.size()};
                    auto store = code.rfind("simdgroup_store(");
                    expect(store != std::string_view::npos) << code;
                    if (store != std::string_view::npos) { expect(code.substr(store, code.find('\n', store) - store).find("_shared") == std::string_view::npos) << code; }
                    expect(executable.plans[0].cost.direct_fragment_stores > 0.0);
                }
            }
            for (auto repeat = 0u; repeat < 2u; repeat++) {
                auto a = values(cfg.m * input_k, 0.13f + static_cast<float>(repeat) * 0.27f);
                auto b = values(input_k * cfg.n, 0.47f - static_cast<float>(repeat) * 0.33f);
                vector<float> sentinel(rows * columns, -17.25f);
                auto left = runtime.upload<float>({cfg.m, input_k}, a);
                auto right = runtime.upload<float>({input_k, cfg.n}, b);
                auto destination = runtime.upload<float>({rows, columns}, sentinel);
                auto history = runtime.upload<float>({rows, columns}, sentinel);
                (*executable.entry)(left, right, destination, history);
                auto actual = runtime.download<float>(destination, rows * columns);
                auto observed = runtime.download<float>(history, rows * columns);
                auto valid = true;
                for (auto row = int64_t{0}; row < rows; row++) {
                    for (auto column = int64_t{0}; column < columns; column++) {
                        auto m = (test.transpose_output ? column : row) - test.row_offset;
                        auto n = (test.transpose_output ? row : column) - test.column_offset;
                        auto written = m >= 0 && m < ceil_div(cfg.m, cfg.bm) * cfg.bm && n >= 0 && n < ceil_div(cfg.n, cfg.bn) * cfg.bn;
                        auto expected = written ? 0.375 : -17.25;
                        if (written && m < cfg.m && n < cfg.n) {
                            for (auto k = int64_t{0}; k < cfg.k; k++) { expected += static_cast<double>(a[m * input_k + k]) * b[k * cfg.n + n]; }
                        }
                        auto index = row * columns + column;
                        valid &= std::isfinite(actual[index]) && std::abs(static_cast<double>(actual[index]) - expected) <= 1e-4 + 2e-5 * std::abs(expected);
                        auto history_value = test.observe_accumulator ? expected : -17.25;
                        valid &= std::isfinite(observed[index]) && std::abs(static_cast<double>(observed[index]) - history_value) <= 1e-4 + 2e-5 * std::abs(history_value);
                    }
                }
                expect(valid) << "direct output, bounds, padding, initializer, and original sink order must be preserved";
            }
        }
    }
}

class AliasMatrixOperandToCarry final : public tvm::tirx::StmtMutator {
private:
    bool _left;
    tvm::tirx::BufferVar _carry;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto previous = _carry;
        if (loop->annotations.count("luisa.tile.mma")) {
            auto column = loop->body.as<tvm::tirx::ForNode>();
            auto sequence = column->body.as<tvm::tirx::SeqStmtNode>();
            auto initial = sequence->seq[0].as<tvm::tirx::BufferStoreNode>();
            _carry = initial->value.as<tvm::tirx::BufferLoadNode>()->buffer;
        }
        auto result = StmtMutator::VisitStmt_(loop);
        _carry = previous;
        return result;
    }
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::BufferStoreNode *store) final {
        auto result = StmtMutator::VisitStmt_(store).as_or_throw<tvm::tirx::BufferStore>();
        auto sum = result->value.as<tvm::tirx::AddNode>();
        if (_carry.defined() && sum != nullptr) {
            auto product = sum->b.as<tvm::tirx::MulNode>();
            if (product != nullptr) {
                auto source = (_left ? product->a : product->b).as<tvm::tirx::BufferLoadNode>();
                auto alias = tvm::tirx::BufferLoad{_carry, source->indices};
                result.CopyOnWrite()->value = sum->a + (_left ? alias * product->b : product->a * alias);
                replacements++;
            }
        }
        return result;
    }

public:
    uint32_t replacements{0u};
    explicit AliasMatrixOperandToCarry(bool left) noexcept : _left{left} {}
    using StmtMutator::operator();
};

void test_accumulator_as_multiplicand(Runtime &runtime) {
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    for (auto self_left : {false, true}) {
        auto kernel = tile_kernel("matrix_carry_operand_alias", [=](TensorView<const float, 2> X, TensorView<float, 2> D) {
                          auto m = axis("m", 8);
                          auto n = axis("n", 8);
                          for (auto &nest : parallel(shape(1), scope)) {
                              auto acc = full<float>(shape(m, n), 0.25f);
                              for (auto &step : nest.pipeline(shape(3), {.stages = 1u})) {
                                  step.stage("load");
                                  auto x = X[coord(step.index() * 8, 0), shape(m, n)];
                                  step.stage("compute");
                                  acc = mma(x, x, acc);
                              }
                              D(coord(0, 0), shape(m, n)).store(acc);
                          }
                      }).capture(tensor_shape(24, 8), tensor_shape(8, 8));
        for (auto [retain, direct] : {std::pair{false, false}, std::pair{true, false}, std::pair{true, true}}) {
            bridge::tirx::PlannerOptions planner;
            planner.retain_accumulators = retain;
            planner.direct_accumulator_store = direct;
            auto native = bridge::tirx::lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            // Frontend value snapshots can introduce an extra operand copy.
            // Construct the legal alias directly, as a later native transform
            // may do, so no incidental copy protects an insufficient proof.
            AliasMatrixOperandToCarry alias{self_left};
            native.value.CopyOnWrite()->body = alias(native.value->body);
            expect(eq(alias.replacements, 1u));
            auto executable = compile_native(runtime, kernel, std::move(native.value), 32u, 256u, planner);
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            for (auto &plan : executable.plans) {
                for (auto &matrix : plan.matrices) { expect(!matrix.persistent_accumulator && !matrix.direct_accumulator_store); }
            }
            for (auto value : {0.0625f, 0.125f}) {
                auto x = runtime.upload<float>({24, 8}, vector<float>(24u * 8u, value));
                auto d = runtime.allocate<float>({8, 8});
                (*executable.entry)(x, d);
                auto actual = runtime.download<float>(d, 64u);
                auto expected = 0.25f * std::pow(1.0f + 8.0f * value, 3.0f);
                expect(std::all_of(actual.begin(), actual.end(), [=](float v) { return std::isfinite(v) && std::abs(v - expected) < 1e-5f; }))
                    << "a multiplicand must observe the new accumulator on every iteration";
            }
        }
    }
}

class InterruptMatrixYield final : public tvm::tirx::StmtMutator {
private:
    bool _break;
    tvm::tirx::PrimVar _iteration;

protected:
    [[nodiscard]] tvm::tirx::Stmt VisitStmt_(const tvm::tirx::ForNode *loop) final {
        auto previous = _iteration;
        if (loop->annotations.count("luisa.tile.pipeline")) { _iteration = loop->loop_var; }
        auto result = StmtMutator::VisitStmt_(loop);
        if (_iteration.defined() && loop->annotations.count("luisa.tile.mma")) {
            tvm::tirx::Stmt control = _break ? tvm::tirx::Stmt{tvm::tirx::Break{tvm::Span{}}} : tvm::tirx::Stmt{tvm::tirx::Continue{tvm::Span{}}};
            result = tvm::tirx::SeqStmt::Flatten(tvm::ffi::Array<tvm::tirx::Stmt>{result, tvm::tirx::IfThenElse{tvm::equal(_iteration, tvm::IntImm::Int64(1)), control}});
            replacements++;
        }
        _iteration = previous;
        return result;
    }

public:
    uint32_t replacements{0u};
    explicit InterruptMatrixYield(bool do_break) noexcept : _break{do_break} {}
    using StmtMutator::operator();
};

void test_interrupted_accumulator_update(Runtime &runtime) {
    auto scope = runtime.target() == "metal" ? exec::Scope::GROUP : exec::Scope::WORKER;
    auto kernel = tile_kernel("matrix_interrupted_yield", [=](TensorView<const float, 2> X, TensorView<float, 2> D) {
                      auto m = axis("m", 8);
                      auto n = axis("n", 8);
                      for (auto &nest : parallel(shape(1), scope)) {
                          auto acc = full<float>(shape(m, n), 0.25f);
                          for (auto &step : nest.pipeline(shape(3), {.stages = 1u})) {
                              step.stage("load");
                              auto x = X[coord(step.index() * 8, 0), shape(m, n)];
                              step.stage("compute");
                              acc = mma(x, x, acc);
                          }
                          D(coord(0, 0), shape(m, n)).store(acc);
                      }
                  }).capture(tensor_shape(24, 8), tensor_shape(8, 8));
    for (auto do_break : {false, true}) {
        for (auto retain : {false, true}) {
            auto native = bridge::tirx::lower(kernel.function());
            expect(native.ok()) << native.error;
            if (!native) { continue; }
            InterruptMatrixYield interrupt{do_break};
            native.value.CopyOnWrite()->body = interrupt(native.value->body);
            expect(eq(interrupt.replacements, 1u));
            bridge::tirx::PlannerOptions planner;
            planner.retain_accumulators = retain;
            auto executable = compile_native(runtime, kernel, std::move(native.value), 32u, 256u, planner);
            if (runtime.target() == "llvm" && !executable.ok()) {
                // The pinned TVMx LLVM emitter has no Break/Continue visitors.
                // Check that known rejection, not a blanket compile-failure
                // exemption. If upstream adds support, exercise the numerical
                // oracle below on CPU too.
                auto unsupported = do_break ? "Do not have a default for tirx.Break" : "Do not have a default for tirx.Continue";
                expect(executable.error.find(unsupported) != std::string::npos) << executable.error;
                continue;
            }
            expect(executable.ok()) << executable.error;
            if (!executable.ok()) { continue; }
            for (auto &plan : executable.plans) {
                for (auto &matrix : plan.matrices) { expect(!matrix.persistent_accumulator && !matrix.direct_accumulator_store); }
            }
            auto x = runtime.upload<float>({24, 8}, vector<float>(24u * 8u, 0.125f));
            auto d = runtime.allocate<float>({8, 8});
            (*executable.entry)(x, d);
            auto actual = runtime.download<float>(d, 64u);
            auto expected = do_break ? 0.375f : 0.5f;
            expect(std::all_of(actual.begin(), actual.end(), [=](float v) { return std::isfinite(v) && std::abs(v - expected) < 1e-5f; }))
                << "MMA result discarded by break/continue must not update the carry";
        }
    }
}

[[nodiscard]] Kernel whole_gemm_kernel(Shape cfg, exec::Scope scope = exec::Scope::AUTOMATIC,
                                       bool epilogue = false) {
    auto definition = tile_kernel("whole_matrix_gemm", [=](TensorView<const float, 2> A,
                                                           TensorView<const float, 2> B,
                                                           TensorView<float, 2> C) {
        auto gm = axis("whole_gm", ceil_div(cfg.m, cfg.bm));
        auto gn = axis("whole_gn", ceil_div(cfg.n, cfg.bn));
        auto m = axis("whole_m", cfg.bm);
        auto n = axis("whole_n", cfg.bn);
        auto k = axis("whole_k", cfg.bk);
        for (auto &nest : parallel(shape(gm, gn), scope)) {
            auto m0 = nest.index(gm) * cfg.bm;
            auto n0 = nest.index(gn) * cfg.bn;
            auto accumulator = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(ceil_div(cfg.k, cfg.bk)), {.stages = 2u})) {
                auto k0 = step.index() * cfg.bk;
                step.stage("load");
                auto a = A[coord(m0, k0), shape(m, k)];
                auto b = B[coord(k0, n0), shape(k, n)];
                step.stage("compute");
                accumulator = mma(a, b, accumulator, cfg.math);
            }
            if (epilogue) { accumulator = accumulator + full<float>(shape(m, n), 1.0f); }
            C(coord(m0, n0), shape(m, n)).store(accumulator);
        }
    });
    return definition.capture(tensor_shape(cfg.m, cfg.k), tensor_shape(cfg.k, cfg.n),
                              tensor_shape(cfg.m, cfg.n));
}

void test_cpu_whole_gemm_library_realization(Runtime &runtime) {
    if (runtime.target() != "llvm") { return; }
    using bridge::tirx::CpuMatrixBackend;
    Shape cfg{17, 19, 13, 4, 16, 8};
    auto kernel = whole_gemm_kernel(cfg);
    auto native = bridge::tirx::lower(kernel.function());
    expect(native.ok()) << native.error;
    if (!native) { return; }
    auto version = native.value->GetAttr<int64_t>("luisa.tile.contract.whole_gemm");
    expect(version.has_value() && version.value() == 1);

    auto aliases = runtime.build(kernel, false, false, true, false, {}, false, false,
                                 CpuMatrixBackend::CBLAS);
    expect(!aliases.ok());
    expect(aliases.error.find("noalias") != luisa::string::npos) << aliases.error;

    auto expect_rejected = [&](const Kernel &rejected) {
        auto lowered = bridge::tirx::lower(rejected.function());
        expect(lowered.ok()) << lowered.error;
        if (lowered) {
            expect(!lowered.value->GetAttr<int64_t>("luisa.tile.contract.whole_gemm"));
        }
        auto executable = runtime.build(rejected, true, false, true, false, {}, false, false,
                                        CpuMatrixBackend::CBLAS);
        expect(!executable.ok());
        expect(executable.error.find("whole-GEMM") != luisa::string::npos) << executable.error;
    };
    expect_rejected(whole_gemm_kernel(cfg, exec::Scope::WORKER));
    expect_rejected(whole_gemm_kernel(cfg, exec::Scope::AUTOMATIC, true));
    auto ordered = cfg;
    ordered.math.allow_reassociation = false;
    expect_rejected(whole_gemm_kernel(ordered));

    auto provider = tvm::ffi::Function::GetGlobal("tvm.contrib.cblas.matmul");
    auto executable = runtime.build(kernel, true, false, true, false, {}, false, false,
                                    CpuMatrixBackend::CBLAS);
    if (!provider) {
        expect(!executable.ok());
        expect(executable.error.find("not registered") != luisa::string::npos) << executable.error;
        return;
    }
    expect(executable.ok()) << executable.error;
    if (!executable.ok()) { return; }
    auto source = executable.module.value()->InspectSource("ll");
    expect(std::string_view{source.data(), source.size()}.find("tvm.contrib.cblas.matmul") !=
           std::string_view::npos);
    auto a_values = values(static_cast<size_t>(cfg.m * cfg.k), 0.13f);
    auto b_values = values(static_cast<size_t>(cfg.k * cfg.n), 0.47f);
    auto a = runtime.upload<float>({cfg.m, cfg.k}, a_values);
    auto b = runtime.upload<float>({cfg.k, cfg.n}, b_values);
    auto c = runtime.allocate<float>({cfg.m, cfg.n});
    (*executable.entry)(a, b, c);
    auto actual = runtime.download<float>(c, static_cast<size_t>(cfg.m * cfg.n));
    auto valid = true;
    for (auto row = int64_t{0}; row < cfg.m; row++) {
        for (auto column = int64_t{0}; column < cfg.n; column++) {
            auto expected = 0.0;
            for (auto inner = int64_t{0}; inner < cfg.k; inner++) {
                expected += static_cast<double>(a_values[row * cfg.k + inner]) *
                            static_cast<double>(b_values[inner * cfg.n + column]);
            }
            auto value = actual[row * cfg.n + column];
            valid &= std::isfinite(value) &&
                     std::abs(static_cast<double>(value) - expected) <=
                         1e-4 + 1e-4 * std::abs(expected);
        }
    }
    expect(valid);
}

}// namespace

int main(int argc, char *argv[]) {
    Runtime runtime{argc > 1 ? argv[1] : "cpu"};
    boost::ut::detail::cfg::parse_arg_with_fallback(argc > 1 ? argc - 1 : argc,
                                                    const_cast<const char **>(argc > 1 ? argv + 1 : argv));
    "tile_matrix_shapes_transposes_and_pipeline_versions"_test = [&] { test_matrix_cases(runtime); };
    "tile_matrix_math_policy_and_participant_gates"_test = [&] { test_matrix_policy_and_participants(runtime); };
    "tile_matrix_exact_threads_use_actual_target_limit"_test = [&] { test_explicit_threads_use_target_capacity(runtime); };
    "tile_matrix_mpp_memory_inputs_and_nonzero_accumulator"_test = [&] { test_mpp_memory_realization(runtime); };
    "tile_matrix_mpp_typed_contract_and_rejections"_test = [&] { test_mpp_typed_contract(runtime); };
    "tile_matrix_mpp_readonly_view_proofs"_test = [&] { test_mpp_readonly_views(runtime); };
    "tile_matrix_mpp_independent_subgroups_and_sync_boundaries"_test = [&] { test_mpp_subgroup_isolation(runtime); };
    "tile_matrix_stale_marker_does_not_tensorize"_test = [&] { test_stale_matrix_marker(runtime); };
    "tile_matrix_literal_initial_and_zero_contraction"_test = [&] { test_literal_initial_and_zero_contraction(runtime); };
    "tile_matrix_worker_local_is_not_a_collective"_test = [&] { test_worker_local_matrix_fallback(runtime); };
    "tile_matrix_mixed_conversion_keeps_reference_types"_test = [&] { test_mixed_input_matrix_fallback(runtime); };
    "tile_cpu_matrix_vectors_and_scalar_tails"_test = [&] { test_cpu_matrix_vectors_and_tails(runtime); };
    "tile_cpu_stack_storage"_test = [&] { test_cpu_stack_storage(runtime); };
    "tile_cpu_readonly_view_gemm"_test = [&] { test_cpu_readonly_view_gemm(runtime); };
    "tile_cpu_readonly_view_proofs"_test = [&] { test_cpu_readonly_view_proofs(runtime); };
    "tile_cpu_readonly_view_aliases"_test = [&] { test_cpu_readonly_view_aliases(runtime); };
    "tile_cpu_cartesian_packing"_test = [&] { test_cpu_cartesian_packing(runtime); };
    "tile_cpu_cartesian_transformed_domains"_test = [&] { test_cpu_cartesian_transformed_domains(runtime); };
    "tile_cpu_matrix_preserves_k_order"_test = [&] { test_cpu_matrix_preserves_k_order(runtime); };
    "tile_matrix_planner_emits_reused_fragments"_test = [&] { test_planned_fragment_reuse(runtime); };
    "tile_matrix_late_prefetch_resources_and_tails"_test = [&] { test_late_matrix_prefetch(runtime); };
    "tile_matrix_late_prefetch_rejects_global_mutation"_test = [&] { test_matrix_prefetch_rejects_global_writes(runtime); };
    "tile_matrix_observed_carry_cannot_be_promoted"_test = [&] { test_observed_accumulator_stays_visible(runtime); };
    "tile_matrix_direct_output_and_observation_guards"_test = [&] { test_direct_accumulator_output(runtime); };
    "tile_matrix_carry_operand_alias_must_not_be_promoted"_test = [&] { test_accumulator_as_multiplicand(runtime); };
    "tile_matrix_interrupted_yield_must_not_update_carry"_test = [&] { test_interrupted_accumulator_update(runtime); };
    "tile_cpu_whole_gemm_library_realization"_test = [&] { test_cpu_whole_gemm_library_realization(runtime); };
}

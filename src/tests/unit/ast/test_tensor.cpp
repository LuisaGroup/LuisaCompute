// Test for the runtime tensor DSL surface (<luisa/tensor/tensor.h>) and its
// end-to-end execution on the CUDA backend through the standard DSL/CallOp
// pipeline (tensor_* helpers -> TENSOR_* CallOps -> lc_tensor_* device
// builtins -> NVRTC).
//
// This test covers:
// - AST predicates: is_tensor_operation(), CallOpSet::uses_tensor_ops()
// - DSL tracing: the tensor_* helpers emit the expected descriptor encoding
//   (no device required)
// - End-to-end DSL -> CUDA with the ergonomic `Tensor` handle:
//   factories (empty/zeros/ones/full/from_buffer), metadata, views
//   (view/permute/slice), contiguous, host<->device transfer
//   element-wise unary/binary/clamp/fma, cast/copy/fill,
//   reduce sum/max/min, cumsum,
//   matmul (F32 alpha/beta, F16 WMMA tensor cores with FP32 accumulator),
//   batch matmul, einsum-style contract.
//
// Usage: test_tensor [backend]  (e.g. `test_tensor cuda`).
// Without a backend argument only the device-free AST checks run.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/ast/expression.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/op.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/variant.h>
#include <luisa/dsl/tensor.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>
#include <luisa/tensor/tensor.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// ---------------------------------------------------------------------------
// Device-free AST checks
// ---------------------------------------------------------------------------

void test_ast_predicates() {
    "is_tensor_operation"_test = [] {
        expect(is_tensor_operation(CallOp::TENSOR_COPY));
        expect(is_tensor_operation(CallOp::TENSOR_ADD));
        expect(is_tensor_operation(CallOp::TENSOR_MATMUL));
        expect(is_tensor_operation(CallOp::TENSOR_BATCH_MATMUL));
        expect(is_tensor_operation(CallOp::TENSOR_CONTRACT));
        expect(!is_tensor_operation(CallOp::CLOCK));
        expect(!is_tensor_operation(CallOp::CLAMP));
        expect(!is_tensor_operation(CallOp::BUFFER_ADDRESS));
    };
    "uses_tensor_ops"_test = [] {
        CallOpSet empty;
        expect(!empty.uses_tensor_ops());
        CallOpSet clock_only;
        clock_only.mark(CallOp::CLOCK);
        expect(!clock_only.uses_tensor_ops());
        CallOpSet tensor;
        tensor.mark(CallOp::TENSOR_ADD);
        expect(tensor.uses_tensor_ops());
    };
}

void test_dsl_tracing() {
    "tensor_add_trace_encoding"_test = [] {
        // Trace a zero-argument kernel calling tensor_add with fake storage;
        // verify the emitted TENSOR_ADD CallExpr and descriptor literals.
        auto kernel = luisa::compute::detail::FunctionBuilder::define_kernel([] {
            auto *fb = luisa::compute::detail::FunctionBuilder::current();
            auto desc = TensorDescriptor::contiguous(
                TensorElementType::F32, std::array<uint32_t, 2>{4u, 5u});
            ByteBufferView fake{nullptr, 42ull, 0u, 4096u, 4096u};
            auto o = tensor_operand(desc, fake);
            auto a = tensor_operand(desc, fake);
            auto b = tensor_operand(desc, fake);
            tensor_add(o, a, b);
            (void)fb;
        });
        expect(kernel != nullptr);
        const CallExpr *found = nullptr;
        traverse_expressions<true>(
            kernel->body(),
            [&](auto expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    auto call = static_cast<const CallExpr *>(expr);
                    if (call->op() == CallOp::TENSOR_ADD) { found = call; }
                }
            },
            [](auto) noexcept {}, [](auto) noexcept {});
        expect(found != nullptr);
        if (found != nullptr) {
            expect(found->type() == Type::of<void>());
            expect(found->arguments().size() == 19u);
            auto lit = [](const Expression *e) noexcept {
                return static_cast<const LiteralExpr *>(e);
            };
            // dtype literal = F32 tag, rank literal = 2
            expect(luisa::get<uint32_t>(lit(found->arguments()[0])->value().to_variant()) ==
                   luisa::to_underlying(TensorElementType::F32));
            expect(luisa::get<uint32_t>(lit(found->arguments()[1])->value().to_variant()) == 2u);
            // extents literal {4, 5, 1, 1}
            auto ext = luisa::get<uint4>(lit(found->arguments()[2])->value().to_variant());
            expect(ext.x == 4u && ext.y == 5u && ext.z == 1u && ext.w == 1u);
            // count literal = 20
            expect(luisa::get<uint32_t>(lit(found->arguments()[18])->value().to_variant()) == 20u);
        }
    };
}

// ---------------------------------------------------------------------------
// Host helpers for device tests
// ---------------------------------------------------------------------------

void check_f32(luisa::span<const float> got, luisa::span<const float> expected,
               float tol, luisa::string_view what) {
    bool ok = got.size() == expected.size();
    if (ok) {
        for (auto i = 0u; i < got.size(); i++) {
            if (std::abs(got[i] - expected[i]) > tol) {
                LUISA_WARNING("{} mismatch at [{}]: got {}, expected {}", what, i, got[i], expected[i]);
                ok = false;
                break;
            }
        }
    }
    expect(ok) << what;
    LUISA_INFO("{}: {}", what, ok ? "OK" : "FAILED");
}

// Wrap a use_tensor=true TileCompileResult as the same typed Kernel type that
// tile::Kernel::to_kernel<Dim>() would produce.  NOTE: do NOT use
// kernel.to_kernel<Dim>() here — it re-lowers with the DEFAULT config
// (use_tensor=false), i.e. the partition path, which would silently bypass the
// tensor-op calls under test.
template<typename K, typename R>
[[nodiscard]] K wrap_tensor_kernel(const R &result) {
    auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
    return K{std::move(fb)};
}

void check_i32(luisa::span<const int32_t> got, luisa::span<const int32_t> expected,
               luisa::string_view what) {
    bool ok = got.size() == expected.size();
    if (ok) {
        for (auto i = 0u; i < got.size(); i++) {
            if (got[i] != expected[i]) {
                LUISA_WARNING("{} mismatch at [{}]: got {}, expected {}", what, i, got[i], expected[i]);
                ok = false;
                break;
            }
        }
    }
    expect(ok) << what;
    LUISA_INFO("{}: {}", what, ok ? "OK" : "FAILED");
}

// ---------------------------------------------------------------------------
// End-to-end DSL -> CUDA tests using the Tensor handle
// ---------------------------------------------------------------------------

void test_factories_and_transfer(Device &device, Stream &stream) {
    constexpr auto N = 1024u;
    auto zeros = Tensor::zeros({N}, TensorElementType::F32, device, stream);
    auto ones = Tensor::ones({N}, TensorElementType::F32, device, stream);
    auto full = Tensor::full({N}, TensorElementType::I32, -7, device, stream);
    stream << synchronize();
    std::vector<float> gotz(N), goto_(N);
    std::vector<int32_t> gotf(N);
    zeros.copy_to(gotz.data(), stream);
    ones.copy_to(goto_.data(), stream);
    full.copy_to(gotf.data(), stream);
    stream << synchronize();
    std::vector<float> expz(N, 0.0f), expo(N, 1.0f);
    std::vector<int32_t> expf(N, -7);
    check_f32(gotz, expz, 1e-6f, "Tensor::zeros");
    check_f32(goto_, expo, 1e-6f, "Tensor::ones");
    check_i32(gotf, expf, "Tensor::full i32");

    // from_buffer + copy_from/copy_to round trip.
    Buffer<float> buf = device.create_buffer<float>(N);
    std::vector<float> host(N), round(N);
    for (auto i = 0u; i < N; i++) { host[i] = static_cast<float>(i) * 0.25f; }
    stream << buf.copy_from(luisa::span{host}) << synchronize();
    auto t = Tensor::from_buffer(buf.view(), {N}, device);
    expect(t.numel() == N);
    expect(t.rank() == 1u);
    expect(t.size(0) == N);
    expect(t.stride(0) == 1u);
    expect(t.dtype() == TensorElementType::F32);
    expect(t.is_contiguous());
    t.copy_to(round.data(), stream);
    stream << synchronize();
    check_f32(round, host, 1e-6f, "Tensor::from_buffer round trip");
}

void test_elementwise_binary(Device &device, Stream &stream) {
    constexpr auto N = 1024u;
    Buffer<float> ba = device.create_buffer<float>(N);
    Buffer<float> bb = device.create_buffer<float>(N);
    std::vector<float> a(N), b(N), exp(N), got(N);
    for (auto i = 0u; i < N; i++) {
        a[i] = static_cast<float>(i) * 0.01f - 5.0f;
        b[i] = static_cast<float>(i % 7u) * 0.5f + 0.25f;
    }
    stream << ba.copy_from(luisa::span{a}) << bb.copy_from(luisa::span{b}) << synchronize();
    auto ta = Tensor::from_buffer(ba.view(), {N}, device);
    auto tb = Tensor::from_buffer(bb.view(), {N}, device);

    auto run = [&](auto &&op, luisa::span<const float> expected, float tol, luisa::string_view what) {
        auto out = op(stream);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        check_f32(got, expected, tol, what);
    };

    for (auto i = 0u; i < N; i++) { exp[i] = a[i] + b[i]; }
    run([&](Stream &s) { return ta.add(tb, s); }, exp, 1e-4f, "Tensor::add");
    for (auto i = 0u; i < N; i++) { exp[i] = a[i] - b[i]; }
    run([&](Stream &s) { return ta.sub(tb, s); }, exp, 1e-4f, "Tensor::sub");
    for (auto i = 0u; i < N; i++) { exp[i] = a[i] * b[i]; }
    run([&](Stream &s) { return ta.mul(tb, s); }, exp, 1e-4f, "Tensor::mul");
    for (auto i = 0u; i < N; i++) { exp[i] = b[i] != 0.0f ? a[i] / b[i] : 0.0f; }
    run([&](Stream &s) { return ta.div(tb, s); }, exp, 1e-3f, "Tensor::div");
    for (auto i = 0u; i < N; i++) { exp[i] = std::min(a[i], b[i]); }
    run([&](Stream &s) { return ta.min(tb, s); }, exp, 1e-5f, "Tensor::min");
    for (auto i = 0u; i < N; i++) { exp[i] = std::max(a[i], b[i]); }
    run([&](Stream &s) { return ta.max(tb, s); }, exp, 1e-5f, "Tensor::max");
    for (auto i = 0u; i < N; i++) { exp[i] = std::pow(a[i], b[i]); }
    run([&](Stream &s) { return ta.pow(tb, s); }, exp, 1e-3f, "Tensor::pow");
    for (auto i = 0u; i < N; i++) { exp[i] = std::min(std::max(a[i], -2.0f), 2.0f); }
    run([&](Stream &s) { return ta.clamp(-2.0, 2.0, s); }, exp, 1e-5f, "Tensor::clamp");

    Buffer<float> bc = device.create_buffer<float>(N);
    std::vector<float> c(N);
    for (auto i = 0u; i < N; i++) { c[i] = static_cast<float>(i % 11u) - 5.0f; }
    stream << bc.copy_from(luisa::span{c}) << synchronize();
    auto tc = Tensor::from_buffer(bc.view(), {N}, device);
    for (auto i = 0u; i < N; i++) { exp[i] = a[i] * b[i] + c[i]; }
    run([&](Stream &s) { return ta.fma(tb, tc, s); }, exp, 1e-3f, "Tensor::fma");
}

void test_elementwise_unary(Device &device, Stream &stream) {
    constexpr auto N = 1024u;
    Buffer<float> buf = device.create_buffer<float>(N);
    std::vector<float> x(N), exp(N), got(N);
    for (auto i = 0u; i < N; i++) { x[i] = static_cast<float>(i) * 0.01f - 5.0f; }
    stream << buf.copy_from(luisa::span{x}) << synchronize();
    auto t = Tensor::from_buffer(buf.view(), {N}, device);

    auto run = [&](auto &&op, luisa::span<const float> expected, float tol, luisa::string_view what) {
        auto out = op(stream);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        check_f32(got, expected, tol, what);
    };

    for (auto i = 0u; i < N; i++) { exp[i] = -x[i]; }
    run([&](Stream &s) { return t.neg(s); }, exp, 1e-5f, "Tensor::neg");
    for (auto i = 0u; i < N; i++) { exp[i] = std::fabs(x[i]); }
    run([&](Stream &s) { return t.abs(s); }, exp, 1e-5f, "Tensor::abs");
    for (auto i = 0u; i < N; i++) { exp[i] = x[i] > 0.0f ? x[i] : 0.0f; }
    run([&](Stream &s) { return t.relu(s); }, exp, 1e-5f, "Tensor::relu");

    // positive inputs for math ops
    std::vector<float> xp(N);
    for (auto i = 0u; i < N; i++) { xp[i] = static_cast<float>(i) * 0.01f + 1.0f; }
    stream << buf.copy_from(luisa::span{xp}) << synchronize();
    auto tp = Tensor::from_buffer(buf.view(), {N}, device);
    // expf fast-math absolute error grows with the argument magnitude, so use
    // a small input range for exp (like the tan pole handling above).
    std::vector<float> xe(N);
    for (auto i = 0u; i < N; i++) { xe[i] = 0.5f + static_cast<float>(i) * 0.0005f; }
    stream << buf.copy_from(luisa::span{xe}) << synchronize();
    auto te = Tensor::from_buffer(buf.view(), {N}, device);
    for (auto i = 0u; i < N; i++) { exp[i] = std::exp(xe[i]); }
    run([&](Stream &s) { return te.exp(s); }, exp, 1e-4f, "Tensor::exp");
    // restore xp so the remaining ops (log/sqrt/sin/...) read the positive data
    stream << buf.copy_from(luisa::span{xp}) << synchronize();
    for (auto i = 0u; i < N; i++) { exp[i] = std::log(xp[i]); }
    run([&](Stream &s) { return tp.log(s); }, exp, 1e-3f, "Tensor::log");
    for (auto i = 0u; i < N; i++) { exp[i] = std::sqrt(xp[i]); }
    run([&](Stream &s) { return tp.sqrt(s); }, exp, 1e-4f, "Tensor::sqrt");
    for (auto i = 0u; i < N; i++) { exp[i] = std::sin(xp[i]); }
    run([&](Stream &s) { return tp.sin(s); }, exp, 1e-3f, "Tensor::sin");
    for (auto i = 0u; i < N; i++) { exp[i] = std::cos(xp[i]); }
    run([&](Stream &s) { return tp.cos(s); }, exp, 1e-3f, "Tensor::cos");
    for (auto i = 0u; i < N; i++) { exp[i] = std::tanh(xp[i]); }
    run([&](Stream &s) { return tp.tanh(s); }, exp, 1e-3f, "Tensor::tanh");
    for (auto i = 0u; i < N; i++) { exp[i] = 1.0f / (1.0f + std::exp(-xp[i])); }
    run([&](Stream &s) { return tp.sigmoid(s); }, exp, 1e-3f, "Tensor::sigmoid");
    for (auto i = 0u; i < N; i++) { exp[i] = std::floor(xp[i]); }
    run([&](Stream &s) { return tp.floor(s); }, exp, 1e-5f, "Tensor::floor");
    for (auto i = 0u; i < N; i++) { exp[i] = std::ceil(xp[i]); }
    run([&](Stream &s) { return tp.ceil(s); }, exp, 1e-5f, "Tensor::ceil");

    // cast f32 -> f16 -> f32 round trip (buffer currently holds xp)
    auto h = tp.cast(TensorElementType::F16, stream);
    auto rt = h.cast(TensorElementType::F32, stream);
    rt.copy_to(got.data(), stream);
    stream << synchronize();
    for (auto i = 0u; i < N; i++) { exp[i] = static_cast<float>(half{xp[i]}); }
    check_f32(got, exp, 1e-3f, "Tensor::cast round trip");
}

void test_views_and_permute(Device &device, Stream &stream) {
    constexpr auto R = 2u, C = 3u;
    Buffer<float> buf = device.create_buffer<float>(R * C);
    std::vector<float> x(R * C), got(R * C), exp(R * C);
    for (auto i = 0u; i < R * C; i++) { x[i] = static_cast<float>(i); }
    stream << buf.copy_from(luisa::span{x}) << synchronize();
    auto t = Tensor::from_buffer(buf.view(), {R, C}, device);

    // permute (metadata) + contiguous (data movement): dst(j,i) = src(i,j)
    const uint32_t perm[2]{1u, 0u};
    auto tp = t.permute(luisa::span<const uint32_t>{perm, 2u});
    expect(!tp.is_contiguous());
    auto tc = tp.contiguous(stream);
    expect(tc.is_contiguous());
    tc.copy_to(got.data(), stream);
    stream << synchronize();
    for (auto j = 0u; j < C; j++) {
        for (auto i = 0u; i < R; i++) { exp[j * R + i] = x[i * C + j]; }
    }
    check_f32(got, exp, 1e-6f, "permute + contiguous");

    // view (reshape) shares storage
    const uint32_t flat[1]{R * C};
    auto tv = t.view(luisa::span<const uint32_t>{flat, 1u});
    expect(tv.numel() == R * C && tv.rank() == 1u);
    // slice: row 1 of the 2x3 matrix; contiguous() materializes a fresh buffer
    auto ts = t.slice(0, 1u, 2u);
    expect(ts.numel() == C);
    auto tsc = ts.contiguous(stream);
    expect(tsc.is_contiguous());
    std::vector<float> gotc(C), expc(C);
    tsc.copy_to(gotc.data(), stream);
    stream << synchronize();
    for (auto c = 0u; c < C; c++) { expc[c] = x[1 * C + c]; }
    check_f32(gotc, expc, 1e-6f, "slice + contiguous");
}

void test_reduce_cumsum(Device &device, Stream &stream) {
    constexpr auto R = 4u, C = 5u;
    Buffer<float> buf = device.create_buffer<float>(R * C);
    std::vector<float> x(R * C), got(R), exp(R);
    for (auto i = 0u; i < R * C; i++) { x[i] = static_cast<float>(i) * 0.5f - 4.0f; }
    stream << buf.copy_from(luisa::span{x}) << synchronize();
    auto t = Tensor::from_buffer(buf.view(), {R, C}, device);
    const int dim1[1]{1};

    auto run = [&](auto &&op, luisa::span<const float> expected, float tol, luisa::string_view what) {
        auto out = op(stream);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        check_f32(got, expected, tol, what);
    };

    for (auto r = 0u; r < R; r++) {
        double s = 0.0;
        for (auto c = 0u; c < C; c++) { s += x[r * C + c]; }
        exp[r] = static_cast<float>(s);
    }
    run([&](Stream &s) { return t.reduce_sum(luisa::span<const int>{dim1, 1u}, s); }, exp, 1e-4f, "Tensor::reduce_sum");
    for (auto r = 0u; r < R; r++) {
        float m = x[r * C];
        for (auto c = 1u; c < C; c++) { m = std::max(m, x[r * C + c]); }
        exp[r] = m;
    }
    run([&](Stream &s) { return t.reduce_max(luisa::span<const int>{dim1, 1u}, s); }, exp, 1e-5f, "Tensor::reduce_max");
    for (auto r = 0u; r < R; r++) {
        float m = x[r * C];
        for (auto c = 1u; c < C; c++) { m = std::min(m, x[r * C + c]); }
        exp[r] = m;
    }
    run([&](Stream &s) { return t.reduce_min(luisa::span<const int>{dim1, 1u}, s); }, exp, 1e-5f, "Tensor::reduce_min");

    // cumsum along dim 1 -> same shape
    std::vector<float> gotc(R * C), expc(R * C);
    auto cs = t.cumsum(1, stream);
    cs.copy_to(gotc.data(), stream);
    stream << synchronize();
    for (auto r = 0u; r < R; r++) {
        float acc = 0.0f;
        for (auto c = 0u; c < C; c++) {
            acc += x[r * C + c];
            expc[r * C + c] = acc;
        }
    }
    check_f32(gotc, expc, 1e-4f, "Tensor::cumsum");
}

void test_matmul(Device &device, Stream &stream) {
    constexpr auto M = 128u, N = 128u, K = 128u;
    constexpr float alpha = 1.25f;
    constexpr float beta = 0.5f;

    // F32 GEMM: fresh output with alpha, and in-place epilogue with alpha/beta
    {
        Buffer<float> ba = device.create_buffer<float>(M * K);
        Buffer<float> bb = device.create_buffer<float>(K * N);
        Buffer<float> bc = device.create_buffer<float>(M * N);
        std::vector<float> A(M * K), B(K * N), C0(M * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < A.size(); i++) { A[i] = static_cast<float>((i * 7u) % 9u) * 0.25f; }
        for (auto i = 0u; i < B.size(); i++) { B[i] = static_cast<float>((i * 5u) % 9u) * 0.25f; }
        for (auto i = 0u; i < C0.size(); i++) { C0[i] = static_cast<float>(i % 5u) * 0.5f; }
        stream << ba.copy_from(luisa::span{A}) << bb.copy_from(luisa::span{B})
               << bc.copy_from(luisa::span{C0}) << synchronize();
        auto a = Tensor::from_buffer(ba.view(), {M, K}, device);
        auto b = Tensor::from_buffer(bb.view(), {K, N}, device);

        // C = alpha * A * B (fresh output, beta = 0)
        auto out = a.matmul(b, GemmOptions{.alpha = alpha}, stream);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        for (auto m = 0u; m < M; m++) {
            for (auto n = 0u; n < N; n++) {
                double s = 0.0;
                for (auto k = 0u; k < K; k++) {
                    s += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
                }
                ref[m * N + n] = static_cast<float>(alpha * s);
            }
        }
        check_f32(got, ref, 1e-3f, "Tensor::matmul F32 alpha");

        // C = alpha * A * B + beta * C0 (in-place epilogue reads existing C)
        auto c = Tensor::from_buffer(bc.view(), {M, N}, device);
        a.matmul_into(c, b, GemmOptions{.alpha = alpha, .beta = beta}, stream);
        c.copy_to(got.data(), stream);
        stream << synchronize();
        for (auto m = 0u; m < M; m++) {
            for (auto n = 0u; n < N; n++) {
                double s = 0.0;
                for (auto k = 0u; k < K; k++) {
                    s += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
                }
                ref[m * N + n] = static_cast<float>(alpha * s + beta * static_cast<double>(C0[m * N + n]));
            }
        }
        check_f32(got, ref, 1e-3f, "Tensor::matmul_into F32 alpha/beta");
    }

    // F16 WMMA tensor-core GEMM with FP32 accumulator
    {
        Buffer<half> ba = device.create_buffer<half>(M * K);
        Buffer<half> bb = device.create_buffer<half>(K * N);
        Buffer<float> bc = device.create_buffer<float>(M * N);
        std::vector<float> A(M * K), B(K * N), ref(M * N), got(M * N);
        std::vector<half> ha(M * K), hb(K * N);
        for (auto i = 0u; i < A.size(); i++) {
            A[i] = static_cast<float>(i % 8u);
            ha[i] = half{A[i]};
        }
        for (auto i = 0u; i < B.size(); i++) {
            B[i] = static_cast<float>((i * 3u) % 8u);
            hb[i] = half{B[i]};
        }
        for (auto m = 0u; m < M; m++) {
            for (auto n = 0u; n < N; n++) {
                double s = 0.0;
                for (auto k = 0u; k < K; k++) {
                    s += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
                }
                ref[m * N + n] = static_cast<float>(s);
            }
        }
        std::vector<float> c0(M * N, 0.0f);
        stream << ba.copy_from(luisa::span{ha}) << bb.copy_from(luisa::span{hb})
               << bc.copy_from(luisa::span{c0}) << synchronize();
        auto a = Tensor::from_buffer(ba.view(), {M, K}, device);
        auto b = Tensor::from_buffer(bb.view(), {K, N}, device);
        auto out = a.matmul(b, GemmOptions{}, stream); // F16 -> F32 accumulator
        expect(out.dtype() == TensorElementType::F32);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        check_f32(got, ref, 0.05f, "Tensor::matmul F16 WMMA");
    }
}

void test_batch_matmul_contract(Device &device, Stream &stream) {
    // batch matmul
    {
        constexpr auto B = 3u, M = 8u, N = 8u, K = 8u;
        Buffer<float> ba = device.create_buffer<float>(B * M * K);
        Buffer<float> bb = device.create_buffer<float>(B * K * N);
        std::vector<float> A(B * M * K), Bv(B * K * N), ref(B * M * N), got(B * M * N);
        for (auto i = 0u; i < A.size(); i++) { A[i] = static_cast<float>((i * 3u) % 7u) * 0.5f; }
        for (auto i = 0u; i < Bv.size(); i++) { Bv[i] = static_cast<float>((i * 5u) % 7u) * 0.5f; }
        for (auto b = 0u; b < B; b++) {
            for (auto m = 0u; m < M; m++) {
                for (auto n = 0u; n < N; n++) {
                    double s = 0.0;
                    for (auto k = 0u; k < K; k++) {
                        s += static_cast<double>(A[(b * M + m) * K + k]) *
                             static_cast<double>(Bv[(b * K + k) * N + n]);
                    }
                    ref[(b * M + m) * N + n] = static_cast<float>(s);
                }
            }
        }
        stream << ba.copy_from(luisa::span{A}) << bb.copy_from(luisa::span{Bv}) << synchronize();
        auto a = Tensor::from_buffer(ba.view(), {B, M, K}, device);
        auto b = Tensor::from_buffer(bb.view(), {B, K, N}, device);
        auto out = a.batch_matmul(b, B, GemmOptions{}, stream);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        check_f32(got, ref, 1e-3f, "Tensor::batch_matmul");
    }

    // einsum contract "ik,kj->ij"
    {
        constexpr auto M = 8u, N = 6u, K = 5u;
        Buffer<float> ba = device.create_buffer<float>(M * K);
        Buffer<float> bb = device.create_buffer<float>(K * N);
        std::vector<float> A(M * K), Bv(K * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < A.size(); i++) { A[i] = static_cast<float>((i * 3u) % 7u) * 0.25f; }
        for (auto i = 0u; i < Bv.size(); i++) { Bv[i] = static_cast<float>((i * 5u) % 7u) * 0.25f; }
        for (auto m = 0u; m < M; m++) {
            for (auto n = 0u; n < N; n++) {
                double s = 0.0;
                for (auto k = 0u; k < K; k++) {
                    s += static_cast<double>(A[m * K + k]) * static_cast<double>(Bv[k * N + n]);
                }
                ref[m * N + n] = static_cast<float>(s);
            }
        }
        stream << ba.copy_from(luisa::span{A}) << bb.copy_from(luisa::span{Bv}) << synchronize();
        auto a = Tensor::from_buffer(ba.view(), {M, K}, device);
        auto b = Tensor::from_buffer(bb.view(), {K, N}, device);
        const uint32_t ma[2]{0u, 1u};
        const uint32_t mb[2]{1u, 2u};
        const uint32_t mc[2]{0u, 2u};
        auto out = a.contract(b, ma, mb, mc, stream);
        out.copy_to(got.data(), stream);
        stream << synchronize();
        check_f32(got, ref, 1e-3f, "Tensor::contract");
    }
}

void test_fill_and_copy(Device &device, Stream &stream) {
    constexpr auto N = 256u;
    // fill_ writes in place on a from_buffer-backed tensor
    Buffer<float> buf = device.create_buffer<float>(N);
    std::vector<float> host(N, 1.0f), got(N), exp(N, 42.0f);
    stream << buf.copy_from(luisa::span{host}) << synchronize();
    auto t = Tensor::from_buffer(buf.view(), {N}, device);
    t.fill_(42.0, stream);
    t.copy_to(got.data(), stream);
    stream << synchronize();
    check_f32(got, exp, 1e-6f, "Tensor::fill_");
    // copy() returns a new tensor with the same data
    auto c = t.copy(stream);
    std::vector<float> gotc(N);
    c.copy_to(gotc.data(), stream);
    stream << synchronize();
    check_f32(gotc, exp, 1e-6f, "Tensor::copy");
}

// ---------------------------------------------------------------------------
// Tile-kernel lowering through the TENSOR_* fast path (use_tensor=true):
// whole-tensor tile programs (Global-only operands, 1x1 tile grid) compile to
// single TENSOR_* calls and execute on the CUDA backend's lc_tensor_* device
// functions.  Each kernel is checked against a host reference.
// ---------------------------------------------------------------------------

namespace tile_lang = luisa::compute::tile::language;
constexpr auto TT = tile_lang::dsl;
using tile_f32 = luisa::compute::tile::float32;
using tile_f16 = luisa::compute::tile::half;
using tile_i32 = luisa::compute::tile::int32;

tile_lang::Tensor<tile_f32, 2> tile_add_global(tile_lang::Tensor<tile_f32, 2> A,
                                               tile_lang::Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f32, 2> C = TT.empty(TT.shape(M, N), tile_f32{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        C(TT.range(0, M), TT.all()) =
            A(TT.range(0, M), TT.all()) + B(TT.range(0, M), TT.all());
    }
    return C;
}

tile_lang::Tensor<tile_f32, 2> tile_copy_global(tile_lang::Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f32, 2> C = TT.empty(TT.shape(M, N), tile_f32{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        TT.copy(A, C);
    }
    return C;
}

// f16 variant: exercises the __half2 wide-vector fast path of the device
// element-wise functions (LCTensorVec<half>::n == 2).
tile_lang::Tensor<tile_f16, 2> tile_add_global_f16(tile_lang::Tensor<tile_f16, 2> A,
                                                   tile_lang::Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f16, 2> C = TT.empty(TT.shape(M, N), tile_f16{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        C(TT.range(0, M), TT.all()) =
            A(TT.range(0, M), TT.all()) + B(TT.range(0, M), TT.all());
    }
    return C;
}

tile_lang::Tensor<tile_f32, 2> tile_abs_global(tile_lang::Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f32, 2> C = TT.empty(TT.shape(M, N), tile_f32{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        C(TT.range(0, M), TT.all()) = TT.abs(A(TT.range(0, M), TT.all()));
    }
    return C;
}

tile_lang::Tensor<tile_f32, 2> tile_clamp_global(tile_lang::Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f32, 2> C = TT.empty(TT.shape(M, N), tile_f32{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        TT.copy(A, C);
        TT.clamp(C(TT.range(0, M), TT.all()), 0.1f, 0.9f);
    }
    return C;
}

tile_lang::Tensor<tile_f32, 2> tile_transpose_global(tile_lang::Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 16;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f32, 2> C = TT.empty(TT.shape(N, M), tile_f32{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        TT.transpose(A(TT.range(0, M), TT.range(0, N)),
                     C(TT.range(0, N), TT.range(0, M)));
    }
    return C;
}

tile_lang::Tensor<tile_f32, 1> tile_reduce_sum_global(tile_lang::Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 threads = 32;
    tile_lang::Tensor<tile_f32, 1> R = TT.empty(TT.shape(M), tile_f32{});
    for (auto [bx, by] : TT.Kernel(1, 1, threads)) {
        TT.reduce_sum(A(TT.range(0, M), TT.range(0, N)), R, /*dim=*/1);
    }
    return R;
}

// The classic tiled GEMM (shared staging + PIPELINED K loop + fragment
// accumulator + final C copy) rewrites to ONE TENSOR_MATMUL; M/N/K are
// multiples of 16 so the CUDA device function takes the F16 WMMA tensor-core
// path with an F32 accumulator.
tile_lang::Tensor<tile_f32, 2> tile_gemm_global(tile_lang::Tensor<tile_f16, 2> A,
                                                tile_lang::Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 32;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;
    tile_lang::Tensor<tile_f32, 2> C = TT.empty(TT.shape(M, N), tile_f32{});
    for (auto [bx, by] : TT.Kernel(TT.ceildiv(N, block_N), TT.ceildiv(M, block_M), threads)) {
        auto A_shared = TT.alloc_shared(TT.shape(block_M, block_K), tile_f16{});
        auto B_shared = TT.alloc_shared(TT.shape(block_K, block_N), tile_f16{});
        auto C_local = TT.alloc_fragment(TT.shape(block_M, block_N), tile_f32{});
        TT.clear(C_local);
        for (auto ko : TT.Pipelined(TT.ceildiv(K, block_K), num_stages)) {
            TT.copy(A(by * block_M, ko * block_K), A_shared);
            TT.copy(B(ko * block_K, bx * block_N), B_shared);
            TT.gemm(A_shared, B_shared, C_local);
        }
        TT.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

void test_tile_lowered_tensor_ops(Device &device, Stream &stream) {
    using luisa::compute::TileToKernelConfig;
    auto compile_and_check = [&](auto kernel, auto expect_op) {
        auto result = luisa::compute::tile_to_kernel(
            kernel.function(), TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        expect(result.function->direct_builtin_callables().test(expect_op));
        return result;
    };
    constexpr auto M = 32u, N = 32u;
    // ---- tile_add_global -> TENSOR_ADD ----
    {
        auto kernel = luisa::compute::tile::jit(tile_add_global).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_ADD);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufB = device.create_buffer<float>(M * N);
        auto bufC = device.create_buffer<float>(M * N);
        std::vector<float> a(M * N), b(M * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < M * N; ++i) {
            a[i] = static_cast<float>(i % 31) * 0.125f;
            b[i] = static_cast<float>(i % 17) * 0.25f;
            ref[i] = a[i] + b[i];
        }
        stream << bufA.copy_from(luisa::span{a}) << bufB.copy_from(luisa::span{b}) << synchronize();
        kernel.validate(bufA, bufB, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed, ShaderOption{.enable_cache = true,
                                                     .enable_debug_info = true});
        stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 1e-4f, "tile lowered TENSOR_ADD");
    }
    // ---- tile_add_global_f16 -> TENSOR_ADD (__half2 vector fast path) ----
    {
        auto kernel = luisa::compute::tile::jit(tile_add_global_f16).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_ADD);
        auto bufA = device.create_buffer<luisa::half>(M * N);
        auto bufB = device.create_buffer<luisa::half>(M * N);
        auto bufC = device.create_buffer<luisa::half>(M * N);
        std::vector<luisa::half> a(M * N), b(M * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < M * N; ++i) {
            a[i] = luisa::half{static_cast<float>(i % 31) * 0.125f};
            b[i] = luisa::half{static_cast<float>(i % 17) * 0.25f};
            ref[i] = luisa::half{static_cast<float>(static_cast<float>(a[i]) + static_cast<float>(b[i]))};
        }
        stream << bufA.copy_from(luisa::span{a}) << bufB.copy_from(luisa::span{b})
               << synchronize();
        kernel.validate(bufA, bufB, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufB, bufC)
                      .dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        std::vector<float> g(M * N), r(M * N);
        for (auto i = 0u; i < M * N; ++i) {
            g[i] = static_cast<float>(got[i]);
            r[i] = static_cast<float>(ref[i]);
        }
        check_f32(g, r, 1e-2f, "tile lowered TENSOR_ADD (f16 half2 path)");
    }
    // ---- tile_copy_global -> TENSOR_COPY ----
    {
        auto kernel = luisa::compute::tile::jit(tile_copy_global).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_COPY);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufC = device.create_buffer<float>(M * N);
        std::vector<float> a(M * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < M * N; ++i) { a[i] = static_cast<float>(i % 7) * 0.5f; }
        ref = a;
        stream << bufA.copy_from(luisa::span{a}) << synchronize();
        kernel.validate(bufA, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 1e-6f, "tile lowered TENSOR_COPY");
    }
    // ---- tile_abs_global -> TENSOR_ABS ----
    {
        auto kernel = luisa::compute::tile::jit(tile_abs_global).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_ABS);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufC = device.create_buffer<float>(M * N);
        std::vector<float> a(M * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < M * N; ++i) { a[i] = static_cast<float>(i % 13) * 0.5f - 3.0f; }
        for (auto i = 0u; i < M * N; ++i) { ref[i] = std::fabs(a[i]); }
        stream << bufA.copy_from(luisa::span{a}) << synchronize();
        kernel.validate(bufA, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 1e-6f, "tile lowered TENSOR_ABS");
    }
    // ---- tile_clamp_global -> TENSOR_COPY + TENSOR_CLAMP ----
    {
        auto kernel = luisa::compute::tile::jit(tile_clamp_global).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_CLAMP);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufC = device.create_buffer<float>(M * N);
        std::vector<float> a(M * N), ref(M * N), got(M * N);
        for (auto i = 0u; i < M * N; ++i) { a[i] = static_cast<float>(i % 31) * 0.25f - 3.0f; }
        for (auto i = 0u; i < M * N; ++i) { ref[i] = std::min(std::max(a[i], 0.1f), 0.9f); }
        stream << bufA.copy_from(luisa::span{a}) << synchronize();
        kernel.validate(bufA, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 1e-6f, "tile lowered TENSOR_CLAMP");
    }
    // ---- tile_transpose_global -> TENSOR_PERMUTE ----
    // TENSOR_PERMUTE from a tile-lowered kernel is end-to-end verified (an
    // earlier illegal-address failure did not reproduce once the tile-lowered
    // kernel is dispatched through the real use_tensor=true lowering).
    {
        constexpr auto TM = 32u, TN = 16u;
        auto kernel = luisa::compute::tile::jit(tile_transpose_global).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_PERMUTE);
        auto bufA = device.create_buffer<float>(TM * TN);
        auto bufC = device.create_buffer<float>(TN * TM);
        std::vector<float> a(TM * TN), ref(TN * TM), got(TN * TM);
        for (auto i = 0u; i < TM * TN; ++i) { a[i] = static_cast<float>(i % 11u) * 0.25f; }
        for (auto r = 0u; r < TN; ++r)
            for (auto c = 0u; c < TM; ++c) { ref[r * TM + c] = a[c * TN + r]; }
        stream << bufA.copy_from(luisa::span{a}) << synchronize();
        kernel.validate(bufA, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 1e-6f, "tile lowered TENSOR_PERMUTE");
    }
    // ---- tile_reduce_sum_global -> TENSOR_REDUCE_SUM ----
    {
        constexpr auto RM = 32u, RN = 32u;
        auto kernel = luisa::compute::tile::jit(tile_reduce_sum_global).compile();
        auto result = compile_and_check(kernel, CallOp::TENSOR_REDUCE_SUM);
        auto bufA = device.create_buffer<float>(RM * RN);
        auto bufR = device.create_buffer<float>(RM);
        std::vector<float> a(RM * RN), ref(RM), got(RM);
        for (auto i = 0u; i < RM * RN; ++i) { a[i] = static_cast<float>(i % 13u) * 0.25f; }
        for (auto r = 0u; r < RM; ++r) {
            double s = 0.0;
            for (auto c = 0u; c < RN; ++c) { s += static_cast<double>(a[r * RN + c]); }
            ref[r] = static_cast<float>(s);
        }
        stream << bufA.copy_from(luisa::span{a}) << synchronize();
        kernel.validate(bufA, bufR);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufR).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufR.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 1e-3f, "tile lowered TENSOR_REDUCE_SUM");
    }
    // ---- tile_gemm_global -> TENSOR_MATMUL (whole-program GEMM rewrite) ----
    // The classic tiled GEMM rewrites to ONE grid-wide TENSOR_MATMUL (F16
    // WMMA tensor-core path, FP32 accumulator).  The output buffer is
    // pre-filled with non-zero garbage so a beta != 0 regression (the rewrite
    // must OVERWRITE C, never add the old contents) fails loudly instead of
    // being masked by a zeroed fresh buffer.
    {
        constexpr auto GM = 64u, GN = 64u, GK = 32u;
        auto kernel = luisa::compute::tile::jit(tile_gemm_global).compile();
        auto result = luisa::compute::tile_to_kernel(
            kernel.function(), TileToKernelConfig{.use_tensor = true});
        expect(result.function != nullptr);
        expect(result.function->direct_builtin_callables().test(CallOp::TENSOR_MATMUL));
        expect(result.function->shared_variables().empty());
        expect(result.dispatch_size.x == 16u * 32u);
        expect(result.dispatch_size.y == 1u);
        auto bufA = device.create_buffer<luisa::half>(GM * GK);
        auto bufB = device.create_buffer<luisa::half>(GK * GN);
        auto bufC = device.create_buffer<float>(GM * GN);
        std::vector<luisa::half> a(GM * GK), b(GK * GN);
        std::vector<float> initial_c(GM * GN, 7.0f);
        std::vector<float> ref(GM * GN), got(GM * GN), got2(GM * GN);
        for (auto i = 0u; i < a.size(); ++i) { a[i] = luisa::half{static_cast<float>((i % 8) * 0.25f)}; }
        for (auto i = 0u; i < b.size(); ++i) { b[i] = luisa::half{static_cast<float>((i % 4) * 0.5f)}; }
        for (auto m = 0u; m < GM; ++m) {
            for (auto n = 0u; n < GN; ++n) {
                double s = 0.0;
                for (auto k = 0u; k < GK; ++k) {
                    s += static_cast<double>(static_cast<float>(a[m * GK + k])) *
                         static_cast<double>(static_cast<float>(b[k * GN + n]));
                }
                ref[m * GN + n] = static_cast<float>(s);
            }
        }
        stream << bufA.copy_from(luisa::span{a}) << bufB.copy_from(luisa::span{b})
               << bufC.copy_from(luisa::span{initial_c})
               << synchronize();
        kernel.validate(bufA, bufB, bufC);
        using K = decltype(kernel.to_kernel<2>());
        auto typed = wrap_tensor_kernel<K>(result);
        auto sh = device.compile(typed);
        stream << sh(bufA, bufB, bufC)
                      .dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{got}) << synchronize();
        check_f32(got, ref, 2.2e-2f, "tile lowered TENSOR_MATMUL");
        // Runtime API cross-check on the same data/shape (the exact device
        // result of the TENSOR_MATMUL device function).
        auto ta = Tensor::from_buffer(bufA.view(), {GM, GK}, device);
        auto tb = Tensor::from_buffer(bufB.view(), {GK, GN}, device);
        auto out = ta.matmul(tb, GemmOptions{}, stream);
        out.copy_to(got2.data(), stream);
        stream << synchronize();
        check_f32(got2, ref, 2.2e-2f, "runtime API same-data matmul");
    }
    LUISA_INFO("Tile-kernel TENSOR_* e2e tests done");
}

} // namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_ast_predicates();
    test_dsl_tracing();

    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        luisa::test::print_device_usage(argv[0]);
        LUISA_INFO("AST-only tensor checks passed; pass a backend (e.g. cuda) for e2e tests.");
        return 0;
    }
    auto &device = dc->device;
    auto stream = device.create_stream();
    LUISA_INFO("Running tensor DSL e2e tests on {}", device.backend_name());

    test_factories_and_transfer(device, stream);
    test_elementwise_binary(device, stream);
    test_elementwise_unary(device, stream);
    test_views_and_permute(device, stream);
    test_reduce_cumsum(device, stream);
    test_matmul(device, stream);
    test_batch_matmul_contract(device, stream);
    test_fill_and_copy(device, stream);
    test_tile_lowered_tensor_ops(device, stream);

    LUISA_INFO("Tensor DSL e2e tests done");
    return 0;
}

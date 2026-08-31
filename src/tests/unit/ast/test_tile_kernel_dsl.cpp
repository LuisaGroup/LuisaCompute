// Test for the tile pseudo kernel in <luisa/dsl/tensor.h>.
// This test covers:
// - tile::Kernel (the tile-DSL analogue of luisa::compute::Kernel in
//   <luisa/dsl/func.h>) tracing a prim function like elementwise_add into a
//   luisa::compute::detail::TileFunctionBuilder
// - tile::Kernel accepting a lambda as well as a function pointer
// - every tile op (T.empty, T.ceildiv, T.Kernel, T.alloc_shared,
//   T.alloc_fragment, T.copy, whole-tile binary + tile-store) emitting the
//   matching TensorStmt into the builder, in program order
// - tile::jit(...).compile() routing through tile::Kernel and keeping the
//   traced builder in the compiled kernel
// - T.Pipelined emitting exactly ONE PipelinedStmt and tracing its body once
//   (guards F1: the range-for must iterate one representative step)
// - device.compile_tile(...) routing a compiled tile kernel through the new
//   TileShader interface; without a native tile backend this exercises the
//   tile_to_kernel + create_shader fallback and dispatches on real device
//   buffers (C = A + B).
//
// The host-side tests run without a device; the device-backed test needs a
// backend argument (e.g. `test_tile_kernel_dsl dx`).

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/tensor.h>
#include <luisa/runtime/tile_shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>

#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// `import tilelang.language as T` -> the constexpr `T` handle.
constexpr auto T = tile::language::dsl;
using namespace tile::language;

using tile_f16 = tile::half;
using tile_f32 = tile::float32;
using tile_i32 = tile::int32;

// Mirrors examples/compute/tensor_stub.cpp elementwise_add (smaller sizes).
Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 block_M = 8, block_N = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});

    // Grid is in blocks: T.Kernel(ceildiv(N, BN), ceildiv(M, BM), threads).
    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_N), tile_f32{});
        auto B_shared = T.alloc_shared(T.shape(block_M, block_N), tile_f32{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});

        T.copy(A(by * block_M, bx * block_N), A_shared);
        T.copy(B(by * block_M, bx * block_N), B_shared);

        // Whole-tile elementwise op -> TileBinaryStmt + TileStoreStmt.
        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);

        T.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

// The lambda spelling of the same kernel (tile::Kernel accepts lambdas too).
auto elementwise_add_lambda = [](Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 32, N = 32;
    constexpr tile_i32 block_M = 8, block_N = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = T.empty(T.shape(M, N), tile_f32{});

    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_N), tile_f32{});
        auto B_shared = T.alloc_shared(T.shape(block_M, block_N), tile_f32{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});

        T.copy(A(by * block_M, bx * block_N), A_shared);
        T.copy(B(by * block_M, bx * block_N), B_shared);

        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);

        T.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
};

// Mirrors examples/compute/tensor_stub.cpp matmul, but with T.Pipelined —
// guards F1: the range-for must emit exactly ONE PipelinedStmt and trace its
// body exactly once (a real lowering would unroll/pipeline all `count` K
// steps, but the traced IR holds one representative copy/copy/gemm trip).
Tensor<tile_f16, 2> pipelined_matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 32;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;

    Tensor<tile_f16, 2> C = T.empty(T.shape(M, N), tile_f16{});

    for (auto [bx, by] : T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads)) {
        auto A_shared = T.alloc_shared(T.shape(block_M, block_K), tile_f16{});
        auto B_shared = T.alloc_shared(T.shape(block_K, block_N), tile_f16{});
        auto C_local = T.alloc_fragment(T.shape(block_M, block_N), tile_f32{});

        T.clear(C_local);

        for (auto ko : T.Pipelined(T.ceildiv(K, block_K), num_stages)) {
            T.copy(A(by * block_M, ko * block_K), A_shared);
            T.copy(B(ko * block_K, bx * block_N), B_shared);
            T.gemm(A_shared, B_shared, C_local);
        }
    }
    return C;
}

// Kernel whose two argument tensors are distinguishable (F32 rank-2 A,
// I32 rank-1 B).  Regression test for the argument-order bug: the argument
// AllocStmts must be emitted in declaration order (A then B).  Previously the
// DSL used std::make_tuple(make_kernel_arg<Args>()...), whose pack expansion
// has an unspecified evaluation order (MSVC evaluates right-to-left), so the
// buffer arguments of two-argument kernels came out swapped — elementwise add
// masked it (A+B is commutative), but matmul (A*B) produced wrong results.
Tensor<tile_i32, 1> two_arg_order_kernel(Tensor<tile_f32, 2> A, Tensor<tile_i32, 1> B) {
    Tensor<tile_i32, 1> C = T.empty(T.shape(4), tile_i32{});
    for (auto bx : T.Kernel(1, 4)) {
        T.copy(B(0), C(0));
    }
    return C;
}

// Kernel exercising the gap-analysis ops (FILL / TRANSPOSE / CLAMP / ATOMIC /
// SYNC / WARP_REDUCE / LOOP_BREAK) — they must emit the matching TileOpKind
// statements in program order.
Tensor<tile_i32, 1> gap_ops_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    Tensor<tile_i32, 1> D = T.empty(T.shape(32), tile_i32{});
    for (auto [bx, by] : T.Kernel(1, 1, 32)) {
        auto A_shared = T.alloc_shared(T.shape(BM, BN), tile_f32{});
        auto T_shared = T.alloc_shared(T.shape(BN, BM), tile_f32{});
        auto F = T.alloc_fragment(T.shape(32), tile_f32{});
        auto v = T.alloc_fragment(T.shape(1), tile_f32{});
        T.copy(A(by * BM, bx * BN), A_shared(BM, BN));
        T.transpose(A_shared(BM, BN), T_shared(BN, BM));
        T.sync_threads();
        T.fill(F(32), 3.5f);
        T.clamp(F(32), 0.1f, 0.9f);
        T.warp_reduce_sum(F(32));
        T.fill(v(1), 7.0f);
        T.warp_reduce_max(v(1));
        T.atomic_store(D, 5);
        T.atomic_add(D, 2);
        T.atomic_max(D, 3);
        T.atomic_min(D, 4);
        T.atomic_or(D, 8);
        T.loop_break();
    }
    return D;
}

// Kernel using the quantized dtype handles (int8 / fp8): the DSL must map
// them to the matching R1 TensorElementType tags (I8 / FP8) on the traced
// AllocStmt operands.
Tensor<tile::int8, 1> quantized_copy_kernel(Tensor<tile::int8, 1> A, Tensor<tile::fp8, 1> B) {
    Tensor<tile::int8, 1> C = T.empty(T.shape(8), tile::int8{});
    Tensor<tile::fp8, 1> D = T.empty(T.shape(8), tile::fp8{});
    for (auto bx : T.Kernel(1, 8)) {
        T.copy(A(0), C(0));
        T.copy(B(0), D(0));
    }
    return C;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "kernel_traces_function_into_tile_function_builder"_test = [] {
        tile::Kernel kernel{elementwise_add};
        auto builder = kernel.function();
        expect(builder != nullptr);
        auto statements = builder->body()->statements();

        const TileOpKind expected[] = {
            TileOpKind::ALLOC,    // A: kernel argument (global tensor)
            TileOpKind::ALLOC,    // B: kernel argument (global tensor)
            TileOpKind::ALLOC,    // C = T.empty(M, N)
            TileOpKind::CEILDIV,  // T.ceildiv(N, block_N)
            TileOpKind::CEILDIV,  // T.ceildiv(M, block_M)
            TileOpKind::KERNEL_2D,// T.Kernel(gx, gy, threads)
            TileOpKind::ALLOC,    // A_shared
            TileOpKind::ALLOC,    // B_shared
            TileOpKind::ALLOC,    // C_local
            TileOpKind::COPY,     // T.copy(A[...], A_shared)
            TileOpKind::COPY,     // T.copy(B[...], B_shared)
            TileOpKind::BINARY,   // A_shared + B_shared
            TileOpKind::STORE,    // C_local = A_shared + B_shared
            TileOpKind::COPY,     // T.copy(C_local, C[...])
        };
        expect(statements.size() == sizeof(expected) / sizeof(expected[0]));
        for (auto i = 0u; i < statements.size(); ++i) {
            expect(statements[i]->op() == expected[i]);
        }

        // spot-check a few statements
        auto *alloc = static_cast<const AllocStmt *>(statements[0]);
        expect(alloc->scope() == TensorScope::Global);
        auto *k2 = static_cast<const Kernel2DStmt *>(statements[5]);
        expect(k2->gx() == 4 && k2->gy() == 4 && k2->threads() == 32);
        auto *binary = static_cast<const TileBinaryStmt *>(statements[11]);
        expect(binary->op() == BinaryOp::ADD);
        expect(binary->lhs() != nullptr && binary->rhs_tensor() != nullptr);
        auto *store = static_cast<const TileStoreStmt *>(statements[12]);
        expect(store->op() == 0);
        expect(store->lhs() != nullptr && store->rhs_tensor() != nullptr);
    };

    "kernel_accepts_lambda"_test = [] {
        tile::Kernel kernel{elementwise_add_lambda};
        auto builder = kernel.function();
        expect(builder != nullptr);
        // same op sequence as the function spelling (14 = 2 arg allocs + body)
        auto statements = builder->body()->statements();
        expect(statements.size() == 14u);
        expect(statements[0]->op() == TileOpKind::ALLOC);
        expect(statements[5]->op() == TileOpKind::KERNEL_2D);
        expect(statements[12]->op() == TileOpKind::STORE);
    };

    "jit_compile_carries_builder"_test = [] {
        // Like tilelang, compile() takes no shape/tile parameters: they are
        // baked into the kernel function itself.
        auto compiled = tile::jit(elementwise_add).compile();
        expect(compiled.function() != nullptr);
        expect(compiled.function()->body()->size() == 14u);
        expect(!compiled.get_kernel_source().empty());
        // the compiled kernel stays callable (stub: no device execution)
        Tensor<tile_f32, 2> A{32, 32};
        Tensor<tile_f32, 2> B{32, 32};
        [[maybe_unused]] auto C = compiled(A, B);
    };

    "pipelined_iterates_one_representative_step"_test = [] {
        // F1: T.Pipelined(count, stages) must emit ONE PipelinedStmt and trace
        // its body exactly once — NOT `count` times.  Before the fix, matmul's
        // copy/copy/gemm trio was re-traced ceildiv(K, block_K) = 4 times and
        // PIPELINED appeared 4 times; now the body appears exactly once.
        tile::Kernel kernel{pipelined_matmul};
        auto builder = kernel.function();
        expect(builder != nullptr);
        auto statements = builder->body()->statements();

        const TileOpKind expected[] = {
            TileOpKind::ALLOC,     // A: kernel argument (global tensor)
            TileOpKind::ALLOC,     // B: kernel argument (global tensor)
            TileOpKind::ALLOC,     // C = T.empty(M, N)
            TileOpKind::CEILDIV,   // T.ceildiv(N, block_N)
            TileOpKind::CEILDIV,   // T.ceildiv(M, block_M)
            TileOpKind::KERNEL_2D, // T.Kernel(gx, gy, threads)
            TileOpKind::ALLOC,     // A_shared
            TileOpKind::ALLOC,     // B_shared
            TileOpKind::ALLOC,     // C_local
            TileOpKind::CLEAR,     // T.clear(C_local)
            TileOpKind::CEILDIV,   // T.ceildiv(K, block_K) (pipelined count)
            TileOpKind::PIPELINED, // T.Pipelined(count, stages) — exactly ONE
            TileOpKind::COPY,      // T.copy(A[...], A_shared)
            TileOpKind::COPY,      // T.copy(B[...], B_shared)
            TileOpKind::GEMM,      // T.gemm(A_shared, B_shared, C_local)
        };
        expect(statements.size() == sizeof(expected) / sizeof(expected[0]));
        for (auto i = 0u; i < statements.size(); ++i) {
            expect(statements[i]->op() == expected[i]);
        }

        auto *pipe = static_cast<const PipelinedStmt *>(statements[11]);
        expect(pipe->count() == 4); // ceildiv(32, 8)
        expect(pipe->stages() == 2);
        // the Pipelined body was traced exactly once (copy/copy/gemm present)
        auto *gemm = static_cast<const GemmStmt *>(statements[14]);
        expect(gemm->a() != nullptr && gemm->b() != nullptr && gemm->c() != nullptr);
    };

    "kernel_arguments_are_traced_in_declaration_order"_test = [] {
        // Regression: make_tuple-style pack expansion evaluated the kernel
        // argument factories in an unspecified order (right-to-left on MSVC),
        // swapping the buffer arguments of two-argument kernels.
        tile::Kernel kernel{two_arg_order_kernel};
        auto builder = kernel.function();
        auto statements = builder->body()->statements();
        auto *alloc_a = static_cast<const AllocStmt *>(statements[0]);
        auto *alloc_b = static_cast<const AllocStmt *>(statements[1]);
        expect(alloc_a->tensor()->dtype() == TensorElementType::F32);
        expect(alloc_a->tensor()->rank() == 2);
        expect(alloc_b->tensor()->dtype() == TensorElementType::I32);
        expect(alloc_b->tensor()->rank() == 1);
    };

    "gap_analysis_ops_emit_matching_statements"_test = [] {
        tile::Kernel kernel{gap_ops_kernel};
        auto builder = kernel.function();
        auto statements = builder->body()->statements();
        // A (f32 global, arg) and D (i32 global, T.empty) are the first two
        // allocs; the rest follows the kernel body in program order.
        const TileOpKind expected[] = {
            TileOpKind::ALLOC,      // A (f32 global, kernel arg)
            TileOpKind::ALLOC,      // D = T.empty(32) (i32 global)
            TileOpKind::KERNEL_2D,  // T.Kernel(1, 1, 32)
            TileOpKind::ALLOC,      // A_shared
            TileOpKind::ALLOC,      // T_shared
            TileOpKind::ALLOC,      // F
            TileOpKind::ALLOC,      // v
            TileOpKind::COPY,       // A -> A_shared
            TileOpKind::TRANSPOSE,  // A_shared -> T_shared
            TileOpKind::SYNC,       // T.sync_threads()
            TileOpKind::FILL,       // T.fill(F, 3.5)
            TileOpKind::CLAMP,      // T.clamp(F, 0.1, 0.9)
            TileOpKind::WARP_REDUCE,// T.warp_reduce_sum(F)
            TileOpKind::FILL,       // T.fill(v, 7.0)
            TileOpKind::WARP_REDUCE,// T.warp_reduce_max(v)
            TileOpKind::ATOMIC,     // T.atomic_store(D, 5)
            TileOpKind::ATOMIC,     // T.atomic_add(D, 2)
            TileOpKind::ATOMIC,     // T.atomic_max(D, 3)
            TileOpKind::ATOMIC,     // T.atomic_min(D, 4)
            TileOpKind::ATOMIC,     // T.atomic_or(D, 8)
            TileOpKind::LOOP_BREAK, // T.loop_break()
        };
        expect(statements.size() == sizeof(expected) / sizeof(expected[0]));
        for (auto i = 0u; i < statements.size(); ++i) {
            expect(statements[i]->op() == expected[i]);
        }
    };

    "quantized_dtypes_trace_to_matching_tags"_test = [] {
        tile::Kernel kernel{quantized_copy_kernel};
        auto builder = kernel.function();
        auto statements = builder->body()->statements();
        // A (int8 arg), B (fp8 arg), C (int8 T.empty), D (fp8 T.empty),
        // then the T.Kernel marker and the two copies.
        expect(statements.size() == 7u);
        auto *alloc_a = static_cast<const AllocStmt *>(statements[0]);
        auto *alloc_b = static_cast<const AllocStmt *>(statements[1]);
        auto *alloc_c = static_cast<const AllocStmt *>(statements[2]);
        auto *alloc_d = static_cast<const AllocStmt *>(statements[3]);
        expect(alloc_a->tensor()->dtype() == TensorElementType::I8);
        expect(alloc_b->tensor()->dtype() == TensorElementType::FP8);
        expect(alloc_c->tensor()->dtype() == TensorElementType::I8);
        expect(alloc_d->tensor()->dtype() == TensorElementType::FP8);
        expect(statements[4]->op() == TileOpKind::KERNEL_1D);
        expect(statements[5]->op() == TileOpKind::COPY);
        expect(statements[6]->op() == TileOpKind::COPY);
        // describe() carries the quantized dtype name
        expect(luisa::string_view{alloc_a->tensor()->describe()}.find("int8") != luisa::string_view::npos);
        expect(luisa::string_view{alloc_b->tensor()->describe()}.find("fp8") != luisa::string_view::npos);
    };

    // Device-backed test: compile a traced tile kernel through
    // device.compile_tile(...) and dispatch it like a regular shader. No
    // backend implements support_tile_compiling() yet, so this exercises the
    // tile_to_kernel + create_shader fallback path and verifies that the
    // lowered kernel actually computes C = A + B on device buffers.
    "compile_tile_runs_fallback_kernel_on_device"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        auto &device = dc->device;
        Stream stream = device.create_stream();

        auto tile_kernel = tile::jit(elementwise_add).compile();
        auto tile_shader = device.compile_tile(tile_kernel);
        expect(tile_shader.valid()) << "compile_tile should produce a valid TileShader";
        // No backend implements native tile compiling yet: expect the regular
        // kernel fallback path (tile_to_kernel + create_shader).
        expect(!tile_shader.is_native_tile_shader());
        // elementwise_add: T.Kernel(4, 4, 32) -> dispatch grid (4 * 32, 4).
        auto xy = tile_shader.dispatch_size_xy();
        expect(xy.x == 128u && xy.y == 4u);

        constexpr auto count = 32u * 32u;
        auto bufA = device.create_buffer<float>(count);
        auto bufB = device.create_buffer<float>(count);
        auto bufC = device.create_buffer<float>(count);
        luisa::vector<float> hA(count), hB(count), hC(count);
        for (auto i = 0u; i < count; ++i) {
            hA[i] = static_cast<float>(i) * 0.5f;
            hB[i] = static_cast<float>(i) * 1.5f + 1.0f;
        }
        stream << bufA.copy_from(luisa::span{hA})
               << bufB.copy_from(luisa::span{hB}) << synchronize();
        stream << tile_shader(bufA, bufB, bufC).dispatch(1u)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        bool ok = true;
        for (auto i = 0u; i < count; ++i) {
            if (std::abs(hC[i] - (hA[i] + hB[i])) > 1e-3f) {
                LUISA_WARNING("compile_tile fallback mismatch at [{}]: got {}, want {}",
                              i, hC[i], hA[i] + hB[i]);
                ok = false;
                break;
            }
        }
        expect(ok) << "tile shader fallback should compute C = A + B";
    };
}

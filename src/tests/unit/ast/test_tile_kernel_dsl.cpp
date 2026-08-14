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
//
// Pure host code: no device / backend is required.

#include "ut/ut.hpp"

#include <cstdint>

#include <luisa/dsl/tensor.h>

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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
}

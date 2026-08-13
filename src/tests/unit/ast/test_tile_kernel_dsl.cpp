// Test for the tile pseudo kernel in <luisa/dsl/tensor.h>.
// This test covers:
// - tile::Kernel (the tile-DSL analogue of luisa::compute::Kernel in
//   <luisa/dsl/func.h>) tracing a prim function like elementwise_add into a
//   luisa::compute::detail::TileFunctionBuilder
// - tile::Kernel accepting a lambda as well as a function pointer
// - every tile op (T.empty, T.ceildiv, T.Kernel, T.alloc_shared,
//   T.alloc_fragment, T.copy, whole-tile binary + tile-store) emitting the
//   matching TensorStmt into the builder, in program order
// - tile::jit(...).compile(...) routing through tile::Kernel and keeping the
//   traced builder in the compiled kernel
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
        auto compiled = tile::jit(elementwise_add).compile(/*M=*/32, /*N=*/32);
        expect(compiled.function() != nullptr);
        expect(compiled.function()->body()->size() == 14u);
        expect(!compiled.get_kernel_source().empty());
        // the compiled kernel stays callable (stub: no device execution)
        Tensor<tile_f32, 2> A{32, 32};
        Tensor<tile_f32, 2> B{32, 32};
        [[maybe_unused]] auto C = compiled(A, B);
    };
}

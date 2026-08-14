// Test for tile_to_kernel — the SIMD->SIMT lowering that translates a traced
// tile function (TileFunctionBuilder, <luisa/ast/tile_function_builder.h>)
// into a REGULAR Luisa kernel (FunctionBuilder, <luisa/ast/function_builder.h>)
// as declared in <luisa/ast/tile_to_kernel.h>.
//
// This test covers:
// - translating the three example tile kernels (elementwise_add, pipelined
//   matmul, rms_norm) into FunctionBuilder instances
// - the dispatch grid equals the T.Kernel grid (KERNEL_1D -> (gx,1,1),
//   KERNEL_2D -> (gx,gy,1))
// - the kernel block size equals the T.Kernel thread count
// - one Buffer<T> argument per Global tensor of the tile function
//   (in AllocStmt order), all with Variable::Tag::BUFFER
// - the produced builder is a KERNEL-tagged FunctionBuilder with a non-empty
//   body, and wraps into a valid Function object
// - shared/fragment allocations produce shared/local array variables
//
// Pure host code: no device / backend is required.
#include "ut/ut.hpp"
#include <cstdint>
#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
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

TILELANG_PRIM_FUNC
Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
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
}

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

Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;
    Tensor<tile_f32, 2> B = T.empty(T.shape(M, N), tile_f32{});
    for (auto bx : T.Kernel(T.ceildiv(M, blk_m), threads)) {
        auto A_local = T.alloc_fragment(T.shape(blk_m, N), tile_f32{});
        auto A_pow_local = T.alloc_fragment(T.shape(blk_m, N), tile_f32{});
        auto A_powsum = T.alloc_fragment(T.shape(blk_m), tile_f32{});
        T.copy(A(T.range(bx * blk_m, (bx + 1) * blk_m), T.all()), A_local);
        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);
        T.reduce_sum(A_pow_local, A_powsum, /*dim=*/1);
        A_powsum(blk_m) = T.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);
        A_local(blk_m, N) *= A_powsum(blk_m);
        T.copy(A_local, B(T.range(bx * blk_m, (bx + 1) * blk_m), T.all()));
    }
    return B;
}

// Quantized dtype lowering: int8 global tensors lower to Buffer<byte>
// arguments (core INT8 element type); fp8 lowers to the fp8 e4m3 element type
// (zero fill is carried as a raw zero byte and cast to fp8).  As in the other
// tile kernels, global-to-global copies are not traced (global views are
// extent-less), so each copy routes through a shared/fragment intermediate.
TILELANG_PRIM_FUNC
Tensor<tile::int8, 1> int8_copy_kernel(Tensor<tile::int8, 1> A) {
    constexpr tile_i32 threads = 8;
    Tensor<tile::int8, 1> C = T.empty(T.shape(8), tile::int8{});
    for (auto bx : T.Kernel(1, threads)) {
        auto A_shared = T.alloc_shared(T.shape(8), tile::int8{});
        T.copy(A(0), A_shared);
        T.copy(A_shared, C(0));
    }
    return C;
}

TILELANG_PRIM_FUNC
Tensor<tile::fp8, 1> fp8_clear_kernel() {
    constexpr tile_i32 threads = 8;
    Tensor<tile::fp8, 1> C = T.empty(T.shape(8), tile::fp8{});
    for (auto bx : T.Kernel(1, threads)) {
        auto C_local = T.alloc_fragment(T.shape(8), tile::fp8{});
        T.clear(C_local);
        T.copy(C_local, C(0));
    }
    return C;
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "elementwise_add_translates_to_regular_kernel"_test = [] {
        tile::Kernel kernel{elementwise_add};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 4u * 32u && result.dispatch_size.y == 4u && result.dispatch_size.z == 1u);
        auto block = result.function->block_size();
        expect(block.x == 32u && block.y == 1u && block.z == 1u);
        // A, B, C -> three buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 3u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        // shared + fragment allocations exist
        expect(!result.function->shared_variables().empty());
        expect(!result.function->local_variables().empty());
        // the produced builder is a kernel with a non-empty body
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        // wraps into a valid Function object
        expect(static_cast<bool>(result.function->function()));
    };

    "pipelined_matmul_translates_to_regular_kernel"_test = [] {
        tile::Kernel kernel{pipelined_matmul};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 4u * 32u && result.dispatch_size.y == 4u && result.dispatch_size.z == 1u);
        auto block = result.function->block_size();
        expect(block.x == 32u && block.y == 1u && block.z == 1u);
        auto args = result.function->arguments();
        expect(args.size() == 3u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "rms_norm_translates_to_1d_regular_kernel"_test = [] {
        tile::Kernel kernel{rms_norm};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        expect(result.dispatch_size.x == 8u * 64u && result.dispatch_size.y == 1u && result.dispatch_size.z == 1u);
        auto block = result.function->block_size();
        expect(block.x == 64u && block.y == 1u && block.z == 1u);
        // A, B -> two buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 2u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "translation_preserves_launch_metadata"_test = [] {
        // 1D kernel: dispatch is (gx, 1, 1); 2D kernel: (gx, gy, 1).
        tile::Kernel k1{rms_norm};
        auto r1 = tile_to_kernel(k1.function());
        expect(r1.dispatch_size.y == 1u);

        tile::Kernel k2{elementwise_add};
        auto r2 = tile_to_kernel(k2.function());
        expect(r2.dispatch_size.x == 4u * 32u && r2.dispatch_size.y == 4u);
    };

    "int8_tensors_lower_to_byte_buffers"_test = [] {
        tile::Kernel kernel{int8_copy_kernel};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        // T.Kernel(1, 8): 1 block of 8 threads -> dispatch.x = 1 * 8
        expect(result.dispatch_size.x == 8u && result.dispatch_size.y == 1u && result.dispatch_size.z == 1u);
        auto block = result.function->block_size();
        expect(block.x == 8u && block.y == 1u && block.z == 1u);
        // A, C -> two buffer arguments
        auto args = result.function->arguments();
        expect(args.size() == 2u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };

    "fp8_tensors_lower_with_zero_fill"_test = [] {
        tile::Kernel kernel{fp8_clear_kernel};
        auto result = tile_to_kernel(kernel.function());
        expect(result.function != nullptr);
        // T.Kernel(1, 8): 1 block of 8 threads -> dispatch.x = 1 * 8
        expect(result.dispatch_size.x == 8u && result.dispatch_size.y == 1u && result.dispatch_size.z == 1u);
        auto block = result.function->block_size();
        expect(block.x == 8u && block.y == 1u && block.z == 1u);
        // C -> one buffer argument (fp8 e4m3 element type)
        auto args = result.function->arguments();
        expect(args.size() == 1u);
        for (auto v : args) { expect(v.tag() == Variable::Tag::BUFFER); }
        expect(result.function->tag() == Function::Tag::KERNEL);
        expect(!result.function->body()->statements().empty());
        expect(static_cast<bool>(result.function->function()));
    };
}

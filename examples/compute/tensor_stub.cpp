// =============================================================================
// tensor_stub.cpp — TileLang-style tile / tensor DSL demo (stub)
// =============================================================================
// A *compilable* adaptation of the pseudo-code in
// `D:/tilelang/dsl_report/tilelang_cpp_tile_style.cpp`, built on the
// header-only stub `include/luisa/dsl/tensor.h`.
//
// The three kernels are written in pure tile style — no threads, no
// `set_block_size`, no `dispatch`, no shared-memory loops, no barriers:
//   * elementwise_add  -> mirrors examples/elementwise/example_elementwise_add.py
//   * matmul           -> mirrors examples/gemm/example_gemm.py (T.Pipelined)
//   * rms_norm         -> mirrors examples/norm/rms_norm.py
//
// Pseudo-code -> valid C++ adaptations (the stub's job):
//   T.empty({M, N}, f32)                  -> LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{})
//   T.alloc_shared({BM, BK}, f16)         -> LuisaTensor.alloc_shared(LuisaTensor.shape(BM, BK), tile_f16{})
//   T.alloc_fragment({blk_m}, f32)        -> LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{})
//   A[by * BM, bx * BN]                   -> A(by * BM, bx * BN)
//                                        (multi-arg operator[] is C++23-only,
//                                         so tile indexing uses operator())
//   A[bx*blk_m : (bx+1)*blk_m, :]         -> A(LuisaTensor.range(bx*blk_m, (bx+1)*blk_m), LuisaTensor.all())
//   f32 / f16 as a value argument         -> tile_f32{} / tile_f16{} (dtype handle)
//   f32(N)                                -> stays (dtype handles are scalars)
//   luisa::compute::tile::jit(matmul).compile(...)    -> stays; logs "kernel.compile"
//
// The stub logs every tile op through the LuisaCompute core logger
// (lc_core / LUISA_INFO).  Running this target must print `kernel.compile`.
// =============================================================================

#include <luisa/dsl/tensor.h>

#include <string>

// TileLang's `import tilelang.language as T` is exposed as the `LuisaTensor`
// constexpr handle (a C++ namespace can only be addressed with `::`, so the
// `LuisaTensor.*` dot syntax comes from the `dsl` handle object).
constexpr auto LuisaTensor = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;

using tile_f16 = luisa::compute::tile::half;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// =============================================================================
// 1. Elementwise add — tiles only
// =============================================================================
TILELANG_PRIM_FUNC
Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 512, N = 512;
    constexpr tile_i32 block_M = 64, block_N = 64;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    // Grid is in *blocks*: LuisaTensor.Kernel(gx, gy) means gx*gy blocks; the
    // range-for binds (bx, by) to each block id (the C++ spelling of
    // `with T.Kernel(...) as (bx, by)`).
    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        // Per-block on-chip staging: global -> shared -> fragment -> global.
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        // Tile copies (global -> shared); a real lowering emits a coalesced
        // SIMT/TMA copy.
        LuisaTensor.copy(A(by * block_M, bx * block_N), A_shared);
        LuisaTensor.copy(B(by * block_M, bx * block_N), B_shared);

        // Whole-tile elementwise op (block style): the (block_M x block_N)
        // tile of C_local becomes the elementwise sum of the two source tiles.
        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);

        // Fragment -> global.
        LuisaTensor.copy(C_local, C(by * block_M, bx * block_N));
    }
    return C;
}

// =============================================================================
// 2. Tiled GEMM with a software pipeline — tiles only
// =============================================================================
TILELANG_PRIM_FUNC
Tensor<tile_f16, 2> matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 1024, N = 1024, K = 1024;
    constexpr tile_i32 block_M = 128, block_N = 128, block_K = 32;
    constexpr tile_i32 threads = 128;
    constexpr tile_i32 num_stages = 3;

    Tensor<tile_f16, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f16{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_K), tile_f16{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_K, block_N), tile_f16{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});// tile_f32 accumulator

        LuisaTensor.clear(C_local);// T.clear

        // Software pipeline: copies and GEMMs are overlapped by the compiler.
        for (auto ko : LuisaTensor.Pipelined(LuisaTensor.ceildiv(K, block_K), num_stages)) {
            LuisaTensor.copy(A(by * block_M, ko * block_K), A_shared);// global -> shared
            LuisaTensor.copy(B(ko * block_K, bx * block_N), B_shared);
            LuisaTensor.gemm(A_shared, B_shared, C_local);// tile GEMM
        }

        // Fused ReLU on the fragment tile (whole-tile op, quickstart.py).
        C_local(block_M, block_N) = LuisaTensor.max(C_local(block_M, block_N), tile_f32(0.0f));

        // Optional device-side debug (TileLang prints from a single thread).
        LuisaTensor.print(C_local, /*msg=*/"C tile:");

        LuisaTensor.copy(C_local, C(by * block_M, bx * block_N));// fragment -> global
    }
    return C;
}

// =============================================================================
// 3. RMSNorm — tiles only
// =============================================================================
Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 512, N = 512;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 128;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_pow_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_powsum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});// per-row scalars

        // Row-slice copy: pseudo `A[bx*blk_m : (bx+1)*blk_m, :]`.
        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()), A_local);

        // Whole-tile square: the (blk_m x N) tile, elementwise.
        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);

        LuisaTensor.reduce_sum(A_pow_local, A_powsum, /*dim=*/1);// row sums

        // Whole-tile per-row scale factor (1-D tile, scalar broadcast).
        A_powsum(blk_m) = LuisaTensor.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);

        // Broadcast: every column of row i is scaled by the scalar A_powsum[i].
        A_local(blk_m, N) *= A_powsum(blk_m);

        LuisaTensor.copy(A_local, B(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));// store row slice
    }
    return B;
}

// CPU reference used only by the host harness below (stub: no data).
Tensor<tile_f16, 2> reference_matmul(const Tensor<tile_f16, 2> &A, const Tensor<tile_f16, 2> &B) {
    LUISA_INFO("[tensor-stub] reference_matmul: stub CPU reference (no data)");
    return Tensor<tile_f16, 2>{};
}

int main() {
    // Same baked sizes as matmul() above; the @jit wrapper infers the target
    // from the tensors on the first call (luisa::compute::tile::jit == @tilelang.jit).
    constexpr tile_i32 M = 1024, N = 1024, K = 1024;

    Tensor<tile_f16, 2> A{M, K};
    Tensor<tile_f16, 2> B{K, N};

    // ---- Trace the tile-style prim functions (host side, stub) -------------
    {
        Tensor<tile_f32, 2> A_f{512, 512};
        Tensor<tile_f32, 2> B_f{512, 512};

        LUISA_INFO("=== tensor-dsl: trace elementwise_add ===");
        auto C_add = elementwise_add(A_f, B_f);
        LUISA_INFO("[tensor-stub] elementwise_add -> {}", C_add.describe());

        LUISA_INFO("=== tensor-dsl: trace rms_norm ===");
        auto C_norm = rms_norm(A_f);
        LUISA_INFO("[tensor-stub] rms_norm -> {}", C_norm.describe());
    }

    // ---- Compile the matmul TIR function into an executable module ---------
    // This logs "kernel.compile" (required output of this example).
    auto matmul_kernel = luisa::compute::tile::jit(matmul).compile(
        /*M=*/M, /*N=*/N, /*K=*/K,
        /*block_M=*/128, /*block_N=*/128, /*block_K=*/32,
        /*threads=*/128, /*num_stages=*/3);

    Tensor<tile_f16, 2> C = matmul_kernel(A, B);// runs on the GPU (stub)

    // Reference & correctness check (Torch here; assert_close in C++).
    auto C_ref = reference_matmul(A, B);
    luisa::compute::tile::testing::assert_close(C, C_ref, /*rtol=*/1e-2f, /*atol=*/1e-2f);

    // Optional: dump the generated CUDA source, no threads visible to us.
    auto cuda_source = matmul_kernel.get_kernel_source();
    luisa::compute::tile::print(cuda_source);

    LUISA_INFO("[tensor-stub] finished: tensor DSL stub traced and kernel.compile done.");
    return 0;
}

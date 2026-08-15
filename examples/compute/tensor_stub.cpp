// =============================================================================
// tensor_stub.cpp — TileLang-style tile / tensor DSL demo (all TileOpKind)
// =============================================================================
// Every tile kernel below is written in the "pure tile" style of the header-
// only stub `include/luisa/dsl/tensor.h` (no threads, no `set_block_size`, no
// shared-memory loops, no barriers) and exercises a distinct TileOpKind set:
//
//   elementwise_add   ALLOC, CEILDIV, KERNEL_2D, COPY, BINARY, STORE
//   pipelined_matmul  ALLOC, CEILDIV, KERNEL_2D, CLEAR, PIPELINED,
//                     COPY, GEMM, MAX, PRINT, STORE
//   rms_norm          KERNEL_1D, COPY, BINARY, STORE, REDUCE_SUM, RSQRT
//   tile_fill         FILL
//   tile_transpose    TRANSPOSE (+ SYNC via explicit T.sync_threads)
//   tile_clamp        CLAMP
//   tile_atomic       ATOMIC (load / store / add / max / min / or)
//   tile_sync         SYNC
//   tile_warp_reduce  WARP_REDUCE (sum / max)
//   loop_break_kernel LOOP_BREAK — traced into the tile IR only; see below.
//   tile_reduce       REDUCE (max / min / abssum / absmax row reductions)
//   tile_scan         CUMSUM / CUMMAX (inclusive prefix scan)
//   tile_min_abs      MIN / ABS (whole-tile elementwise ops)
//   tile_vote_shuffle ANY_OF / ALL_OF / SHUFFLE (xor / up / down)
//
// Each kernel is (1) traced with tile::jit(...).compile(),
// (2) lowered to a REGULAR Luisa kernel with `tile_to_kernel`
// (<luisa/ast/tile_to_kernel.h>), (3) compiled on the backend named on the
// command line (`dx` / `vk` / `cuda` / ...), (4) dispatched on real buffers,
// and (5) checked against a host-side reference computation.  The required
// `kernel.compile` log line is produced by the jit() compile below.
//
// LOOP_BREAK is the one op that is traced but NOT lowered/dispatched: the
// lowering emits `break_()`, which needs an enclosing loop.  The flat tile IR
// only has a loop inside T.Pipelined, whose body scanner stops at the first
// statement that does not touch shared memory — so a top-level
// `T.loop_break()` would escape to kernel scope and be rejected by backends.
//
// The multiple-T.Kernel guard (one launch per tile function) is demonstrated
// at the end; pass `--trigger-guard` to hit it (it aborts the process).
// =============================================================================

#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

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
// 1. Elementwise add — ALLOC / CEILDIV / KERNEL_2D / COPY / BINARY / STORE
// =============================================================================
Tensor<tile_f32, 2> elementwise_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    // Grid is in *blocks*: LuisaTensor.Kernel(gx, gy) means gx*gy blocks; the
    // range-for binds (bx, by) to each block id (the C++ spelling of
    // `with T.Kernel(...) as (bx, by)`).  Tracing visits one representative
    // block (bx, by) = (0, 0); tile_to_kernel reconstructs each block's base
    // offset from block_id().
    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        // Tile copies (global -> shared); a real lowering emits a coalesced
        // SIMT/TMA copy.
        LuisaTensor.copy(A(by * block_M, bx * block_N), A_shared(block_M, block_N));
        LuisaTensor.copy(B(by * block_M, bx * block_N), B_shared(block_M, block_N));

        // Whole-tile elementwise op (block style): the (block_M x block_N)
        // tile of C_local becomes the elementwise sum of the two source tiles.
        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);

        // Fragment -> global.
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// =============================================================================
// 2. Tiled GEMM with a software pipeline — CLEAR / PIPELINED / GEMM / MAX /
//    PRINT / STORE (plus ALLOC / CEILDIV / KERNEL_2D / COPY)
// =============================================================================
Tensor<tile_f16, 2> pipelined_matmul(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 64, N = 64, K = 64;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 8;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 2;

    Tensor<tile_f16, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f16{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_K), tile_f16{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_K, block_N), tile_f16{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});// f32 accumulator

        LuisaTensor.clear(C_local);// T.clear

        // Software pipeline: copies and GEMMs are overlapped by the compiler.
        for (auto ko : LuisaTensor.Pipelined(LuisaTensor.ceildiv(K, block_K), num_stages)) {
            LuisaTensor.copy(A(by * block_M, ko * block_K), A_shared(block_M, block_K));// global -> shared
            LuisaTensor.copy(B(ko * block_K, bx * block_N), B_shared(block_K, block_N));
            LuisaTensor.gemm(A_shared(block_M, block_K), B_shared(block_K, block_N), C_local(block_M, block_N));// tile GEMM
        }

        // Fused ReLU on the fragment tile (whole-tile op).
        C_local(block_M, block_N) = LuisaTensor.max(C_local(block_M, block_N), 0.0f);

        // Optional device-side debug (prints tile[0] from thread 0).
        LuisaTensor.print(C_local(block_M, block_N), "C tile:");

        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));// fragment -> global
    }
    return C;
}

// =============================================================================
// 3. RMSNorm — KERNEL_1D / COPY / BINARY / STORE / REDUCE_SUM / RSQRT
// =============================================================================
Tensor<tile_f32, 2> rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_pow_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_powsum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});// per-row scalars

        // Row-slice copy: pseudo `A[bx*blk_m : (bx+1)*blk_m, :]`.
        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, N));

        // Whole-tile square: the (blk_m x N) tile, elementwise.
        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);

        LuisaTensor.reduce_sum(A_pow_local(blk_m, N), A_powsum(blk_m), /*dim=*/1);// row sums

        // Whole-tile per-row scale factor (1-D tile, scalar broadcast).
        A_powsum(blk_m) = LuisaTensor.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);

        // Broadcast: every column of row i is scaled by the scalar A_powsum[i].
        A_local(blk_m, N) *= A_powsum(blk_m);

        LuisaTensor.copy(A_local(blk_m, N),
                         B(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));// store row slice
    }
    return B;
}

// =============================================================================
// 4. T.fill — FILL
// =============================================================================
// Fill a per-thread fragment tile with a constant and copy it out.
Tensor<tile_f32, 1> tile_fill_kernel() {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> C = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto F = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.fill(F(N), 3.5f);
        LuisaTensor.copy(F(N), C(0));
    }
    return C;
}

// =============================================================================
// 5. T.transpose — TRANSPOSE (shared-memory 2D transpose)
// =============================================================================
Tensor<tile_f32, 2> tile_transpose_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(BM, BN), tile_f32{});
        auto T_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(BN, BM), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), A_shared(BM, BN));// global -> shared
        LuisaTensor.transpose(A_shared(BM, BN), T_shared(BN, BM));// dst[j, i] = src[i, j]
        LuisaTensor.sync_threads();// explicit block barrier (SYNC)
        LuisaTensor.copy(T_shared(BN, BM), B(by * BM, bx * BN));// shared -> global
    }
    return B;
}

// =============================================================================
// 6. T.clamp — CLAMP (in-place elementwise clamp into [lo, hi])
// =============================================================================
Tensor<tile_f32, 2> tile_clamp_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), C_local(BM, BN));// global -> fragment
        LuisaTensor.clamp(C_local(BM, BN), 0.1f, 0.9f);// in-place clamp
        LuisaTensor.copy(C_local(BM, BN), C(by * BM, bx * BN));// fragment -> global
    }
    return C;
}

// =============================================================================
// 7. T.atomic_* — ATOMIC (load / store / add / max / min / or on an int tile)
// =============================================================================
// One block of 32 threads; the whole D tile is partitioned once per atomic
// statement, so thread i performs the op on element i.  The sequence
//   load -> store 5 -> add 2 -> max 3 -> min 4 -> or 8
// leaves every element at ((((5 + 2) max 3) min 4) | 8) = (7 min 4) | 8 =
// 4 | 8 = 12.
Tensor<tile_i32, 1> tile_atomic_kernel() {
    constexpr tile_i32 N = 32;
    constexpr tile_i32 threads = 32;

    Tensor<tile_i32, 1> D = LuisaTensor.empty(LuisaTensor.shape(N), tile_i32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        LuisaTensor.atomic_load(D);// no-op read (return value is discarded)
        LuisaTensor.atomic_store(D, 5);
        LuisaTensor.atomic_add(D, 2);
        LuisaTensor.atomic_max(D, 3);
        LuisaTensor.atomic_min(D, 4);
        LuisaTensor.atomic_or(D, 8);
    }
    return D;
}

// =============================================================================
// 8. T.sync_threads — SYNC (explicit block barrier between shared accesses)
// =============================================================================
Tensor<tile_f32, 2> tile_sync_kernel(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(BM, BN), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), A_shared(BM, BN));// global -> shared
        LuisaTensor.sync_threads();// block-wide barrier
        LuisaTensor.copy(A_shared(BM, BN), C(by * BM, bx * BN));// shared -> global
    }
    return C;
}

// =============================================================================
// 9. T.warp_reduce_sum / max — WARP_REDUCE (register-level warp reduction)
// =============================================================================
// The warp-reduce result currently has no consumer in the tile IR, so it is
// computed into a throw-away register; the kernel also copies the reduced
// fragment out so the whole program is verifiable.
Tensor<tile_f32, 1> tile_warp_reduce_kernel() {
    constexpr tile_i32 N = 1;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> W = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto v = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.fill(v(N), 7.0f);
        LuisaTensor.warp_reduce_sum(v(N));// result discarded by the lowering
        LuisaTensor.warp_reduce_max(v(N));
        LuisaTensor.copy(v(N), W(0));// thread 0 stores the value -> verifiable
    }
    return W;
}

// =============================================================================
// 10. T.loop_break — LOOP_BREAK (traced only; see the file header)
// =============================================================================
Tensor<tile_f32, 2> loop_break_kernel(Tensor<tile_f32, 2> A) {
    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(8, 8), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, 32)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
        // A top-level break is representable in the tile IR but not yet
        // lowerable (see the file header): the lowering's `break_()` needs an
        // enclosing loop that the flat tile IR cannot place it in.
        LuisaTensor.loop_break();
    }
    return B;
}

// =============================================================================
// 12. T.reduce_max / reduce_min / reduce_abssum / reduce_absmax — REDUCE
// =============================================================================
// Row-wise reductions of a fragment tile (the generic TileLang reduce family).
Tensor<tile_f32, 1> tile_reduce_kernel(Tensor<tile_f32, 2> A,
                                       Tensor<tile_f32, 1> Bmax,
                                       Tensor<tile_f32, 1> Bmin,
                                       Tensor<tile_f32, 1> Babssum) {
    constexpr tile_i32 M = 64, N = 64;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 64;

    Tensor<tile_f32, 1> Babsmax = LuisaTensor.empty(LuisaTensor.shape(M), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto v_max = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});
        auto v_min = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});
        auto v_abssum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});
        auto v_absmax = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, N));

        LuisaTensor.reduce_max(A_local(blk_m, N), v_max(blk_m), /*dim=*/1);
        LuisaTensor.reduce_min(A_local(blk_m, N), v_min(blk_m), /*dim=*/1);
        LuisaTensor.reduce_abssum(A_local(blk_m, N), v_abssum(blk_m), /*dim=*/1);
        LuisaTensor.reduce_absmax(A_local(blk_m, N), v_absmax(blk_m), /*dim=*/1);

        LuisaTensor.copy(v_max(blk_m), Bmax(bx * blk_m));
        LuisaTensor.copy(v_min(blk_m), Bmin(bx * blk_m));
        LuisaTensor.copy(v_abssum(blk_m), Babssum(bx * blk_m));
        LuisaTensor.copy(v_absmax(blk_m), Babsmax(bx * blk_m));
    }
    return Babsmax;
}

// =============================================================================
// 13. T.cumsum / T.cummax — CUMSUM / CUMMAX (inclusive prefix scan)
// =============================================================================
Tensor<tile_f32, 1> tile_scan_kernel(Tensor<tile_f32, 1> A, Tensor<tile_f32, 1> S) {
    constexpr tile_i32 N = 64;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> Mx = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto S_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        auto M_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_local(N));// global -> fragment
        LuisaTensor.cumsum(A_local(N), S_local(N), /*dim=*/0);// inclusive prefix sum
        LuisaTensor.cummax(A_local(N), M_local(N), /*dim=*/0);// inclusive prefix max
        LuisaTensor.copy(S_local(N), S(0));// fragment -> global
        LuisaTensor.copy(M_local(N), Mx(0));
    }
    return Mx;
}

// =============================================================================
// 14. T.min / T.abs — MIN / ABS (whole-tile elementwise ops)
// =============================================================================
Tensor<tile_f32, 2> tile_min_abs_kernel(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 BM = 8, BN = 8;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(BM, BN), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});
        auto M_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});
        auto Abs_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(BM, BN), tile_f32{});

        LuisaTensor.copy(A(by * BM, bx * BN), A_local(BM, BN));// global -> fragment
        M_local(BM, BN) = LuisaTensor.min(A_local(BM, BN), 0.5f);// elementwise min
        Abs_local(BM, BN) = LuisaTensor.abs(A_local(BM, BN));// elementwise abs
        LuisaTensor.copy(M_local(BM, BN), B(by * BM, bx * BN));// fragment -> global
        LuisaTensor.copy(Abs_local(BM, BN), C(by * BM, bx * BN));
    }
    return C;
}

// =============================================================================
// 15. T.any_of / T.all_of / T.shfl_* — ANY_OF / ALL_OF / SHUFFLE
// =============================================================================
// The vote/shuffle results have no consumer in the tile IR (they are computed
// into throw-away registers, like WARP_REDUCE); the filled fragment is copied
// out so the whole program is verifiable.
Tensor<tile_f32, 1> tile_vote_shuffle_kernel() {
    constexpr tile_i32 N = 2;
    constexpr tile_i32 threads = 32;

    Tensor<tile_f32, 1> W = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto v = LuisaTensor.alloc_fragment(LuisaTensor.shape(N), tile_f32{});
        LuisaTensor.fill(v(N), 1.0f);
        LuisaTensor.any_of(v(N));// block vote: any element != 0
        LuisaTensor.all_of(v(N));// block vote: all elements != 0
        LuisaTensor.shfl_xor(v(N), 1);// warp shuffle (result discarded)
        LuisaTensor.shfl_up(v(N), 1);
        LuisaTensor.shfl_down(v(N), 1);
        LuisaTensor.copy(v(N), W(0));// verifiable output
    }
    return W;
}

// =============================================================================
// 16. Multiple-T.Kernel guard — an INVALID tile function (opt-in trigger).
//     A tile function maps to exactly ONE kernel launch (TileLang emits one
//     `__global__` per `T.Kernel`), so tracing a second T.Kernel must be
//     rejected: jit(...).compile() derives the SIMT launch metadata and logs
//     an error + aborts when the body contains more than one T.Kernel.
// =============================================================================
Tensor<tile_f32, 2> two_kernels(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    for (auto [bx, by] : LuisaTensor.Kernel(4, 4, 32)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
    }
    for (auto [bx, by] : LuisaTensor.Kernel(8, 8, 64)) {
        LuisaTensor.copy(A(0, 0), B(0, 0));
    }
    return B;
}

int main(int argc, char *argv[]) {
    using namespace luisa::compute;// Kernel / Device / Context / detail for the translation test
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    auto backend = argc > 1 && argv != nullptr && argv[1] != nullptr ? luisa::string_view{argv[1]} : luisa::string_view{};
    auto trigger_guard = argc > 2 && argv != nullptr && argv[2] != nullptr &&
                         luisa::string_view{argv[2]} == "--trigger-guard";

    // =========================================================================
    // Trace every tile kernel and lower it with tile_to_kernel.  Structural
    // checks (dispatch grid, block size, buffer argument count) run for all
    // kernels; device compilation + dispatch + host verification run when a
    // backend name is passed on the command line.
    // =========================================================================
    auto same_u3 = [](luisa::uint3 a, luisa::uint3 b) noexcept {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    };
    auto translate_and_verify = [&](luisa::string_view name,
                                    luisa::shared_ptr<const luisa::compute::detail::TileFunctionBuilder> const &tile_fn,
                                    luisa::uint3 expected_dispatch, luisa::uint3 expected_block,
                                    size_t expected_buffers) -> TileCompileResult {
        LUISA_INFO("=== tensor-dsl: tile_to_kernel({}) ===", name);
        auto result = tile_to_kernel(tile_fn);
        LUISA_ASSERT(result.function != nullptr,
                     "[tensor-stub] tile_to_kernel({}) produced a null FunctionBuilder.", name);
        LUISA_ASSERT(same_u3(result.dispatch_size, expected_dispatch),
                     "[tensor-stub] tile_to_kernel({}) dispatch mismatch: got ({},{},{}), want ({},{},{}).",
                     name, result.dispatch_size.x, result.dispatch_size.y, result.dispatch_size.z,
                     expected_dispatch.x, expected_dispatch.y, expected_dispatch.z);
        auto block = result.function->block_size();
        LUISA_ASSERT(same_u3(block, expected_block),
                     "[tensor-stub] tile_to_kernel({}) block-size mismatch: got ({},{},{}), want ({},{},{}).",
                     name, block.x, block.y, block.z, expected_block.x, expected_block.y, expected_block.z);
        auto arg_count = result.function->arguments().size();
        LUISA_ASSERT(arg_count == expected_buffers,
                     "[tensor-stub] tile_to_kernel({}) buffer-argument count mismatch: got {}, want {}.",
                     name, arg_count, expected_buffers);
        LUISA_INFO("[tensor-stub] tile_to_kernel({}) -> FunctionBuilder dispatch=({},{},{}), "
                   "block=({},{},{}), {} buffer argument(s), body has {} statement(s).",
                   name, result.dispatch_size.x, result.dispatch_size.y, result.dispatch_size.z,
                   block.x, block.y, block.z, arg_count,
                   result.function->body()->statements().size());
        return result;
    };

    auto trace_and_verify = [&]<typename F>(luisa::string_view name, F &&fn,
                                            luisa::uint3 expected_dispatch,
                                            luisa::uint3 expected_block,
                                            size_t expected_buffers) {
        auto kernel = luisa::compute::tile::jit(std::forward<F>(fn)).compile();
        LUISA_INFO("[tensor-stub] {} traced {} statements: [{}]",
                   name, kernel.function()->body()->size(), kernel.describe());
        auto result = translate_and_verify(name, kernel.function(),
                                           expected_dispatch, expected_block, expected_buffers);
        return std::make_pair(std::move(kernel), std::move(result));
    };

    // =========================================================================
    // Device pass: compile + dispatch + host verification (backend given).
    // =========================================================================
    if (backend.empty()) {
        LUISA_INFO("=== tensor-dsl: structural verification only ===");
        trace_and_verify("elementwise_add", elementwise_add,
                         luisa::uint3{128u, 4u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
        trace_and_verify("pipelined_matmul", pipelined_matmul,
                         luisa::uint3{128u, 4u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
        trace_and_verify("rms_norm", rms_norm,
                         luisa::uint3{512u, 1u, 1u}, luisa::uint3{64u, 1u, 1u}, 2u);
        trace_and_verify("tile_fill", tile_fill_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
        trace_and_verify("tile_transpose", tile_transpose_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);
        trace_and_verify("tile_clamp", tile_clamp_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);
        trace_and_verify("tile_atomic", tile_atomic_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
        trace_and_verify("tile_sync", tile_sync_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);
        trace_and_verify("tile_warp_reduce", tile_warp_reduce_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
        trace_and_verify("tile_reduce", tile_reduce_kernel,
                         luisa::uint3{512u, 1u, 1u}, luisa::uint3{64u, 1u, 1u}, 5u);
        trace_and_verify("tile_scan", tile_scan_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
        trace_and_verify("tile_min_abs", tile_min_abs_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
        trace_and_verify("tile_vote_shuffle", tile_vote_shuffle_kernel,
                         luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
        luisa::compute::tile::Kernel loop_break_kernel_obj{loop_break_kernel};
        LUISA_INFO("[tensor-stub] loop_break traced {} statements: [{}] (not lowered: "
                   "break_() requires an enclosing loop, see the file header)",
                   loop_break_kernel_obj.function()->body()->size(), loop_break_kernel_obj.describe());
        LUISA_INFO("[tensor-stub] no backend given: translation verified structurally only "
                   "(pass a backend name, e.g. 'dx'/'vk', to also compile, dispatch and verify).");
    } else {
        LUISA_INFO("=== tensor-dsl: compile + dispatch + verify on backend '{}' ===", backend);
        Context ctx(executable);
        Device device = ctx.create_device(backend);
        auto stream = device.create_stream();

        auto check = [](luisa::string_view name, float err, float tol) {
            LUISA_INFO("[tensor-stub] {} runtime check: max error = {}", name, err);
            LUISA_ASSERT(err < tol, "{} produced wrong results on the device (max error {} >= {}).",
                         name, err, tol);
        };

        // ---- elementwise_add: C = A + B -------------------------------------
        {
            auto [elementwise_kernel, elementwise_result] = trace_and_verify(
                "elementwise_add", elementwise_add,
                luisa::uint3{128u, 4u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
            constexpr uint32_t M = 64u, N = 64u;
            auto bufA = device.create_buffer<float>(M * N);
            auto bufB = device.create_buffer<float>(M * N);
            auto bufC = device.create_buffer<float>(M * N);
            luisa::vector<float> hA(M * N), hB(M * N), hC(M * N), hRef(M * N);
            for (auto i = 0u; i < M * N; ++i) {
                hA[i] = static_cast<float>(i) * 0.5f;
                hB[i] = static_cast<float>(i) * 1.5f + 1.0f;
                hRef[i] = hA[i] + hB[i];
            }
            stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

            // Typed path: tile::jit(...).compile().to_kernel<Dim>() carries the
            // buffer element types from the tile function signature automatically.
            elementwise_kernel.validate(bufA, bufB, bufC);
            auto typed_elementwise = elementwise_kernel.to_kernel<2>();
            auto sh = device.compile(typed_elementwise);
            stream << sh(bufA, bufB, bufC).dispatch(elementwise_result.dispatch_size.x, elementwise_result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < M * N; ++i) { err = luisa::max(err, luisa::abs(hC[i] - hRef[i])); }
            check("elementwise_add", err, 1e-3f);
        }

        // ---- pipelined_matmul: C = max(A @ B, 0) ----------------------------
        {
            auto [matmul_kernel, matmul_result] = trace_and_verify(
                "pipelined_matmul", pipelined_matmul,
                luisa::uint3{128u, 4u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
            constexpr uint32_t M = 64u, N = 64u, K = 64u;
            auto bufA = device.create_buffer<luisa::half>(M * K);
            auto bufB = device.create_buffer<luisa::half>(K * N);
            auto bufC = device.create_buffer<luisa::half>(M * N);
            luisa::vector<luisa::half> hA(M * K), hB(K * N), hC(M * N);
            // f16-exact inputs so the f32 host reference is meaningful.
            for (auto i = 0u; i < M * K; ++i) { hA[i] = luisa::half{static_cast<float>((i % 8)) * 0.25f}; }
            for (auto i = 0u; i < K * N; ++i) { hB[i] = luisa::half{static_cast<float>((i % 4)) * 0.5f}; }
            stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

            // Typed path: tile::jit(...).compile().to_kernel<Dim>() carries the
            // buffer element types from the tile function signature automatically.
            matmul_kernel.validate(bufA, bufB, bufC);
            auto typed_matmul = matmul_kernel.to_kernel<2>();
            auto sh = device.compile(typed_matmul);
            stream << sh(bufA, bufB, bufC).dispatch(matmul_result.dispatch_size.x, matmul_result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto r = 0u; r < M; ++r) {
                for (auto c = 0u; c < N; ++c) {
                    auto s = 0.0f;
                    for (auto k = 0u; k < K; ++k) {
                        s += static_cast<float>(hA[r * K + k]) * static_cast<float>(hB[k * N + c]);
                    }
                    auto ref = luisa::max(s, 0.0f);
                    err = luisa::max(err, luisa::abs(static_cast<float>(hC[r * N + c]) - ref));
                }
            }
            check("pipelined_matmul", err, 1e-2f);
        }

        // ---- rms_norm: B[r][c] = A[r][c] * rsqrt(sum_c A[r][c]^2 / N + 1e-12) --
        {
            auto [rms_kernel, rms_result] = trace_and_verify(
                "rms_norm", rms_norm,
                luisa::uint3{512u, 1u, 1u}, luisa::uint3{64u, 1u, 1u}, 2u);
            constexpr uint32_t M = 64u, N = 64u;
            auto bufA = device.create_buffer<float>(M * N);
            auto bufB = device.create_buffer<float>(M * N);
            luisa::vector<float> hA(M * N), hB(M * N);
            for (auto i = 0u; i < M * N; ++i) { hA[i] = static_cast<float>(i) * 0.5f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            rms_kernel.validate(bufA, bufB);
            auto typed_rms = rms_kernel.to_kernel<1>();
            auto sh = device.compile(typed_rms);
            stream << sh(bufA, bufB).dispatch(rms_result.dispatch_size.x)
                   << bufB.copy_to(luisa::span{hB}) << synchronize();
            auto err = 0.0f;
            for (auto r = 0u; r < M; ++r) {
                auto s = 0.0f;
                for (auto c = 0u; c < N; ++c) { s += hA[r * N + c] * hA[r * N + c]; }
                auto scale = 1.0f / luisa::sqrt(s / static_cast<float>(N) + 1e-12f);
                for (auto c = 0u; c < N; ++c) {
                    err = luisa::max(err, luisa::abs(hB[r * N + c] - hA[r * N + c] * scale));
                }
            }
            check("rms_norm", err, 1e-3f);
        }

        // ---- tile_fill: C[i] = 3.5 ------------------------------------------
        {
            auto [fill_kernel, fill_result] = trace_and_verify(
                "tile_fill", tile_fill_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
            constexpr uint32_t N = 64u;
            auto bufC = device.create_buffer<float>(N);
            luisa::vector<float> hC(N);
            fill_kernel.validate(bufC);
            auto typed_fill = fill_kernel.to_kernel<1>();
            auto sh = device.compile(typed_fill);
            stream << sh(bufC).dispatch(fill_result.dispatch_size.x)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < N; ++i) { err = luisa::max(err, luisa::abs(hC[i] - 3.5f)); }
            check("tile_fill", err, 1e-5f);
        }

        // ---- tile_transpose: B[i][j] = A[j][i] ------------------------------
        {
            auto [transpose_kernel, transpose_result] = trace_and_verify(
                "tile_transpose", tile_transpose_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufB = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hB(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i); }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            transpose_kernel.validate(bufA, bufB);
            auto typed_transpose = transpose_kernel.to_kernel<2>();
            auto sh = device.compile(typed_transpose);
            stream << sh(bufA, bufB).dispatch(transpose_result.dispatch_size.x, transpose_result.dispatch_size.y)
                   << bufB.copy_to(luisa::span{hB}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < BM; ++i) {
                for (auto j = 0u; j < BN; ++j) {
                    err = luisa::max(err, luisa::abs(hB[i * BN + j] - hA[j * BM + i]));
                }
            }
            check("tile_transpose", err, 1e-5f);
        }

        // ---- tile_clamp: C[i] = clamp(A[i], 0.1, 0.9) ----------------------
        {
            auto [clamp_kernel, clamp_result] = trace_and_verify(
                "tile_clamp", tile_clamp_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufC = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hC(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i % 16) * 0.1f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            clamp_kernel.validate(bufA, bufC);
            auto typed_clamp = clamp_kernel.to_kernel<2>();
            auto sh = device.compile(typed_clamp);
            stream << sh(bufA, bufC).dispatch(clamp_result.dispatch_size.x, clamp_result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < BM * BN; ++i) {
                auto ref = luisa::clamp(hA[i], 0.1f, 0.9f);
                err = luisa::max(err, luisa::abs(hC[i] - ref));
            }
            check("tile_clamp", err, 1e-5f);
        }

        // ---- tile_atomic: D[i] = 15 -----------------------------------------
        {
            auto [atomic_kernel, atomic_result] = trace_and_verify(
                "tile_atomic", tile_atomic_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
            constexpr uint32_t N = 32u;
            auto bufD = device.create_buffer<int>(N);
            luisa::vector<int> hD(N, 0);
            stream << bufD.copy_from(luisa::span{hD}) << synchronize();

            atomic_kernel.validate(bufD);
            auto typed_atomic = atomic_kernel.to_kernel<1>();
            auto sh = device.compile(typed_atomic);
            stream << sh(bufD).dispatch(atomic_result.dispatch_size.x)
                   << bufD.copy_to(luisa::span{hD}) << synchronize();
            auto err = 0;
            for (auto i = 0u; i < N; ++i) { err = luisa::max(err, luisa::abs(hD[i] - 12)); }
            LUISA_INFO("[tensor-stub] tile_atomic runtime check: max |D-12| = {}", err);
            LUISA_ASSERT(err == 0, "tile_atomic produced wrong results on the device (max |D-12| = {}).", err);
        }

        // ---- tile_sync: C = A ----------------------------------------------
        {
            auto [sync_kernel, sync_result] = trace_and_verify(
                "tile_sync", tile_sync_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufC = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hC(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i) * 0.25f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            sync_kernel.validate(bufA, bufC);
            auto typed_sync = sync_kernel.to_kernel<2>();
            auto sh = device.compile(typed_sync);
            stream << sh(bufA, bufC).dispatch(sync_result.dispatch_size.x, sync_result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < BM * BN; ++i) { err = luisa::max(err, luisa::abs(hC[i] - hA[i])); }
            check("tile_sync", err, 1e-5f);
        }

        // ---- tile_warp_reduce: W[0] = 7.0 -----------------------------------
        {
            auto [warp_reduce_kernel_obj, warp_reduce_result] = trace_and_verify(
                "tile_warp_reduce", tile_warp_reduce_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
            constexpr uint32_t N = 1u;
            auto bufW = device.create_buffer<float>(N);
            luisa::vector<float> hW(N);
            warp_reduce_kernel_obj.validate(bufW);
            auto typed_warp_reduce = warp_reduce_kernel_obj.to_kernel<1>();
            auto sh = device.compile(typed_warp_reduce);
            stream << sh(bufW).dispatch(warp_reduce_result.dispatch_size.x)
                   << bufW.copy_to(luisa::span{hW}) << synchronize();
            auto err = luisa::abs(hW[0] - 7.0f);
            check("tile_warp_reduce", err, 1e-5f);
        }

        // ---- tile_reduce: row-wise max/min/abssum/absmax ----------------------
        {
            auto [reduce_kernel, reduce_result] = trace_and_verify(
                "tile_reduce", tile_reduce_kernel,
                luisa::uint3{512u, 1u, 1u}, luisa::uint3{64u, 1u, 1u}, 5u);
            constexpr uint32_t M = 64u, N = 64u;
            auto bufA = device.create_buffer<float>(M * N);
            auto bufMax = device.create_buffer<float>(M);
            auto bufMin = device.create_buffer<float>(M);
            auto bufAbsSum = device.create_buffer<float>(M);
            auto bufAbsMax = device.create_buffer<float>(M);
            luisa::vector<float> hA(M * N), hMax(M), hMin(M), hAbsSum(M), hAbsMax(M);
            for (auto i = 0u; i < M * N; ++i) {
                // mixed-sign inputs: negatives exercise min/abs, the spread
                // separates max from absmax
                hA[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
            }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            reduce_kernel.validate(bufA, bufMax, bufMin, bufAbsSum, bufAbsMax);
            auto typed_reduce = reduce_kernel.to_kernel<1>();
            auto sh = device.compile(typed_reduce);
            stream << sh(bufA, bufMax, bufMin, bufAbsSum, bufAbsMax).dispatch(reduce_result.dispatch_size.x)
                   << bufMax.copy_to(luisa::span{hMax}) << bufMin.copy_to(luisa::span{hMin})
                   << bufAbsSum.copy_to(luisa::span{hAbsSum}) << bufAbsMax.copy_to(luisa::span{hAbsMax})
                   << synchronize();
            auto err = 0.0f;
            for (auto r = 0u; r < M; ++r) {
                auto ref_max = -1e30f, ref_min = 1e30f, ref_abssum = 0.0f, ref_absmax = 0.0f;
                for (auto c = 0u; c < N; ++c) {
                    auto v = hA[r * N + c];
                    ref_max = luisa::max(ref_max, v);
                    ref_min = luisa::min(ref_min, v);
                    ref_abssum += luisa::abs(v);
                    ref_absmax = luisa::max(ref_absmax, luisa::abs(v));
                }
                err = luisa::max(err, luisa::abs(hMax[r] - ref_max));
                err = luisa::max(err, luisa::abs(hMin[r] - ref_min));
                err = luisa::max(err, luisa::abs(hAbsSum[r] - ref_abssum));
                err = luisa::max(err, luisa::abs(hAbsMax[r] - ref_absmax));
            }
            check("tile_reduce", err, 1e-3f);
        }

        // ---- tile_scan: S = inclusive prefix sum, Mx = inclusive prefix max ----
        {
            auto [scan_kernel, scan_result] = trace_and_verify(
                "tile_scan", tile_scan_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
            constexpr uint32_t N = 64u;
            auto bufA = device.create_buffer<float>(N);
            auto bufS = device.create_buffer<float>(N);
            auto bufMx = device.create_buffer<float>(N);
            luisa::vector<float> hA(N), hS(N), hMx(N);
            for (auto i = 0u; i < N; ++i) { hA[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.5f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            scan_kernel.validate(bufA, bufS, bufMx);
            auto typed_scan = scan_kernel.to_kernel<1>();
            auto sh = device.compile(typed_scan);
            stream << sh(bufA, bufS, bufMx).dispatch(scan_result.dispatch_size.x)
                   << bufS.copy_to(luisa::span{hS}) << bufMx.copy_to(luisa::span{hMx}) << synchronize();
            auto err = 0.0f;
            auto run_sum = 0.0f, run_max = -1e30f;
            for (auto i = 0u; i < N; ++i) {
                run_sum += hA[i];
                run_max = luisa::max(run_max, hA[i]);
                err = luisa::max(err, luisa::abs(hS[i] - run_sum));
                err = luisa::max(err, luisa::abs(hMx[i] - run_max));
            }
            check("tile_scan", err, 1e-3f);
        }

        // ---- tile_min_abs: B = min(A, 0.5), C = abs(A) ----------------------
        {
            auto [min_abs_kernel, min_abs_result] = trace_and_verify(
                "tile_min_abs", tile_min_abs_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufB = device.create_buffer<float>(BM * BN);
            auto bufC = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hB(BM * BN), hC(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) {
                hA[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;
            }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            min_abs_kernel.validate(bufA, bufB, bufC);
            auto typed_min_abs = min_abs_kernel.to_kernel<2>();
            auto sh = device.compile(typed_min_abs);
            stream << sh(bufA, bufB, bufC).dispatch(min_abs_result.dispatch_size.x, min_abs_result.dispatch_size.y)
                   << bufB.copy_to(luisa::span{hB}) << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < BM * BN; ++i) {
                err = luisa::max(err, luisa::abs(hB[i] - luisa::min(hA[i], 0.5f)));
                err = luisa::max(err, luisa::abs(hC[i] - luisa::abs(hA[i])));
            }
            check("tile_min_abs", err, 1e-5f);
        }

        // ---- tile_vote_shuffle: W = 1.0 (votes/shuffles exercised) -----------
        {
            auto [vote_shuffle_kernel, vote_shuffle_result] = trace_and_verify(
                "tile_vote_shuffle", tile_vote_shuffle_kernel,
                luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);
            constexpr uint32_t N = 2u;
            auto bufW = device.create_buffer<float>(N);
            luisa::vector<float> hW(N);
            vote_shuffle_kernel.validate(bufW);
            auto typed_vote_shuffle = vote_shuffle_kernel.to_kernel<1>();
            auto sh = device.compile(typed_vote_shuffle);
            stream << sh(bufW).dispatch(vote_shuffle_result.dispatch_size.x)
                   << bufW.copy_to(luisa::span{hW}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < N; ++i) { err = luisa::max(err, luisa::abs(hW[i] - 1.0f)); }
            check("tile_vote_shuffle", err, 1e-5f);
        }

        LUISA_INFO("[tensor-stub] all {} translated kernels compiled, dispatched and verified on '{}'.",
                   (size_t)13, backend);
    }

    // =========================================================================
    // Optional: trigger the multiple-T.Kernel guard (aborts the process).
    // =========================================================================
    if (trigger_guard) {
        LUISA_INFO("=== tensor-dsl: trigger the multiple-T.Kernel guard ===");
        auto invalid_kernel = luisa::compute::tile::jit(two_kernels).compile();// aborts here
        (void)invalid_kernel;
    }

    LUISA_INFO("[tensor-stub] finished: all tile kernels traced, lowered, compiled and verified.");
    return 0;
}

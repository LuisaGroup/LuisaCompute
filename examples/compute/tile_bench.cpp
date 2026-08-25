// =============================================================================
// tile_bench.cpp — performance benchmark for the tile_to_kernel lowering
// =============================================================================
// Benchmark companion of tensor_stub.cpp: the same TileLang-style "pure tile"
// DSL kernels, at realistic sizes, timed on the backend named on the command
// line (`cuda` / `dx` / `vk`).  It exercises the block-partitioned / warp-
// collective fast paths emitted by tile_to_kernel (see lc_optimize):
//
// bench_gemm GEMM with block-partitioned C tile + register accumulators
// (PIPELINED / COPY / CLEAR / GEMM / STORE)
// bench_rms_norm row reduction with warp all-reduce (REDUCE_SUM / RSQRT /
// BINARY / STORE) + staged global->fragment copies
// bench_scan inclusive prefix scan with WARP_PREFIX_SUM (CUMSUM)
// bench_add whole-tile elementwise add (COPY / BINARY / STORE)
//
// section 5: one micro-benchmark per remaining tile operator, every container
// 8192 x 512 = 4,194,304 elements (>4096) so memory/scheduling shortages in
// the lowering show up as low GB/s instead of hiding under launch latency:
// bench_copy / bench_clear / bench_fill / bench_clamp / bench_saxpy /
// bench_transpose / bench_reduce_{sum,max,min,abssum,absmax} / bench_cummax /
// bench_atomic_add / bench_{exp,sqrt,tanh,erf} / bench_pow
//
// Every kernel is (1) traced with tile::jit(...).compile(), (2) lowered with
// tile_to_kernel, (3) dispatched `iters` times and timed, and (4) checked
// against a host-side reference (sampled for the large GEMM).
// =============================================================================

#include <luisa/dsl/tensor.h>
#include <luisa/ast/tile_to_kernel.h>
#include <luisa/dsl/func.h>
#include <luisa/core/clock.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <string>
#include <cmath>
#include <limits>

constexpr auto LuisaTensor = luisa::compute::tile::language::dsl;
using luisa::compute::tile::language::Tensor;

using tile_f16 = luisa::compute::tile::half;
using tile_f32 = luisa::compute::tile::float32;
using tile_i32 = luisa::compute::tile::int32;

// =============================================================================
// 1. GEMM: C = A @ B (f16 in, f16 out, f32 accumulate), software pipeline
// =============================================================================
Tensor<tile_f16, 2> bench_gemm(Tensor<tile_f16, 2> A, Tensor<tile_f16, 2> B) {
    constexpr tile_i32 M = 512, N = 512, K = 512;
    constexpr tile_i32 block_M = 16, block_N = 16, block_K = 32;
    constexpr tile_i32 threads = 256;
    constexpr tile_i32 num_stages = 1;

    Tensor<tile_f16, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f16{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_K), tile_f16{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_K, block_N), tile_f16{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.clear(C_local);
        for (auto ko : LuisaTensor.Pipelined(LuisaTensor.ceildiv(K, block_K), num_stages)) {
            LuisaTensor.copy(A(by * block_M, ko * block_K), A_shared(block_M, block_K));
            LuisaTensor.copy(B(ko * block_K, bx * block_N), B_shared(block_K, block_N));
            LuisaTensor.gemm(A_shared(block_M, block_K), B_shared(block_K, block_N), C_local(block_M, block_N));
        }
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// =============================================================================
// 1b. Large GEMM: C = A @ B (4096 x 4096 x 4096, f32)
//     Exercises the warp-K-split path in tile_to_kernel: block_K = 256 hits
//     the K >= 256 gate, the 8x8 C tile gives MT*NT = 4 < threads = 32,
//     C_local is a small fragment -> single-warp barrier-free write-back, and
//     the shared tiles stay at 16 KB total (Vulkan-safe).
// =============================================================================
Tensor<tile_f32, 2> bench_gemm_4096(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 4096, N = 4096, K = 4096;
    constexpr tile_i32 block_M = 8, block_N = 8, block_K = 256;
    constexpr tile_i32 threads = 32;
    constexpr tile_i32 num_stages = 1;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_K), tile_f32{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_K, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.clear(C_local);
        for (auto ko : LuisaTensor.Pipelined(LuisaTensor.ceildiv(K, block_K), num_stages)) {
            LuisaTensor.copy(A(by * block_M, ko * block_K), A_shared(block_M, block_K));
            LuisaTensor.copy(B(ko * block_K, bx * block_N), B_shared(block_K, block_N));
            LuisaTensor.gemm(A_shared(block_M, block_K), B_shared(block_K, block_N), C_local(block_M, block_N));
        }
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// =============================================================================
// 2. RMSNorm: B[r][c] = A[r][c] * rsqrt(mean_c A[r][c]^2 + eps)
// =============================================================================
Tensor<tile_f32, 2> bench_rms_norm(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 2048, N = 256;
    constexpr tile_i32 blk_m = 4;
    constexpr tile_i32 threads = 128;

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_pow_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto A_powsum = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, N));
        A_pow_local(blk_m, N) = A_local(blk_m, N) * A_local(blk_m, N);
        LuisaTensor.reduce_sum(A_pow_local(blk_m, N), A_powsum(blk_m), /*dim=*/1);
        A_powsum(blk_m) = LuisaTensor.rsqrt(A_powsum(blk_m) / tile_f32(N) + 1e-12f);
        A_local(blk_m, N) *= A_powsum(blk_m);
        LuisaTensor.copy(A_local(blk_m, N),
                         B(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));
    }
    return B;
}

// =============================================================================
// 3. Inclusive prefix scan along rows (CUMSUM on shared tiles)
// =============================================================================
Tensor<tile_f32, 2> bench_scan(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 1024, N = 256;
    constexpr tile_i32 blk_m = 8;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> S = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(M, blk_m), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(blk_m, N), tile_f32{});
        auto S_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(blk_m, N), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_shared(blk_m, N));
        LuisaTensor.cumsum(A_shared(blk_m, N), S_shared(blk_m, N), /*dim=*/1);
        LuisaTensor.copy(S_shared(blk_m, N),
                         S(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));
    }
    return S;
}

// =============================================================================
// 3b. 1D inclusive prefix scan (one long line, many warps) — exercises the
//     two-pass block scan in tile_to_kernel (line_count = 1 < nw = 8 for
//     256 threads; the warp-per-line partition would leave 7 warps idle).
// =============================================================================
Tensor<tile_f32, 1> bench_scan_1d(Tensor<tile_f32, 1> A) {
    constexpr tile_i32 N = 2048;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 1> S = LuisaTensor.empty(LuisaTensor.shape(N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(1, threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(N), tile_f32{});
        auto S_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(N), tile_f32{});

        LuisaTensor.copy(A(0), A_shared(N));
        LuisaTensor.cumsum(A_shared(N), S_shared(N), /*dim=*/0);
        LuisaTensor.copy(S_shared(N), S(0));
    }
    return S;
}

// =============================================================================
// 4. Elementwise add: C = A + B
// =============================================================================
Tensor<tile_f32, 2> bench_add(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 2048, N = 2048;
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(N, block_N), LuisaTensor.ceildiv(M, block_M), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto B_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.copy(A(by * block_M, bx * block_N), A_shared(block_M, block_N));
        LuisaTensor.copy(B(by * block_M, bx * block_N), B_shared(block_M, block_N));
        C_local(block_M, block_N) = A_shared(block_M, block_N) + B_shared(block_M, block_N);
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// Global-only whole-tensor variants that exercise the TENSOR_* fast path
// directly (pure-Global operands, 1x1 tile grid so the whole tensor is the
// op domain).  These kernels ERROR on the default partition path (extent-less
// global views), so the benchmark gates them behind --tensor; the "before"
// counterpart is the staged kernel above (same math / byte count).
Tensor<tile_f32, 2> bench_add_global(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 M = 2048, N = 2048;
    constexpr tile_i32 threads = 256;
    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        C(LuisaTensor.range(0, M), LuisaTensor.all()) =
            A(LuisaTensor.range(0, M), LuisaTensor.all()) +
            B(LuisaTensor.range(0, M), LuisaTensor.all());
    }
    return C;
}

Tensor<tile_f32, 2> bench_copy_global(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 2048, N = 2048;
    constexpr tile_i32 threads = 256;
    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(M, N), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        LuisaTensor.copy(A, C);
    }
    return C;
}

// Global-only whole-tensor variants (TENSOR_* fast path) for the section-5
// operators; the whole container is OP_M x OP_N with a 1x1 tile grid so the
// tensor is the op domain.  These kernels ERROR on the default partition path
// (extent-less global views), so the benchmark gates them behind --tensor.
constexpr tile_i32 OP_M = 8192, OP_N = 512;// 4,194,304 elements per container
Tensor<tile_f32, 2> bench_clamp_global(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 threads = 256;
    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        LuisaTensor.copy(A, C);
        LuisaTensor.clamp(C(LuisaTensor.range(0, OP_M), LuisaTensor.all()), -0.5f, 0.5f);
    }
    return C;
}

Tensor<tile_f32, 2> bench_unary_exp_global(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 threads = 256;
    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        C(LuisaTensor.range(0, OP_M), LuisaTensor.all()) =
            LuisaTensor.exp(A(LuisaTensor.range(0, OP_M), LuisaTensor.all()));
    }
    return C;
}

Tensor<tile_f32, 1> bench_reduce_sum_global(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 threads = 256;
    Tensor<tile_f32, 1> R = LuisaTensor.empty(LuisaTensor.shape(OP_M), tile_f32{});
    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        LuisaTensor.reduce_sum(A(LuisaTensor.range(0, OP_M), LuisaTensor.range(0, OP_N)),
                               R, /*dim=*/1);
    }
    return R;
}

// =============================================================================
// 5. Per-operator micro-benchmarks (op isolation).
// =============================================================================
// Every remaining tile operator gets its own kernel below so the report can
// attribute time to a single lowering path.  All containers are huge
// (>= 8192 elements, most 8192 x 512 = 4,194,304 — 1024x the 4096 floor) so
// that memory-system / scheduling shortages in the lowering show up as low
// GB/s instead of being hidden by launch latency.
//
// bench_copy         pure global -> shared -> global copy (COPY bandwidth)
// bench_clear        whole-tile clear + store (CLEAR)
// bench_fill         constant fill + store (FILL)
// bench_clamp        in-place clamp to [lo, hi] (CLAMP)
// bench_saxpy        C = alpha*A + B (chained BINARY mul/add)
// bench_transpose    shared-memory 2D transpose (TRANSPOSE + SYNC)
// bench_reduce_op<N> row reductions (REDUCE_SUM/MAX/MIN/ABSSUM/ABSMAX)
// bench_cummax       row-wise inclusive max scan (CUMMAX)
// bench_atomic_add   one atomic fetch-add per element (ATOMIC)
// bench_unary_op<N>  exp / sqrt / tanh / erf (FAST_MATH / IEEE math)
// bench_pow          C = pow(A, 0.5) (binary IEEE math POW)
  // =============================================================================

  // ---- 5a. pure copy: C = A staged through shared ---------------------------
Tensor<tile_f32, 2> bench_copy(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 block_M = 32, block_N = 32;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto T_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(block_M, block_N), tile_f32{});
        LuisaTensor.copy(A(by * block_M, bx * block_N), T_shared(block_M, block_N));
        LuisaTensor.copy(T_shared(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- 5b. clear: C = 0 (fragment clear + store) ----------------------------
Tensor<tile_f32, 2> bench_clear() {
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        LuisaTensor.clear(C_local);
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- 5c. fill: C = 3.5 (fragment fill + store) ----------------------------
Tensor<tile_f32, 2> bench_fill() {
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        LuisaTensor.fill(C_local(block_M, block_N), 3.5f);
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- 5d. clamp: C = clamp(A, -0.5, 0.5) ----------------------------------
Tensor<tile_f32, 2> bench_clamp(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        LuisaTensor.copy(A(by * block_M, bx * block_N), C_local(block_M, block_N));
        LuisaTensor.clamp(C_local(block_M, block_N), -0.5f, 0.5f);
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- 5e. saxpy: C = 2.5 * A + B (chained BINARY ops) ----------------------
Tensor<tile_f32, 2> bench_saxpy(Tensor<tile_f32, 2> A, Tensor<tile_f32, 2> B) {
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto B_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.copy(A(by * block_M, bx * block_N), A_local(block_M, block_N));
        LuisaTensor.copy(B(by * block_M, bx * block_N), B_local(block_M, block_N));
        // C = A / 0.4 + B  (x/0.4 == x*2.5; the DSL has tile*scalar via '/' only)
        C_local(block_M, block_N) = A_local(block_M, block_N) / 0.4f + B_local(block_M, block_N);
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- 5f. transpose: B = A^T (72x72 = 5184 elements, one block) -------------
// Two lowering limitations shape this kernel (both are findings in
// perf_report.md):
//   1. a multi-block tiled transpose (write block position swapped w.r.t. the
//      read block position) is NOT expressible — every global access binds
//      axis-0 to block_id().y and axis-1 to block_id().x, so only diagonal
//      blocks would land correctly.  The whole container is therefore
//      transposed from ONE block.
//   2. _partition_2d under-covered a tile when threads % min(threads, cols)
//      != 0 (e.g. 72 cols x 256 threads leaves rows == 3 (mod 4) missing
//      their last 32 columns).  The lowered flattened-linear fallback now
//      covers every cell (verified: threads = 256 on this exact shape gives
//      max sampled error 0), and threads = 288 = 4 x 72 remains exact.
// 72x72 = 5184 elements (> 4096), two 20.7 KB shared tiles = 41.5 KB,
// within the 48 KB maxComputeSharedMemorySize of this device (RTX 4060).
Tensor<tile_f32, 2> bench_transpose(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 M = 72, N = 72;// 5184 elements per container (> 4096)
    constexpr tile_i32 threads = 288;// 4 x 72: exact _partition_2d coverage

    Tensor<tile_f32, 2> B = LuisaTensor.empty(LuisaTensor.shape(N, M), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(1, 1, threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(M, N), tile_f32{});
        auto T_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(N, M), tile_f32{});

        LuisaTensor.copy(A(0, 0), A_shared(M, N));
        LuisaTensor.transpose(A_shared(M, N), T_shared(N, M));
        LuisaTensor.sync_threads();
        LuisaTensor.copy(T_shared(N, M), B(0, 0));
    }
    return B;
}

// ---- 5g. row reductions: sum / max / min / abssum / absmax ----------------
template<int Tag>// 0=sum 1=max 2=min 3=abssum 4=absmax
Tensor<tile_f32, 1> bench_reduce_op(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 blk_m = 4;
    constexpr tile_i32 threads = 128;

    Tensor<tile_f32, 1> R = LuisaTensor.empty(LuisaTensor.shape(OP_M), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_M, blk_m), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m, OP_N), tile_f32{});
        auto v = LuisaTensor.alloc_fragment(LuisaTensor.shape(blk_m), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_local(blk_m, OP_N));
        if constexpr (Tag == 0) {
            LuisaTensor.reduce_sum(A_local(blk_m, OP_N), v(blk_m), /*dim=*/1);
        } else if constexpr (Tag == 1) {
            LuisaTensor.reduce_max(A_local(blk_m, OP_N), v(blk_m), /*dim=*/1);
        } else if constexpr (Tag == 2) {
            LuisaTensor.reduce_min(A_local(blk_m, OP_N), v(blk_m), /*dim=*/1);
        } else if constexpr (Tag == 3) {
            LuisaTensor.reduce_abssum(A_local(blk_m, OP_N), v(blk_m), /*dim=*/1);
        } else {
            LuisaTensor.reduce_absmax(A_local(blk_m, OP_N), v(blk_m), /*dim=*/1);
        }
        LuisaTensor.copy(v(blk_m), R(bx * blk_m));
    }
    return R;
}

// ---- 5h. row-wise inclusive max scan (CUMMAX on shared tiles) -------------
Tensor<tile_f32, 2> bench_cummax(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 blk_m = 4;
    constexpr tile_i32 threads = 128;

    Tensor<tile_f32, 2> S = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_M, blk_m), threads)) {
        auto A_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(blk_m, OP_N), tile_f32{});
        auto S_shared = LuisaTensor.alloc_shared(LuisaTensor.shape(blk_m, OP_N), tile_f32{});

        LuisaTensor.copy(A(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()),
                         A_shared(blk_m, OP_N));
        LuisaTensor.cummax(A_shared(blk_m, OP_N), S_shared(blk_m, OP_N), /*dim=*/1);
        LuisaTensor.copy(S_shared(blk_m, OP_N),
                         S(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()));
    }
    return S;
}

// ---- 5i. atomic add: D[i] += 1, one fetch-add per element (i32) -----------
Tensor<tile_i32, 2> bench_atomic_add() {
    constexpr tile_i32 blk_m = 32;
    constexpr tile_i32 threads = 256;

    Tensor<tile_i32, 2> D = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_i32{});

    for (auto bx : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_M, blk_m), threads)) {
        LuisaTensor.atomic_add(D(LuisaTensor.range(bx * blk_m, (bx + 1) * blk_m), LuisaTensor.all()), 1);
    }
    return D;
}

// ---- 5j. unary math: exp / sqrt / tanh / erf ------------------------------
template<int Tag>// 0=exp 1=sqrt 2=tanh 3=erf
Tensor<tile_f32, 2> bench_unary_op(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.copy(A(by * block_M, bx * block_N), A_local(block_M, block_N));
        if constexpr (Tag == 0) {
            C_local(block_M, block_N) = LuisaTensor.exp(A_local(block_M, block_N));
        } else if constexpr (Tag == 1) {
            C_local(block_M, block_N) = LuisaTensor.sqrt(A_local(block_M, block_N));
        } else if constexpr (Tag == 2) {
            C_local(block_M, block_N) = LuisaTensor.tanh(A_local(block_M, block_N));
        } else {
            C_local(block_M, block_N) = LuisaTensor.erf(A_local(block_M, block_N));
        }
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

// ---- 5k. pow: C = pow(A, 0.5) (binary IEEE math) ---------------------------
Tensor<tile_f32, 2> bench_pow(Tensor<tile_f32, 2> A) {
    constexpr tile_i32 block_M = 16, block_N = 16;
    constexpr tile_i32 threads = 256;

    Tensor<tile_f32, 2> C = LuisaTensor.empty(LuisaTensor.shape(OP_M, OP_N), tile_f32{});

    for (auto [bx, by] : LuisaTensor.Kernel(LuisaTensor.ceildiv(OP_N, block_N), LuisaTensor.ceildiv(OP_M, block_M), threads)) {
        auto A_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto E_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});
        auto C_local = LuisaTensor.alloc_fragment(LuisaTensor.shape(block_M, block_N), tile_f32{});

        LuisaTensor.copy(A(by * block_M, bx * block_N), A_local(block_M, block_N));
        LuisaTensor.fill(E_local(block_M, block_N), 0.5f);
        C_local(block_M, block_N) = LuisaTensor.pow(A_local(block_M, block_N), E_local(block_M, block_N));
        LuisaTensor.copy(C_local(block_M, block_N), C(by * block_M, bx * block_N));
    }
    return C;
}

int main(int argc, char *argv[]) {
    using namespace luisa::compute;
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    luisa::string_view backend = argc > 1 && argv != nullptr && argv[1] != nullptr ? luisa::string_view{argv[1]} : luisa::string_view{"cuda"};
    uint32_t iters = 20u;
    if (argc > 2 && argv[2] != nullptr) { iters = static_cast<uint32_t>(std::max(1, atoi(argv[2]))); }
    // --tensor: lower every eligible bench through the TENSOR_* CallOp path
    // (TileToKernelConfig::use_tensor).  Ineligible kernels keep the
    // partition path, so the two runs stay apples-to-apples per bench.
    bool use_tensor = false;
    for (auto i = 3; i < argc; ++i) {
        if (argv[i] != nullptr && luisa::string_view{argv[i]} == "--tensor") { use_tensor = true; }
    }
    TileToKernelConfig tile_config{.use_tensor = use_tensor};
    // NOTE: do NOT dispatch kernel.to_kernel<2>() when use_tensor is enabled —
    // it re-lowers with the DEFAULT config (use_tensor=false, partition path)
    // and its dispatch_size disagrees with the tensor-op lowering (e.g. the
    // rewritten GEMM uses tiles_m*tiles_n*threads).  The typed kernel below is
    // built from the SAME `result` (the tile_config-aware lowering) so the
    // --tensor path is actually benchmarked.

    Context ctx(executable);
    Device device = ctx.create_device(backend);
    Stream stream = device.create_stream();
    luisa::Clock clock;

    auto report = [&](luisa::string_view name, double ms, double gflops, uint32_t count) {
        if (gflops > 0.0) {
            LUISA_INFO("[tile-bench] {} : {:.3f} ms/iter ({:.2f} GFLOP/s, {} iters)",
                       name, ms, gflops, count);
        } else {
            LUISA_INFO("[tile-bench] {} : {:.3f} ms/iter ({} iters)", name, ms, count);
        }
    };
    auto check = [](luisa::string_view name, float err, float tol) {
        LUISA_INFO("[tile-bench] {} correctness: max sampled error = {}", name, err);
        LUISA_ASSERT(err < tol, "{} produced wrong results (max error {} >= {}).", name, err, tol);
    };
    // bandwidth-flavoured report: bytes moved per iteration -> GB/s
    auto report_bw = [&](luisa::string_view name, double ms, double bytes, uint32_t count) {
        LUISA_INFO("[tile-bench] {} : {:.3f} ms/iter ({:.2f} GB/s, {} iters)",
                   name, ms, bytes / (ms * 1e6), count);
    };

    // ---- bench_gemm ----------------------------------------------------------
    {
        constexpr uint32_t M = 512u, N = 512u, K = 512u;
        auto kernel = luisa::compute::tile::jit(bench_gemm).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<luisa::half>(M * K);
        auto bufB = device.create_buffer<luisa::half>(K * N);
        auto bufC = device.create_buffer<luisa::half>(M * N);
        luisa::vector<luisa::half> hA(M * K), hB(K * N), hC(M * N);
        for (auto i = 0u; i < M * K; ++i) { hA[i] = luisa::half{static_cast<float>((i % 8)) * 0.25f}; }
        for (auto i = 0u; i < K * N; ++i) { hB[i] = luisa::half{static_cast<float>((i % 4)) * 0.5f}; }
        stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

        kernel.validate(bufA, bufB, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        // warmup + verify
        stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto s = 0u; s < 4096u; ++s) {// sampled reference (512^3 host GEMM is slow)
            auto r = (s * 37u + 11u) % M;
            auto c = (s * 91u + 7u) % N;
            auto acc = 0.0f;
            for (auto k = 0u; k < K; ++k) {
                acc += static_cast<float>(hA[r * K + k]) * static_cast<float>(hB[k * N + c]);
            }
            err = luisa::max(err, luisa::abs(static_cast<float>(hC[r * N + c]) - acc));
        }
        check("bench_gemm", err, 2e-2f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        auto ms = clock.toc() / iters;
        report("bench_gemm", ms, 2.0 * M * N * K / (ms * 1e6), iters);
    }

    // ---- bench_gemm_4096 -----------------------------------------------------
    // 4096x4096x4096 f32 GEMM. This config hits the warp-K-split path in
    // tile_to_kernel (block_K = 256, MT*NT = 4 < threads = 32, small fragment
    // C_local -> single-warp barrier-free write-back, 16 KB shared -> safe on
    // Vulkan). Correctness is checked
    // against a double-precision host reference on sampled points; performance
    // is reported in GFLOP/s. Default iteration count is capped to keep the
    // benchmark fast (each iteration is ~137 GFLOP).
    {
        constexpr uint32_t M = 4096u, N = 4096u, K = 4096u;
        auto kernel = luisa::compute::tile::jit(bench_gemm_4096).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(M * K);
        auto bufB = device.create_buffer<float>(K * N);
        auto bufC = device.create_buffer<float>(M * N);
        luisa::vector<float> hA(M * K), hB(K * N), hC(M * N);
        for (auto i = 0u; i < M * K; ++i) {
            auto x = (i * 2654435761u) >> 13u;
            hA[i] = static_cast<float>(x & 0x3FFu) / 512.0f - 1.0f;// [-1, 1)
        }
        for (auto i = 0u; i < K * N; ++i) {
            auto x = (i * 40503u) >> 11u;
            hB[i] = static_cast<float>(x & 0x3FFu) / 512.0f - 1.0f;// [-1, 1)
        }
        stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

        kernel.validate(bufA, bufB, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        // warmup + verify
        stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        // err = max over checked points of (|device - double_reference| / tol),
        // tol = 2e-3 + 1e-5*|ref| (f32 accumulation of 4096 terms).
        auto err = 0.0f;
        luisa::vector<luisa::string> mismatches;
        auto check_point = [&](uint32_t r, uint32_t c) {
            if (!std::isfinite(hC[r * N + c])) {
                if (mismatches.size() < 8u) { mismatches.emplace_back(luisa::format("({},{}) non-finite {}", r, c, hC[r * N + c])); }
                err = 1e9f; return; }
            double acc = 0.0;
            for (auto k = 0u; k < K; ++k) {
                acc += static_cast<double>(hA[r * K + k]) * static_cast<double>(hB[k * N + c]);
            }
            auto diff = std::abs(static_cast<double>(hC[r * N + c]) - acc);
            auto tol = 2e-3 + 1e-5 * std::abs(acc);
            if (diff > tol && mismatches.size() < 8u) {
                mismatches.emplace_back(luisa::format("({},{}) dev={} ref={} diff={} tol={}",
                                                       r, c, hC[r * N + c], static_cast<float>(acc),
                                                       diff, tol));
            }
            err = luisa::max(err, static_cast<float>(diff / tol));
        };
        // two columns per row (every row covered) + four full rows
        for (auto r = 0u; r < M; ++r) {
            check_point(r, (r * 2654435761u) % N);
            check_point(r, ((r * 2654435761u) + 2047u) % N);
        }
        for (auto r : {0u, M / 4u, M / 2u, 3u * M / 4u}) {
            for (auto c = 0u; c < N; ++c) { check_point(r, c); }
        }
        for (auto &m : mismatches) { LUISA_WARNING("[bench_gemm_4096] mismatch: {}", m); }
        check("bench_gemm_4096", err, 1.0f);

        auto iters_4096 = iters < 5u ? iters : 5u;
        clock.tic();
        for (auto i = 0u; i < iters_4096; ++i) {
            stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        auto ms = clock.toc() / iters_4096;
        report("bench_gemm_4096", ms, 2.0 * M * N * K / (ms * 1e6), iters_4096);
    }

    // ---- bench_rms_norm ------------------------------------------------------
    {
        constexpr uint32_t M = 2048u, N = 256u;
        auto kernel = luisa::compute::tile::jit(bench_rms_norm).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufB = device.create_buffer<float>(M * N);
        luisa::vector<float> hA(M * N), hB(M * N);
        for (auto i = 0u; i < M * N; ++i) { hA[i] = static_cast<float>(i % 97) * 0.05f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufB);
        auto typed = kernel.to_kernel<1>();
        auto sh = device.compile(typed);
        stream << sh(bufA, bufB).dispatch(result.dispatch_size.x)
               << bufB.copy_to(luisa::span{hB}) << synchronize();
        auto err = 0.0f;
        for (auto r = 0u; r < M; r += 64u) {
            auto s = 0.0f;
            for (auto c = 0u; c < N; ++c) { s += hA[r * N + c] * hA[r * N + c]; }
            auto scale = 1.0f / luisa::sqrt(s / static_cast<float>(N) + 1e-12f);
            for (auto c = 0u; c < N; ++c) {
                err = luisa::max(err, luisa::abs(hB[r * N + c] - hA[r * N + c] * scale));
            }
        }
        check("bench_rms_norm", err, 1e-3f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufB).dispatch(result.dispatch_size.x);
        }
        stream << synchronize();
        report("bench_rms_norm", clock.toc() / iters, 0.0, iters);
    }

    // ---- bench_scan ----------------------------------------------------------
    {
        constexpr uint32_t M = 1024u, N = 256u;
        auto kernel = luisa::compute::tile::jit(bench_scan).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufS = device.create_buffer<float>(M * N);
        luisa::vector<float> hA(M * N), hS(M * N);
        for (auto i = 0u; i < M * N; ++i) { hA[i] = static_cast<float>(i % 13) * 0.25f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufS);
        auto typed = kernel.to_kernel<1>();
        auto sh = device.compile(typed);
        stream << sh(bufA, bufS).dispatch(result.dispatch_size.x)
               << bufS.copy_to(luisa::span{hS}) << synchronize();
        auto err = 0.0f;
        for (auto r = 0u; r < M; r += 32u) {
            auto acc = 0.0f;
            for (auto c = 0u; c < N; ++c) {
                acc += hA[r * N + c];
                err = luisa::max(err, luisa::abs(hS[r * N + c] - acc));
            }
        }
        check("bench_scan", err, 1e-2f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufS).dispatch(result.dispatch_size.x);
        }
        stream << synchronize();
        report("bench_scan", clock.toc() / iters, 0.0, iters);
    }

    // ---- bench_scan_1d -------------------------------------------------------
    // 1D inclusive scan of a single 2048-element line with 256 threads. The
    // default warp-per-line partition leaves 7 of 8 warps idle, so the
    // lowering switches to the two-pass block scan (plan 2.8).  All-ones input
    // keeps the prefix sums exact in f32 so the reference is exact.
    {
        constexpr uint32_t N = 2048u;
        auto kernel = luisa::compute::tile::jit(bench_scan_1d).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(N);
        auto bufS = device.create_buffer<float>(N);
        luisa::vector<float> hA(N, 1.0f), hS(N);
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufS);
        auto typed = kernel.to_kernel<1>();
        auto sh = device.compile(typed);
        stream << sh(bufA, bufS).dispatch(result.dispatch_size.x)
               << bufS.copy_to(luisa::span{hS}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < N; ++i) {
            err = luisa::max(err, luisa::abs(hS[i] - static_cast<float>(i + 1u)));
        }
        check("bench_scan_1d", err, 1e-5f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufS).dispatch(result.dispatch_size.x);
        }
        stream << synchronize();
        report("bench_scan_1d", clock.toc() / iters, 0.0, iters);
    }

    // ---- bench_add -----------------------------------------------------------
    {
        constexpr uint32_t M = 2048u, N = 2048u;
        auto kernel = luisa::compute::tile::jit(bench_add).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(M * N);
        auto bufB = device.create_buffer<float>(M * N);
        auto bufC = device.create_buffer<float>(M * N);
        luisa::vector<float> hA(M * N), hB(M * N), hC(M * N);
        for (auto i = 0u; i < M * N; ++i) {
            hA[i] = static_cast<float>(i % 31) * 0.125f;
            hB[i] = static_cast<float>(i % 17) * 0.25f;
        }
        stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

        kernel.validate(bufA, bufB, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < M * N; i += 4099u) {
            err = luisa::max(err, luisa::abs(hC[i] - (hA[i] + hB[i])));
        }
        check("bench_add", err, 1e-3f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        auto ms = clock.toc() / iters;
        report("bench_add", ms, static_cast<double>(M) * N / (ms * 1e6), iters);
    }

    // ---- bench_add_global / bench_copy_global (tensor-op path only) ----------
    // Pure-Global whole-tensor variants; the default partition path rejects
    // extent-less global views, so these run only under --tensor.
    constexpr uint32_t OM = 8192u, ON = 512u;// 4,194,304 elements per container
    constexpr double op_elems = static_cast<double>(OM) * static_cast<double>(ON);
    if (use_tensor) {
        {
            constexpr uint32_t M = 2048u, N = 2048u;
            auto kernel = luisa::compute::tile::jit(bench_add_global).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            auto bufA = device.create_buffer<float>(M * N);
            auto bufB = device.create_buffer<float>(M * N);
            auto bufC = device.create_buffer<float>(M * N);
            luisa::vector<float> hA(M * N), hB(M * N), hC(M * N);
            for (auto i = 0u; i < M * N; ++i) {
                hA[i] = static_cast<float>(i % 31) * 0.125f;
                hB[i] = static_cast<float>(i % 17) * 0.25f;
            }
            stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

            kernel.validate(bufA, bufB, bufC);
            using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
            auto sh = device.compile(typed);
            stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < M * N; i += 4099u) {
                err = luisa::max(err, luisa::abs(hC[i] - (hA[i] + hB[i])));
            }
            check("bench_add_global", err, 1e-3f);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
            }
            stream << synchronize();
            auto ms = clock.toc() / iters;
            report("bench_add_global", ms, static_cast<double>(M) * N / (ms * 1e6), iters);
        }
        {
            constexpr uint32_t M = 2048u, N = 2048u;
            auto kernel = luisa::compute::tile::jit(bench_copy_global).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            auto bufA = device.create_buffer<float>(M * N);
            auto bufC = device.create_buffer<float>(M * N);
            luisa::vector<float> hA(M * N), hC(M * N);
            for (auto i = 0u; i < M * N; ++i) { hA[i] = static_cast<float>(i % 7u) * 0.5f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            kernel.validate(bufA, bufC);
            using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
            auto sh = device.compile(typed);
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < M * N; i += 4099u) { err = luisa::max(err, luisa::abs(hC[i] - hA[i])); }
            check("bench_copy_global", err, 1e-6f);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
            }
            stream << synchronize();
            auto ms = clock.toc() / iters;
            report("bench_copy_global", ms, 2.0 * static_cast<double>(M) * N / (ms * 1e6), iters);
        }
        // ---- bench_clamp_global / bench_unary_exp_global / bench_reduce_sum_global
        {
            auto kernel = luisa::compute::tile::jit(bench_clamp_global).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            auto bufA = device.create_buffer<float>(OM * ON);
            auto bufC = device.create_buffer<float>(OM * ON);
            luisa::vector<float> hA(OM * ON), hC(OM * ON);
            for (auto i = 0u; i < OM * ON; ++i) { hA[i] = static_cast<float>(i % 31u) * 0.25f - 3.0f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            kernel.validate(bufA, bufC);
            using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
            auto sh = device.compile(typed);
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < OM * ON; i += 4099u) {
                auto ref = luisa::clamp(hA[i], -0.5f, 0.5f);
                err = luisa::max(err, luisa::abs(hC[i] - ref));
            }
            check("bench_clamp_global", err, 1e-6f);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
            }
            stream << synchronize();
            auto ms = clock.toc() / iters;
            report_bw("bench_clamp_global", ms, 8.0 * op_elems, iters);
        }
        {
            auto kernel = luisa::compute::tile::jit(bench_unary_exp_global).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            auto bufA = device.create_buffer<float>(OM * ON);
            auto bufC = device.create_buffer<float>(OM * ON);
            luisa::vector<float> hA(OM * ON), hC(OM * ON);
            for (auto i = 0u; i < OM * ON; ++i) { hA[i] = 0.25f + static_cast<float>(i % 100u) * 0.001f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            kernel.validate(bufA, bufC);
            using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
            auto sh = device.compile(typed);
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < OM * ON; i += 4099u) {
                err = luisa::max(err, luisa::abs(hC[i] - std::exp(hA[i])));
            }
            check("bench_unary_exp_global", err, 1e-3f);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
            }
            stream << synchronize();
            auto ms = clock.toc() / iters;
            report_bw("bench_unary_exp_global", ms, 8.0 * op_elems, iters);
        }
        {
            auto kernel = luisa::compute::tile::jit(bench_reduce_sum_global).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            auto bufA = device.create_buffer<float>(OM * ON);
            auto bufR = device.create_buffer<float>(OM);
            luisa::vector<float> hA(OM * ON), hR(OM);
            for (auto i = 0u; i < OM * ON; ++i) { hA[i] = static_cast<float>(i % 13u) * 0.25f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            kernel.validate(bufA, bufR);
            using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
            auto sh = device.compile(typed);
            stream << sh(bufA, bufR).dispatch(result.dispatch_size.x, result.dispatch_size.y)
                   << bufR.copy_to(luisa::span{hR}) << synchronize();
            auto err = 0.0f;
            for (auto r = 0u; r < OM; r += 64u) {
                auto s = 0.0f;
                for (auto c = 0u; c < ON; ++c) { s += hA[r * ON + c]; }
                err = luisa::max(err, luisa::abs(hR[r] - s));
            }
            check("bench_reduce_sum_global", err, 1e-3f);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufR).dispatch(result.dispatch_size.x, result.dispatch_size.y);
            }
            stream << synchronize();
            auto ms = clock.toc() / iters;
            report_bw("bench_reduce_sum_global", ms, 4.0 * op_elems, iters);
        }
    }

    // =========================================================================
    // Per-operator micro-benchmarks (section 5 kernels above).  Every container
    // is 8192 x 512 = 4,194,304 elements so memory-system shortages are visible.
    // =========================================================================
    // ---- bench_copy ----------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_copy).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hA(OM * ON), hC(OM * ON);
        for (auto i = 0u; i < OM * ON; ++i) { hA[i] = static_cast<float>(i % 7u) * 0.5f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < OM * ON; i += 4099u) { err = luisa::max(err, luisa::abs(hC[i] - hA[i])); }
        check("bench_copy", err, 1e-6f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        report_bw("bench_copy", clock.toc() / iters, 2.0 * 4.0 * op_elems, iters);
    }

    // ---- bench_clear ---------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_clear).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hC(OM * ON, 1.0f);

        kernel.validate(bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < OM * ON; i += 4099u) { err = luisa::max(err, luisa::abs(hC[i])); }
        check("bench_clear", err, 1e-6f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        report_bw("bench_clear", clock.toc() / iters, 4.0 * op_elems, iters);
    }

    // ---- bench_fill ----------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_fill).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hC(OM * ON);

        kernel.validate(bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < OM * ON; i += 4099u) { err = luisa::max(err, luisa::abs(hC[i] - 3.5f)); }
        check("bench_fill", err, 1e-6f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        report_bw("bench_fill", clock.toc() / iters, 4.0 * op_elems, iters);
    }

    // ---- bench_clamp ---------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_clamp).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hA(OM * ON), hC(OM * ON);
        for (auto i = 0u; i < OM * ON; ++i) { hA[i] = static_cast<float>(i % 31u) * 0.25f - 3.0f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < OM * ON; i += 4099u) {
            auto ref = luisa::clamp(hA[i], -0.5f, 0.5f);
            err = luisa::max(err, luisa::abs(hC[i] - ref));
        }
        check("bench_clamp", err, 1e-6f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        report_bw("bench_clamp", clock.toc() / iters, 8.0 * op_elems, iters);
    }

    // ---- bench_saxpy ---------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_saxpy).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufB = device.create_buffer<float>(OM * ON);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hA(OM * ON), hB(OM * ON), hC(OM * ON);
        for (auto i = 0u; i < OM * ON; ++i) {
            hA[i] = static_cast<float>(i % 31u) * 0.125f;
            hB[i] = static_cast<float>(i % 17u) * 0.25f;
        }
        stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

        kernel.validate(bufA, bufB, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < OM * ON; i += 4099u) {
            err = luisa::max(err, luisa::abs(hC[i] - (hA[i] / 0.4f + hB[i])));
        }
        check("bench_saxpy", err, 1e-5f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufB, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        auto ms = clock.toc() / iters;
        report("bench_saxpy", ms, 2.0 * op_elems / (ms * 1e6), iters);
        report_bw("bench_saxpy", ms, 12.0 * op_elems, iters);
    }

    // ---- bench_transpose -----------------------------------------------------
    // Full-matrix transpose of a 72x72 (5184-element) container from a single
    // 288-thread block (41.5 KB shared). Multi-block tiled transpose is not
    // expressible today and non-divisor tile shapes hit the _partition_2d
    // coverage bug (see the kernel comment) — both are lowering findings in
    // perf_report.md, so this kernel is latency-bound by design.
    {
        constexpr uint32_t TM = 72u, TN = 72u;// 5184 elements each (> 4096)
        auto kernel = luisa::compute::tile::jit(bench_transpose).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(TM * TN);
        auto bufB = device.create_buffer<float>(TM * TN);
        luisa::vector<float> hA(TM * TN), hB(TM * TN);
        for (auto i = 0u; i < TM * TN; ++i) { hA[i] = static_cast<float>(i % 19u) * 0.5f - 4.0f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufB);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufA, bufB).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufB.copy_to(luisa::span{hB}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < TM; ++i) {
            for (auto j = 0u; j < TN; ++j) {
                err = luisa::max(err, luisa::abs(hB[j * TM + i] - hA[i * TN + j]));
            }
        }
        check("bench_transpose", err, 1e-6f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufB).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        report_bw("bench_transpose", clock.toc() / iters,
                  2.0 * 4.0 * static_cast<double>(TM) * TN, iters);
    }

    // ---- bench_reduce_{sum,max,min,abssum,absmax} ----------------------------
    {
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufR = device.create_buffer<float>(OM);
        luisa::vector<float> hA(OM * ON), hR(OM);
        for (auto i = 0u; i < OM * ON; ++i) {
            auto x = (i * 2654435761u) >> 13u;
            hA[i] = static_cast<float>(x & 0x3FFu) / 512.0f - 1.0f;// [-1, 1)
        }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        auto run_reduce = [&]<int Tag>(const char *name, float tol) {
            auto kernel = luisa::compute::tile::jit(bench_reduce_op<Tag>).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            kernel.validate(bufA, bufR);
            auto typed = kernel.template to_kernel<1>();
            auto sh = device.compile(typed);
            stream << sh(bufA, bufR).dispatch(result.dispatch_size.x)
                   << bufR.copy_to(luisa::span{hR}) << synchronize();
            auto err = 0.0f;
            for (auto r = 0u; r < OM; r += 64u) {
                double acc = 0.0;
                double aacc = 0.0;
                float mx = -std::numeric_limits<float>::infinity();
                float mn = std::numeric_limits<float>::infinity();
                float amx = 0.0f;
                for (auto c = 0u; c < ON; ++c) {
                    auto v = hA[r * ON + c];
                    acc += static_cast<double>(v);
                    aacc += static_cast<double>(v < 0 ? -v : v);
                    mx = luisa::max(mx, v);
                    mn = luisa::min(mn, v);
                    amx = luisa::max(amx, luisa::abs(v));
                }
                float ref{};
                if constexpr (Tag == 0) { ref = static_cast<float>(acc); }
                else if constexpr (Tag == 1) { ref = mx; }
                else if constexpr (Tag == 2) { ref = mn; }
                else if constexpr (Tag == 3) { ref = static_cast<float>(aacc); }
                else { ref = amx; }
                err = luisa::max(err, luisa::abs(hR[r] - ref));
            }
            check(name, err, tol);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufR).dispatch(result.dispatch_size.x);
            }
            stream << synchronize();
            report_bw(name, clock.toc() / iters, 4.0 * op_elems, iters);
        };
        run_reduce.operator()<0>("bench_reduce_sum", 0.1f);
        run_reduce.operator()<1>("bench_reduce_max", 1e-6f);
        run_reduce.operator()<2>("bench_reduce_min", 1e-6f);
        // abssum rows sum |x| <= 512 -> absolute tolerance 0.1
        run_reduce.operator()<3>("bench_reduce_abssum", 0.1f);
        run_reduce.operator()<4>("bench_reduce_absmax", 1e-6f);
    }

    // ---- bench_cummax --------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_cummax).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufS = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hA(OM * ON), hS(OM * ON);
        for (auto i = 0u; i < OM * ON; ++i) { hA[i] = static_cast<float>(i % 13u) * 0.25f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufS);
        auto typed = kernel.to_kernel<1>();
        auto sh = device.compile(typed);
        stream << sh(bufA, bufS).dispatch(result.dispatch_size.x)
               << bufS.copy_to(luisa::span{hS}) << synchronize();
        auto err = 0.0f;
        for (auto r = 0u; r < OM; r += 128u) {
            auto mx = -std::numeric_limits<float>::infinity();
            for (auto c = 0u; c < ON; ++c) {
                mx = luisa::max(mx, hA[r * ON + c]);
                err = luisa::max(err, luisa::abs(hS[r * ON + c] - mx));
            }
        }
        check("bench_cummax", err, 1e-6f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufS).dispatch(result.dispatch_size.x);
        }
        stream << synchronize();
        report_bw("bench_cummax", clock.toc() / iters, 8.0 * op_elems, iters);
    }

    // ---- bench_atomic_add ----------------------------------------------------
    // Each dispatch performs exactly one fetch-add per element, so after k
    // dispatches D[i] == k exactly — verified both after warmup and after the
    // timing loop.
    {
        auto kernel = luisa::compute::tile::jit(bench_atomic_add).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufD = device.create_buffer<int>(OM * ON);
        luisa::vector<int> hD(OM * ON, 0);
        stream << bufD.copy_from(luisa::span{hD}) << synchronize();

        kernel.validate(bufD);
        auto typed = kernel.to_kernel<1>();
        auto sh = device.compile(typed);
        uint32_t dispatch_count = 0u;
        stream << sh(bufD).dispatch(result.dispatch_size.x) << synchronize();
        dispatch_count++;
        stream << bufD.copy_to(luisa::span{hD}) << synchronize();
        auto err = 0;
        for (auto i = 0u; i < OM * ON; i += 4099u) { err = luisa::max(err, luisa::abs(hD[i] - static_cast<int>(dispatch_count))); }
        check("bench_atomic_add", static_cast<float>(err), 0.5f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufD).dispatch(result.dispatch_size.x);
            dispatch_count++;
        }
        stream << synchronize();
        auto ms = clock.toc() / iters;
        stream << bufD.copy_to(luisa::span{hD}) << synchronize();
        auto err2 = 0;
        for (auto i = 0u; i < OM * ON; i += 4099u) { err2 = luisa::max(err2, luisa::abs(hD[i] - static_cast<int>(dispatch_count))); }
        LUISA_INFO("[tile-bench] bench_atomic_add post-run check: max |D-{}| = {}", dispatch_count, err2);
        LUISA_ASSERT(err2 == 0, "bench_atomic_add lost updates (max |D-{}| = {}).", dispatch_count, err2);
        report_bw("bench_atomic_add", ms, 8.0 * op_elems, iters);
    }

    // ---- bench_{exp,sqrt,tanh,erf} -------------------------------------------
    {
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hA(OM * ON), hC(OM * ON);
        for (auto i = 0u; i < OM * ON; ++i) {
            hA[i] = static_cast<float>(i % 32u) * 0.125f - 2.0f;// [-2, 1.875]
        }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        auto run_unary = [&]<int Tag>(const char *name, float tol) {
            auto kernel = luisa::compute::tile::jit(bench_unary_op<Tag>).compile();
            auto result = tile_to_kernel(kernel.function(), tile_config);
            kernel.validate(bufA, bufC);
            auto typed = kernel.template to_kernel<2>();
            auto sh = device.compile(typed);
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < OM * ON; i += 4099u) {
                auto x = hA[i];
                float ref{};
                if constexpr (Tag == 0) { ref = std::exp(x); }
                else if constexpr (Tag == 1) { ref = std::sqrt(luisa::max(x, 0.0f)); }
                else if constexpr (Tag == 2) { ref = std::tanh(x); }
                else { ref = std::erf(x); }
                err = luisa::max(err, luisa::abs(hC[i] - ref));
            }
            check(name, err, tol);

            clock.tic();
            for (auto i = 0u; i < iters; ++i) {
                stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
            }
            stream << synchronize();
            report_bw(name, clock.toc() / iters, 8.0 * op_elems, iters);
        };
        run_unary.operator()<0>("bench_exp", 2e-3f);
        run_unary.operator()<1>("bench_sqrt", 1e-5f);
        run_unary.operator()<2>("bench_tanh", 1e-3f);
        run_unary.operator()<3>("bench_erf", 1e-3f);
    }

    // ---- bench_pow -----------------------------------------------------------
    {
        auto kernel = luisa::compute::tile::jit(bench_pow).compile();
        auto result = tile_to_kernel(kernel.function(), tile_config);
        auto bufA = device.create_buffer<float>(OM * ON);
        auto bufC = device.create_buffer<float>(OM * ON);
        luisa::vector<float> hA(OM * ON), hC(OM * ON);
        for (auto i = 0u; i < OM * ON; ++i) { hA[i] = static_cast<float>(i % 31u) * 0.125f + 0.25f; }
        stream << bufA.copy_from(luisa::span{hA}) << synchronize();

        kernel.validate(bufA, bufC);
        using KTensor = decltype(kernel.to_kernel<2>());
        auto fb = luisa::const_pointer_cast<const ::luisa::compute::detail::FunctionBuilder>(result.function);
        KTensor typed{std::move(fb)};
        auto sh = device.compile(typed);
        stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y)
               << bufC.copy_to(luisa::span{hC}) << synchronize();
        auto err = 0.0f;
        for (auto i = 0u; i < OM * ON; i += 4099u) {
            err = luisa::max(err, luisa::abs(hC[i] - std::pow(hA[i], 0.5f)));
        }
        check("bench_pow", err, 1e-3f);

        clock.tic();
        for (auto i = 0u; i < iters; ++i) {
            stream << sh(bufA, bufC).dispatch(result.dispatch_size.x, result.dispatch_size.y);
        }
        stream << synchronize();
        report_bw("bench_pow", clock.toc() / iters, 8.0 * op_elems, iters);
    }

    LUISA_INFO("[tile-bench] all benchmarks passed on backend '{}'.", backend);
    return 0;
}

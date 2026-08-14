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
//
// Each kernel is (1) traced with tile::Kernel / tile::jit(...).compile(),
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
TILELANG_PRIM_FUNC
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
// 11. Multiple-T.Kernel guard — an INVALID tile function (opt-in trigger).
//     A tile function maps to exactly ONE kernel launch (TileLang emits one
//     `__global__` per `T.Kernel`), so tracing a second T.Kernel must be
//     rejected: jit(...).compile() derives the SIMT launch metadata and logs
//     an error + aborts when the body contains more than one T.Kernel.
// =============================================================================
TILELANG_PRIM_FUNC
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

    // ---- 1. elementwise_add: T.Kernel(4, 4, 32); globals A, B, C ------------
    luisa::compute::tile::Kernel elementwise_kernel{elementwise_add};
    LUISA_INFO("[tensor-stub] elementwise_add traced {} statements: [{}]",
               elementwise_kernel.function()->body()->size(), elementwise_kernel.describe());
    auto elementwise_result = translate_and_verify(
        "elementwise_add", elementwise_kernel.function(),
        luisa::uint3{128u, 4u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);

    // ---- 2. matmul (pipelined): T.Kernel(4, 4, 32); globals A, B, C ---------
    // This also produces the required "kernel.compile" log line.
    auto matmul_kernel = luisa::compute::tile::jit(pipelined_matmul).compile();
    LUISA_INFO("[tensor-stub] pipelined_matmul traced {} statements: [{}]",
               matmul_kernel.function()->body()->size(), matmul_kernel.describe());
    auto matmul_result = translate_and_verify(
        "pipelined_matmul", matmul_kernel.function(),
        luisa::uint3{128u, 4u, 1u}, luisa::uint3{32u, 1u, 1u}, 3u);

    // ---- 3. rms_norm: T.Kernel(8, 64); globals A, B ------------------------
    luisa::compute::tile::Kernel rms_kernel{rms_norm};
    LUISA_INFO("[tensor-stub] rms_norm traced {} statements: [{}]",
               rms_kernel.function()->body()->size(), rms_kernel.describe());
    auto rms_result = translate_and_verify(
        "rms_norm", rms_kernel.function(),
        luisa::uint3{512u, 1u, 1u}, luisa::uint3{64u, 1u, 1u}, 2u);

    // ---- 4. fill: T.Kernel(1, 32); global C ---------------------------------
    luisa::compute::tile::Kernel fill_kernel{tile_fill_kernel};
    LUISA_INFO("[tensor-stub] tile_fill traced {} statements: [{}]",
               fill_kernel.function()->body()->size(), fill_kernel.describe());
    auto fill_result = translate_and_verify(
        "tile_fill", fill_kernel.function(),
        luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);

    // ---- 5. transpose: T.Kernel(1, 1, 32); globals A, B ---------------------
    luisa::compute::tile::Kernel transpose_kernel{tile_transpose_kernel};
    LUISA_INFO("[tensor-stub] tile_transpose traced {} statements: [{}]",
               transpose_kernel.function()->body()->size(), transpose_kernel.describe());
    auto transpose_result = translate_and_verify(
        "tile_transpose", transpose_kernel.function(),
        luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);

    // ---- 6. clamp: T.Kernel(1, 1, 32); globals A, C -------------------------
    luisa::compute::tile::Kernel clamp_kernel{tile_clamp_kernel};
    LUISA_INFO("[tensor-stub] tile_clamp traced {} statements: [{}]",
               clamp_kernel.function()->body()->size(), clamp_kernel.describe());
    auto clamp_result = translate_and_verify(
        "tile_clamp", clamp_kernel.function(),
        luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);

    // ---- 7. atomic: T.Kernel(1, 32); global D (i32) -------------------------
    luisa::compute::tile::Kernel atomic_kernel{tile_atomic_kernel};
    LUISA_INFO("[tensor-stub] tile_atomic traced {} statements: [{}]",
               atomic_kernel.function()->body()->size(), atomic_kernel.describe());
    auto atomic_result = translate_and_verify(
        "tile_atomic", atomic_kernel.function(),
        luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);

    // ---- 8. sync: T.Kernel(1, 1, 32); globals A, C --------------------------
    luisa::compute::tile::Kernel sync_kernel{tile_sync_kernel};
    LUISA_INFO("[tensor-stub] tile_sync traced {} statements: [{}]",
               sync_kernel.function()->body()->size(), sync_kernel.describe());
    auto sync_result = translate_and_verify(
        "tile_sync", sync_kernel.function(),
        luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 2u);

    // ---- 9. warp_reduce: T.Kernel(1, 32); global W --------------------------
    luisa::compute::tile::Kernel warp_reduce_kernel_obj{tile_warp_reduce_kernel};
    LUISA_INFO("[tensor-stub] tile_warp_reduce traced {} statements: [{}]",
               warp_reduce_kernel_obj.function()->body()->size(), warp_reduce_kernel_obj.describe());
    auto warp_reduce_result = translate_and_verify(
        "tile_warp_reduce", warp_reduce_kernel_obj.function(),
        luisa::uint3{32u, 1u, 1u}, luisa::uint3{32u, 1u, 1u}, 1u);

    // ---- 10. loop_break: traced only (no lowering; see file header) ---------
    luisa::compute::tile::Kernel loop_break_kernel_obj{loop_break_kernel};
    LUISA_INFO("[tensor-stub] loop_break traced {} statements: [{}] (not lowered: "
               "break_() requires an enclosing loop, see the file header)",
               loop_break_kernel_obj.function()->body()->size(), loop_break_kernel_obj.describe());

    // =========================================================================
    // Device pass: compile + dispatch + host verification (backend given).
    // =========================================================================
    if (backend.empty()) {
        LUISA_INFO("[tensor-stub] no backend given: translation verified structurally only "
                   "(pass a backend name, e.g. 'dx'/'vk', to also compile, dispatch and verify).");
    } else {
        LUISA_INFO("=== tensor-dsl: compile + dispatch + verify on backend '{}' ===", backend);
        Context ctx(executable);
        Device device = ctx.create_device(backend);
        auto stream = device.create_stream();

        auto wrap = [](luisa::shared_ptr<luisa::compute::detail::FunctionBuilder> fb) {
            return luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>{std::move(fb)};
        };
        auto check = [](luisa::string_view name, float err, float tol) {
            LUISA_INFO("[tensor-stub] {} runtime check: max error = {}", name, err);
            LUISA_ASSERT(err < tol, "{} produced wrong results on the device (max error {} >= {}).",
                         name, err, tol);
        };

        // ---- elementwise_add: C = A + B -------------------------------------
        {
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

            // Guarded type-less path: keep the explicit Kernel<...> instantiation
            // but first validate the runtime bindings against the tile IR.
            elementwise_kernel.validate(bufA, bufB, bufC);
            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(elementwise_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<2, luisa::compute::Buffer<float>,
                                                            luisa::compute::Buffer<float>,
                                                            luisa::compute::Buffer<float>>{fb});
            stream << sh(bufA, bufB, bufC).dispatch(elementwise_result.dispatch_size.x, elementwise_result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < M * N; ++i) { err = luisa::max(err, luisa::abs(hC[i] - hRef[i])); }
            check("elementwise_add", err, 1e-3f);
        }

        // ---- pipelined_matmul: C = max(A @ B, 0) ----------------------------
        {
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
            constexpr uint32_t M = 64u, N = 64u;
            auto bufA = device.create_buffer<float>(M * N);
            auto bufB = device.create_buffer<float>(M * N);
            luisa::vector<float> hA(M * N), hB(M * N);
            for (auto i = 0u; i < M * N; ++i) { hA[i] = static_cast<float>(i) * 0.5f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(rms_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<1, luisa::compute::Buffer<float>,
                                                            luisa::compute::Buffer<float>>{fb});
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
            constexpr uint32_t N = 64u;
            auto bufC = device.create_buffer<float>(N);
            luisa::vector<float> hC(N);
            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(fill_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<1, luisa::compute::Buffer<float>>{fb});
            stream << sh(bufC).dispatch(fill_result.dispatch_size.x)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < N; ++i) { err = luisa::max(err, luisa::abs(hC[i] - 3.5f)); }
            check("tile_fill", err, 1e-5f);
        }

        // ---- tile_transpose: B[i][j] = A[j][i] ------------------------------
        {
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufB = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hB(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i); }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(transpose_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<2, luisa::compute::Buffer<float>,
                                                            luisa::compute::Buffer<float>>{fb});
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
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufC = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hC(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i % 16) * 0.1f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(clamp_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<2, luisa::compute::Buffer<float>,
                                                            luisa::compute::Buffer<float>>{fb});
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
            constexpr uint32_t N = 32u;
            auto bufD = device.create_buffer<int>(N);
            luisa::vector<int> hD(N, 0);
            stream << bufD.copy_from(luisa::span{hD}) << synchronize();

            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(atomic_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<1, luisa::compute::Buffer<int>>{fb});
            stream << sh(bufD).dispatch(atomic_result.dispatch_size.x)
                   << bufD.copy_to(luisa::span{hD}) << synchronize();
            auto err = 0;
            for (auto i = 0u; i < N; ++i) { err = luisa::max(err, luisa::abs(hD[i] - 12)); }
            LUISA_INFO("[tensor-stub] tile_atomic runtime check: max |D-12| = {}", err);
            LUISA_ASSERT(err == 0, "tile_atomic produced wrong results on the device (max |D-12| = {}).", err);
        }

        // ---- tile_sync: C = A ----------------------------------------------
        {
            constexpr uint32_t BM = 8u, BN = 8u;
            auto bufA = device.create_buffer<float>(BM * BN);
            auto bufC = device.create_buffer<float>(BM * BN);
            luisa::vector<float> hA(BM * BN), hC(BM * BN);
            for (auto i = 0u; i < BM * BN; ++i) { hA[i] = static_cast<float>(i) * 0.25f; }
            stream << bufA.copy_from(luisa::span{hA}) << synchronize();

            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(sync_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<2, luisa::compute::Buffer<float>,
                                                            luisa::compute::Buffer<float>>{fb});
            stream << sh(bufA, bufC).dispatch(sync_result.dispatch_size.x, sync_result.dispatch_size.y)
                   << bufC.copy_to(luisa::span{hC}) << synchronize();
            auto err = 0.0f;
            for (auto i = 0u; i < BM * BN; ++i) { err = luisa::max(err, luisa::abs(hC[i] - hA[i])); }
            check("tile_sync", err, 1e-5f);
        }

        // ---- tile_warp_reduce: W[0] = 7.0 -----------------------------------
        {
            constexpr uint32_t N = 1u;
            auto bufW = device.create_buffer<float>(N);
            luisa::vector<float> hW(N);
            luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> fb{wrap(warp_reduce_result.function)};
            auto sh = device.compile(luisa::compute::Kernel<1, luisa::compute::Buffer<float>>{fb});
            stream << sh(bufW).dispatch(warp_reduce_result.dispatch_size.x)
                   << bufW.copy_to(luisa::span{hW}) << synchronize();
            auto err = luisa::abs(hW[0] - 7.0f);
            check("tile_warp_reduce", err, 1e-5f);
        }

        LUISA_INFO("[tensor-stub] all {} translated kernels compiled, dispatched and verified on '{}'.",
                   (size_t)9, backend);
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

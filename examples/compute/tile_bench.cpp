// =============================================================================
// tile_bench.cpp — performance benchmark for the tile_to_kernel lowering
// =============================================================================
// Benchmark companion of tensor_stub.cpp: the same TileLang-style "pure tile"
// DSL kernels, at realistic sizes, timed on the backend named on the command
// line (`cuda` / `dx` / `vk`).  It exercises the block-partitioned / warp-
// collective fast paths emitted by tile_to_kernel (see lc_optimize):
//
//   bench_gemm     GEMM with block-partitioned C tile + register accumulators
//                  (PIPELINED / COPY / CLEAR / GEMM / STORE)
//   bench_rms_norm row reduction with warp all-reduce (REDUCE_SUM / RSQRT /
//                  BINARY / STORE) + staged global->fragment copies
//   bench_scan     inclusive prefix scan with WARP_PREFIX_SUM (CUMSUM)
//   bench_add      whole-tile elementwise add (COPY / BINARY / STORE)
//
// Every kernel is (1) traced with tile::jit(...).compile(), (2) lowered with
// tile_to_kernel, (3) dispatched `iters` times and timed, and (4) checked
// against a host-side reference (sampled for the large GEMM).
// =============================================================================

#include <luisa/dsl/tensor.h>
#include <luisa/dsl/tile_to_kernel.h>
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

int main(int argc, char *argv[]) {
    using namespace luisa::compute;
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    luisa::string_view backend = argc > 1 && argv != nullptr && argv[1] != nullptr ? luisa::string_view{argv[1]} : luisa::string_view{"cuda"};
    uint32_t iters = 20u;
    if (argc > 2 && argv[2] != nullptr) { iters = static_cast<uint32_t>(std::max(1, atoi(argv[2]))); }

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

    // ---- bench_gemm ----------------------------------------------------------
    {
        constexpr uint32_t M = 512u, N = 512u, K = 512u;
        auto kernel = luisa::compute::tile::jit(bench_gemm).compile();
        auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
        auto bufA = device.create_buffer<luisa::half>(M * K);
        auto bufB = device.create_buffer<luisa::half>(K * N);
        auto bufC = device.create_buffer<luisa::half>(M * N);
        luisa::vector<luisa::half> hA(M * K), hB(K * N), hC(M * N);
        for (auto i = 0u; i < M * K; ++i) { hA[i] = luisa::half{static_cast<float>((i % 8)) * 0.25f}; }
        for (auto i = 0u; i < K * N; ++i) { hB[i] = luisa::half{static_cast<float>((i % 4)) * 0.5f}; }
        stream << bufA.copy_from(luisa::span{hA}) << bufB.copy_from(luisa::span{hB}) << synchronize();

        kernel.validate(bufA, bufB, bufC);
        auto typed = kernel.to_kernel<2>();
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
        auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
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
        auto typed = kernel.to_kernel<2>();
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
        auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
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
        auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
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
        auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
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
        auto result = tile_to_kernel(kernel.function(), TileToKernelConfig{});
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
        auto typed = kernel.to_kernel<2>();
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

    LUISA_INFO("[tile-bench] all benchmarks passed on backend '{}'.", backend);
    return 0;
}

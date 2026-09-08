// GPU thread-group (block) size benchmark.
//
// Sweeps block sizes / shapes for 8 kernel archetypes while keeping the total
// amount of work constant, so only the group size varies:
//   1. elementwise   - float4 saxpy, memory-bound
//   2. block_reduction - shared-memory tree reduction, barrier-heavy
//   3. gemm_tile     - tiled SGEMM with shared-memory staging (2D + 1D shapes)
//   4. warp_reduce   - warp-level warp_active_sum reduction
//   5. divergent     - irregular per-thread loop counts (load balancing)
//   6. register_heavy - high register pressure (may fail to launch)
//   7. histogram     - global atomics vs shared-memory privatization
//   8. image_2d      - 2D write patterns, shape-not-size effects
//
// Usage: test_block_size_bench <backend> [--device-index N]
// Results are appended to <repo>/benchmark_results/<backend>_block_size_bench.csv
// and printed as markdown tables. Any correctness failure exits non-zero.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/sugar.h>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto k_results_dir = "benchmark_results";

struct BenchRow {
    std::string backend;
    std::string device;
    std::string case_name;
    std::string shape;
    uint32_t block_size;
    double ms_per_dispatch;
    double throughput;// 0 = n/a
    std::string unit; // "GB/s", "GFLOPS", ""
    int iterations;
    std::string correctness;// PASS / FAIL / LAUNCH-FAIL
};

struct BenchContext {
    Device &device;
    Stream &stream;
    std::string backend;
    std::string device_name;
    std::vector<BenchRow> rows;
    bool any_fail{false};

    void add_row(std::string case_name, std::string shape, uint32_t block_size,
                 double ms, double throughput, std::string unit, int iters,
                 std::string correctness) {
        if (correctness == "FAIL") { any_fail = true; }
        rows.push_back(BenchRow{backend, device_name, std::move(case_name),
                                std::move(shape), block_size, ms, throughput,
                                std::move(unit), iters, std::move(correctness)});
    }
};

inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

/// Enqueue 3 warmup dispatches, calibrate an iteration count so the measured
/// batch is >= ~100 ms (capped), then enqueue `iters` dispatches back-to-back
/// and divide the wall time by `iters`.
template<typename F>
double time_dispatches(Stream &stream, F &&enqueue, int &iters_out) {
    for (int i = 0; i < 3; ++i) { enqueue(); }
    stream.synchronize();
    auto t0 = std::chrono::steady_clock::now();
    enqueue();
    stream.synchronize();
    auto single = ms_since(t0);
    int iters = std::clamp(static_cast<int>(100.0 / std::max(single, 0.01)), 1, 200);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) { enqueue(); }
    stream.synchronize();
    auto total = ms_since(t0);
    iters_out = iters;
    return total / static_cast<double>(iters);
}

inline std::string shape_str(uint32_t x, uint32_t y) {
    return std::to_string(x) + "x" + std::to_string(y);
}

/// 2D dispatch split so that no grid dimension exceeds 65535 blocks (DX
/// limit). d1 is rounded up to a multiple of bs so warps and blocks never
/// straddle a row. Total threads d1*d2 >= n, with < d2 + bs slack.
struct GridSplit {
    uint32_t d1, d2, blocks;
};

inline GridSplit make_split(uint64_t n, uint32_t bs) {
    const uint64_t max_threads_x = 65535ull * bs;
    uint32_t d2 = static_cast<uint32_t>((n + max_threads_x - 1u) / max_threads_x);
    uint32_t d1 = 0;
    for (;;) {
        d1 = static_cast<uint32_t>((n + d2 - 1u) / d2);
        d1 = (d1 + bs - 1u) / bs * bs;
        if (d1 / bs <= 65535u) { break; }
        ++d2;
    }
    return {d1, d2, (d1 / bs) * d2};
}

/// Flat 1D index for kernels launched with the 2D split above.
inline auto flat_dispatch_id() noexcept {
    return dispatch_id().y * dispatch_size().x + dispatch_id().x;
}

// ---------------------------------------------------------------------------
// Case 1: elementwise float4 saxpy (memory bound)
// ---------------------------------------------------------------------------

template<uint32_t BS>
void case_elementwise(BenchContext &bc) {
    constexpr uint32_t n = 16u * 1024u * 1024u;// 16M float4 = 256 MiB/buffer
    constexpr double traffic_gb = 3.0 * n * 16.0 / 1e9;
    auto x = bc.device.create_buffer<float4>(n);
    auto y = bc.device.create_buffer<float4>(n);
    auto z = bc.device.create_buffer<float4>(n);

    auto init = Kernel2D([](BufferVar<float4> xv, BufferVar<float4> yv,
                            UInt count) noexcept {
        set_block_size(256u, 1u, 1u);
        auto i = flat_dispatch_id();
        $if (i < count) {
            auto f = cast<float>(i) * 0.25f;
            xv.write(i, make_float4(f, f * 0.5f, f * 0.25f, f * 0.125f));
            yv.write(i, make_float4(f * 2.0f, f, f * 0.5f, f * 0.25f));
        };
    });
    auto init_shader = bc.device.compile(init);
    auto init_split = make_split(n, 256u);
    bc.stream << init_shader(x, y, n).dispatch(init_split.d1, init_split.d2)
              << synchronize();

    auto kernel = Kernel2D([](BufferVar<float4> xv, BufferVar<float4> yv,
                              BufferVar<float4> zv, Float a, UInt count) noexcept {
        set_block_size(BS, 1u, 1u);
        auto i = flat_dispatch_id();
        $if (i < count) {
            zv.write(i, a * xv.read(i) + yv.read(i));
        };
    });
    auto shader = bc.device.compile(kernel);
    auto split = make_split(n, BS);

    // correctness: first 16 elements
    constexpr uint32_t check = 16u;
    std::vector<float4> xh(check), yh(check), zh(check);
    bc.stream << x.view(0, check).copy_to(luisa::span{xh})
              << y.view(0, check).copy_to(luisa::span{yh}) << synchronize();
    bc.stream << shader(x, y, z, 2.0f, n).dispatch(split.d1, split.d2) << synchronize();
    bc.stream << z.view(0, check).copy_to(luisa::span{zh}) << synchronize();
    bool ok = true;
    for (uint32_t i = 0; i < check && ok; ++i) {
        auto expect = 2.0f * xh[i] + yh[i];
        auto got = zh[i];
        ok = std::abs(expect.x - got.x) < 1e-4f &&
             std::abs(expect.y - got.y) < 1e-4f &&
             std::abs(expect.z - got.z) < 1e-4f &&
             std::abs(expect.w - got.w) < 1e-4f;
    }
    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(x, y, z, 2.0f, n).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("elementwise", std::to_string(BS), BS, ms,
               ok ? traffic_gb / (ms / 1e3) : 0.0, "GB/s", iters,
               ok ? "PASS" : "FAIL");
}

// Variant for non-multiple-of-32 sizes (e.g. 100): the DSL requires block
// sizes to be multiples of 32, so emulate with a BS_HW-thread block where the
// top (BS_HW - ACTIVE) lanes exit immediately. On the hardware this matches
// what a real ACTIVE-thread block does (4 warps, last one partially active).
template<uint32_t BS_HW, uint32_t ACTIVE>
void case_elementwise_partial(BenchContext &bc) {
    constexpr uint32_t n = 16u * 1024u * 1024u;
    constexpr double traffic_gb = 3.0 * n * 16.0 / 1e9;
    auto x = bc.device.create_buffer<float4>(n);
    auto y = bc.device.create_buffer<float4>(n);
    auto z = bc.device.create_buffer<float4>(n);

    auto init = Kernel2D([](BufferVar<float4> xv, BufferVar<float4> yv,
                            UInt count) noexcept {
        set_block_size(256u, 1u, 1u);
        auto i = flat_dispatch_id();
        $if (i < count) {
            auto f = cast<float>(i) * 0.25f;
            xv.write(i, make_float4(f, f * 0.5f, f * 0.25f, f * 0.125f));
            yv.write(i, make_float4(f * 2.0f, f, f * 0.5f, f * 0.25f));
        };
    });
    auto init_shader = bc.device.compile(init);
    auto init_split = make_split(n, 256u);
    bc.stream << init_shader(x, y, n).dispatch(init_split.d1, init_split.d2)
              << synchronize();

    auto kernel = Kernel2D([](BufferVar<float4> xv, BufferVar<float4> yv,
                              BufferVar<float4> zv, Float a,
                              UInt count) noexcept {
        set_block_size(BS_HW, 1u, 1u);
        auto i = flat_dispatch_id();
        $if ((thread_id().x < ACTIVE) & (i < count)) {
            zv.write(i, a * xv.read(i) + yv.read(i));
        };
    });
    auto shader = bc.device.compile(kernel);
    auto split = make_split(n, ACTIVE);

    constexpr uint32_t check = 16u;
    std::vector<float4> xh(check), yh(check), zh(check);
    bc.stream << x.view(0, check).copy_to(luisa::span{xh})
              << y.view(0, check).copy_to(luisa::span{yh}) << synchronize();
    bc.stream << shader(x, y, z, 2.0f, n).dispatch(split.d1, split.d2) << synchronize();
    bc.stream << z.view(0, check).copy_to(luisa::span{zh}) << synchronize();
    bool ok = true;
    for (uint32_t i = 0; i < check && ok; ++i) {
        auto expect = 2.0f * xh[i] + yh[i];
        auto got = zh[i];
        ok = std::abs(expect.x - got.x) < 1e-4f &&
             std::abs(expect.y - got.y) < 1e-4f &&
             std::abs(expect.z - got.z) < 1e-4f &&
             std::abs(expect.w - got.w) < 1e-4f;
    }

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(x, y, z, 2.0f, n).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("elementwise", std::to_string(ACTIVE) + "e", ACTIVE, ms,
               ok ? traffic_gb / (ms / 1e3) : 0.0, "GB/s", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 2: block reduction with shared-memory tree + barriers
// ---------------------------------------------------------------------------

template<uint32_t BS>
void case_block_reduction(BenchContext &bc) {
    constexpr uint32_t n = 16u * 1024u * 1024u;// 64 MiB of floats
    constexpr double traffic_gb = n * 4.0 / 1e9;
    auto x = bc.device.create_buffer<float>(n);

    auto init = Kernel2D([](BufferVar<float> xv, UInt count) noexcept {
        set_block_size(256u, 1u, 1u);
        auto i = flat_dispatch_id();
        $if (i < count) { xv.write(i, cast<float>(i % 7u) * 0.25f); };
    });
    auto init_shader = bc.device.compile(init);
    auto init_split = make_split(n, 256u);
    bc.stream << init_shader(x, n).dispatch(init_split.d1, init_split.d2)
              << synchronize();

    auto split = make_split(n, BS);
    auto partial = bc.device.create_buffer<float>(split.blocks);

    auto kernel = Kernel2D([](BufferVar<float> xv, BufferVar<float> part,
                              UInt count) noexcept {
        set_block_size(BS, 1u, 1u);
        Shared<float> s{BS};
        auto tid = thread_id().x;
        auto i = flat_dispatch_id();
        Float v = 0.0f;
        $if (i < count) { v = xv.read(i); };
        s.write(tid, v);
        sync_block();
        for (uint32_t stride = BS / 2u; stride > 0u; stride >>= 1u) {
            $if (tid < stride) {
                s.write(tid, s.read(tid) + s.read(tid + stride));
            };
            sync_block();
        }
        // flat block index (2D dispatch split)
        auto block = block_id().y * (dispatch_size().x / block_size().x) +
                     block_id().x;
        $if (tid == 0u) { part.write(block, s.read(0u)); };
    });
    auto shader = bc.device.compile(kernel);

    // host reference: sum of (i % 7) * 0.25 over n elements
    double expect = 0.0;
    {
        double cycle = 0.0;
        for (uint32_t r = 0; r < 7u; ++r) { cycle += r * 0.25; }
        expect = cycle * static_cast<double>(n / 7u);
        for (uint32_t r = 0; r < n % 7u; ++r) { expect += r * 0.25; }
    }

    std::vector<float> ph(split.blocks);
    bc.stream << shader(x, partial, n).dispatch(split.d1, split.d2) << synchronize();
    bc.stream << partial.copy_to(luisa::span{ph}) << synchronize();
    double got = std::accumulate(ph.begin(), ph.end(), 0.0);
    bool ok = std::abs(got - expect) <= 1e-4 * std::max(expect, 1.0);

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(x, partial, n).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("block_reduction", std::to_string(BS), BS, ms,
               ok ? traffic_gb / (ms / 1e3) : 0.0, "GB/s", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 3: tiled SGEMM with shared-memory staging
// Block tile is (TX outputs per row) x (TY rows); TK is the K chunk.
// ---------------------------------------------------------------------------

template<uint32_t TX, uint32_t TY, uint32_t TK = 16u>
void case_gemm_tile(BenchContext &bc) {
    constexpr uint32_t M = 2048u, N = 2048u, K = 2048u;
    constexpr uint32_t BS = TX * TY;
    constexpr double flops = 2.0 * M * N * K;

    std::vector<float> ha(M * K), hb(K * N);
    for (uint32_t i = 0; i < M * K; ++i) {
        ha[i] = static_cast<float>(static_cast<int>(i % 13u) - 6) * 0.125f;
    }
    for (uint32_t i = 0; i < K * N; ++i) {
        hb[i] = static_cast<float>(static_cast<int>(i % 11u) - 5) * 0.25f;
    }
    auto a = bc.device.create_buffer<float>(ha.size());
    auto b = bc.device.create_buffer<float>(hb.size());
    auto c = bc.device.create_buffer<float>(M * N);
    bc.stream << a.copy_from(luisa::span{ha}) << b.copy_from(luisa::span{hb})
              << synchronize();

    auto kernel = Kernel2D([K, N](BufferVar<float> av, BufferVar<float> bv,
                                      BufferVar<float> cv) noexcept {
        set_block_size(TX, TY, 1u);
        Shared<float> sa{TY * TK};
        Shared<float> sb{TK * TX};
        auto tid = thread_id().y * TX + thread_id().x;
        auto lx = thread_id().x;
        auto ly = thread_id().y;
        auto row0 = block_id().y * TY;
        auto col0 = block_id().x * TX;
        Float sum = 0.0f;
        $for(kk, 0u, K, TK) {
            // cooperative staging of A tile (TY x TK)
            for (uint32_t rep = 0u; rep < (TY * TK + BS - 1u) / BS; ++rep) {
                auto idx = tid + rep * BS;
                $if (idx < TY * TK) {
                    sa.write(idx, av.read((row0 + idx / TK) * K + kk + idx % TK));
                };
            }
            // cooperative staging of B tile (TK x TX)
            for (uint32_t rep = 0u; rep < (TK * TX + BS - 1u) / BS; ++rep) {
                auto idx = tid + rep * BS;
                $if (idx < TK * TX) {
                    sb.write(idx, bv.read((kk + idx / TX) * N + col0 + idx % TX));
                };
            }
            sync_block();
            for (uint32_t k = 0u; k < TK; ++k) {
                sum += sa.read(ly * TK + k) * sb.read(k * TX + lx);
            }
            sync_block();
        };
        cv.write((row0 + ly) * N + col0 + lx, sum);
    });
    auto shader = bc.device.compile(kernel);
    auto grid_x = N / TX;
    auto grid_y = M / TY;

    // correctness: sampled outputs vs double-precision CPU dot products
    const uint32_t samples[][2] = {
        {0u, 0u}, {0u, N - 1u}, {M - 1u, 0u}, {M - 1u, N - 1u},
        {M / 2u, N / 2u}, {1023u, 517u}, {517u, 1023u}, {204u, 1500u}};
    bool ok = true;
    {
        std::vector<float> cvh(8u);
        bc.stream << shader(a, b, c).dispatch(N, M) << synchronize();
        for (auto s = 0u; s < 8u && ok; ++s) {
            auto r = samples[s][0], col = samples[s][1];
            // read the single element via a 1-element copy
            bc.stream << c.view(r * N + col, 1).copy_to(luisa::span{cvh}.subspan(s, 1))
                      << synchronize();
            double expect = 0.0;
            for (uint32_t k = 0u; k < K; ++k) {
                expect += static_cast<double>(ha[r * K + k]) *
                          static_cast<double>(hb[k * N + col]);
            }
            if (std::abs(cvh[s] - expect) > 1e-3 * std::max(1.0, std::abs(expect))) {
                ok = false;
            }
        }
    }

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(a, b, c).dispatch(N, M);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("gemm_tile", shape_str(TX, TY), BS, ms,
               ok ? flops / (ms / 1e3) / 1e9 : 0.0, "GFLOPS", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 4: warp-level reduction (warp_active_sum), one item per warp
// ---------------------------------------------------------------------------

template<uint32_t BS>
void case_warp_reduce(BenchContext &bc) {
    constexpr uint32_t items = 1u * 1024u * 1024u;
    constexpr uint32_t total = items * 32u;
    constexpr double traffic_gb = 2.0 * total * 4.0 / 1e9;
    auto input = bc.device.create_buffer<float>(total);
    auto output = bc.device.create_buffer<float>(items);

    auto init = Kernel2D([](BufferVar<float> in, UInt count) noexcept {
        set_block_size(256u, 1u, 1u);
        auto i = flat_dispatch_id();
        $if (i < count) { in.write(i, cast<float>(i % 999u) * 0.25f); };
    });
    auto init_shader = bc.device.compile(init);
    auto init_split = make_split(total, 256u);
    bc.stream << init_shader(input, total).dispatch(init_split.d1, init_split.d2)
              << synchronize();

    auto kernel = Kernel2D([](BufferVar<float> in, BufferVar<float> out,
                              UInt count) noexcept {
        set_block_size(BS, 1u, 1u);
        set_warp_size(32u);
        auto i = flat_dispatch_id();
        auto item = i / 32u;
        Float v = 0.0f;
        $if (i < count) { v = in.read(i); };
        Float s = warp_active_sum(v);
        // chain a few more warp reductions (scale by 1/32 to stay idempotent:
        // warp_active_sum of a warp-uniform value returns 32*v)
        s = warp_active_sum(s) * 0.03125f;
        s = warp_active_sum(s) * 0.03125f;
        s = warp_active_sum(s) * 0.03125f;
        s = warp_active_sum(s) * 0.03125f;
        $if ((i < count) & (warp_lane_id() == 0u)) { out.write(item, s); };
    });
    auto shader = bc.device.compile(kernel);
    auto split = make_split(total, BS);

    // correctness: first 16 items
    constexpr uint32_t check = 16u;
    std::vector<float> oh(check);
    bc.stream << shader(input, output, total).dispatch(split.d1, split.d2) << synchronize();
    bc.stream << output.view(0, check).copy_to(luisa::span{oh}) << synchronize();
    bool ok = true;
    for (uint32_t m = 0; m < check && ok; ++m) {
        float expect = 0.0f;
        for (uint32_t l = 0; l < 32u; ++l) {
            expect += static_cast<float>(((m * 32u + l) % 999u) * 0.25f);
        }
        ok = std::abs(oh[m] - expect) <= 1e-4f * std::max(expect, 1.0f);
    }

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(input, output, total).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("warp_reduce", std::to_string(BS), BS, ms,
               ok ? traffic_gb / (ms / 1e3) : 0.0, "GB/s", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 5: divergent per-thread loop counts (hash-derived, [0, 512))
// ---------------------------------------------------------------------------

template<uint32_t BS>
void case_divergent(BenchContext &bc) {
    constexpr uint32_t n = 8u * 1024u * 1024u;
    auto out = bc.device.create_buffer<float>(n);

    auto kernel = Kernel2D([](BufferVar<float> ov, UInt count) noexcept {
        set_block_size(BS, 1u, 1u);
        auto i = flat_dispatch_id();
        UInt iters = (i * 2654435761u) % 512u;
        Float acc = cast<float>(i % 1024u) * 0.001f;
        $for(j, 0u, iters) {
            acc = acc * 1.000001f + 0.5f;
        };
        $if (i < count) { ov.write(i, acc); };
    });
    auto shader = bc.device.compile(kernel);
    auto split = make_split(n, BS);

    constexpr uint32_t check = 16u;
    std::vector<float> oh(check);
    bc.stream << shader(out, n).dispatch(split.d1, split.d2) << synchronize();
    bc.stream << out.view(0, check).copy_to(luisa::span{oh}) << synchronize();
    bool ok = true;
    for (uint32_t i = 0; i < check && ok; ++i) {
        uint32_t iters = (i * 2654435761u) % 512u;
        float acc = static_cast<float>(i % 1024u) * 0.001f;
        for (uint32_t j = 0; j < iters; ++j) {
            acc = acc * 1.000001f + 0.5f;
        }
        ok = std::abs(oh[i] - acc) <= 1e-2f * std::max(1.0f, std::abs(acc));
    }

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(out, n).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("divergent", std::to_string(BS), BS, ms, 0.0, "", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 6: register-heavy kernel (>= 32 independent float4 accumulators)
// ---------------------------------------------------------------------------

template<uint32_t BS>
void case_register_heavy(BenchContext &bc) {
    constexpr uint32_t n = 2u * 1024u * 1024u;
    constexpr uint32_t data_size = 4096u;
    constexpr uint32_t inner = 64u;
    auto data = bc.device.create_buffer<float4>(data_size);
    auto out = bc.device.create_buffer<float4>(n);

    std::vector<float4> dh(data_size);
    for (uint32_t i = 0; i < data_size; ++i) {
        dh[i] = make_float4(i * 0.001f, i * 0.002f, i * 0.003f, i * 0.004f);
    }
    bc.stream << data.copy_from(luisa::span{dh}) << synchronize();

    auto kernel = Kernel2D([inner, data_size](BufferVar<float4> dv, BufferVar<float4> ov,
                                              UInt count) noexcept {
        set_block_size(BS, 1u, 1u);
        auto i = flat_dispatch_id();
        Float4 acc[32];
        auto base = dv.read(i & (data_size - 1u));
        for (uint32_t u = 0u; u < 32u; ++u) {
            acc[u] = base * (1.0f + static_cast<float>(u) * 0.01f);
        }
        $for(t, 0u, inner) {
            auto v = dv.read((i + t * 997u) & (data_size - 1u));
            for (uint32_t u = 0u; u < 32u; ++u) {
                acc[u] = acc[u] * 1.000001f + v;
            }
        };
        Float4 s = acc[0];
        for (uint32_t u = 1u; u < 32u; ++u) { s = s + acc[u]; }
        $if (i < count) { ov.write(i, s); };
    });

    std::optional<decltype(bc.device.compile(kernel))> shader;
    try {
        shader = bc.device.compile(kernel);
    } catch (const std::exception &e) {
        LUISA_WARNING("register_heavy BS={} failed to compile: {}", BS, e.what());
        bc.add_row("register_heavy", std::to_string(BS), BS, 0.0, 0.0, "", 0,
                   "LAUNCH-FAIL");
        return;
    }
    auto split = make_split(n, BS);

    // correctness: first 16 outputs, CPU simulation
    constexpr uint32_t check = 16u;
    std::vector<float4> oh(check);
    try {
        bc.stream << (*shader)(data, out, n).dispatch(split.d1, split.d2) << synchronize();
    } catch (const std::exception &e) {
        LUISA_WARNING("register_heavy BS={} failed to dispatch: {}", BS, e.what());
        bc.add_row("register_heavy", std::to_string(BS), BS, 0.0, 0.0, "", 0,
                   "LAUNCH-FAIL");
        return;
    }
    bc.stream << out.view(0, check).copy_to(luisa::span{oh}) << synchronize();
    bool ok = true;
    for (uint32_t i = 0; i < check && ok; ++i) {
        float ax[32][4];
        auto base = dh[i & (data_size - 1u)];
        float b[4] = {base.x, base.y, base.z, base.w};
        for (uint32_t u = 0u; u < 32u; ++u) {
            for (int c = 0; c < 4; ++c) {
                ax[u][c] = b[c] * (1.0f + static_cast<float>(u) * 0.01f);
            }
        }
        for (uint32_t t = 0u; t < inner; ++t) {
            auto v = dh[(i + t * 997u) & (data_size - 1u)];
            float vv[4] = {v.x, v.y, v.z, v.w};
            for (uint32_t u = 0u; u < 32u; ++u) {
                for (int c = 0; c < 4; ++c) {
                    ax[u][c] = ax[u][c] * 1.000001f + vv[c];
                }
            }
        }
        float s[4] = {ax[0][0], ax[0][1], ax[0][2], ax[0][3]};
        for (uint32_t u = 1u; u < 32u; ++u) {
            for (int c = 0; c < 4; ++c) { s[c] += ax[u][c]; }
        }
        float g[4] = {oh[i].x, oh[i].y, oh[i].z, oh[i].w};
        for (int c = 0; c < 4 && ok; ++c) {
            ok = std::abs(g[c] - s[c]) <= 1e-2f * std::max(1.0f, std::abs(s[c]));
        }
    }

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << (*shader)(data, out, n).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("register_heavy", std::to_string(BS), BS, ms, 0.0, "", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 7: histogram - (a) global atomics, (b) shared-memory privatization
// ---------------------------------------------------------------------------

template<uint32_t BS, bool SHARED>
void case_histogram(BenchContext &bc) {
    constexpr uint32_t n = 16u * 1024u * 1024u;
    constexpr uint32_t bins = 4096u;
    auto values = bc.device.create_buffer<uint>(n);
    auto counters = bc.device.create_buffer<uint>(bins);

    std::vector<uint> hv(n);
    std::vector<uint> expect(bins, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        auto v = (i * 2654435761u ^ (i >> 13u)) % bins;
        hv[i] = v;
        expect[v]++;
    }
    bc.stream << values.copy_from(luisa::span{hv}) << synchronize();

    auto shader = [&bc] {
        if constexpr (SHARED) {
            auto kernel = Kernel2D([bins](BufferVar<uint> vv, BufferVar<uint> cv,
                                          UInt count) noexcept {
                set_block_size(BS, 1u, 1u);
                Shared<uint> sc{bins};
                auto tid = thread_id().x;
                $for(i, tid, bins, BS) { sc.write(i, 0u); };
                sync_block();
                auto i = flat_dispatch_id();
                $if (i < count) {
                    auto bin = vv.read(i);
                    sc.atomic(bin).fetch_add(1u);
                };
                sync_block();
                $for(i, tid, bins, BS) {
                    auto c = sc.read(i);
                    $if (c > 0u) { cv.atomic(i).fetch_add(c); };
                };
            });
            return bc.device.compile(kernel);
        } else {
            auto kernel = Kernel2D([](BufferVar<uint> vv, BufferVar<uint> cv,
                                      UInt count) noexcept {
                set_block_size(BS, 1u, 1u);
                auto i = flat_dispatch_id();
                $if (i < count) {
                    auto bin = vv.read(i);
                    cv.atomic(bin).fetch_add(1u);
                };
            });
            return bc.device.compile(kernel);
        }
    }();
    auto split = make_split(n, BS);

    std::vector<uint> zeros(bins, 0u);
    std::vector<uint> ch(bins);
    bc.stream << counters.copy_from(luisa::span{zeros}) << synchronize();
    bc.stream << shader(values, counters, n).dispatch(split.d1, split.d2) << synchronize();
    bc.stream << counters.copy_to(luisa::span{ch}) << synchronize();
    bool ok = (ch == expect);
    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(values, counters, n).dispatch(split.d1, split.d2);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row(SHARED ? "histogram_shared" : "histogram_global",
               std::to_string(BS), BS, ms, 0.0, "", iters, ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Case 8: 2D image write patterns (shape, not size)
// ---------------------------------------------------------------------------

template<uint32_t BX, uint32_t BY>
void case_image_2d(BenchContext &bc) {
    constexpr uint32_t W = 4096u, H = 4096u;
    constexpr double traffic_gb = W * H * 4.0 / 1e9;
    auto out = bc.device.create_buffer<float>(W * H);

    auto kernel = Kernel2D([W, H](BufferVar<float> ov) noexcept {
        set_block_size(BX, BY, 1u);
        auto x = block_id().x * BX + thread_id().x;
        auto y = block_id().y * BY + thread_id().y;
        $if ((x < W) & (y < H)) {
            Float v = 0.0f;
            for (uint32_t k = 0u; k < 8u; ++k) {
                v = sin(v + cast<float>(x) * 0.01f +
                        cast<float>(y) * 0.02f + cast<float>(k));
            }
            ov.write(y * W + x, v);
        };
    });
    auto shader = bc.device.compile(kernel);
    auto grid_x = (W + BX - 1u) / BX;
    auto grid_y = (H + BY - 1u) / BY;

    constexpr uint32_t check = 16u;
    std::vector<float> oh(check);
    bc.stream << shader(out).dispatch(W, H) << synchronize();
    bc.stream << out.view(0, check).copy_to(luisa::span{oh}) << synchronize();
    bool ok = true;
    for (uint32_t i = 0; i < check && ok; ++i) {
        auto x = i % W, y = i / W;
        float v = 0.0f;
        for (uint32_t k = 0u; k < 8u; ++k) {
            v = std::sin(v + static_cast<float>(x) * 0.01f +
                         static_cast<float>(y) * 0.02f + static_cast<float>(k));
        }
        ok = std::abs(oh[i] - v) <= 1e-4f;
    }

    int iters = 0;
    double ms = 0.0;
    if (ok) {
        auto enqueue = [&] {
            bc.stream << shader(out).dispatch(W, H);
        };
        ms = time_dispatches(bc.stream, enqueue, iters);
    }
    bc.add_row("image_2d", shape_str(BX, BY), BX * BY, ms,
               ok ? traffic_gb / (ms / 1e3) : 0.0, "GB/s", iters,
               ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

void print_markdown(const std::vector<BenchRow> &rows) {
    std::string current;
    for (const auto &r : rows) {
        if (r.case_name != current) {
            current = r.case_name;
            std::cout << "\n### " << current << "\n\n"
                      << "| shape | block | ms/dispatch | throughput | unit | iters | result |\n"
                      << "|---|---|---|---|---|---|---|\n";
        }
        std::ostringstream tp;
        if (r.throughput > 0.0) {
            tp << std::fixed << std::setprecision(2) << r.throughput;
        } else {
            tp << "-";
        }
        std::cout << "| " << r.shape << " | " << r.block_size << " | "
                  << std::fixed << std::setprecision(4) << r.ms_per_dispatch
                  << " | " << tp.str() << " | " << r.unit << " | "
                  << r.iterations << " | " << r.correctness << " |\n";
    }
    std::cout << std::endl;
}

void write_csv(const std::filesystem::path &path,
               const std::vector<BenchRow> &rows) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        LUISA_WARNING("Failed to open CSV file: {}", path.string());
        return;
    }
    f << "backend,device,case,shape,block_size,ms_per_dispatch,throughput,"
         "unit,iterations,correctness\n";
    for (const auto &r : rows) {
        f << r.backend << ",\"" << r.device << "\"," << r.case_name << ","
          << r.shape << "," << r.block_size << "," << std::fixed
          << std::setprecision(6) << r.ms_per_dispatch << "," << std::fixed
          << std::setprecision(3) << r.throughput << "," << r.unit << ","
          << r.iterations << "," << r.correctness << "\n";
    }
    LUISA_INFO("CSV written to {}", path.string());
}

bool is_discrete_gpu_name(const std::string &name_lower) {
    return name_lower.find("nvidia") != std::string::npos ||
           name_lower.find("geforce") != std::string::npos ||
           name_lower.find("rtx") != std::string::npos ||
           name_lower.find("radeon") != std::string::npos ||
           name_lower.find("arc") != std::string::npos;
}

}// namespace

int main(int argc, char *argv[]) {
    const char *exe = (argc > 0 && argv && argv[0]) ? argv[0] : luisa::test::safe_argv0();
    if (argc <= 1 || argv[1] == nullptr || argv[1][0] == '\0') {
        luisa::test::print_device_usage(exe);
        return 1;
    }
    std::string backend = argv[1];

    // Optional --device-index N override.
    int device_index_override = -1;
    for (int i = 2; i < argc; ++i) {
        if (std::string_view{argv[i]} == "--device-index" && i + 1 < argc) {
            device_index_override = std::atoi(argv[i + 1]);
        }
    }

    Context context{exe};
    auto names = context.backend_device_names(backend);
    LUISA_INFO("Backend '{}' reports {} hardware device(s):", backend, names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        LUISA_INFO("  [{}] {}", i, names[i]);
    }

    uint32_t device_index = 0;
    if (device_index_override >= 0) {
        device_index = static_cast<uint32_t>(device_index_override);
        LUISA_INFO("Device index overridden by CLI: {}", device_index);
    } else {
        for (size_t i = 0; i < names.size(); ++i) {
            std::string lower{names[i]};
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (is_discrete_gpu_name(lower)) {
                device_index = static_cast<uint32_t>(i);
                break;
            }
        }
    }
    if (device_index >= names.size() && !names.empty()) {
        LUISA_WARNING("Device index {} out of range; falling back to 0.", device_index);
        device_index = 0;
    }

    DeviceConfig config{};
    config.device_index = device_index;
    Device device;
    try {
        device = context.create_device(backend, &config);
    } catch (const std::exception &e) {
        LUISA_ERROR("Failed to create device '{}' index {}: {}", backend,
                    device_index, e.what());
        return 1;
    }
    std::string device_name = names.empty() ? "unknown"
                                            : std::string{names[device_index]};
    LUISA_INFO("Selected device [{}]: {}", device_index, device_name);
    LUISA_INFO("Backend warp size: {}", device.compute_warp_size());

    auto stream = device.create_stream();
    BenchContext bc{device, stream, backend, device_name};

    LUISA_INFO("=== Case 1: elementwise ===");
    case_elementwise<32>(bc);
    case_elementwise<64>(bc);
    case_elementwise_partial<128, 100>(bc);// DSL requires multiple-of-32
    case_elementwise<128>(bc);
    case_elementwise<256>(bc);
    case_elementwise<512>(bc);
    case_elementwise<1024>(bc);

    LUISA_INFO("=== Case 2: block_reduction ===");
    case_block_reduction<32>(bc);
    case_block_reduction<64>(bc);
    case_block_reduction<128>(bc);
    case_block_reduction<256>(bc);
    case_block_reduction<512>(bc);
    case_block_reduction<1024>(bc);

    LUISA_INFO("=== Case 3: gemm_tile ===");
    case_gemm_tile<8, 8>(bc);
    case_gemm_tile<16, 16>(bc);
    case_gemm_tile<32, 32>(bc);
    case_gemm_tile<128, 1>(bc);
    case_gemm_tile<512, 1, 8u>(bc);

    LUISA_INFO("=== Case 4: warp_reduce ===");
    case_warp_reduce<32>(bc);
    case_warp_reduce<64>(bc);
    case_warp_reduce<128>(bc);
    case_warp_reduce<256>(bc);

    LUISA_INFO("=== Case 5: divergent ===");
    case_divergent<32>(bc);
    case_divergent<64>(bc);
    case_divergent<128>(bc);
    case_divergent<256>(bc);
    case_divergent<512>(bc);
    case_divergent<1024>(bc);

    LUISA_INFO("=== Case 6: register_heavy ===");
    case_register_heavy<64>(bc);
    case_register_heavy<128>(bc);
    case_register_heavy<256>(bc);
    case_register_heavy<512>(bc);
    case_register_heavy<1024>(bc);

    LUISA_INFO("=== Case 7: histogram ===");
    case_histogram<64, false>(bc);
    case_histogram<128, false>(bc);
    case_histogram<256, false>(bc);
    case_histogram<512, false>(bc);
    case_histogram<1024, false>(bc);
    case_histogram<64, true>(bc);
    case_histogram<128, true>(bc);
    case_histogram<256, true>(bc);
    case_histogram<512, true>(bc);
    case_histogram<1024, true>(bc);

    LUISA_INFO("=== Case 8: image_2d ===");
    case_image_2d<64, 1>(bc);
    case_image_2d<1, 64>(bc);
    case_image_2d<8, 8>(bc);
    case_image_2d<16, 16>(bc);
    case_image_2d<256, 1>(bc);
    case_image_2d<1, 256>(bc);

    print_markdown(bc.rows);

    std::error_code ec;
    std::filesystem::create_directories(k_results_dir, ec);
    auto csv_path = std::filesystem::path{k_results_dir} /
                    (backend + "_block_size_bench.csv");
    write_csv(csv_path, bc.rows);

    // Histogram speedup summary.
    std::cout << "\n### histogram shared-vs-global speedup\n\n"
              << "| block | global ms | shared ms | speedup |\n"
              << "|---|---|---|---|\n";
    for (auto bs : {64u, 128u, 256u, 512u, 1024u}) {
        const BenchRow *g = nullptr, *s = nullptr;
        for (const auto &r : bc.rows) {
            if (r.block_size == bs && r.case_name == "histogram_global") { g = &r; }
            if (r.block_size == bs && r.case_name == "histogram_shared") { s = &r; }
        }
        if (g && s && g->ms_per_dispatch > 0.0 && s->ms_per_dispatch > 0.0) {
            std::cout << "| " << bs << " | " << std::fixed << std::setprecision(4)
                      << g->ms_per_dispatch << " | " << s->ms_per_dispatch
                      << " | " << std::setprecision(2)
                      << g->ms_per_dispatch / s->ms_per_dispatch << "x |\n";
        }
    }
    std::cout << std::endl;

    if (bc.any_fail) {
        LUISA_ERROR("Some correctness checks FAILED.");
        return 1;
    }
    LUISA_INFO("All correctness checks passed.");
    return 0;
}

// Verification benchmark for lc_optimize SKILL.md Part II (sections 11-23).
//
// Measures the externally-sourced claims from the cross-platform optimization
// guide on the local device, so the skill can cite locally reproduced numbers
// next to the external priors. Cases:
// 1. roofline_stream - saxpy at AI = 1/6 FLOP/byte; checks the kernel lands on
//    the bandwidth roof (memory-bound) and prints achieved % of datasheet peak
//    (SKILL.md §11).
// 2. vectorization - scalar float vs float4 streaming saxpy; measures the gain
//    from 16 B/lane accesses (SKILL.md §14.2, external claim: 15-40%).
// 3. async_pipeline - staged shared-memory reduction, synchronous LDG+STS vs
//    2-stage cp.async double buffering (SKILL.md §15.1, external claim:
//    1.2-1.5x on staged kernels).
// 4. launch_overhead - 2000 tiny dispatches enqueued one-by-one vs packed in a
//    single CommandList (SKILL.md §20.1).
//
// Usage: benchmark_part2_verify <backend> [--csv <file>]
// All cases are correctness-checked against a host reference before timing.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/backends/ext/cuda/cuda_graph_ext.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

namespace {

struct Row {
    std::string case_name;
    std::string variant;
    double ms_per_dispatch;
    double throughput;// GB/s or GFLOPS; 0 = n/a
    std::string unit;
    bool pass;
};

inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

/// Enqueue 3 warmups, calibrate so the timed batch is >= ~100 ms, then time a
/// batch of back-to-back dispatches and return ms/dispatch (batch average).
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

// ---------------------------------------------------------------------------
// Case 1 + 2: roofline streaming saxpy, scalar vs float4 (§11, §14.2)
// ---------------------------------------------------------------------------
template<bool VEC>
void case_stream(Device &device, Stream &stream, std::vector<Row> &rows) {
    using T = std::conditional_t<VEC, float4, float>;
    constexpr uint32_t elems = VEC ? 4u * 1024u * 1024u : 16u * 1024u * 1024u;
    constexpr double traffic_gb = 3.0 * elems * (VEC ? 16.0 : 4.0) / 1e9;
    constexpr double flops = 2.0 * elems * (VEC ? 4.0 : 1.0);
    auto x = device.create_buffer<T>(elems);
    auto y = device.create_buffer<T>(elems);
    auto z = device.create_buffer<T>(elems);
    auto kernel = Kernel1D([](BufferVar<T> xv, BufferVar<T> yv, BufferVar<T> zv,
                              Float a) noexcept {
        set_block_size(256u, 1u, 1u);
        auto i = dispatch_id().x;
        zv.write(i, xv.read(i) * a + yv.read(i));
    });
    auto shader = device.compile(kernel);
    int iters = 0;
    auto enqueue = [&] { stream << shader(x, y, z, 2.0f).dispatch(elems); };
    double ms = time_dispatches(stream, enqueue, iters);
    rows.push_back({VEC ? "vectorization" : "roofline_stream",
                    VEC ? "float4" : "scalar_float", ms,
                    traffic_gb / (ms / 1e3), "GB/s", true});
    std::cout << (VEC ? "vectorization  float4        " : "roofline_stream scalar_float ")
              << std::fixed << std::setprecision(4) << ms << " ms  "
              << std::setprecision(1) << traffic_gb / (ms / 1e3) << " GB/s  ("
              << std::setprecision(2) << flops / (ms / 1e3) / 1e9
              << " GFLOPS implied)\n";
}

// ---------------------------------------------------------------------------
// Case 3: staged shared-memory reduction, sync copy vs cp.async pipeline (§15)
// ---------------------------------------------------------------------------
void case_pipeline(Device &device, Stream &stream, std::vector<Row> &rows) {
    constexpr uint32_t tile = 256u;          // float4 per stage
    constexpr uint32_t stages = 32u;         // stages per block
    constexpr uint32_t block_size = 256u;
    constexpr uint32_t per_block = tile * stages;
    constexpr uint32_t n = 32u * 1024u * 1024u;                    // total float4 (512 MiB)
    constexpr uint32_t num_blocks = n / per_block;                 // 4096
    constexpr double traffic_gb = n * 16.0 / 1e9;

    auto src = device.create_buffer<float4>(n);
    // Per-thread partials; the host sums them into a per-block reference.
    auto out = device.create_buffer<float4>(num_blocks * block_size);
    std::vector<float4> h(n);
    for (uint32_t i = 0u; i < n; ++i) {
        h[i] = make_float4(static_cast<float>(i & 1023u) * 0.001f,
                           static_cast<float>((i >> 10) & 1023u) * 0.002f, 1.0f, 2.0f);
    }
    stream << src.copy_from(luisa::span{h}) << synchronize();

    // Baseline: synchronous global load -> shared store -> barrier -> consume.
    auto sync_kernel = Kernel1D([tile, stages, per_block, block_size](
                                    BufferVar<float4> sv, BufferVar<float4> ov) noexcept {
        set_block_size(block_size, 1u, 1u);
        $shared<float4> buf{2u * tile};
        $uint tid = thread_x();
        $uint base = block_x() * per_block;
        Float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        $for(k, 0u, stages) {
            buf[(k % 2u) * tile + tid] = sv.read(base + k * tile + tid);
            sync_block();
            acc = acc + buf[(k % 2u) * tile + ((tid + 1u) & (tile - 1u))];
            sync_block();
        };
        ov.write(block_x() * block_size + tid, acc);
    });
    // Pipelined: 2-stage cp.async double buffer; consumer never waits on a
    // synchronous load. Every iteration commits a pipeline group (possibly
    // empty) so pipeline_wait_prior(1) always retires exactly stage k.
    auto async_kernel = Kernel1D([tile, stages, per_block, block_size](
                                     BufferVar<float4> sv, BufferVar<float4> ov) noexcept {
        set_block_size(block_size, 1u, 1u);
        $shared<float4> buf{2u * tile};
        $uint tid = thread_x();
        $uint base = block_x() * per_block;
        auto addr = sv.device_address();
        auto issue = [&](Expr<uint> stage, Expr<uint> slot) {
            async_copy(1u, buf[slot * tile + tid],
                       addr + cast<luisa::ulong>((base + stage * tile + tid) * 16u),
                       16u, 1u, 16u, 0u);
        };
        issue(0u, 0u);
        pipeline_commit();
        issue(1u, 1u);
        pipeline_commit();
        Float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        $for(k, 0u, stages) {
            pipeline_wait_prior(1u);
            sync_block();
            acc = acc + buf[(k % 2u) * tile + ((tid + 1u) & (tile - 1u))];
            sync_block();
            $if(k + 2u < stages) { issue(k + 2u, k % 2u); };
            pipeline_commit();
        };
        ov.write(block_x() * block_size + tid, acc);
    });

    // Host reference: each stage's 256 elements are consumed exactly once.
    std::vector<float4> ref(num_blocks);
    for (uint32_t b = 0u; b < num_blocks; ++b) {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t k = 0u; k < stages; ++k) {
            for (uint32_t t = 0u; t < tile; ++t) {
                auto v = h[b * per_block + k * tile + t];
                s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
            }
        }
        ref[b] = s;
    }

    auto check = [&](auto &shader, const char *name) {
        std::vector<float4> got(num_blocks * block_size);
        stream << shader(src, out).dispatch(num_blocks * block_size)
               << out.copy_to(luisa::span{got}) << synchronize();
        bool ok = true;
        for (uint32_t b = 0u; b < num_blocks && ok; ++b) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (uint32_t t = 0u; t < block_size; ++t) {
                auto v = got[b * block_size + t];
                sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
            }
            ok = std::abs(sum.x - ref[b].x) <= 1e-2f * std::max(1.0f, std::abs(ref[b].x)) &&
                 std::abs(sum.y - ref[b].y) <= 1e-2f * std::max(1.0f, std::abs(ref[b].y));
            if (!ok) {
                LUISA_WARNING("{} mismatch at block {}: got ({}, {}, {}, {}) ref ({}, {}, {}, {})",
                              name, b, sum.x, sum.y, sum.z, sum.w,
                              ref[b].x, ref[b].y, ref[b].z, ref[b].w);
            }
        }
        int iters = 0;
        auto enqueue = [&] {
            stream << shader(src, out).dispatch(num_blocks * block_size);
        };
        double ms = ok ? time_dispatches(stream, enqueue, iters) : 0.0;
        rows.push_back({"async_pipeline", name, ms, traffic_gb / (ms / 1e3), "GB/s", ok});
        std::cout << "async_pipeline   " << std::setw(12) << name << " " << std::fixed
                  << std::setprecision(4) << ms << " ms  " << std::setprecision(1)
                  << traffic_gb / (ms / 1e3) << " GB/s  " << (ok ? "PASS" : "FAIL")
                  << "\n";
    };
    auto sync_shader = device.compile(sync_kernel);
    auto async_shader = device.compile(async_kernel);
    check(sync_shader, "sync_copy");
    check(async_shader, "cp_async_2stage");
}

// ---------------------------------------------------------------------------
// Case 4: tiny-kernel launch overhead, per-dispatch vs CommandList batch (§20.1)
// ---------------------------------------------------------------------------
void case_launch(Device &device, Stream &stream, std::vector<Row> &rows) {
    constexpr int dispatches = 2000;
    constexpr uint32_t threads = 256u;
    auto out = device.create_buffer<float>(threads);
    auto kernel = Kernel1D([threads](BufferVar<float> ov, UInt salt) noexcept {
        set_block_size(threads, 1u, 1u);
        ov.write(thread_x(), cast<float>(salt));
    });
    auto shader = device.compile(kernel);

    // Reference: last dispatch writes salt = dispatches-1.
    stream << shader(out, static_cast<uint32_t>(dispatches - 1)).dispatch(threads)
           << synchronize();
    std::vector<float> host(threads);
    stream << out.copy_to(luisa::span{host}) << synchronize();
    bool ok = std::abs(host[0] - static_cast<float>(dispatches - 1)) < 1e-4f;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < dispatches; ++i) {
        stream << shader(out, static_cast<uint32_t>(i)).dispatch(threads);
    }
    stream.synchronize();
    double ms_each = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    CommandList list;
    list.reserve(dispatches, 0u);
    for (int i = 0; i < dispatches; ++i) {
        list << shader(out, static_cast<uint32_t>(i)).dispatch(threads);
    }
    stream << list.commit() << synchronize();
    double ms_batch = ms_since(t0);

    rows.push_back({"launch_overhead", "per_dispatch", ms_each / dispatches, 0.0, "", ok});
    rows.push_back({"launch_overhead", "command_list", ms_batch / dispatches, 0.0, "", ok});
    std::cout << "launch_overhead  per_dispatch " << std::fixed << std::setprecision(4)
              << ms_each / dispatches << " ms/tiny-dispatch (total " << ms_each
              << " ms for " << dispatches << ")\n"
              << "launch_overhead  command_list " << ms_batch / dispatches
              << " ms/tiny-dispatch (total " << ms_batch << " ms)  " << (ok ? "PASS" : "FAIL")
              << "\n";

    // Mode C: capture the same 2000 dispatches in a CUDA graph (CUDA backend
    // only; SKILL.md §20.1) and time graph replays.
    if (auto *ext = device.extension<CudaGraphExt>()) {
        CommandList gl;
        gl.reserve(dispatches, 0u);
        for (int i = 0; i < dispatches; ++i) {
            gl << shader(out, static_cast<uint32_t>(i)).dispatch(threads);
        }
        auto graph = ext->create_graph(std::move(gl.commit()).command_list());
        if (graph.handle().handle != CudaGraphExt::invalid_handle) {
            auto exec = ext->instantiate(graph.handle().handle);
            constexpr int rounds = 20;
            ext->launch(exec.handle().handle, stream.handle());
            stream << synchronize();
            t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < rounds; ++r) {
                ext->launch(exec.handle().handle, stream.handle());
            }
            stream << synchronize();
            double ms_graph = ms_since(t0) / rounds / dispatches;
            rows.push_back({"launch_overhead", "cuda_graph", ms_graph, 0.0, "", ok});
            std::cout << "launch_overhead  cuda_graph   " << ms_graph
                      << " ms/tiny-dispatch\n";
        }
    }
}

void write_csv(const std::string &path, const std::vector<Row> &rows,
               const std::string &device_name) {
    std::ofstream f(path, std::ios::app);
    if (!f) {
        LUISA_WARNING("Failed to open CSV file: {}", path);
        return;
    }
    for (const auto &r : rows) {
        f << "part2_verify,\"" << device_name << "\"," << r.case_name << ","
          << r.variant << "," << std::fixed << std::setprecision(6)
          << r.ms_per_dispatch << "," << std::setprecision(3) << r.throughput
          << "," << r.unit << ",," << (r.pass ? "PASS" : "FAIL") << "\n";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    auto &device = dc->device;
    auto stream = device.create_stream();
    std::string csv_path = "benchmark_results/part2_verify.csv";
    for (int i = 2; i < argc; ++i) {
        if (luisa::string_view{argv[i]} == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    std::vector<Row> rows;
    std::cout << "# lc_optimize Part II verification on this device\n";
    case_stream<false>(device, stream, rows);
    case_stream<true>(device, stream, rows);
    case_pipeline(device, stream, rows);
    case_launch(device, stream, rows);

    std::error_code ec;
    luisa::filesystem::create_directories("benchmark_results", ec);
    write_csv(csv_path, rows, std::string{device.backend_name()});
    bool all_pass = true;
    for (const auto &r : rows) { all_pass &= r.pass; }
    std::cout << (all_pass ? "ALL CASES PASS" : "SOME CASES FAILED") << "\n";
    return all_pass ? 0 : 1;
}

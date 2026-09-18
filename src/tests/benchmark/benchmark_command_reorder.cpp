// Benchmark for the backend command-reordering pass implemented in
// src/backends/common/command_reorder_visitor.h (Vulkan and DirectX backends).
//
// What is measured
// ----------------
// One command batch holding `dispatches` independent, small but deliberately
// compute-heavy dispatches. Each dispatch writes its own sub-range of one
// destination buffer (`Buffer::view().subview(...)`) and reads one shared,
// read-only input buffer. Because those ranges never alias, the reorder pass
// merges the whole batch into ONE barrier-free layer: the backend records a
// single barrier for the batch and the GPU can overlap the dispatches. The same
// batch is then measured with reordering switched off, where every command
// keeps a layer of its own, so the backend emits a barrier per command and the
// dispatches serialize.
//
// Each thread runs a long serial chain of transcendentals (plus a block-local
// shared-memory reduction), so a batch costs milliseconds of *device* time
// while host-side command recording stays negligible. That keeps the comparison
// focused on what the extra barriers cost on the GPU, which is exactly what
// command reordering is meant to remove.
//
// Usage
// -----
//   benchmark_command_reorder <backend:dx|vk> [mode] [dispatches] [threads] [iters] [rounds] [verbose]
//
// Defaults: mode=0, dispatches=16, threads=256, iters=4096, rounds=5, verbose=0.
// Passing a non-zero `verbose` raises the log level so the backend reports how
// many reorder layers each batch actually produced (1 merged layer vs. one
// layer per command).
//
// mode 0 (disjoint): dispatch j writes sub-range j of the destination buffer;
// the batch has no hazards at all, so the ideal layer count is 1.
// mode 1 (rotate): dispatch j writes sub-range (j % 16), i.e. sixteen
// sub-ranges written round-robin. Each sub-range forms a genuine
// write-after-write chain, so the ideal layer count is ceil(dispatches / 16).
// A range tracker that keeps per-range precision (instead of collapsing to a
// union view) keeps those chains independent; a collapsing one serializes far
// more of the batch. This mode is the regression test for that precision.
//
// Examples:
//   xmake run benchmark_command_reorder vk
//   xmake run benchmark_command_reorder dx 1 64 256 4096 7
//
// The switch itself is the per-device `CommandReorderExt`
// (include/luisa/backends/ext/command_reorder_ext.h). Setting
// LUISA_DISABLE_COMMAND_REORDER=1 forces reordering off process-wide and wins
// over this benchmark, which then reports near-identical times for both groups.
#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"
#include <luisa/backends/ext/command_reorder_ext.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
using namespace luisa;
using namespace luisa::compute;

namespace {

/// Sub-ranges that mode 1 (rotate) cycles over.
constexpr size_t kRotateRanges = 16u;
/// Threads per block for the heavy kernel; also the size of the block-local
/// shared-memory reduction.
constexpr uint32_t kBlockSize = 32u;

struct Options {
    size_t mode{0u};
    size_t dispatches{16u};
    size_t threads{256u};
    size_t iters{4096u};
    size_t rounds{5u};
};

[[nodiscard]] size_t parse_uint(const char *text, const char *name, size_t minimum) {
    auto input = std::string_view{text};
    size_t value{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && value >= minimum,
                 "{} must be an integer >= {}, got \"{}\".", name, minimum, text);
    return value;
}
[[nodiscard]] size_t parse_positive(const char *text, const char *name) {
    return parse_uint(text, name, 1u);
}

struct Statistics {
    double min{};
    double median{};
    double mean{};
    double max{};
};

[[nodiscard]] Statistics summarize(luisa::vector<double> samples) {
    LUISA_ASSERT(!samples.empty(), "No samples to summarize.");
    std::sort(samples.begin(), samples.end());
    auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return {samples.front(),
            samples[samples.size() / 2u],
            sum / static_cast<double>(samples.size()),
            samples.back()};
}

/// One command batch: `dispatches` dispatches, each writing its own sub-range
/// of `dst` and reading the whole of `src`. `salt` differs per dispatch so that
/// every sub-range ends up with distinguishable contents. `Shader` here is the
/// typed `device.compile()` result (a `Shader<N, Args...>` wrapper), hence the
/// template parameter.
template<typename ShaderT>
[[nodiscard]] CommandList build_batch(const ShaderT &shader,
                                      const Buffer<float> &src,
                                      const Buffer<float4> &dst,
                                      const Options &opt) {
    CommandList list;
    list.reserve(opt.dispatches, 0u);
    for (auto j = size_t{0}; j < opt.dispatches; j++) {
        auto range_index = opt.mode == 0u ? j : j % kRotateRanges;
        auto dst_view = dst.view().subview(range_index * opt.threads, opt.threads);
        list << shader(src.view(), dst_view, static_cast<uint32_t>(opt.iters),
                       static_cast<uint32_t>(j))
                    .dispatch(static_cast<uint32_t>(opt.threads));
    }
    return list;
}

/// Submit `rounds` batches with the reorder switch set to `reorder_enabled` and
/// return the wall-clock milliseconds of each batch (submission + sync).
template<typename ShaderT>
[[nodiscard]] luisa::vector<double> measure_group(Stream &stream,
                                                  CommandReorderExt *reorder_ext,
                                                  const ShaderT &shader,
                                                  const Buffer<float> &src,
                                                  const Buffer<float4> &dst,
                                                  const Options &opt,
                                                  bool reorder_enabled) {
    reorder_ext->set_command_reorder_enabled(reorder_enabled);
    if (auto effective = reorder_ext->command_reorder_enabled(); effective != reorder_enabled) {
        // A process-wide override (LUISA_DISABLE_COMMAND_REORDER) wins over the
        // runtime switch, so say so instead of measuring the same thing twice.
        LUISA_WARNING("Reorder switch: requested {}, effective {}.", reorder_enabled, effective);
    }
    // Warm up: the first submission of a shader also touches driver-side
    // pipeline setup, which must not be attributed to either group.
    stream << build_batch(shader, src, dst, opt).commit() << synchronize();
    luisa::vector<double> samples;
    samples.reserve(opt.rounds);
    for (auto r = size_t{0}; r < opt.rounds; r++) {
        Clock clock;
        stream << build_batch(shader, src, dst, opt).commit() << synchronize();
        samples.emplace_back(clock.toc());
    }
    return samples;
}

[[nodiscard]] luisa::vector<float4> download(Stream &stream, const Buffer<float4> &dst) {
    luisa::vector<float4> host(dst.size());
    stream << dst.copy_to(luisa::span{host}) << synchronize();
    return host;
}

[[nodiscard]] bool bitwise_equal(luisa::span<const float4> a, luisa::span<const float4> b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

void print_row(luisa::string_view name, const Statistics &s) {
    std::ostringstream cells;
    cells << std::fixed << std::setprecision(3) << std::setw(11) << s.min << std::setw(11)
          << s.median << std::setw(11) << s.mean << std::setw(11) << s.max;
    LUISA_INFO("{:<24}{}", name, cells.str());
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_ERROR_WITH_LOCATION(
            "Usage: {} <backend:dx|vk> [dispatches] [threads] [iters] [rounds] [verbose]",
            argv[0]);
        return 1;
    }
    Options opt;
    if (argc > 2) {
        opt.mode = parse_uint(argv[2], "mode", 0u);
    }
    if (argc > 3) {
        opt.dispatches = parse_positive(argv[3], "dispatches");
    }
    if (argc > 4) {
        opt.threads = parse_positive(argv[4], "threads");
    }
    if (argc > 5) {
        opt.iters = parse_positive(argv[5], "iters");
    }
    if (argc > 6) {
        opt.rounds = parse_positive(argv[6], "rounds");
    }
    auto verbose = argc > 7 ? parse_positive(argv[7], "verbose") != 0u : false;
    LUISA_ASSERT(opt.mode <= 1u, "mode must be 0 (disjoint) or 1 (rotate).");
    LUISA_ASSERT(opt.dispatches <= 1024u,
                 "dispatches must be <= 1024 to bound buffer memory and batch size.");
    LUISA_ASSERT(opt.threads % kBlockSize == 0u,
                 "threads must be a multiple of the kernel block size ({}).", kBlockSize);

    if (verbose) {
        log_level_verbose();
    } else {
        log_level_info();
    }
    auto [context, device] = test::create_device(argc, argv);
    auto stream = device.create_stream();

    auto *reorder_ext = device.extension<CommandReorderExt>();
    if (reorder_ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Backend \"{}\" does not implement CommandReorderExt: this benchmark needs "
            "the vk or dx backend.",
            device.backend_name());
        return 1;
    }
    LUISA_INFO("CommandReorderExt present; initial state: {}.",
               reorder_ext->command_reorder_enabled() ? "enabled" : "disabled");
    // Probe the runtime switch: if a process-wide override already forced it
    // off, both groups would measure the serialized baseline.
    reorder_ext->set_command_reorder_enabled(true);
    if (!reorder_ext->command_reorder_enabled()) {
        LUISA_WARNING("Command reordering is forced off by LUISA_DISABLE_COMMAND_REORDER, "
                      "which overrides the runtime switch: both groups measure the serialized "
                      "baseline and the reported speedup is meaningless.");
    }
    reorder_ext->set_command_reorder_enabled(false);

    // Input: read-only and shared by every dispatch of the batch, so it never
    // splits the batch into separate layers.
    auto src = device.create_buffer<float>(opt.threads);
    // Output: one disjoint sub-range per dispatch (mode 0) or sixteen
    // sub-ranges written round-robin (mode 1).
    auto range_count = opt.mode == 0u ? opt.dispatches : kRotateRanges;
    auto dst = device.create_buffer<float4>(range_count * opt.threads);
    {
        luisa::vector<float> host_src(opt.threads);
        for (size_t i = 0; i < opt.threads; i++) {
            host_src[i] = std::sin(static_cast<float>(i) * 0.017f) * 2.f + 0.75f;
        }
        stream << src.copy_from(luisa::span{host_src}) << synchronize();
    }

    // A latency-bound transcendental chain: every iteration feeds the next one,
    // so a single thread only keeps its own dependency chain busy and the
    // dispatch as a whole is a good candidate to overlap with its neighbours.
    // The block-local shared-memory reduction adds real per-block work without
    // introducing any cross-dispatch dependency.
    Kernel1D heavy = [](BufferFloat input, BufferFloat4 output,
                        Var<uint> iterations, Var<uint> salt) noexcept {
        set_block_size(kBlockSize, 1u, 1u);
        auto tid = dispatch_id().x;
        auto lane = thread_x();
        Shared<float> scratch{kBlockSize};
        Var<float> x = input.read(tid) + salt.cast<float>() * 0.03125f;
        Var<float> y = x * 0.5f + 1.0f;
        Var<float4> acc = make_float4(0.f);
        for (auto i : dynamic_range(iterations)) {
            auto s = sin(x);
            auto c = cos(y);
            auto p = s * c;
            Var<float> r0 = sqrt(abs(x) + 1.0f);
            acc += make_float4(s, c, p, r0);
            Var<float> t0 = exp(-abs(s));
            Var<float> t1 = log(abs(y) + 1.0f);
            Var<float> t2 = atan2(s, c);
            Var<float> t3 = p * p + i.cast<float>() * 0.000125f;
            acc += make_float4(t0, t1, t2, t3);
            x = x * 0.5f + s;
            y = y * 0.5f + c;
        }
        scratch[lane] = length(acc);
        sync_block();
        Var<float> reduction = 0.f;
        for (auto k = 0u; k < kBlockSize; k++) {
            reduction += scratch[k];
        }
        output.write(tid, acc + reduction);
    };
    // Cache bypassed so a measurement run never touches the shader cache on disk
    // (a cache hit reads a file inside Device::compile). The round-trip itself is
    // covered by test_vk_cuda_kernel_launch, which compiles with the cache on.
    auto shader = device.compile(heavy, ShaderOption{.enable_cache = false});

    LUISA_INFO("Batch shape: mode {} ({}), {} dispatches x {} threads x {} iterations "
               "({} threads/batch, block size {}).",
               opt.mode, opt.mode == 0u ? "disjoint" : "rotate over 16 sub-ranges",
               opt.dispatches, opt.threads, opt.iters,
               opt.dispatches * opt.threads, kBlockSize);

    // Reference run with reordering disabled: one command per layer, i.e. the
    // strictly ordered baseline. Everything afterwards must match it bit for
    // bit, otherwise the merged dispatches are racing.
    reorder_ext->set_command_reorder_enabled(false);
    stream << build_batch(shader, src, dst, opt).commit() << synchronize();
    auto reference = download(stream, dst);

    // The two groups alternate every round so that clock ramping and thermal
    // drift hit them equally.
    luisa::vector<double> on_samples;
    luisa::vector<double> off_samples;
    for (auto round = size_t{0}; round < opt.rounds; round++) {
        auto on = measure_group(stream, reorder_ext, shader, src, dst, opt, true);
        auto off = measure_group(stream, reorder_ext, shader, src, dst, opt, false);
        on_samples.insert(on_samples.end(), on.begin(), on.end());
        off_samples.insert(off_samples.end(), off.begin(), off.end());
        LUISA_INFO("round {}: reorder on {:.3f} ms, off {:.3f} ms", round + 1, on.front(), off.front());
    }

    // Correctness: both configurations must reproduce the strictly ordered
    // reference exactly, and each dispatch must have written its own sub-range.
    reorder_ext->set_command_reorder_enabled(true);
    stream << build_batch(shader, src, dst, opt).commit() << synchronize();
    auto reordered = download(stream, dst);
    reorder_ext->set_command_reorder_enabled(false);
    stream << build_batch(shader, src, dst, opt).commit() << synchronize();
    auto serialized = download(stream, dst);
    reorder_ext->set_command_reorder_enabled(true);

    auto all_finite = std::all_of(reordered.begin(), reordered.end(), [](float4 v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
    });
    auto span_at = [&](luisa::span<const float4> s, size_t j) {
        return s.subspan(j * opt.threads, opt.threads);
    };
    // Mode 1 rewrites the same sub-ranges repeatedly, so only the bitwise
    // equality with the serialized reference proves ordering there.
    auto distinct_ranges = true;
    if (opt.mode == 0u) {
        for (auto j = size_t{1}; j < opt.dispatches; j++) {
            if (bitwise_equal(span_at(luisa::span<const float4>{reordered}, j - 1u),
                              span_at(luisa::span<const float4>{reordered}, j))) {
                distinct_ranges = false;
                break;
            }
        }
    }
    auto matches_reference =
        bitwise_equal(luisa::span<const float4>{reordered}, luisa::span<const float4>{reference}) &&
        bitwise_equal(luisa::span<const float4>{serialized}, luisa::span<const float4>{reference});

    auto on = summarize(on_samples);
    auto off = summarize(off_samples);
    LUISA_INFO("---------------------------------------------------------------------------");
    LUISA_INFO("{:<24}{:>11}{:>11}{:>11}{:>11}", "group", "min(ms)", "median", "mean", "max");
    print_row("reorder enabled", on);
    print_row("reorder disabled", off);
    LUISA_INFO("---------------------------------------------------------------------------");
    LUISA_INFO("Speedup (median, disabled / enabled): {:.2f}x", off.median / on.median);
    LUISA_INFO("Reorder layers per batch: enabled ~1, disabled {} (one per command).",
               opt.dispatches);
    LUISA_INFO("Checks: reordered == strictly ordered result = {}, outputs finite = {}, "
               "one distinct sub-range per dispatch = {}.",
               matches_reference, all_finite, distinct_ranges);

    if (!matches_reference || !all_finite || !distinct_ranges) {
        LUISA_ERROR_WITH_LOCATION(
            "Command reorder benchmark validation FAILED: the reordered batch must "
            "reproduce the strictly ordered result exactly, produce finite outputs and "
            "write one distinct sub-range per dispatch.");
    }
    if (on.median > off.median) {
        LUISA_WARNING("Reordering did not help in this run (on {:.3f} ms > off {:.3f} ms); "
                      "check the batch shape or driver scheduling.",
                      on.median, off.median);
    } else {
        LUISA_INFO("Command reorder benchmark validation PASSED.");
    }
    return 0;
}

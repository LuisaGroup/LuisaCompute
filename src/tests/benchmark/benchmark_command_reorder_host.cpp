// Host-side benchmark for the backend command-reordering pass implemented in
// src/backends/common/command_reorder_visitor.h (Vulkan and DirectX backends).
//
// What is measured
// ----------------
// Pure *host* submission cost of command batches. The kernel is trivial (one
// write per thread over a single block), so the GPU is never the bottleneck
// and the wall-clock time of the submission loop is dominated by host work:
// the CommandReorderVisitor's hazard analysis plus the backend's per-layer
// barrier bookkeeping. Batches are submitted back to back without
// synchronization so the GPU pipeline stays full and does not throttle the
// measurement.
//
// Two stress profiles (mode):
//   0 = disjoint: every dispatch writes its own sub-range of one buffer, so a
//      good range tracker merges the whole batch into ONE layer. Watching the
//      per-batch host cost collapse after the range-tracking optimization is
//      the point of this mode.
//   1 = shared: every dispatch writes the SAME range. That is a genuine
//      write-after-write hazard, so each dispatch gets its own layer and the
//      backend emits a barrier per command. This measures the per-layer host
//      cost directly and must not regress.
//   2 = rotate: dispatches write sixteen sub-ranges round-robin, a genuine but
//      bounded hazard (ideal: ceil(dispatches/16) layers). A range tracker
//      that loses per-range precision serializes much more of the batch.
//
// Usage
// -----
//   benchmark_command_reorder_host <backend:dx|vk> [mode] [dispatches] [batches] [threads] [rounds] [verbose]
//
// Defaults: mode=0, dispatches=256, batches=200, threads=32, rounds=5, verbose=0.
//
// Examples:
//   xmake run benchmark_command_reorder_host vk
//   xmake run benchmark_command_reorder_host dx 1 256 200 32 5
// Everything this benchmark added for the Vulkan profiling work (the
// descriptor-path A/B switches, their device config and the restore-cost report)
// is gated by the submit profiler's private compile-time macro
// LUISA_VULKAN_PROFILE_SUBMIT (default 0). The reorder measurement itself is
// untouched and always available. The macro is a source-level opt-in: it is
// deliberately not an XMake/CMake option and not an environment variable.
#if !defined(LUISA_VULKAN_PROFILE_SUBMIT)
#define LUISA_VULKAN_PROFILE_SUBMIT 0
#endif

#include "ut/ut.hpp"// boost.ut cfg used by test_device.h
#include "test_device.h"
#include <luisa/backends/ext/command_reorder_ext.h>
// The A/B switch is a VulkanDeviceConfigExt and the profiler report reads Vulkan
// capability keys, so both only exist in a profiler-enabled build.
#if LUISA_VULKAN_PROFILE_SUBMIT
#include <luisa/backends/ext/vk_config_ext.h>
#endif// LUISA_VULKAN_PROFILE_SUBMIT
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/string.h>
#include <algorithm>
#include <charconv>
#if LUISA_VULKAN_PROFILE_SUBMIT
#include <cstdlib>// std::strtoull, profiler report only
#include <cstring>// std::strcmp, the A/B command line only
#endif// LUISA_VULKAN_PROFILE_SUBMIT
#include <numeric>
using namespace luisa;
using namespace luisa::compute;

namespace {

struct Options {
    size_t mode{0u};        // 0 = disjoint sub-ranges, 1 = shared range (max layering)
    size_t dispatches{256u};
    size_t batches{200u};
    size_t threads{32u};
    size_t rounds{5u};
};

[[nodiscard]] size_t parse_uint(const char *text, const char *name, size_t minimum) {
    auto input = luisa::string_view{text};
    size_t value{};
    auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    LUISA_ASSERT(result.ec == std::errc{} && result.ptr == input.data() + input.size() && value >= minimum,
                 "{} must be an integer >= {}, got \"{}\".", name, minimum, text);
    return value;
}
[[nodiscard]] size_t parse_positive(const char *text, const char *name) {
    return parse_uint(text, name, 1u);
}

#if LUISA_VULKAN_PROFILE_SUBMIT
// Plan P2.2 A/B switch: `--descriptor-update-template` (anywhere after the backend
// argument) answers VulkanDeviceConfigExt::enable_descriptor_update_template()
// with true, so every per-dispatch set-0 update goes through a cached
// VkDescriptorUpdateTemplate. Without it the library default applies, which is the
// pre-P2.2 vkUpdateDescriptorSets call - the path that measured *faster* on this
// driver. Nothing else changes, so two runs of the same binary differ only in the
// descriptor-update path, which is what makes the phase-2 measurement an A/B
// instead of a session-to-session comparison.
 // Plan P2.2 / P2.3 A/B harness: either switch can be turned on independently,
 // so one binary in one session can compare the three set-0 paths (classic
 // vkUpdateDescriptorSets, cached update templates, push descriptors) on the
 // same workload. Nothing else changes, which is what makes the phase-2
 // measurement an A/B instead of a session-to-session comparison.
 class DescriptorPathConfig final : public VulkanDeviceConfigExt {
     bool _templates;
     bool _push;

 public:
     explicit DescriptorPathConfig(bool templates, bool push) noexcept
         : _templates{templates}, _push{push} {}
     [[nodiscard]] bool enable_descriptor_update_template() const noexcept override {
         return _templates;
     }
     [[nodiscard]] bool enable_push_descriptor() const noexcept override {
         return _push;
     }
 };
#endif// LUISA_VULKAN_PROFILE_SUBMIT

struct Statistics {
    double min{};
    double median{};
    double mean{};
};

[[nodiscard]] Statistics summarize(luisa::vector<double> samples) {
    LUISA_ASSERT(!samples.empty(), "No samples to summarize.");
    luisa::sort(samples.begin(), samples.end());
    auto sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return {samples.front(), samples[samples.size() / 2u],
            sum / static_cast<double>(samples.size())};
}

/// One command batch: `dispatches` trivial dispatches, either into disjoint
/// sub-ranges of `dst` (mode 0), all into the same leading range (mode 1), or
/// round-robin over sixteen sub-ranges (mode 2).
template<typename ShaderT>
[[nodiscard]] CommandList build_batch(const ShaderT &shader,
                                      const Buffer<float4> &dst,
                                      const Options &opt) {
    CommandList list;
    list.reserve(opt.dispatches, 0u);
    auto dispatch_size = static_cast<uint32_t>(opt.threads);
    for (auto j = size_t{0}; j < opt.dispatches; j++) {
        auto offset = opt.mode == 0u   ? j * opt.threads :
                      opt.mode == 1u   ? 0u :
                      /* mode == 2u */ (j % 16u) * opt.threads;
        auto dst_view = dst.view().subview(offset, opt.threads);
        list << shader(dst_view, static_cast<uint32_t>(j)).dispatch(dispatch_size);
    }
    return list;
}

/// Submit `batches` batches back to back (no synchronization between them, so
/// host submission is the bottleneck) and return the host milliseconds of each
/// repetition's submission loop.
template<typename ShaderT>
[[nodiscard]] luisa::vector<double> measure_group(Stream &stream,
                                                  CommandReorderExt *reorder_ext,
                                                  const ShaderT &shader,
                                                  const Buffer<float4> &dst,
                                                  const Options &opt,
                                                  bool reorder_enabled) {
    reorder_ext->set_command_reorder_enabled(reorder_enabled);
    if (auto effective = reorder_ext->command_reorder_enabled(); effective != reorder_enabled) {
        // A process-wide override (LUISA_DISABLE_COMMAND_REORDER) wins over the
        // runtime switch, so say so instead of measuring the same thing twice.
        LUISA_WARNING("Reorder switch: requested {}, effective {}.", reorder_enabled, effective);
    }
    // Warm up: driver-side pipeline setup plus allocator steady state must not
    // be attributed to either group.
    stream << build_batch(shader, dst, opt).commit() << synchronize();
    luisa::vector<double> samples;
    samples.reserve(opt.rounds);
    for (auto r = size_t{0}; r < opt.rounds; r++) {
        Clock clock;
        for (auto b = size_t{0}; b < opt.batches; b++) {
            stream << build_batch(shader, dst, opt).commit();
        }
        samples.emplace_back(clock.toc());
    }
    stream << synchronize();
    return samples;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LUISA_ERROR_WITH_LOCATION(
            "Usage: {} <backend:dx|vk> [mode] [dispatches] [batches] [threads] [rounds] [verbose]",
            argv[0]);
        return 1;
    }
#if LUISA_VULKAN_PROFILE_SUBMIT
    // Strip the P2.2 switch (and only it) before the positional arguments are
    // read, so `... vk 0 256 200 32 5 --descriptor-update-template` and
    // `... vk 0 256 200 32 5` describe the same workload.
 auto descriptor_update_template = false;
 auto push_descriptor = false;
 auto classic_descriptor = false;
    {
        auto kept = 0;
        for (auto i = 0; i < argc; ++i) {
 if (std::strcmp(argv[i], "--descriptor-update-template") == 0) {
 descriptor_update_template = true;
 continue;
 }
 // Plan P2.3: `--push-descriptor` asks for set-0 push descriptors,
 // `--classic-descriptor` forces the classic allocate/update/bind path on a
 // device that could push. Without either flag the library default applies.
 if (std::strcmp(argv[i], "--push-descriptor") == 0) {
 push_descriptor = true;
 continue;
 }
 if (std::strcmp(argv[i], "--classic-descriptor") == 0) {
 push_descriptor = false;
 classic_descriptor = true;
 continue;
 }
            argv[kept++] = argv[i];
        }
        argv[kept] = nullptr;
        argc = kept;
    }
#endif// LUISA_VULKAN_PROFILE_SUBMIT

    Options opt;
    if (argc > 2) {
        opt.mode = parse_uint(argv[2], "mode", 0u);
    }
    if (argc > 3) {
        opt.dispatches = parse_positive(argv[3], "dispatches");
    }
    if (argc > 4) {
        opt.batches = parse_positive(argv[4], "batches");
    }
    if (argc > 5) {
        opt.threads = parse_positive(argv[5], "threads");
    }
    if (argc > 6) {
        opt.rounds = parse_positive(argv[6], "rounds");
    }
    auto verbose = argc > 7 ? parse_positive(argv[7], "verbose") != 0u : false;
    LUISA_ASSERT(opt.mode <= 2u, "mode must be 0 (disjoint sub-ranges), 1 (shared range) or 2 (rotate).");
    LUISA_ASSERT(opt.threads % 32u == 0u, "threads must be a multiple of the block size (32).");

    if (verbose) {
        log_level_verbose();
    } else {
        log_level_info();
    }
#if LUISA_VULKAN_PROFILE_SUBMIT
 // P2.2/P2.3 A/B: with `--descriptor-update-template` every per-dispatch set-0
    // update goes through a cached VkDescriptorUpdateTemplate; without it the
    // library default (legacy vkUpdateDescriptorSets, the measured-faster path)
    // runs. Both use the same creation helper otherwise.
 auto descriptor_config_requested =
 descriptor_update_template || push_descriptor || classic_descriptor;
 DeviceConfig device_config;
 if (descriptor_config_requested) {
 auto use_push = push_descriptor && !classic_descriptor;
 if (push_descriptor && classic_descriptor) {
 LUISA_WARNING("'--push-descriptor' wins over '--classic-descriptor'; both "
 "cannot be measured in one run.");
 }
 device_config.extension =
 luisa::make_unique<DescriptorPathConfig>(descriptor_update_template, use_push);
 }
 auto device_context = test::create_device_from_ut(
 argc, argv, descriptor_config_requested ? &device_config : nullptr);
    if (!device_context) { return 1; }
    auto [context, device] = std::move(*device_context);
 LUISA_INFO("descriptor updates: {} | set-0 binding: {} "
 "(maxPushDescriptors {})",
 device.query("vk_descriptor_update_template_path"),
 device.query("vk_push_descriptor_path"),
 device.query("vk_push_descriptor_max_count"));
#else
 // The descriptor-path A/B harness and the report above belong to the gated block:
 // a default build creates the device exactly as it always did - no config
 // extension and no descriptor-path query.
 auto [context, device] = test::create_device(argc, argv);
#endif// LUISA_VULKAN_PROFILE_SUBMIT
    auto stream = device.create_stream();

    auto *reorder_ext = device.extension<CommandReorderExt>();
    if (reorder_ext == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Backend \"{}\" does not implement CommandReorderExt: this benchmark needs "
            "the vk or dx backend.",
            device.backend_name());
        return 1;
    }

    auto dst = device.create_buffer<float4>(
        opt.mode == 0u ? opt.dispatches * opt.threads :
        opt.mode == 1u ? opt.threads :
                         16u * opt.threads);

    // One write per thread over one small block: the GPU is idle, so only the
    // host path is measured.
    Kernel1D tiny = [](BufferFloat4 output, Var<uint> salt) noexcept {
        set_block_size(32u, 1u, 1u);
        output.write(dispatch_id().x, make_float4(salt.cast<float>()));
    };
    // The shader cache round-trip is broken in this machine's release build
    // (pre-existing: a fresh cache entry crashes on read-back), so bypass it.
    auto shader = device.compile(tiny, ShaderOption{.enable_cache = false});

    LUISA_INFO("Host benchmark: mode {} ({}), {} dispatches/batch x {} batches, "
               "{} threads/dispatch, {} timed rounds.",
               opt.mode, opt.mode == 0u   ? "disjoint sub-ranges" :
                         opt.mode == 1u   ? "shared range" :
                                            "rotate over 16 sub-ranges",
               opt.dispatches, opt.batches, opt.threads, opt.rounds);

    luisa::vector<double> on_samples;
    luisa::vector<double> off_samples;
    for (auto round = size_t{0}; round < opt.rounds; round++) {
        auto on = measure_group(stream, reorder_ext, shader, dst, opt, true);
        auto off = measure_group(stream, reorder_ext, shader, dst, opt, false);
        on_samples.insert(on_samples.end(), on.begin(), on.end());
        off_samples.insert(off_samples.end(), off.begin(), off.end());
    }

    auto on = summarize(on_samples);
    auto off = summarize(off_samples);
    auto per_batch = [](double total_ms, const Options &o) {
        return total_ms / static_cast<double>(o.batches);
    };
    LUISA_INFO("----------------------------------------------------------------------------------");
    LUISA_INFO("{:<24}{:>14}{:>14}{:>14}{:>16}", "group", "min(ms)", "median", "mean", "ms/batch(med)");
    LUISA_INFO("{:<24}{:>14.3f}{:>14.3f}{:>14.3f}{:>16.4f}", "reorder enabled",
               on.min, on.median, on.mean, per_batch(on.median, opt));
    LUISA_INFO("{:<24}{:>14.3f}{:>14.3f}{:>14.3f}{:>16.4f}", "reorder disabled",
               off.min, off.median, off.mean, per_batch(off.median, opt));
    LUISA_INFO("----------------------------------------------------------------------------------");
    LUISA_INFO("Host cost ratio (median, disabled / enabled): {:.2f}x",
               off.median / on.median);

#if LUISA_VULKAN_PROFILE_SUBMIT
 // Plan phase-3 gate (§5.1): the same run, in a profiler-enabled build, attributes
 // the host encode window. The restore barrier of F5 is the one cost phase 3 could
 // remove, so it is reported as ns per restore, ns per batch and as a fraction of
 // the measured host encode time. The whole block exists only with
 // LUISA_VULKAN_PROFILE_SUBMIT=1 (without it there are no counters to read), and
 // the counters are cumulative for the process, so the timing groups above should
 // be measured in a run built without the macro.
    if (device.query("vk_profile_enabled") == "1") {
        auto read_counter = [&device](const char *key) {
            auto text = device.query(key);
            return text.empty() ? uint64_t{0u} :
                                  std::strtoull(text.c_str(), nullptr, 10);
        };
        auto restores = read_counter("vk_profile_restore_calls");
        auto restore_ns = read_counter("vk_profile_restore_host_ns");
        auto encode_ns = read_counter("vk_profile_host_encode_ns");
        auto batches = read_counter("vk_profile_command_batches");
        auto state_buffers = read_counter("vk_profile_restore_state_buffers");
        auto state_mips = read_counter("vk_profile_restore_state_texture_mips");
        LUISA_INFO("----------------------------------------------------------------------------------");
        LUISA_INFO("vk_profile_report = {}", device.query("vk_profile_report"));
        LUISA_INFO("restore attribution: {} calls in {} batches ({} buffer / {} "
                   "image / {} memory structs), {:.2f} us/call, {:.3f} us/batch, "
                   "{:.3f} % of the {:.2f} ms host encode window",
                   restores, batches,
                   read_counter("vk_profile_restore_buffer_barriers"),
                   read_counter("vk_profile_restore_image_barriers"),
                   read_counter("vk_profile_restore_memory_barriers"),
                   restores == 0u ? 0.0 :
                                    static_cast<double>(restore_ns) / 1000.0 /
                                        static_cast<double>(restores),
                   batches == 0u ? 0.0 :
                                   static_cast<double>(restore_ns) / 1000.0 /
                                       static_cast<double>(batches),
                   encode_ns == 0u ? 0.0 :
                                     100.0 * static_cast<double>(restore_ns) /
                                         static_cast<double>(encode_ns),
                   static_cast<double>(encode_ns) / 1.0e6);
        LUISA_INFO("restore tracker state consumed: {} buffers, {} texture mips "
                   "({:.2f} buffers / {:.2f} mips per batch)",
                   state_buffers, state_mips,
                   restores == 0u ? 0.0 : static_cast<double>(state_buffers) / static_cast<double>(restores),
                   restores == 0u ? 0.0 : static_cast<double>(state_mips) / static_cast<double>(restores));
    }
#endif// LUISA_VULKAN_PROFILE_SUBMIT
    return 0;
}

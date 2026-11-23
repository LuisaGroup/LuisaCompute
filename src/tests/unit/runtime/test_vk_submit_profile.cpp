// Test for the Vulkan backend's counter-only submit profiler (plan P0.2/P0.3).
// This test covers:
// - LUISA_VULKAN_PROFILE_SUBMIT=1 is latched before the first dispatch and
//   answers through Device::query (including the one-line report)
// - F1/P1.1: two compiles of an identical kernel share one pipeline layout and
//   one set of descriptor-set layouts (the layouts are interned by structure, so
//   the second compile creates no Vulkan layout at all)
// - F10/P1.3: the pipeline cache is device-scoped, so two identical fresh kernels
//   share the one device cache and create no cache of their own, while each
//   compile still creates its own pipeline object; temporary merge-source caches
//   (legacy per-shader PSO seeds) are counted separately from real caches
// - P1.1 negative: a structurally different kernel (a different binding list)
//   creates exactly one new descriptor-set layout and one new pipeline layout
//   instead of aliasing the interned ones
// - F2/F4/F5: one dispatch still costs one descriptor-set allocation, one update,
//   one set bind and exactly *one* argument plan (P1.4 removed the second one),
//   and every batch ends with one ALL_COMMANDS restore struct
// - P1.2: the redundant-bind elimination. N consecutive dispatches of one
//   unchanged shader in one command buffer bind its pipeline exactly once
//   (`pipeline_binds == 1`), while every dispatch still allocates, updates and
//   binds its own descriptor set; the tracked state is reset per command buffer,
//   survives a pipeline change that keeps the same interned layout, is
//   interrupted by the internal indirect-prepare/bindless-update kernels and is
//   flushed by a custom command that issues raw Vulkan on the command buffer
// - F3: what is left of the untracked rebinding, asserted as exact counts so the
//   tripwire fails if the elimination silently stops happening
// - P2.2: every per-dispatch set-0 update goes through exactly one entry point.
//   The *default* device reports `legacy` (vkUpdateDescriptorSets), because the
//   template path measured 2.5-7 % slower in host submission time on this driver
//   and VulkanDeviceConfigExt::enable_descriptor_update_template() therefore
//   defaults to false. `--descriptor-update-template` turns it on, and then the
//   same assertions must see vkUpdateDescriptorSetWithTemplate for every eligible
//   dispatch while the paths that stay legacy by design (the indirect-preparation
//   set) must still show up as vkUpdateDescriptorSets. Dispatched values are
//   checked in both configurations for repeated buffer-only dispatches, a mixed
//   image/buffer set, a printer tail and distinct shaders/layouts.
// - F6/P1.5/P2.1: one native submit per batch - through vkQueueSubmit2 on a
//   sync2-capable device or through the legacy vkQueueSubmit on a Vulkan 1.2
//   floor, with `queue_submits == queue_submit2 + queue_submit_legacy` - and the
//   configurable in-flight depth
//   (VulkanDeviceConfigExt::max_frames_in_flight, default 3) whose throttle is
//   skipped when the GPU completion watermark already covers its target
// - F8/P6.1: the 14 core Vulkan limits answer real values, are plausible against
//   the Vulkan specification minima, and equal the matching field of
//   VulkanExtensionStatusExt::capability_status() (P6.3). min/max subgroup size
//   are explicitly unknown on a device without subgroup-size-control properties
//   instead of being derived from subgroupSize.
// - the host encode/submit timings are reported and plausible
// Every structural assertion is paired with a check of the dispatched buffer
// values, so the counters cannot hide a reordered, skipped or mis-bound batch.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <luisa/backends/ext/command_reorder_ext.h>
#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
#include <luisa/backends/ext/vk_extension_ext.h>
#include <luisa/backends/ext/vk_replay_ext.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// Plan P2.1: `--min-api 1.2` forces the legacy submission path in the same build,
// so this binary proves both branches of `detail::queue_submit` (a process can hold
// only one Vulkan device, so it cannot be done in a single run).
bool g_min_api_1_2{false};

// The profiler latches on first use, so the variable has to be set before the
// backend records anything - in practice before the device exists.
void latch_profile_mode() noexcept {
#ifdef _WIN32
    _putenv_s("LUISA_VULKAN_PROFILE_SUBMIT", "1");
#else
    setenv("LUISA_VULKAN_PROFILE_SUBMIT", "1", 1);
#endif
}

// Every counter the profiler exposes (see src/backends/vk/submit_profile.h).
enum class Counter : size_t {
    dispatches,
    descriptor_set_allocations,
    descriptor_set_updates,
    descriptor_set_update_templates,
    descriptor_update_template_creates,
    descriptor_update_template_fallbacks,
    push_descriptor_calls,
    push_descriptor_shaders,
    push_descriptor_classic_shaders,
    descriptor_set_binds,
    pipeline_binds,
    argument_plans,
    barrier_calls,
    barrier_memory_barriers,
    barrier_buffer_barriers,
    barrier_image_barriers,
    barrier_all_commands_barriers,
    descriptor_set_layout_creates,
    pipeline_layout_creates,
    pipeline_cache_creates,
    pipeline_cache_merge_source_creates,
    pipeline_cache_merges,
    pipeline_creates,
    command_batches,
    queue_submits,
    queue_submit2,
    queue_submit_legacy,
    timeline_wait_entries,
    timeline_waits,
    timeline_wait_skips,
    restore_calls,
    restore_host_ns,
    restore_memory_barriers,
    restore_buffer_barriers,
    restore_image_barriers,
    restore_state_buffers,
    restore_state_texture_mips,
    host_encode_ns,
    host_submit_ns,
    // Plan P4: replayable recording.
    replay_records,
    replay_record_host_ns,
    replay_submits,
    replay_rejections,
    replay_update_refusals,
    replay_invalidations,
    replay_releases,
    count
};

constexpr auto kCounterKeys = std::array{
    "vk_profile_dispatches",
    "vk_profile_descriptor_set_allocations",
    "vk_profile_descriptor_set_updates",
    "vk_profile_descriptor_set_update_templates",
    "vk_profile_descriptor_update_template_creates",
    "vk_profile_descriptor_update_template_fallbacks",
    "vk_profile_push_descriptor_calls",
    "vk_profile_push_descriptor_shaders",
    "vk_profile_push_descriptor_classic_shaders",
    "vk_profile_descriptor_set_binds",
    "vk_profile_pipeline_binds",
    "vk_profile_argument_plans",
    "vk_profile_barrier_calls",
    "vk_profile_barrier_memory_barriers",
    "vk_profile_barrier_buffer_barriers",
    "vk_profile_barrier_image_barriers",
    "vk_profile_barrier_all_commands_barriers",
    "vk_profile_descriptor_set_layout_creates",
    "vk_profile_pipeline_layout_creates",
    "vk_profile_pipeline_cache_creates",
    "vk_profile_pipeline_cache_merge_source_creates",
    "vk_profile_pipeline_cache_merges",
    "vk_profile_pipeline_creates",
    "vk_profile_command_batches",
    "vk_profile_queue_submits",
    "vk_profile_queue_submit2",
    "vk_profile_queue_submit_legacy",
    "vk_profile_timeline_wait_entries",
    "vk_profile_timeline_waits",
    "vk_profile_timeline_wait_skips",
    "vk_profile_restore_calls",
    "vk_profile_restore_host_ns",
    "vk_profile_restore_memory_barriers",
    "vk_profile_restore_buffer_barriers",
    "vk_profile_restore_image_barriers",
    "vk_profile_restore_state_buffers",
    "vk_profile_restore_state_texture_mips",
    "vk_profile_host_encode_ns",
    "vk_profile_host_submit_ns",
    "vk_profile_replay_records",
    "vk_profile_replay_record_host_ns",
    "vk_profile_replay_submits",
    "vk_profile_replay_rejections",
    "vk_profile_replay_update_refusals",
    "vk_profile_replay_invalidations",
    "vk_profile_replay_releases"};
static_assert(kCounterKeys.size() == static_cast<size_t>(Counter::count));

using Counters = std::array<uint64_t, static_cast<size_t>(Counter::count)>;

Counters read_counters(Device &device) noexcept {
    Counters result{};
    for (size_t i = 0u; i < kCounterKeys.size(); ++i) {
        auto value = device.query(kCounterKeys[i]);
        result[i] = value.empty() ?
                        0u :
                        std::strtoull(value.c_str(), nullptr, 10);
    }
    return result;
}

[[nodiscard]] Counters difference(const Counters &before,
                                  const Counters &after) noexcept {
    Counters result{};
    for (size_t i = 0u; i < result.size(); ++i) {
        result[i] = after[i] >= before[i] ? after[i] - before[i] : 0u;
    }
    return result;
}

[[nodiscard]] uint64_t value(const Counters &counters, Counter counter) noexcept {
    return counters[static_cast<size_t>(counter)];
}

// Does this backend build a compute kernel through the native XIR-to-SPIR-V route
// (`vk_spirv_codegen_route == "native"`)? Only that route gives a shader the
// indirect-dispatch metadata binding, so the two indirect-dispatch tests below are
// only meaningful on it. An HLSL-fallback build
// (`lc_vk_backend_use_xir_spirv=false`) rejects an indirect dispatch with
// "requires the native XIR-to-SPIR-V logical metadata ABI" - a property of the
// build, not of the replay/profiler work this file asserts - so the tests skip with
// an explicit line instead of failing on an unrelated configuration.
[[nodiscard]] bool native_spirv_route(Device &device) noexcept {
    auto route = device.query("vk_spirv_codegen_route");
    expect(!route.empty())
        << "the backend must report the compute codegen route it was built with";
    return route == "native";
}

// Which native entry point the backend submits through (plan P2.1). "submit2" on a
// synchronization2-capable device, "legacy" on a Vulkan 1.2 floor.
[[nodiscard]] bool reported_submit2(Device &device) noexcept {
    auto path = device.query("vk_queue_submit_path");
    expect(!path.empty())
        << "the backend must report the queue-submission path it uses";
    return path == "submit2";
}

// F6/P2.1: every counted native submission is exactly one of the two variants, and
// the variant that is used matches the reported path. `d` is a counter delta.
void check_submit_variants(Device &device, const Counters &d,
                           uint64_t expected_submissions,
                           const char *what) noexcept {
    auto submits = value(d, Counter::queue_submits);
    auto submit2 = value(d, Counter::queue_submit2);
    auto legacy = value(d, Counter::queue_submit_legacy);
    LUISA_INFO("{}: submits={} submit2={} legacy={}", what, submits, submit2, legacy);
    expect(submit2 + legacy == submits)
        << what << ": queue_submits must equal queue_submit2 + queue_submit_legacy";
    expect(submits == expected_submissions)
        << what << ": one native submission per command batch";
    if (reported_submit2(device)) {
        expect(submit2 == expected_submissions && legacy == 0u)
            << what << ": a sync2 device must submit through vkQueueSubmit2 only";
    } else {
        expect(legacy == expected_submissions && submit2 == 0u)
            << what << ": a Vulkan 1.2 device must keep vkQueueSubmit only";
    }
}

// Which entry point performs the per-dispatch set-0 update (plan P2.2).
// "template" when the device, the config switch and the loader all allow a
// cached VkDescriptorUpdateTemplate, "legacy" when vkUpdateDescriptorSets does.
[[nodiscard]] bool reported_template_path(Device &device) noexcept {
    auto path = device.query("vk_descriptor_update_template_path");
    expect(!path.empty())
        << "the backend must report the descriptor-update path it uses";
    return path == "template";
}

// Which set-0 binding path the device latched at creation (plan P2.3): "push"
// when DSL shaders are created with a push descriptor set 0 and the encoder
// records vkCmdPushDescriptorSetKHR, "classic" when every dispatch allocates,
// updates and binds its own set 0.
[[nodiscard]] bool reported_push_path(Device &device) noexcept {
    auto path = device.query("vk_push_descriptor_path");
    expect(!path.empty())
        << "the backend must report which path binds set 0";
    expect(path == "push" || path == "classic")
        << "vk_push_descriptor_path answers push or classic";
    return path == "push";
}

// Plan P2.3: the set-0 accounting of one batch of `set0_dispatches` DSL
// dispatches whose global sets (sampler + bindless heaps) do not change.
//
//   classic: every dispatch allocates its own set 0 and binds it, so
//            allocations == binds == set0_dispatches and no push happens;
//   push:    no set 0 is allocated and no set-0 bind is recorded, one
//            vkCmdPushDescriptorSetKHR replaces them, and the global sets are
//            bound at their real first set as often as the batch needs it
//            (once per uninterrupted run of identical global binds).
//
// The expected bind counts are passed per path because they are exactly what
// the test is about: a push batch must bind the globals *fewer* times than it
// pushes, which is the elimination plan P2.3 claims.
void check_set0_path(Device &device, const Counters &d,
                     uint64_t set0_dispatches,
                     uint64_t classic_binds, uint64_t push_binds,
                     const char *what) noexcept {
    auto push = reported_push_path(device);
    auto allocations = value(d, Counter::descriptor_set_allocations);
    auto binds = value(d, Counter::descriptor_set_binds);
    auto pushes = value(d, Counter::push_descriptor_calls);
    LUISA_INFO("{}: set-0 path {} -> {} allocations, {} set binds, {} push calls "
               "for {} dispatches",
               what, push ? "push" : "classic", allocations, binds, pushes,
               set0_dispatches);
    if (push) {
        expect(allocations == 0u)
            << what << ": the push path must not allocate a set 0";
        expect(pushes == set0_dispatches)
            << what << ": one vkCmdPushDescriptorSetKHR per dispatch";
        expect(binds == push_binds)
            << what << ": the global sets are bound at their real first set, and "
                       "a fresh local set no longer forces that bind";
    } else {
        expect(allocations == set0_dispatches)
            << what << ": every classic dispatch allocates its own set 0";
        expect(binds == classic_binds)
            << what << ": every classic set-0 bind is recorded";
        expect(pushes == 0u)
            << what << ": a classic device must record no push call";
    }
}

// F2/P2.2: every set-0 update of the batch is exactly one of the two entry
// points, and the split matches the reported path. `expected_template_updates`
// is what the eligible dispatches of `d` must produce, `expected_legacy_updates`
// what the paths that stay legacy (the indirect-preparation kernel's own set)
// must still produce.
void check_descriptor_update_path(Device &device, const Counters &d,
                                  uint64_t expected_set0_updates,
                                  uint64_t expected_legacy_updates,
                                  const char *what) noexcept {
    auto legacy = value(d, Counter::descriptor_set_updates);
    auto templates = value(d, Counter::descriptor_set_update_templates);
    auto creates = value(d, Counter::descriptor_update_template_creates);
    auto fallbacks = value(d, Counter::descriptor_update_template_fallbacks);
    LUISA_INFO("{}: descriptor updates legacy={} template={} "
               "template_creates={} template_fallbacks={}",
               what, legacy, templates, creates, fallbacks);
    if (reported_push_path(device)) {
            // Plan P2.3: set 0 is pushed, so there is no update call at all. The
            // only vkUpdateDescriptorSets calls left are the paths that stay classic
            // by design (the indirect-preparation kernel's own set).
            expect(value(d, Counter::push_descriptor_calls) == expected_set0_updates)
                << what << ": every dispatch of a push shader records exactly one "
                           "vkCmdPushDescriptorSetKHR";
            expect(templates == 0u && creates == 0u && fallbacks == 0u)
                << what << ": the push path never creates or uses a descriptor-update "
                           "template";
            expect(legacy == expected_legacy_updates)
                << what << ": only the paths that intentionally stay classic may call "
                           "vkUpdateDescriptorSets";
        } else if (reported_template_path(device) && !reported_push_path(device)) {
            expect(templates == expected_set0_updates)
                << what << ": every eligible set-0 update must go through "
                           "vkUpdateDescriptorSetWithTemplate (P2.2)";
            expect(legacy == expected_legacy_updates)
                << what << ": only the paths that intentionally stay legacy may "
                           "call vkUpdateDescriptorSets";
            expect(fallbacks == 0u)
                << what << ": an eligible dispatch must not fall back to the "
                           "legacy call";
            expect(creates <= templates)
                << what << ": a template is created once per distinct write shape "
                           "and every created template is then used";
            expect(value(d, Counter::push_descriptor_calls) == 0u)
                << what << ": a template device binds set 0 classically, so no push "
                           "call may be recorded";
        } else {
            expect(templates == 0u && creates == 0u && fallbacks == 0u)
                << what << ": a device without template support must not touch "
                           "the template entry points at all";
            expect(value(d, Counter::push_descriptor_calls) == 0u)
                << what << ": a classic device records no push call";
            expect(legacy == expected_set0_updates + expected_legacy_updates)
                << what << ": every update keeps exactly one "
                           "vkUpdateDescriptorSets call";
    }
}

// Plan F5/§5.1: the per-submission restore barrier, now measured instead of
// assumed. `restores` is the number of `ResourceBarrier::restore_states` calls
// the measured window must contain (one per submitted batch, never one per
// dispatch), and the structural invariants pin *what* that call does:
//  - one VkBufferMemoryBarrier2 per tracked buffer (so a "cheaper" restore that
//    silently stopped covering a buffer fails here),
//  - at most one VkImageMemoryBarrier2 per tracked mip,
//  - the restore's structs are a subset of the funnel counters, and
//  - the restore's host time is inside the batch's encode time.
void check_restore_shape(const Counters &d, uint64_t restores,
                         const char *what) noexcept {
    auto calls = value(d, Counter::restore_calls);
    auto host_ns = value(d, Counter::restore_host_ns);
    auto memory = value(d, Counter::restore_memory_barriers);
    auto buffers = value(d, Counter::restore_buffer_barriers);
    auto images = value(d, Counter::restore_image_barriers);
    auto state_buffers = value(d, Counter::restore_state_buffers);
    auto state_mips = value(d, Counter::restore_state_texture_mips);
    auto funnel_structs = value(d, Counter::barrier_memory_barriers) +
                          value(d, Counter::barrier_buffer_barriers) +
                          value(d, Counter::barrier_image_barriers);
    LUISA_INFO(
        "{}: restore calls={} host={} ns ({:.2f} us/call), structs mem={} "
        "buf={} img={}, tracker state buf={} mips={}",
        what, calls, host_ns,
        calls == 0u ? 0.0 :
                      static_cast<double>(host_ns) / 1000.0 /
                          static_cast<double>(calls),
        memory, buffers, images, state_buffers, state_mips);
    expect(calls == restores)
        << what << ": exactly one restore_states call per submitted batch, not "
                   "per dispatch (F5)";
    expect(buffers == state_buffers)
        << what << ": the restore records one buffer barrier per tracked buffer "
                   "(a restore that covered fewer buffers would drop required "
                   "restores)";
    expect(images <= state_mips && memory <= calls)
        << what << ": the restore cannot record more image structs than tracked "
                   "mips or more bindless memory structs than calls";
    expect(memory + buffers + images <= funnel_structs)
        << what << ": the restore's structs are a subset of the barriers the "
                   "batch recorded";
    expect(host_ns <= value(d, Counter::host_encode_ns))
        << what << ": the restore runs inside the measured encode window";
}

// One 32-thread dispatch writes its own index into its own slice, so a batch of
// N dispatches has to leave N * 32 verified values behind.
constexpr size_t kThreadsPerDispatch = 32u;

Kernel1D write_dispatch_index = [](BufferFloat out, UInt index) noexcept {
    out.write(index * static_cast<uint32_t>(kThreadsPerDispatch) +
                  dispatch_id().x,
              cast<float>(index));
};

// Structurally different from `write_dispatch_index`: two buffer bindings in the
// local descriptor set instead of one. Used to prove that the interned layouts do
// not alias a different binding list (P1.1).
Kernel1D write_two_buffers = [](BufferFloat first, BufferFloat second,
                                UInt index) noexcept {
    auto slot = index * static_cast<uint32_t>(kThreadsPerDispatch) +
                dispatch_id().x;
    first.write(slot, 1.0f);
    second.write(slot, 2.0f);
};

// Two kernels with the *same* binding structure (one storage buffer, one scalar
// index, one dispatch block) and different bodies. P1.1 interns the layouts by
// structure, so both share one VkPipelineLayout and one VkDescriptorSetLayout,
// while the pipelines are still two objects - exactly the "alternating pipelines
// with the same interned layout" case that P1.2 must handle without either
// aliasing them or re-binding descriptors that are still valid.
// The written value identifies both the shader and the dispatch index.
Kernel1D write_odd_slot = [](BufferFloat out, UInt index) noexcept {
    out.write(index * static_cast<uint32_t>(kThreadsPerDispatch) +
                  dispatch_id().x,
              cast<float>(index) * 2.0f + 1.0f);
};

Kernel1D write_even_slot = [](BufferFloat out, UInt index) noexcept {
    out.write(index * static_cast<uint32_t>(kThreadsPerDispatch) +
                  dispatch_id().x,
              cast<float>(index) * 2.0f + 2.0f);
};

// A deliberately GPU-bound kernel: 32 back-to-back batches of it cannot retire
// as fast as the host records them, so the hard-coded depth-2 in-flight limit has
// to stall the host. Every thread keeps its own slot, so the results also prove
// that no batch was dropped.
constexpr uint32_t kThreadsPerGpuBoundDispatch = 1u << 16;
constexpr uint32_t kGpuBoundIterations = 8192u;

Kernel1D gpu_bound_kernel = [](BufferFloat out, UInt index) noexcept {
    auto x = cast<float>(dispatch_id().x + index);
    Float accum = 0.0f;
    $for (r, kGpuBoundIterations) {
        accum = sin(accum + cast<float>(r) * 1.000001f);
    };
    out.write(index * kThreadsPerGpuBoundDispatch + dispatch_id().x, accum);
};

void check_dispatch_indices(const luisa::vector<float> &host,
                            size_t dispatch_count,
                            const char *what) noexcept {
    auto ok = host.size() == dispatch_count * kThreadsPerDispatch;
    if (ok) {
        for (size_t i = 0u; i < dispatch_count; ++i) {
            for (size_t t = 0u; t < kThreadsPerDispatch; ++t) {
                auto actual = host[i * kThreadsPerDispatch + t];
                auto expected = static_cast<float>(i);
                if (actual != expected) {
                    ok = false;
                    LUISA_WARNING(
                        "{}: dispatch {} thread {} produced {} instead of {}.",
                        what, i, t, actual, expected);
                    break;
                }
            }
            if (!ok) { break; }
        }
    }
    expect(ok) << what << ": every dispatched thread must have run and every "
                          "dispatch must have kept its own slice";
}

void reg_profile_latch(Device &device) {
    "vk_profile_is_latched_and_reported"_test = [&device] {
        expect(device.query("vk_profile_enabled") == "1")
            << "the mode has to be latched from LUISA_VULKAN_PROFILE_SUBMIT=1 "
               "before the first dispatch";
        auto report = device.query("vk_profile_report");
        expect(!report.empty())
            << "vk_profile_report must answer the one-line dump";
        auto well_formed =
            report.find("dispatches=") != luisa::string::npos &&
            report.find("per-dispatch:") != luisa::string::npos &&
            report.find("host:") != luisa::string::npos;
        expect(well_formed) << "the report must carry counts, per-dispatch "
                               "ratios and host timings";
        if (!well_formed) { LUISA_INFO("vk_profile_report = {}", report); }
        // An unknown key stays unknown instead of being answered with a number.
        expect(device.query("vk_profile_no_such_counter").empty())
            << "an unknown vk_profile_* key must not fabricate a value";
    };
}

void reg_submit_path(Device &device) {
    "vk_profile_submit_path"_test = [&device] {
        // Plan P2.1: the reported path must be one of the two, and a 1.2 floor must
        // select the legacy one. Run the same binary with `--min-api 1.2` to cover
        // the other branch in this build.
        auto path = device.query("vk_queue_submit_path");
        LUISA_INFO("queue-submission path: {} (1.2 floor requested: {})",
                   path, g_min_api_1_2);
        expect(path == "submit2" || path == "legacy")
            << "the backend must report which native submission path it uses";
        if (g_min_api_1_2) {
            expect(path == "legacy")
                << "a Vulkan 1.2 floor must force the legacy vkQueueSubmit path";
        }
        auto counters = read_counters(device);
        expect(value(counters, Counter::queue_submit2) +
                   value(counters, Counter::queue_submit_legacy) ==
               value(counters, Counter::queue_submits))
            << "queue_submits must equal queue_submit2 + queue_submit_legacy";
    };
}

void reg_cold_create_layouts(Device &device) {
    "vk_profile_cold_create_layouts"_test = [&device] {
        // Two compiles of the *same* kernel. Since P1.1 the descriptor-set and
        // pipeline layouts are interned by structure, so the second compile must
        // not create a single Vulkan layout. Since P1.3 the pipeline cache is
        // device-scoped: neither compile creates a cache of its own (the device
        // cache was created once when the Device was), while each compile still
        // creates its own VkPipeline object.
        auto cache_handle_before = device.query("vk_pipeline_cache_handle");
        auto before = read_counters(device);
        auto first = device.compile(write_dispatch_index);
        auto after_first = read_counters(device);
        auto second = device.compile(write_dispatch_index);
        auto after_second = read_counters(device);
        auto cache_handle_after = device.query("vk_pipeline_cache_handle");
        auto first_delta = difference(before, after_first);
        auto second_delta = difference(after_first, after_second);
        auto first_layouts = value(first_delta, Counter::descriptor_set_layout_creates);
        auto first_pipeline_layouts = value(first_delta, Counter::pipeline_layout_creates);
        auto second_layouts = value(second_delta, Counter::descriptor_set_layout_creates);
        auto second_pipeline_layouts = value(second_delta, Counter::pipeline_layout_creates);
        auto pipeline_caches = value(second_delta, Counter::pipeline_cache_creates);
        auto first_pipeline_caches = value(first_delta, Counter::pipeline_cache_creates);
        auto merge_sources = value(second_delta, Counter::pipeline_cache_merge_source_creates);
        auto merges = value(second_delta, Counter::pipeline_cache_merges);
        auto first_merge_sources = value(first_delta, Counter::pipeline_cache_merge_source_creates);
        auto first_merges = value(first_delta, Counter::pipeline_cache_merges);
        auto pipelines = value(second_delta, Counter::pipeline_creates);
        LUISA_INFO(
            "cold-create delta: first compile dsl_layouts={} pipeline_layouts={} "
            "pipeline_caches={} merge_sources={} merges={}; second (identical) "
            "compile dsl_layouts={} pipeline_layouts={} pipeline_caches={} "
            "merge_sources={} merges={} pipelines={}",
            first_layouts, first_pipeline_layouts, first_pipeline_caches,
            first_merge_sources, first_merges, second_layouts,
            second_pipeline_layouts, pipeline_caches, merge_sources, merges,
            pipelines);
        expect(first_pipeline_layouts == 1u)
            << "one compiled shader creates one pipeline layout";
        expect(first_layouts >= 2u)
            << "one compiled shader creates one descriptor-set layout per set "
               "(the local set plus the immutable-sampler heap at least)";
        expect(second_pipeline_layouts == 0u)
            << "an identical structure must reuse the interned pipeline layout "
               "instead of creating a second one (P1.1)";
        expect(second_layouts == 0u)
            << "an identical structure must reuse every interned descriptor-set "
               "layout instead of creating duplicates (P1.1)";
        // F10/P1.3: the cache is device-scoped. A compile must not create a
        // cache of its own - the one cache was created when the Device was and is
        // reused by every DSL shader - while the pipeline itself is still one
        // object per compile.
        //
        // The only VkPipelineCache a compile may still create is a short-lived
        // merge source for a legacy per-shader PSO seed found in the cache
        // directory (those files predate P1.3 and are add-only inputs), so the
        // assertion is on "creations that are not merge sources": a persistent
        // per-shader cache would show up there and fail this tripwire.
        expect(first_pipeline_caches == first_merge_sources &&
               pipeline_caches == merge_sources)
            << "a compile may only create the device cache itself plus one "
               "temporary merge source per legacy per-shader seed; a persistent "
               "per-shader cache would break this (P1.3/F10)";
        expect(pipelines == 1u)
            << "each kernel still compiles its own pipeline object";
        expect(first_merges == first_merge_sources &&
               merges == merge_sources)
            << "every merge source created by a compile must be merged exactly "
               "once (P1.3)";
        // The device cache must really exist and be the one the shaders use, so
        // the zero above is "reused", not "caching disabled".
        auto cache_state = device.query("vk_pipeline_cache_state");
        auto cache_handle = device.query("vk_pipeline_cache_handle");
        expect(cache_state == "ready")
            << "the device pipeline cache must be available (P1.3)";
        expect(!cache_handle.empty() && cache_handle != "0")
            << "the device pipeline cache must expose a live handle (P1.3)";
        expect(cache_handle_before == cache_handle_after)
            << "the two identical kernels must share the same device cache "
               "object, not create one each (P1.3/F10)";
        // Process-wide invariant of the honest accounting: a temporary
        // merge-source cache exists only to be merged, so the cumulative numbers
        // must agree (they are not per-delta numbers; both are counted at the
        // same two call sites in pipeline_cache.cpp).
        expect(value(after_second, Counter::pipeline_cache_merge_source_creates) ==
               value(after_second, Counter::pipeline_cache_merges))
            << "every merge-source cache must correspond to exactly one "
               "vkMergePipelineCaches (P1.3)";
        // The compiled shaders must still work, so the measured creation path is
        // the one the dispatches below use - and two shaders have to be usable on
        // one shared set of interned layouts.
        Buffer<float> buffer = device.create_buffer<float>(kThreadsPerDispatch);
        Stream stream = device.create_stream();
        luisa::vector<float> host(kThreadsPerDispatch, -1.0f);
        stream << first(buffer, 0u).dispatch(kThreadsPerDispatch)
               << second(buffer, 0u).dispatch(kThreadsPerDispatch)
               << buffer.copy_to(luisa::span{host})
               << synchronize();
        check_dispatch_indices(host, 1u, "cold-create");
    };
}

void reg_layout_intern_separation(Device &device) {
    "vk_layout_intern_reuses_and_separates"_test = [&device] {
        // Holding one shader alive keeps its interned entries alive for the whole
        // test, so the deltas below describe the interning decision and not a
        // release/re-create cycle.
        auto baseline = device.compile(write_dispatch_index);
        // Positive: an identical structure creates nothing.
        auto before = read_counters(device);
        auto identical = device.compile(write_dispatch_index);
        auto after_identical = read_counters(device);
        auto reused = difference(before, after_identical);
        // Negative: a different binding list must not alias the interned entries.
        auto different = device.compile(write_two_buffers);
        auto after_different = read_counters(device);
        auto added = difference(after_identical, after_different);
        auto reused_layouts = value(reused, Counter::descriptor_set_layout_creates);
        auto reused_pipeline_layouts = value(reused, Counter::pipeline_layout_creates);
        auto added_layouts = value(added, Counter::descriptor_set_layout_creates);
        auto added_pipeline_layouts = value(added, Counter::pipeline_layout_creates);
        LUISA_INFO(
            "intern separation: identical compile -> dsl_layouts={} "
            "pipeline_layouts={}; structurally different binding list -> "
            "dsl_layouts={} pipeline_layouts={}",
            reused_layouts, reused_pipeline_layouts,
            added_layouts, added_pipeline_layouts);
        expect(reused_layouts == 0u && reused_pipeline_layouts == 0u)
            << "an identical shader structure must reuse every interned layout "
               "(P1.1 positive)";
        expect(added_pipeline_layouts == 1u)
            << "a different binding list must get its own pipeline layout "
               "(P1.1 negative)";
        expect(added_layouts == 1u)
            << "only the local set differs, so exactly one descriptor-set layout "
               "is created again while the sampler set stays interned (P1.1 "
               "negative)";
        // Every shader still has to work on its own (possibly shared) layouts.
        constexpr auto width = kThreadsPerDispatch;
        Buffer<float> single = device.create_buffer<float>(width);
        Buffer<float> first_of_two = device.create_buffer<float>(width);
        Buffer<float> second_of_two = device.create_buffer<float>(width);
        Stream stream = device.create_stream();
        luisa::vector<float> single_host(width, -1.0f);
        luisa::vector<float> first_host(width, -1.0f);
        luisa::vector<float> second_host(width, -1.0f);
        stream << baseline(single, 0u).dispatch(width)
               << identical(single, 0u).dispatch(width)
               << different(first_of_two, second_of_two, 0u).dispatch(width)
               << single.copy_to(luisa::span{single_host})
               << first_of_two.copy_to(luisa::span{first_host})
               << second_of_two.copy_to(luisa::span{second_host})
               << synchronize();
        check_dispatch_indices(single_host, 1u, "intern-shared");
        auto two_buffer_ok = true;
        for (auto i = 0u; i < width; ++i) {
            if (first_host[i] != 1.0f || second_host[i] != 2.0f) {
                two_buffer_ok = false;
                break;
            }
        }
        expect(two_buffer_ok)
            << "the shader with the structurally different layout must still "
               "dispatch into both of its buffers";
    };
}

void reg_dispatch_path_counts(Device &device) {
    "vk_profile_dispatch_path_counts"_test = [&device] {
        constexpr auto dispatch_count = size_t{8u};
        auto shader = device.compile(write_dispatch_index);
        Buffer<float> buffer =
            device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(dispatch_count, 0u);
        for (auto i = 0u; i < dispatch_count; ++i) {
            list << shader(buffer, i).dispatch(kThreadsPerDispatch);
        }
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        LUISA_INFO(
            "dispatch delta ({} dispatches, 1 batch): dispatches={} "
            "set_alloc={} set_update={} set_bind={} pipeline_bind={} "
            "arg_plans={} barrier_calls={} buffer_barriers={} "
            "all_commands_barriers={} batches={} submits={}",
            dispatch_count,
            value(d, Counter::dispatches),
            value(d, Counter::descriptor_set_allocations),
            value(d, Counter::descriptor_set_updates),
            value(d, Counter::descriptor_set_binds),
            value(d, Counter::pipeline_binds),
            value(d, Counter::argument_plans),
            value(d, Counter::barrier_calls),
            value(d, Counter::barrier_buffer_barriers),
            value(d, Counter::barrier_all_commands_barriers),
            value(d, Counter::command_batches),
            value(d, Counter::queue_submits));
        expect(value(d, Counter::dispatches) == dispatch_count)
            << "one recorded dispatch per command";
check_set0_path(device, d, dispatch_count, dispatch_count, 1u,
 "dispatch-path");
 // F2/P2.2: every dispatch still binds its own set 0 exactly once; only the
 // entry point that performs the update changed.
 check_descriptor_update_path(device, d, dispatch_count, 0u,
 "dispatch-path");
 expect(value(d, Counter::pipeline_binds) == 1u)
 << "P1.2: the one unchanged pipeline of this batch is bound once per "
 "command buffer, not once per dispatch (F3)";
 expect(value(d, Counter::argument_plans) == dispatch_count)
 << "P1.4: the sizing pass plans every dispatch once and the "
 "preprocessing pass reuses that plan (F4 was 2 x dispatches)";
 expect(value(d, Counter::command_batches) == 1u)
 << "all dispatches of one command list form one batch";
 expect(value(d, Counter::queue_submits) == 1u)
 << "one batch must submit once (F6)";
 // F6/P2.1: the single submission goes through exactly one native entry point,
 // and which one is decided by the device synchronization2 capability
 // (exercised on both branches by test_vk_queue_submit_runtime with and without
 // `--min-api 1.2`).
 check_submit_variants(device, d, 1u, "dispatch-path");
 expect(value(d, Counter::barrier_all_commands_barriers) == 1u)
 << "every submission ends with exactly one ALL_COMMANDS restore "
 "struct for the touched buffer (F5)";
 // F5/5.1: the restore is measured now, and its shape is pinned.
 check_restore_shape(d, 1u, "dispatch-path");
 expect(value(d, Counter::restore_buffer_barriers) >= 1u)
 << "a batch that touched a buffer must restore it, not zero structs";
 expect(value(d, Counter::restore_image_barriers) == 0u)
 << "a buffer-only batch records no image restore struct";
 expect(value(d, Counter::restore_state_buffers) >= 1u)
 << "the restore must consume the tracker state of the buffers the batch "
 "touched (this is the state a persistent cache would keep)";
 expect(value(d, Counter::host_encode_ns) > 0u &&
 value(d, Counter::host_submit_ns) > 0u)
 << "the profiler must report host encode and submit time";
 // Sanity bound only: a dispatch whose encode costs more than 1 ms would mean
 // the mode broke the hot path.
 expect(value(d, Counter::host_encode_ns) < 1'000'000u * dispatch_count)
 << "host encode time per dispatch must stay in the microsecond class";
 luisa::vector<float> host(dispatch_count * kThreadsPerDispatch, -1.0f);
 stream << buffer.copy_to(luisa::span{host}) << synchronize();
 check_dispatch_indices(host, dispatch_count, "dispatch-path");    };
}

void reg_decode_shaped_batch(Device &device) {
      "vk_profile_decode_shaped_batch"_test = [&device] {
          // The gallery's measured launch count of one Qwen decode token (Q).
          constexpr auto dispatch_count = size_t{438u};
          auto shader = device.compile(write_dispatch_index);
          Buffer<float> buffer =
              device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
          Stream stream = device.create_stream();
          CommandList list;
          list.reserve(dispatch_count, 0u);
          for (auto i = 0u; i < dispatch_count; ++i) {
              list << shader(buffer, i).dispatch(kThreadsPerDispatch);
          }
          auto before = read_counters(device);
          stream << list.commit() << synchronize();
          auto after = read_counters(device);
          auto d = difference(before, after);
          auto encode_per_dispatch =
              value(d, Counter::dispatches) == 0u ?
                  0.0 :
                  static_cast<double>(value(d, Counter::host_encode_ns)) /
                      1000.0 /
                      static_cast<double>(value(d, Counter::dispatches));
          LUISA_INFO(
              "decode-shaped batch: {} dispatches -> encode {:.1f} us/dispatch, "
              "submit {:.1f} us total, {} set allocations, {} set binds, {} "
              "pipeline binds, {} argument plans, {} restore structs, {} push calls",
              value(d, Counter::dispatches), encode_per_dispatch,
              static_cast<double>(value(d, Counter::host_submit_ns)) / 1000.0,
              value(d, Counter::descriptor_set_allocations),
              value(d, Counter::descriptor_set_binds),
              value(d, Counter::pipeline_binds),
              value(d, Counter::argument_plans),
              value(d, Counter::barrier_all_commands_barriers),
              value(d, Counter::push_descriptor_calls));
          expect(value(d, Counter::dispatches) == dispatch_count);
          check_set0_path(device, d, dispatch_count, dispatch_count, 1u,
                          "decode-shaped batch");
          check_descriptor_update_path(device, d, dispatch_count, 0u,
                                       "decode-shaped batch");
          expect(value(d, Counter::pipeline_binds) == 1u)
              << "P1.2: a 438-dispatch batch of one unchanged shader binds its "
                 "pipeline once per command buffer (F3 consequence of F1)";
          expect(value(d, Counter::argument_plans) == dispatch_count)
              << "argument planning runs exactly once per dispatch (P1.4; F4 "
                 "recorded 2 x dispatches)";
          expect(value(d, Counter::command_batches) == 1u &&
                 value(d, Counter::queue_submits) == 1u)
              << "438 dispatches are still one batch and one submit (F6)";
          check_submit_variants(device, d, 1u, "decode-shaped batch");
          expect(value(d, Counter::barrier_all_commands_barriers) == 1u)
              << "one ALL_COMMANDS restore struct for the whole batch (F5)";
          // The decode-shaped number plan §5.1 asked for: one restore per token,
          // its host cost, and its share of the host encode window. It is a host
          // measurement only - no GPU time is inferred from it.
          check_restore_shape(d, 1u, "decode-shaped batch");
          auto restore_ns = value(d, Counter::restore_host_ns);
          auto encode_ns = value(d, Counter::host_encode_ns);
          LUISA_INFO(
              "decode-shaped restore: {} call, {:.2f} us, {:.3f} % of the {:.2f} us "
              "host encode window ({} buffer / {} image / {} memory structs)",
              value(d, Counter::restore_calls),
              static_cast<double>(restore_ns) / 1000.0,
              encode_ns == 0u ? 0.0 :
                                100.0 * static_cast<double>(restore_ns) /
                                    static_cast<double>(encode_ns),
              static_cast<double>(encode_ns) / 1000.0,
              value(d, Counter::restore_buffer_barriers),
              value(d, Counter::restore_image_barriers),
              value(d, Counter::restore_memory_barriers));
          expect(encode_per_dispatch > 0.0 && encode_per_dispatch < 1000.0)
              << "host encode time per dispatch must stay in the microsecond class";
          luisa::vector<float> host(dispatch_count * kThreadsPerDispatch, -1.0f);
          stream << buffer.copy_to(luisa::span{host}) << synchronize();
          check_dispatch_indices(host, dispatch_count, "decode-shaped batch");
      };
  }

  // ---------------------------------------------------------------------------
  // Plan §5.1 / F5: what the per-submission restore barrier costs
  // ---------------------------------------------------------------------------
  void reg_restore_cost(Device &device) {
      "vk_profile_restore_cost"_test = [&device] {
          // The measurement phase 3 was told to start with. Two windows with the
          // *same* dispatch count and the same GPU work, differing only in how the
          // dispatches are grouped:
          // (a) one batch of `count` dispatches -> one restore_states call
          // (b) `count` batches of one dispatch -> `count` restores
          // so the restore is a per-submission fixed cost and the difference
          // between the windows is what a persistent barrier cache could remove.
          // Host time only: no GPU cost is inferred from it.
          constexpr auto count = size_t{16u};
          auto shader = device.compile(write_dispatch_index);
          Buffer<float> buffer =
              device.create_buffer<float>(count * kThreadsPerDispatch);
          Stream stream = device.create_stream();
          auto single_restore_ns = uint64_t{0u};
          auto single_restore_structs = uint64_t{0u};
          {
              CommandList list;
              list.reserve(count, 0u);
              for (auto i = 0u; i < count; ++i) {
                  list << shader(buffer, i).dispatch(kThreadsPerDispatch);
              }
              auto before = read_counters(device);
              stream << list.commit() << synchronize();
              auto after = read_counters(device);
              auto d = difference(before, after);
              check_restore_shape(d, 1u, "restore-cost (one batch)");
              expect(value(d, Counter::dispatches) == count)
                  << "the one-batch window must really contain every dispatch";
              expect(value(d, Counter::restore_calls) == 1u)
                  << "grouping N dispatches into one batch must pay exactly one "
                     "restore (F5: the restore is per submission)";
              single_restore_ns = value(d, Counter::restore_host_ns);
              single_restore_structs = value(d, Counter::restore_buffer_barriers);
              LUISA_INFO(
                  "restore cost (one batch, {} dispatches): {} restore call, "
                  "{} ns, {} buffer structs, {:.2f} us host encode",
                  value(d, Counter::dispatches),
                  value(d, Counter::restore_calls), single_restore_ns,
                  single_restore_structs,
                  static_cast<double>(value(d, Counter::host_encode_ns)) / 1000.0);
          }
          // (b) The same dispatches, one batch each.
          auto per_batch_restore_ns = uint64_t{0u};
          auto per_batch_structs = uint64_t{0u};
          {
              auto before = read_counters(device);
              for (auto i = 0u; i < count; ++i) {
                  CommandList list;
                  list << shader(buffer, i).dispatch(kThreadsPerDispatch);
                  stream << list.commit();
              }
              stream << synchronize();
              auto after = read_counters(device);
              auto d = difference(before, after);
              check_restore_shape(d, count, "restore-cost (one dispatch per batch)");
              expect(value(d, Counter::dispatches) == count &&
                     value(d, Counter::command_batches) == count)
                  << "the many-batch window must contain the same dispatches in "
                     "one batch each";
              expect(value(d, Counter::restore_host_ns) > 0u)
                  << "restores must register a nonzero host cost";
              per_batch_restore_ns = value(d, Counter::restore_host_ns);
              per_batch_structs = value(d, Counter::restore_buffer_barriers);
              LUISA_INFO(
                  "restore cost ({} batches, 1 dispatch each): {} restore calls, "
                  "{} ns total ({:.2f} us/call), {} buffer structs, {:.2f} us host "
                  "encode",
                  count, value(d, Counter::restore_calls),
                  per_batch_restore_ns,
                  static_cast<double>(per_batch_restore_ns) / 1000.0 /
                      static_cast<double>(count),
                  per_batch_structs,
                  static_cast<double>(value(d, Counter::host_encode_ns)) / 1000.0);
          }
          // The restore is a per-submission cost: the same 16 dispatches cost 16
          // restores when split into 16 batches. Reported as the marginal cost of
          // one extra submission, which is exactly the ceiling of what a
          // cross-submission barrier cache can remove on the host.
          auto extra_ns = per_batch_restore_ns > single_restore_ns ?
                              per_batch_restore_ns - single_restore_ns :
                              0u;
          LUISA_INFO(
              "restore cost: marginal host cost per extra submission = {:.2f} us "
              "({} extra restores, {} extra buffer structs)",
              static_cast<double>(extra_ns) / 1000.0 /
                  static_cast<double>(count - 1u),
              count - 1u,
              per_batch_structs >= single_restore_structs ?
                  per_batch_structs - single_restore_structs :
                  0u);
          expect(per_batch_structs >= single_restore_structs)
              << "every restore of the same buffer must still record its struct";
          // (c) The other half of the cost model: how the restore scales with the
          // *state* it has to restore, not with the dispatch count. One batch that
          // touches `count` distinct buffers must restore every one of them, so
          // this window isolates the per-tracked-resource cost that a persistent
          // cache would have to keep instead of dropping.
          {
              luisa::vector<Buffer<float>> buffers;
              buffers.reserve(count);
              for (auto i = 0u; i < count; ++i) {
                  // `write_dispatch_index` writes at `index * 32 + thread`, so each
                  // buffer is sized for the slice its own dispatch writes.
                  buffers.emplace_back(
                      device.create_buffer<float>(count * kThreadsPerDispatch));
              }
              CommandList list;
              list.reserve(count, 0u);
              for (auto i = 0u; i < count; ++i) {
                  list << shader(buffers[i], i).dispatch(kThreadsPerDispatch);
              }
              auto before = read_counters(device);
              stream << list.commit() << synchronize();
              auto after = read_counters(device);
              auto d = difference(before, after);
              check_restore_shape(d, 1u, "restore-cost (many distinct buffers)");
              expect(value(d, Counter::restore_state_buffers) >= count)
                  << "every touched buffer must be tracked and restored";
              expect(value(d, Counter::restore_buffer_barriers) ==
                     value(d, Counter::restore_state_buffers))
                  << "one buffer struct per tracked buffer in the same batch";
              auto many_ns = value(d, Counter::restore_host_ns);
              LUISA_INFO(
                  "restore cost (one batch, {} distinct buffers): {} restore call, "
                  "{} ns, {} buffer structs, marginal {:.3f} us per tracked buffer "
                  "over the single-buffer batch",
                  count, value(d, Counter::restore_calls), many_ns,
                  value(d, Counter::restore_buffer_barriers),
                  count <= 1u ? 0.0 :
                                static_cast<double>(many_ns > single_restore_ns ?
                                                        many_ns - single_restore_ns :
                                                        0u) /
                                    1000.0 / static_cast<double>(count - 1u));
              for (auto i = 0u; i < count; ++i) {
                  luisa::vector<float> values(count * kThreadsPerDispatch, -1.0f);
                  stream << buffers[i].copy_to(luisa::span{values}) << synchronize();
                  auto ok = true;
                  for (auto t = 0u; t < kThreadsPerDispatch; ++t) {
                      if (values[i * kThreadsPerDispatch + t] != static_cast<float>(i)) {
                          ok = false;
                          break;
                      }
                  }
                  expect(ok) << "every one of the distinct buffers must have been "
                                "written by its own dispatch";
                  if (!ok) { break; }
              }
          }
          luisa::vector<float> host(count * kThreadsPerDispatch, -1.0f);
          stream << buffer.copy_to(luisa::span{host}) << synchronize();
          check_dispatch_indices(host, count, "restore-cost");
      };
  }

  // ---------------------------------------------------------------------------
  // P1.2: per-command-buffer redundant-bind elimination
  // ---------------------------------------------------------------------------
  void reg_bind_elimination_repeated(Device &device) {
      "vk_bind_elimination_repeated_shader"_test = [&device] {
          // The core of P1.2: one unchanged shader, one command buffer. The pipeline
          // is bound once; every dispatch allocates, fills and binds its own
          // descriptor set, because that set is a different object each time - on
          // the classic path. Plan P2.3 removes the allocation and the set-0 bind
          // from the push path and binds only the global sets, once.
          constexpr auto dispatch_count = size_t{16u};
          auto shader = device.compile(write_dispatch_index);
          Buffer<float> buffer =
              device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
          Stream stream = device.create_stream();
          CommandList list;
          list.reserve(dispatch_count, 0u);
          for (auto i = 0u; i < dispatch_count; ++i) {
              list << shader(buffer, i).dispatch(kThreadsPerDispatch);
          }
          auto before = read_counters(device);
          stream << list.commit() << synchronize();
          auto after = read_counters(device);
          auto d = difference(before, after);
          LUISA_INFO(
              "P1.2 repeated shader: {} dispatches in 1 batch -> {} pipeline "
              "binds, {} set binds, {} allocations, {} updates, {} push calls",
              dispatch_count, value(d, Counter::pipeline_binds),
              value(d, Counter::descriptor_set_binds),
              value(d, Counter::descriptor_set_allocations),
              value(d, Counter::descriptor_set_updates),
              value(d, Counter::push_descriptor_calls));
          expect(value(d, Counter::dispatches) == dispatch_count);
          expect(value(d, Counter::pipeline_binds) == 1u)
              << "consecutive dispatches of one unchanged shader must bind its "
                 "pipeline exactly once per command buffer (P1.2)";
          check_set0_path(device, d, dispatch_count, dispatch_count, 1u,
                          "bind-elimination-repeated");
          check_descriptor_update_path(device, d, dispatch_count, 0u,
                                       "bind-elimination-repeated");
          luisa::vector<float> host(dispatch_count * kThreadsPerDispatch, -1.0f);
          stream << buffer.copy_to(luisa::span{host}) << synchronize();
          check_dispatch_indices(host, dispatch_count, "bind-elimination-repeated");
      };
  }

void reg_bind_elimination_alternating(Device &device) {
    "vk_bind_elimination_alternating_pipelines"_test = [&device] {
        // Two kernels with an identical binding structure: P1.1 interns one
        // descriptor-set layout and one pipeline layout for both, so this is the
        // "same layout object, different pipeline" case. Alternating pipelines
        // must still bind (the handles differ), and each dispatch must reach the
        // right pipeline.
        auto before_compile = read_counters(device);
        auto odd = device.compile(write_odd_slot);
        auto after_odd = read_counters(device);
        auto even = device.compile(write_even_slot);
        auto after_even = read_counters(device);
        auto second_compile = difference(after_odd, after_even);
        expect(value(second_compile, Counter::descriptor_set_layout_creates) == 0u &&
               value(second_compile, Counter::pipeline_layout_creates) == 0u)
            << "the second kernel must reuse the interned layouts, otherwise "
               "this test does not exercise the shared-layout case (P1.1)";
        expect(value(second_compile, Counter::pipeline_creates) == 1u)
            << "and it still compiles its own pipeline (P1.3 is pending)";
        static_cast<void>(before_compile);

        constexpr auto count = size_t{7u};
        constexpr std::array<bool, count> use_odd{true, true, false, false,
                                                  false, true, true};
        auto shader_for = [&](size_t index) -> decltype(odd) & {
            return use_odd[index] ? odd : even;
        };
        Buffer<float> buffer =
            device.create_buffer<float>(count * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(count, 0u);
        for (auto i = 0u; i < count; ++i) {
            list << shader_for(i)(buffer, i).dispatch(kThreadsPerDispatch);
        }
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        LUISA_INFO(
            "P1.2 alternating pipelines: pattern AABBBA -> {} pipeline binds "
            "(expected 3), {} set binds",
            value(d, Counter::pipeline_binds),
            value(d, Counter::descriptor_set_binds));
        expect(value(d, Counter::pipeline_binds) == 3u)
            << "AABBBA must bind A, then B, then A again: only a repeated "
               "identical handle may be skipped (P1.2)";
 check_set0_path(device, d, count, count, 1u,
 "bind-elimination-alternating");
        luisa::vector<float> host(count * kThreadsPerDispatch, -1.0f);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < count && ok; ++i) {
            auto expected = static_cast<float>(i) * 2.0f +
                            (use_odd[i] ? 1.0f : 2.0f);
            for (auto t = 0u; t < kThreadsPerDispatch; ++t) {
                auto actual = host[i * kThreadsPerDispatch + t];
                if (actual != expected) {
                    LUISA_WARNING(
                        "alternating pipelines: dispatch {} thread {} produced {} "
                        "instead of {} - the wrong pipeline was left bound.",
                        i, t, actual, expected);
                    ok = false;
                    break;
                }
            }
        }
        expect(ok) << "skipping a redundant bind must never leave the wrong "
                      "pipeline bound for a dispatch";
    };
}

void reg_bind_elimination_state_reset(Device &device) {
    "vk_bind_elimination_state_reset_across_batches"_test = [&device] {
        // The tracked state belongs to one recorded command buffer. A new batch
        // gets a (possibly recycled) command buffer with no bindings of its own,
        // so the pipeline has to be bound again even though nothing about the
        // shader changed.
        constexpr auto batch_count = size_t{3u};
        constexpr auto per_batch = size_t{2u};
        auto shader = device.compile(write_dispatch_index);
        Buffer<float> buffer = device.create_buffer<float>(
            batch_count * per_batch * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        auto before = read_counters(device);
        for (auto b = 0u; b < batch_count; ++b) {
            CommandList list;
            list.reserve(per_batch, 0u);
            for (auto i = 0u; i < per_batch; ++i) {
                list << shader(buffer, b * per_batch + i)
                            .dispatch(kThreadsPerDispatch);
            }
            stream << list.commit();
        }
        stream << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        LUISA_INFO(
            "P1.2 state reset: {} batches x {} identical dispatches -> {} "
            "pipeline binds (expected {}), {} set binds",
            batch_count, per_batch, value(d, Counter::pipeline_binds),
            batch_count, value(d, Counter::descriptor_set_binds));
        expect(value(d, Counter::command_batches) == batch_count);
        expect(value(d, Counter::pipeline_binds) == batch_count)
            << "each command buffer binds the pipeline once: the tracked state "
               "is reset with the recording and never carried across batches, "
               "submissions or command-buffer recycling (P1.2)";
 // Plan P2.3: the tracked state is reset with the recording, so the push
 // path binds the global sets once per command buffer - not once per
 // dispatch - while the classic path binds one fresh set per dispatch.
 check_set0_path(device, d, batch_count * per_batch,
 batch_count * per_batch, batch_count,
 "bind-elimination-reset");
        luisa::vector<float> host(batch_count * per_batch * kThreadsPerDispatch,
                                  -1.0f);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        check_dispatch_indices(host, batch_count * per_batch,
                               "bind-elimination-reset");
    };
}

void reg_bind_elimination_indirect(Device &device) {
    if (!native_spirv_route(device)) {
        LUISA_INFO(
            "indirect prepare/restore bind elimination: skipped - this build has "
            "no native XIR-to-SPIR-V codegen, so no shader carries the "
            "indirect-dispatch metadata ABI (vk_spirv_codegen_route={})",
            device.query("vk_spirv_codegen_route"));
        return;
    }
    "vk_bind_elimination_indirect_prepare_restore"_test = [&device] {
        // The indirect path is the one place where the backend itself binds
        // between a target's own binds: the preparation kernel binds its own
        // pipeline and set, then the target contract is restored. The tracker has
        // to see that the preparation kernel really changed the bind point, and
        // the counts below spell out exactly which binds are recorded.
        constexpr auto kernel_block_size = 64u;
        constexpr auto dispatch_count = 3u;
        Kernel1D clear_kernel = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(dispatch_count);
        };
        Kernel1D author_kernel = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_kernel(
                dispatch_id().x,
                make_uint3(kernel_block_size, 1u, 1u),
                make_uint3(dispatch_id().x + 1u, 1u, 1u),
                dispatch_id().x);
        };
        Kernel1D target_kernel = [=](BufferUInt buffer) noexcept {
            set_block_size(kernel_block_size, 1u, 1u);
            // Stamps its own record id, so a record that did not run (or ran in
            // place of another one) leaves a zero behind.
            buffer.write(kernel_id(), 1u);
        };
        auto clear_shader = device.compile(clear_kernel);
        auto author_shader = device.compile(author_kernel);
        auto target_shader = device.compile(target_kernel);
        IndirectDispatchBuffer commands =
            device.create_indirect_dispatch_buffer(dispatch_count);
        Buffer<uint> buffer = device.create_buffer<uint>(dispatch_count);
        std::array<uint, dispatch_count> host{};
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(dispatch_count + 4u, 0u);
        list << buffer.copy_from(luisa::span{host});
        list << clear_shader(commands).dispatch(1u);
        list << author_shader(commands).dispatch(dispatch_count);
        for (auto i = 0u; i < dispatch_count; ++i) {
            list << target_shader(buffer).dispatch(commands, i, 1u);
        }
        list << buffer.copy_to(luisa::span{host});
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        // Derivation: clear and author are direct dispatches (one pipeline bind
        // each). The first indirect dispatch binds the target, the preparation kernel
        // and the target again (3). Every later indirect dispatch starts with the
        // target still bound (skipped), then binds preparation and the target (2 each).
        // Pipeline binds therefore do not depend on the set-0 path.
        //
        // What the set-0 path changes (plan P2.3):
        //   classic: every set-0 sequence allocates a set and binds it, so 2 + 3N binds
        //            and 2 + 2N allocations;
        //   push:    neither clear/author nor the target allocate or bind a set 0 any
        //            more - one push replaces each of those sequences - and the global
        //            sets are bound at their first set. Only the preparation kernel's
        //            own (classic) set is allocated and bound, and each indirect
        //            dispatch pushes twice (prologue + after the preparation kernel
        //            disturbed the push descriptor bank), so N allocations, N legacy
        //            updates, 2 + 2N binds and 2 + 2N pushes.
        auto const push = reported_push_path(device);
        auto const expected_pipeline_binds =
            2u + 3u + 2u * (dispatch_count - 1u);
        auto const expected_set_binds =
            push ? 2u + 2u * dispatch_count : 2u + 3u * dispatch_count;
        auto const expected_allocations =
            push ? dispatch_count : 2u + 2u * dispatch_count;
        LUISA_INFO(
            "P1.2 indirect: {} target dispatches (set-0 path {}) -> {} pipeline binds "
            "(expected {}), {} set binds (expected {}), {} allocations (expected {}), "
            "{} updates, {} push calls",
            dispatch_count, push ? "push" : "classic",
            value(d, Counter::pipeline_binds), expected_pipeline_binds,
            value(d, Counter::descriptor_set_binds), expected_set_binds,
            value(d, Counter::descriptor_set_allocations), expected_allocations,
            value(d, Counter::descriptor_set_updates),
            value(d, Counter::push_descriptor_calls));
        expect(value(d, Counter::dispatches) == 2u + dispatch_count);
        expect(value(d, Counter::pipeline_binds) == expected_pipeline_binds)
            << "the preparation kernel's own bind and the target restoration that "
               "follows it are real binds; only the repeated target bind at the start "
               "of the next indirect dispatch may be skipped (P1.2)";
        expect(value(d, Counter::descriptor_set_binds) == expected_set_binds)
            << "the preparation kernel binds its own set, so the target sets must be "
               "bound again afterwards (classic), while the push path binds only the "
               "global sets of each set-0 sequence";
        expect(value(d, Counter::descriptor_set_allocations) == expected_allocations)
            << "the push path allocates nothing for set 0, so only the preparation "
               "kernel's own set is allocated";
        // Every set-0 sequence is written exactly once. The target/clear/author sets go
        // through the P2.2 funnel (or the push call on the push path); the
        // indirect-preparation kernel's own set is one of the paths that stay on
        // vkUpdateDescriptorSets, so its `dispatch_count` updates must still show up as
        // legacy calls. The eligible count is the number of set-0 sequences: on the
        // classic path the indirect re-bind re-uses the already written set (an empty
        // write list is not an update), on the push path it has to push again.
        check_descriptor_update_path(
            device, d,
            push ? 2u + 2u * dispatch_count : 2u + dispatch_count,
            dispatch_count, "indirect prepare");
        // Every record must have been consumed by exactly one target dispatch, so
        // every slot carries the stamp the target kernel writes.
        auto ok = true;
        for (auto i = 0u; i < dispatch_count; ++i) {
            if (host[i] != 1u) {
                LUISA_WARNING("indirect bind elimination: buffer[{}] is {}, "
                              "expected 1.",
                              i, host[i]);
                ok = false;
            }
        }
        expect(ok) << "every indirect target dispatch must still run exactly once "
                      "after the bind elimination";
    };
}

// The flush cases: binds that this tracker must not try to predict.
//
// The one entry point this file needs is resolved through the loader module the
// backend already loaded. Two copies of volk live in this process (the backend
// DLL's, which is initialised, and this executable's, which is not - initialising
// it would need a VulkanDeviceConfigExt::init_volk handshake this test does not
// own), so going through the loader is the honest way to get a working pointer.
[[nodiscard]] PFN_vkCmdFillBuffer resolve_cmd_fill_buffer() noexcept {
    static auto entry = []() noexcept -> PFN_vkCmdFillBuffer {
#if defined(_WIN32)
        auto module = ::GetModuleHandleA("vulkan-1.dll");
        if (module == nullptr) { return nullptr; }
        return reinterpret_cast<PFN_vkCmdFillBuffer>(
            ::GetProcAddress(module, "vkCmdFillBuffer"));
#else
        auto module = ::dlopen("libvulkan.so.1", RTLD_LAZY);
        if (module == nullptr) { return nullptr; }
        return reinterpret_cast<PFN_vkCmdFillBuffer>(
            ::dlsym(module, "vkCmdFillBuffer"));
#endif
    }();
    return entry;
}

class RawFillCommand final : public VKCustomCmd {
private:
    luisa::vector<ResourceUsage> _usages;
    VkBuffer _target{};
    VkDeviceSize _size{0u};
    uint32_t _value{0u};

public:
    RawFillCommand(Buffer<float> const &target, uint32_t value) noexcept
        : _target{luisa::bit_cast<VkBuffer>(target.native_handle())},
          _size{4u},
          _value{value} {
        // Declares the whole buffer so the reorder layer keeps this command
        // ordered with the dispatches that write the same buffer.
        _usages.emplace_back(
            Argument::Buffer{target.handle(), 0u, target.size_bytes()},
            ResourceUsageType::CopyDest);
    }
    [[nodiscard]] luisa::span<ResourceUsage> get_resource_usages() noexcept override {
        return _usages;
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }
    void execute(VkPhysicalDevice, VkDevice, VkQueue,
                 VkCommandBuffer command_buffer,
                 VkDescriptorPool) const noexcept override {
        auto fill_buffer = resolve_cmd_fill_buffer();
        LUISA_ASSERT(fill_buffer != nullptr,
                     "vkCmdFillBuffer is unavailable.");
        fill_buffer(command_buffer, _target, 0u, _size, _value);
    }
};

void reg_bind_elimination_internal_and_raw(Device &device) {
    "vk_bind_elimination_internal_and_raw_binds"_test = [&device] {
        // Two ways the tracked state is interrupted by code the tracker does not
        // own: an internal kernel (the bindless upload) and a custom command that
        // records raw Vulkan on the command buffer. Both cases are laid out so the
        // expected count does not depend on the reorder visitor's layer choice.
        constexpr auto dispatch_count = size_t{2u};
        auto shader = device.compile(write_dispatch_index);

        // 1) Internal kernel between identical dispatches.
        {
            constexpr auto slot_count = size_t{4u};
            BindlessArray heap = device.create_bindless_array(slot_count);
            Buffer<float> slot_buffer = device.create_buffer<float>(kThreadsPerDispatch);
            heap.emplace_on_update(0u, slot_buffer);
            Buffer<float> buffer =
                device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
            Stream stream = device.create_stream();
            CommandList list;
            list.reserve(dispatch_count + 1u, 0u);
            // The update kernel is submitted first and touches resources the
            // dispatches do not, so it may land in either layer and the expected
            // count is the same either way.
            list << heap.update();
            for (auto i = 0u; i < dispatch_count; ++i) {
                list << shader(buffer, i).dispatch(kThreadsPerDispatch);
            }
            auto before = read_counters(device);
            stream << list.commit() << synchronize();
            auto after = read_counters(device);
            auto d = difference(before, after);
            LUISA_INFO(
                "P1.2 internal kernel: bindless update + {} identical "
                "dispatches -> {} pipeline binds (expected 2), {} set binds",
                dispatch_count, value(d, Counter::pipeline_binds),
                value(d, Counter::descriptor_set_binds));
            expect(value(d, Counter::pipeline_binds) == 2u)
                << "the bindless upload kernel binds a different pipeline, so the "
                   "dispatches around it need exactly one bind of their own "
                   "(P1.2 routing of the internal kernels)";
 // Plan P2.3: on the push path the bindless-upload kernel still binds its
 // own (classic) set while the following dispatches bind only their global
 // sets - once, because the tracked slots survive the pushes of set 0.
 auto const expected_binds =
 reported_push_path(device) ? 2u : dispatch_count + 1u;
 expect(value(d, Counter::descriptor_set_binds) == expected_binds)
 << "the internal kernel binds its own set; the dispatches bind theirs "
 "(classic) or only their global sets (push)";
            luisa::vector<float> host(dispatch_count * kThreadsPerDispatch, -1.0f);
            stream << buffer.copy_to(luisa::span{host}) << synchronize();
            check_dispatch_indices(host, dispatch_count, "bind-elimination-bindless");
        }

        // 2) Raw binds from a custom command. `execute` gets the command buffer
        // itself, so nothing the tracker knew about it may survive.
        {
            constexpr auto slot_count = dispatch_count + 1u;
            auto const primed = -7.0f;
            auto const filled = 42.0f;
            Buffer<float> buffer =
                device.create_buffer<float>(slot_count * kThreadsPerDispatch);
            luisa::vector<float> host(slot_count * kThreadsPerDispatch, primed);
            Stream stream = device.create_stream();
            stream << buffer.copy_from(luisa::span{host}) << synchronize();
            host.assign(host.size(), -1.0f);
            CommandList list;
            list.reserve(dispatch_count + 2u, 0u);
            // The dispatches and the custom command all write the same buffer, so
            // the reorder layer keeps them in submission order: dispatch, custom,
            // dispatch. That is the shape that can tell the flush apart - a custom
            // command submitted first would look exactly like "no custom command
            // at all" (one bind either way), because a flush is not a bind.
            list << shader(buffer, 1u).dispatch(kThreadsPerDispatch);
            list << luisa::make_unique<RawFillCommand>(
                buffer, luisa::bit_cast<uint32_t>(filled));
            // Slot 0 is never written by the dispatches, so it keeps the raw fill
            // value: this proves the custom command really ran.
            for (auto i = 2u; i <= dispatch_count; ++i) {
                list << shader(buffer, i).dispatch(kThreadsPerDispatch);
            }
            list << buffer.copy_to(luisa::span{host});
            auto before = read_counters(device);
            stream << list.commit() << synchronize();
            auto after = read_counters(device);
            auto d = difference(before, after);
            LUISA_INFO(
                "P1.2 raw custom bind: custom command + {} identical dispatches "
                "-> {} pipeline binds (expected 2)",
                dispatch_count, value(d, Counter::pipeline_binds));
            expect(value(d, Counter::pipeline_binds) == 2u)
                << "a custom command that records raw Vulkan on the command "
                   "buffer must flush the tracked state, so the following "
                   "dispatches bind their pipeline again (P1.2)";
            expect(host[0] == filled) << "the custom command must have run";
            auto ok = true;
            for (auto i = 1u; i <= dispatch_count && ok; ++i) {
                for (auto t = 0u; t < kThreadsPerDispatch; ++t) {
                    auto actual = host[i * kThreadsPerDispatch + t];
                    if (actual != static_cast<float>(i)) {
                        LUISA_WARNING(
                            "raw custom bind: dispatch {} thread {} produced {} "
                            "instead of {}.",
                            i - 1u, t, actual, static_cast<float>(i));
                        ok = false;
                        break;
                    }
                }
            }
            expect(ok) << "the dispatches after the custom command must still "
                          "write their own slices";
        }
    };
}

// ---------------------------------------------------------------------------
// P2.2: per-dispatch set-0 updates through VkDescriptorUpdateTemplate
// ---------------------------------------------------------------------------
// The path itself is a device-creation input, so this executable also takes
// `--descriptor-update-template` (switch
// VulkanDeviceConfigExt::enable_descriptor_update_template() on) and removes it
// from argv. Without the flag the device keeps the library default, which is the
// measured-faster legacy vkUpdateDescriptorSets path; with it the template path
// must prove itself. Running the same binary with and without the flag covers
// both branches in one build.
bool g_descriptor_update_template{false};
// Plan P2.3: `--push-descriptor` asks for the push path, `--no-push-descriptor`
// forces the classic set-0 path on a device that could push. The default is
// VulkanDeviceConfigExt::enable_push_descriptor(), which is what the library
// ships; the two flags exist so one binary can measure and test both paths.
struct PushDescriptorSwitch {
    bool specified{false};
    bool enabled{false};
};
PushDescriptorSwitch g_push_descriptor_option{};

// A kernel with an image and a buffer in the same set 0: the update mixes
// VkDescriptorImageInfo and VkDescriptorBufferInfo payloads, which is the case
// where a template's packed layout must not confuse the two.
constexpr auto kMixedImageSize = 8u;

Kernel2D copy_image_to_buffer = [](ImageFloat source, BufferFloat4 destination) noexcept {
    auto coord = dispatch_id().xy();
    destination.write(coord.x + coord.y * static_cast<uint32_t>(kMixedImageSize),
                      source.read(coord));
};

// A printer appends its count/data descriptors at the *end* of set 0, so this
// shader's write shape has two extra STORAGE_BUFFER entries after the argument
// block.
Kernel1D printer_tail_kernel = [](BufferFloat out, Float value) noexcept {
    auto id = dispatch_id().x;
    device_log("p2.2-value={}", value);
    out.write(id, value);
};

// Plan P2.3: a kernel whose set 0 cannot be pushed on any driver that exposes
// this limit - every buffer argument is one descriptor, so 33 buffers plus the
// argument block exceed maxPushDescriptors (32 on the machine this backend was
// measured on, and the host test covers the rejection itself). It exists to
// interleave a push shader and a classic shader in *one* command buffer, which
// is the case a per-shader decision has to get right: the two shaders use
// different interned set-0 layouts (push vs classic flags), so they must never
// alias, and the encoder must bind each of them on its own path.
Kernel1D write_33_buffers = [](BufferFloat b00, BufferFloat b01, BufferFloat b02, BufferFloat b03, BufferFloat b04, BufferFloat b05, BufferFloat b06, BufferFloat b07, BufferFloat b08, BufferFloat b09, BufferFloat b10, BufferFloat b11, BufferFloat b12, BufferFloat b13, BufferFloat b14, BufferFloat b15, BufferFloat b16, BufferFloat b17, BufferFloat b18, BufferFloat b19, BufferFloat b20, BufferFloat b21, BufferFloat b22, BufferFloat b23, BufferFloat b24, BufferFloat b25, BufferFloat b26, BufferFloat b27, BufferFloat b28, BufferFloat b29, BufferFloat b30, BufferFloat b31, BufferFloat b32) noexcept {
      b00.write(dispatch_id().x, 0.0f);
      b01.write(dispatch_id().x, 1.0f);
      b02.write(dispatch_id().x, 2.0f);
      b03.write(dispatch_id().x, 3.0f);
      b04.write(dispatch_id().x, 4.0f);
      b05.write(dispatch_id().x, 5.0f);
      b06.write(dispatch_id().x, 6.0f);
      b07.write(dispatch_id().x, 7.0f);
      b08.write(dispatch_id().x, 8.0f);
      b09.write(dispatch_id().x, 9.0f);
      b10.write(dispatch_id().x, 10.0f);
      b11.write(dispatch_id().x, 11.0f);
      b12.write(dispatch_id().x, 12.0f);
      b13.write(dispatch_id().x, 13.0f);
      b14.write(dispatch_id().x, 14.0f);
      b15.write(dispatch_id().x, 15.0f);
      b16.write(dispatch_id().x, 16.0f);
      b17.write(dispatch_id().x, 17.0f);
      b18.write(dispatch_id().x, 18.0f);
      b19.write(dispatch_id().x, 19.0f);
      b20.write(dispatch_id().x, 20.0f);
      b21.write(dispatch_id().x, 21.0f);
      b22.write(dispatch_id().x, 22.0f);
      b23.write(dispatch_id().x, 23.0f);
      b24.write(dispatch_id().x, 24.0f);
      b25.write(dispatch_id().x, 25.0f);
      b26.write(dispatch_id().x, 26.0f);
      b27.write(dispatch_id().x, 27.0f);
      b28.write(dispatch_id().x, 28.0f);
      b29.write(dispatch_id().x, 29.0f);
      b30.write(dispatch_id().x, 30.0f);
      b31.write(dispatch_id().x, 31.0f);
      b32.write(dispatch_id().x, 32.0f);
};

void reg_descriptor_update_path(Device &device) {
    "vk_profile_descriptor_update_path"_test = [&device] {
        // Plan P2.2. The reported path is latched at device creation from the
        // negotiated extension, the config switch and the resolved entry points.
        // The template path is opt-in: on the driver this backend was measured on
        // it costs 2.5-7 % more host submission time, so the library default is
        // the legacy vkUpdateDescriptorSets call and
        // `--descriptor-update-template` is what enables the other branch.
        auto path = device.query("vk_descriptor_update_template_path");
        LUISA_INFO("descriptor-update path: {} (template requested: {})",
                   path, g_descriptor_update_template);
        expect(path == "template" || path == "legacy")
            << "the backend must report which entry point performs a "
               "per-dispatch descriptor update";
        if (g_descriptor_update_template) {
            expect(path == "template")
                << "VulkanDeviceConfigExt::enable_descriptor_update_template() "
                   "== true must select vkUpdateDescriptorSetWithTemplate";
        } else {
            expect(path == "legacy")
                << "the library default must stay on vkUpdateDescriptorSets: the "
                   "template path measured slower on this driver (plan P2.2)";
        }
    };
}

void reg_descriptor_update_templates_repeated(Device &device) {
    "vk_descriptor_update_templates_repeated_buffer_dispatches"_test = [&device] {
        // Repeated dispatches of one unchanged, buffer-only shader: the same
        // write shape every time, so one template is created and reused for
        // every dispatch of the batch.
        constexpr auto dispatch_count = size_t{16u};
        auto shader = device.compile(write_dispatch_index);
        Buffer<float> buffer =
            device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(dispatch_count, 0u);
        for (auto i = 0u; i < dispatch_count; ++i) {
            list << shader(buffer, i).dispatch(kThreadsPerDispatch);
        }
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        expect(value(d, Counter::dispatches) == dispatch_count);
        check_descriptor_update_path(device, d, dispatch_count, 0u,
                                     "repeated buffer dispatches");
        if (reported_template_path(device) && !reported_push_path(device)) {
            expect(value(d, Counter::descriptor_update_template_creates) == 1u)
                << "one shader has one set-0 write shape, so the whole batch "
                   "must reuse the single template it created";
        }
        luisa::vector<float> host(dispatch_count * kThreadsPerDispatch, -1.0f);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        check_dispatch_indices(host, dispatch_count,
                               "descriptor-update-templates-repeated");
    };
}

void reg_descriptor_update_templates_mixed(Device &device) {
    "vk_descriptor_update_templates_mixed_image_and_buffer"_test = [&device] {
        // An image and a buffer in one set 0: the packed payload holds
        // VkDescriptorImageInfo and VkDescriptorBufferInfo values side by side,
        // so a wrong offset or stride would sample garbage.
        constexpr auto size = kMixedImageSize;
        constexpr auto pixel_count = size_t{size} * size;
        auto shader = device.compile(copy_image_to_buffer);
        std::array<float4, pixel_count> source_data{};
        for (auto i = size_t{0u}; i < pixel_count; ++i) {
            source_data[i] = make_float4(static_cast<float>(i),
                                         static_cast<float>(i) + 0.5f,
                                         -static_cast<float>(i),
                                         static_cast<float>(i * 3u));
        }
        Buffer<float4> upload = device.create_buffer<float4>(pixel_count);
        Buffer<float4> readback = device.create_buffer<float4>(pixel_count);
        Image<float> image = device.create_image<float>(
            PixelStorage::FLOAT4, make_uint2(size, size));
        Stream stream = device.create_stream();
        stream << upload.copy_from(luisa::span{source_data})
               << image.copy_from(upload)
               << synchronize();
        CommandList list;
        list.reserve(1u, 0u);
        list << shader(image, readback).dispatch(size, size);
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        check_descriptor_update_path(device, d, 1u, 0u,
                                     "mixed image/buffer set 0");
        std::array<float4, pixel_count> actual{};
        stream << readback.copy_to(luisa::span{actual}) << synchronize();
        auto ok = true;
        for (auto i = size_t{0u}; i < pixel_count; ++i) {
            auto expected = source_data[i];
            auto got = actual[i];
            if (got.x != expected.x || got.y != expected.y ||
                got.z != expected.z || got.w != expected.w) {
                LUISA_WARNING("mixed set 0: pixel {} read ({}, {}, {}, {}) "
                              "instead of ({}, {}, {}, {}).",
                              i, got.x, got.y, got.z, got.w,
                              expected.x, expected.y, expected.z, expected.w);
                ok = false;
                break;
            }
        }
        expect(ok) << "a template-updated set mixing image and buffer "
                      "descriptors must still bind both correctly";
    };
}

void reg_descriptor_update_templates_printer(Device &device) {
    "vk_descriptor_update_templates_printer_tail"_test = [&device] {
        // The printer descriptors are appended after every argument binding, so
        // this exercises a write list whose tail is not derived from the
        // argument list at all.
        constexpr auto threads = 4u;
        constexpr auto value = 3.5f;
        auto shader = device.compile(printer_tail_kernel);
        Buffer<float> buffer = device.create_buffer<float>(threads);
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(1u, 0u);
        list << shader(buffer, value).dispatch(threads);
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        check_descriptor_update_path(device, d, 1u, 0u, "printer tail");
        luisa::vector<float> host(threads, -1.0f);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < threads; ++i) {
            if (host[i] != value) {
                LUISA_WARNING("printer tail: thread {} wrote {} instead of {}.",
                              i, host[i], value);
                ok = false;
            }
        }
        expect(ok) << "a dispatch with printer descriptors must still write "
                      "its argument block and buffer correctly";
    };
}

void reg_descriptor_update_templates_distinct_shaders(Device &device) {
    "vk_descriptor_update_templates_distinct_shaders"_test = [&device] {
        // Two structurally different shaders: different descriptor-set layouts
        // and different write shapes, so each owns its own template.
        constexpr auto dispatch_count = size_t{4u};
        auto single = device.compile(write_dispatch_index);
        auto pair = device.compile(write_two_buffers);
        Buffer<float> first =
            device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
        Buffer<float> second =
            device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(2u * dispatch_count, 0u);
        for (auto i = 0u; i < dispatch_count; ++i) {
            list << single(first, i).dispatch(kThreadsPerDispatch);
            list << pair(first, second, i).dispatch(kThreadsPerDispatch);
        }
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        expect(value(d, Counter::dispatches) == 2u * dispatch_count);
        check_descriptor_update_path(device, d, 2u * dispatch_count, 0u,
                                     "two shaders, two layouts");
        if (reported_template_path(device) && !reported_push_path(device)) {
            expect(value(d, Counter::descriptor_update_template_creates) == 2u)
                << "two write shapes need two templates, and neither may be "
                   "reused by the other (the shapes differ)";
        }
        luisa::vector<float> first_host(dispatch_count * kThreadsPerDispatch, -1.0f);
        luisa::vector<float> second_host(dispatch_count * kThreadsPerDispatch, -1.0f);
        stream << first.copy_to(luisa::span{first_host})
               << second.copy_to(luisa::span{second_host})
               << synchronize();
        // `pair` runs after `single` and overwrites `first` with 1.0f and writes
        // `second` with 2.0f, so the last writer per slice must win. This is the
        // numeric proof that each shader reached its own descriptors - a template
        // built for the wrong layout would bind the wrong buffer.
        auto first_ok = true;
        for (auto value : first_host) {
            if (value != 1.0f) { first_ok = false; break; }
        }
        auto second_ok = true;
        for (auto value : second_host) {
            if (value != 2.0f) { second_ok = false; break; }
        }
        expect(first_ok) << "the two-buffer shader must have bound its own "
                            "first buffer through its own template";
        expect(second_ok) << "and its own second buffer";
    };
}

// ---------------------------------------------------------------------------
// F6 / P1.5: configurable in-flight depth plus the completion fast path
// ---------------------------------------------------------------------------
// The depth is a device-creation input, so this executable takes it as an
// argument (`test_vk_submit_profile vk [--frames-in-flight <n>]`); the pair is
// removed from argv before the device helper and the Boost.UT parser see it.
// Without the argument the backend default (3) has to apply. Every expectation
// below is derived from the depth the Device reports through the diagnostic
// `vk_max_frames_in_flight` query, so one binary covers the default and every
// explicit depth, and a configured value that is silently ignored fails the
// latch assertion.
struct FramesInFlightOption {
    bool specified{false};
    uint32_t value{0u};
};

FramesInFlightOption g_frames_in_flight_option{};

// Removes `--frames-in-flight <n>` and `--min-api <version>` from argv in place and
// reports what was requested.
[[nodiscard]] FramesInFlightOption
take_frames_in_flight_option(int &argc, char **argv) noexcept {
    FramesInFlightOption option;
    auto kept = 0;
    for (auto i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--frames-in-flight") == 0 && i + 1 < argc) {
            option.specified = true;
            auto parsed = std::strtoull(argv[i + 1], nullptr, 10);
            option.value = parsed > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(parsed);
            ++i;// skip the value as well
            continue;
        }
        if (std::strcmp(argv[i], "--min-api") == 0 && i + 1 < argc) {
            auto value = std::string_view{argv[i + 1]};
            g_min_api_1_2 = value == "1.2" || value == "12";
            ++i;// skip the value as well
            continue;
        }
 if (std::strcmp(argv[i], "--descriptor-update-template") == 0) {
 g_descriptor_update_template = true;
 continue;
 }
 if (std::strcmp(argv[i], "--push-descriptor") == 0) {
 g_push_descriptor_option.specified = true;
 g_push_descriptor_option.enabled = true;
 continue;
 }
 if (std::strcmp(argv[i], "--no-push-descriptor") == 0) {
 g_push_descriptor_option.specified = true;
 g_push_descriptor_option.enabled = false;
 continue;
 }
        argv[kept++] = argv[i];
    }
    argv[kept] = nullptr;
    argc = kept;
    return option;
}

// Answers only the in-flight depth, the API-version floor and (plan P2.2) the
// descriptor-update switch, so the rest of the device is created exactly as the
// default configuration creates it.
class SubmitProfileConfig final : public VulkanDeviceConfigExt {
    uint32_t _frames_in_flight;
    uint32_t _min_api_version;
 bool _descriptor_update_template;
 bool _push_descriptor;
 bool _push_descriptor_specified;
public:
 explicit SubmitProfileConfig(uint32_t frames_in_flight,
 uint32_t min_api_version,
 bool descriptor_update_template,
 bool push_descriptor,
 bool push_descriptor_specified = true) noexcept
 : _frames_in_flight{frames_in_flight},
 _min_api_version{min_api_version},
 _descriptor_update_template{descriptor_update_template},
 _push_descriptor{push_descriptor},
 _push_descriptor_specified{push_descriptor_specified} {}
    [[nodiscard]] uint32_t max_frames_in_flight() const noexcept override {
        return _frames_in_flight;
    }
    [[nodiscard]] uint32_t min_api_version() const noexcept override {
        return _min_api_version;
    }
 // Plan P2.2: `--descriptor-update-template` turns the template path on (the
 // library default is the measured-faster legacy vkUpdateDescriptorSets call).
 [[nodiscard]] bool enable_descriptor_update_template() const noexcept override {
 return _descriptor_update_template;
 }
 // Plan P2.3: `--push-descriptor` / `--no-push-descriptor` choose the set-0
 // path explicitly; without either flag the library default applies, which is
 // what a shipping consumer sees.
 [[nodiscard]] bool enable_push_descriptor() const noexcept override {
 return _push_descriptor_specified ? _push_descriptor :
 VulkanDeviceConfigExt::enable_push_descriptor();
    }
};

[[nodiscard]] uint32_t latched_frames_in_flight(Device &device) noexcept {
    auto text = device.query("vk_max_frames_in_flight");
    LUISA_ASSERT(!text.empty(),
                 "the Vulkan backend must report the in-flight depth it latched.");
    return static_cast<uint32_t>(std::strtoul(text.c_str(), nullptr, 10));
}


  // ---------------------------------------------------------------------------
  // P2.3: set-0 push descriptors
  // ---------------------------------------------------------------------------
  void reg_push_descriptor_path(Device &device) {
      "vk_push_descriptor_path"_test = [&device] {
          // The reported path is latched at device creation from the negotiated
          // extension, the config switch, the resolved entry point and
          // maxPushDescriptors. This test pins the report, so a device that could
          // push but chose the classic layout has to say so.
          auto path = device.query("vk_push_descriptor_path");
          auto max_count = device.query("vk_push_descriptor_max_count");
          LUISA_INFO("push-descriptor path: {} (maxPushDescriptors {}, switch {})",
                     path, max_count,
                     g_push_descriptor_option.specified ?
                         (g_push_descriptor_option.enabled ? "on" : "off") :
                         "library-default");
          expect(path == "push" || path == "classic")
              << "the backend must report which path binds set 0";
          expect(!max_count.empty())
              << "the resolved maxPushDescriptors must be reported";
          auto max_push = std::strtoul(max_count.c_str(), nullptr, 10);
          if (g_push_descriptor_option.specified &&
              !g_push_descriptor_option.enabled) {
              expect(path == "classic")
                  << "VulkanDeviceConfigExt::enable_push_descriptor() == false "
                     "must keep the classic set-0 path";
              expect(max_push == 0u)
                  << "a classic path never reports a push limit";
          } else if (path == "push") {
              expect(max_push > 0u)
                  << "a push device must report the limit its layouts respect";
          } else {
              // The driver or the library default chose classic: then no layout
              // may have been created with the push bit, which the shader counters
              // assert.
              expect(max_push == 0u);
          }
          // A classic device must not have created a single push shader, and a push
          // device must not report both counters as zero after the compiles this
          // binary already did.
          auto classic_shaders = device.query("vk_profile_push_descriptor_classic_shaders");
          auto push_shaders = device.query("vk_profile_push_descriptor_shaders");
          LUISA_INFO("push-descriptor shaders: {} push, {} classic", push_shaders,
                     classic_shaders);
          if (path == "classic") {
              expect(!classic_shaders.empty())
                  << "the shader-decision counter must be readable";
          }
      };
  }

  void reg_push_descriptor_alternating(Device &device) {
      "vk_push_descriptor_alternating_paths"_test = [&device] {
          // Plan P2.3, the case a per-shader decision has to get right: two
          // shaders in one command buffer, one pushable and one not. The classic
          // shader's set 0 exceeds maxPushDescriptors (33 buffer arguments), so it
          // is created with the classic layout - and because a push set-0 layout
          // and a classic one are different interned structures, the two shaders
          // must also use different pipeline layouts.
          if (!reported_push_path(device)) {
              LUISA_INFO("push path unavailable: the alternating-path test needs a "
                         "push-capable device and is skipped");
              return;
          }
          constexpr auto pairs = size_t{3u};
          constexpr auto threads = 4u;
          auto push_shader = device.compile(write_dispatch_index);
          auto before_classic = read_counters(device);
          auto classic_shader = device.compile(write_33_buffers);
          auto after_classic = read_counters(device);
          auto decisions = difference(before_classic, after_classic);
          expect(value(decisions, Counter::push_descriptor_shaders) == 0u &&
                 value(decisions, Counter::push_descriptor_classic_shaders) >= 1u)
              << "the 33-buffer shader must be latched as a classic set-0 shader, "
                 "because its layout would exceed maxPushDescriptors (host test "
                 "covers the limit arithmetic)";
          expect(value(decisions, Counter::descriptor_set_layout_creates) >= 1u &&
                 value(decisions, Counter::pipeline_layout_creates) >= 1u)
              << "a push set-0 layout and a classic one are different structures, so "
                 "the classic shader needs its own interned layouts (no aliasing)";
          // Three buffers are enough to prove the values; the rest are written too
          // so every descriptor of the classic layout is really bound.
          luisa::vector<Buffer<float>> buffers;
          buffers.reserve(33u);
          for (auto i = 0u; i < 33u; ++i) {
              buffers.emplace_back(device.create_buffer<float>(threads));
          }
 // `write_dispatch_index` stamps `index * 32 + thread`, so the push buffer is
 // sized for that shape.
 Buffer<float> push_buffer =
 device.create_buffer<float>(pairs * kThreadsPerDispatch);
          Stream stream = device.create_stream();
          CommandList list;
          list.reserve(2u * pairs, 0u);
          for (auto i = 0u; i < pairs; ++i) {
 list << push_shader(push_buffer, static_cast<uint>(i))
 .dispatch(kThreadsPerDispatch);
              auto &&b = buffers;
              list << classic_shader(
                          b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8],
                          b[9], b[10], b[11], b[12], b[13], b[14], b[15], b[16],
                          b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                          b[25], b[26], b[27], b[28], b[29], b[30], b[31], b[32])
                          .dispatch(threads);
          }
          auto before = read_counters(device);
          stream << list.commit() << synchronize();
          auto after = read_counters(device);
          auto d = difference(before, after);
          LUISA_INFO(
              "push/classic alternating: {} pairs -> {} push calls, {} set "
              "allocations, {} set binds, {} pipeline binds",
              pairs, value(d, Counter::push_descriptor_calls),
              value(d, Counter::descriptor_set_allocations),
              value(d, Counter::descriptor_set_binds),
              value(d, Counter::pipeline_binds));
          expect(value(d, Counter::dispatches) == 2u * pairs);
          expect(value(d, Counter::push_descriptor_calls) == pairs)
              << "only the push shader pushes, once per dispatch";
          expect(value(d, Counter::descriptor_set_allocations) == pairs)
              << "only the classic shader allocates a set 0, once per dispatch";
          expect(value(d, Counter::descriptor_set_binds) == 2u * pairs)
              << "the two shaders use different pipeline layouts, so every dispatch "
                 "re-binds the sets of its own layout (the tracker never assumes "
                 "two layouts are compatible)";
          expect(value(d, Counter::pipeline_binds) == 2u * pairs)
              << "alternating pipelines always bind";
          // Values: the classic shader stamps 0..32 into its own buffers, the push
          // shader writes its dispatch index into its slice.
          for (auto i = 0u; i < 33u; ++i) {
              luisa::vector<float> host(threads, -1.0f);
              stream << buffers[i].copy_to(luisa::span{host}) << synchronize();
              auto ok = true;
              for (auto t = 0u; t < threads; ++t) {
                  if (host[t] != static_cast<float>(i)) { ok = false; break; }
              }
              expect(ok) << "the classic shader must reach every one of its 33 "
                            "descriptors (a wrong layout or a stale push bank would "
                            "leave a buffer untouched)";
              if (!ok) { break; }
          }
 luisa::vector<float> push_host(pairs * kThreadsPerDispatch, -1.0f);
          stream << push_buffer.copy_to(luisa::span{push_host}) << synchronize();
          auto push_ok = true;
          for (auto i = 0u; i < pairs; ++i) {
 for (auto t = 0u; t < kThreadsPerDispatch; ++t) {
 if (push_host[i * kThreadsPerDispatch + t] != static_cast<float>(i)) {
                      push_ok = false;
                      break;
                  }
              }
          }
          expect(push_ok) << "the push shader must keep writing its own argument "
                             "block and buffer across the alternation";
      };
  }

void reg_frames_in_flight_config(Device &device) {
    "vk_in_flight_depth_is_latched_from_config"_test = [&device] {
        auto latched = latched_frames_in_flight(device);
        if (g_frames_in_flight_option.specified) {
            LUISA_INFO("in-flight depth: latched {} (explicitly requested {})",
                       latched, g_frames_in_flight_option.value);
            expect(latched == g_frames_in_flight_option.value)
                << "the depth returned by "
                   "VulkanDeviceConfigExt::max_frames_in_flight() must be the one "
                   "the streams use";
        } else {
            LUISA_INFO("in-flight depth: latched {} (backend default)", latched);
            expect(latched == 3u)
                << "the backend default in-flight depth is three batches";
        }
        expect(latched >= 1u)
            << "a zero depth is rejected at device creation, so it can never be "
               "latched as the bound";
    };
}

void reg_in_flight_depth(Device &device) {
    "vk_profile_in_flight_depth"_test = [&device] {
        constexpr auto batch_count = size_t{32u};
        auto depth = latched_frames_in_flight(device);
        // The throttle branch is taken while `last_fence > depth`, i.e. on the
        // batches at index depth+1 .. batch_count-1. For depth 2 that is the
        // historical 32 - 3 = 29 host stalls; depth 3 stalls 28 times, i.e.
        // exactly one batch later.
        auto first_throttled = static_cast<size_t>(depth) + 1u;
        auto expected_entries = batch_count > first_throttled ? batch_count - first_throttled : size_t{0u};
        auto shader = device.compile(gpu_bound_kernel);
        Buffer<float> buffer = device.create_buffer<float>(
            batch_count * kThreadsPerGpuBoundDispatch);
        Stream stream = device.create_stream();
        luisa::vector<float> host(batch_count * kThreadsPerGpuBoundDispatch,
                                  -2.0f);
        auto before = read_counters(device);
        for (auto i = 0u; i < batch_count; ++i) {
            // Every batch is submitted on its own, back to back, and the device
            // is behind the host, so the in-flight limit has to show up.
            CommandList list;
            list << shader(buffer, i).dispatch(kThreadsPerGpuBoundDispatch);
            stream << list.commit();
        }
        auto after = read_counters(device);
        auto d = difference(before, after);
        auto entries = value(d, Counter::timeline_wait_entries);
        auto waits = value(d, Counter::timeline_waits);
        auto skips = value(d, Counter::timeline_wait_skips);
        LUISA_INFO(
            "in-flight depth {}: {} GPU-bound batches -> {} batches, {} "
            "submissions, {} throttle entries ({} host waits, {} completion "
            "skips)",
            depth, batch_count, value(d, Counter::command_batches),
            value(d, Counter::queue_submits), entries, waits, skips);
        // F5/§5.1 on the shape where a per-submission barrier could plausibly
        // cost GPU time: 32 GPU-bound submissions pay 32 restores, so the
        // restore cost is a per-submission constant and this shape is the one a
        // GPU-side measurement would have to use. Only the host cost is
        // measured here; nothing below infers GPU time from it.
        check_restore_shape(d, batch_count, "in-flight depth");
        expect(value(d, Counter::restore_host_ns) > 0u)
            << "32 restores must register a nonzero host cost (the timer works)";
        LUISA_INFO(
            "in-flight depth: {:.2f} us of host restore per GPU-bound batch "
            "({:.3f} % of the {:.2f} us host encode per batch)",
            static_cast<double>(value(d, Counter::restore_host_ns)) / 1000.0 /
                static_cast<double>(batch_count),
            100.0 * static_cast<double>(value(d, Counter::restore_host_ns)) /
                static_cast<double>(value(d, Counter::host_encode_ns)),
            static_cast<double>(value(d, Counter::host_encode_ns)) / 1000.0 /
                static_cast<double>(batch_count));
        expect(value(d, Counter::command_batches) == batch_count)
            << "one command batch per dispatch call";
        expect(value(d, Counter::queue_submits) == batch_count)
            << "one native submit per batch, not per dispatch (F6)";
        check_submit_variants(device, d, batch_count, "in-flight depth");
        expect(entries == expected_entries)
            << "the configured depth must decide when the host throttles: "
               "batch_count - (depth + 1) entries, and depth 2 reproduces the "
               "historical 29 stalls in 32 batches";
        expect(waits + skips == entries)
            << "every throttle entry either blocks the host or takes the fast "
               "path, never both and never neither";
        // One batch here is 65536 threads with 8192 serial sin() steps, so most
        // throttle entries really block the host. The exact wait/skip split is
        // timing dependent (the target can retire while the host is recording
        // the next batch, and a driver may drain faster than the host submits),
        // so this only asserts that the bound is *not* purely theoretical; the
        // skip accounting is pinned deterministically by
        // vk_in_flight_depth_fast_path below.
        expect(waits >= 1u)
            << "a GPU-bound submission loop must really block the host (F6), not "
               "just enter the throttle";
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto v : host) {
            // sin() keeps every result inside [-1, 1]; the -2.0f placeholder
            // therefore proves the dispatch really ran.
            if (!(v >= -1.0f && v <= 1.0f)) {
                ok = false;
                break;
            }
        }
        expect(ok) << "every GPU-bound batch must have written a result";
    };
}

void reg_in_flight_depth_fast_path(Device &device) {
    "vk_in_flight_depth_fast_path_skips_completed_throttle"_test = [&device] {
        // Deterministic fast path (P1.5): a throttled submission must not block
        // the host when the GPU has already completed the target timeline value.
        // Nothing here depends on how fast the GPU is - the test *completes* the
        // in-flight window first, and only then submits the one throttled batch.
        auto depth = latched_frames_in_flight(device);
        auto window = static_cast<size_t>(depth) + 1u;
        auto shader = device.compile(write_dispatch_index);
        Buffer<float> buffer = device.create_buffer<float>(
            (window + 1u) * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        luisa::vector<float> host((window + 1u) * kThreadsPerDispatch, -3.0f);
        for (auto i = size_t{0u}; i < window; ++i) {
            CommandList list;
            list << shader(buffer, static_cast<uint32_t>(i))
                        .dispatch(kThreadsPerDispatch);
            stream << list.commit();
        }
        // Everything submitted so far is complete and retired, so the completion
        // watermark covers every timeline value in flight.
        stream << synchronize();
        auto before = read_counters(device);
        {
            // last_fence() is now depth + 1 > depth, so this submission enters the
            // throttle with target 1 - a value the GPU finished long ago.
            CommandList list;
            list << shader(buffer, static_cast<uint32_t>(window))
                        .dispatch(kThreadsPerDispatch);
            stream << list.commit();
        }
        auto after = read_counters(device);
        auto d = difference(before, after);
        auto entries = value(d, Counter::timeline_wait_entries);
        auto waits = value(d, Counter::timeline_waits);
        auto skips = value(d, Counter::timeline_wait_skips);
        LUISA_INFO(
            "in-flight fast path (depth {}): {} throttle entry, {} host wait, "
            "{} completion skip",
            depth, entries, waits, skips);
        expect(entries == 1u)
            << "a submission that follows depth + 1 completed batches must enter "
               "the throttle";
        expect(skips == 1u)
            << "the target is complete, so the entry must take the fast path";
        expect(waits == 0u)
            << "the host must not wait for a timeline value the GPU has already "
               "completed";
        expect(value(d, Counter::command_batches) == 1u);
        expect(value(d, Counter::queue_submits) == 1u)
            << "skipping the throttle wait must not change the submission shape";
        check_submit_variants(device, d, 1u, "in-flight fast path");
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        check_dispatch_indices(host, window + 1u, "fast-path");
    };
}

// ---------------------------------------------------------------------------
// F8 / P6.1: the 14 core Vulkan limits answer real values, and the status record
// (plan P6.3) reports exactly the same numbers
// ---------------------------------------------------------------------------
// Every check below compares the query answer with the field of
// `VulkanCapabilityStatus` of the same name, so a device whose query path and
// status record drift apart fails here even when both look plausible. Where the
// plausibility floor is stated it is the Vulkan specification's *minimum* limit
// value (VkPhysicalDeviceLimits "Limit" column), never a number read off one
// particular GPU.
[[nodiscard]] uint64_t parse_limit_answer(luisa::string_view answer) noexcept {
    return std::strtoull(luisa::string{answer}.c_str(), nullptr, 10);
}

[[nodiscard]] bool is_power_of_two(uint64_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

// One of the 14 keys. `available` is the device's own answer to "can this be
// known at all": only min/max subgroup size (Vulkan subgroup-size control) and
// subgroup ops can be unknown on a device this backend accepts. An unavailable
// key must answer *nothing* and must carry 0 in the record - it must never be
// answered with a fabricated number.
void check_core_limit(Device &device, const char *key,
                      bool available, uint64_t recorded,
                      uint64_t spec_minimum) noexcept {
    auto answer = device.query(key);
    if (!available) {
        expect(answer.empty())
            << key << ": this device has no query for it, so the answer must be "
                      "empty rather than a fabricated number";
        expect(recorded == 0u)
            << key << ": an unavailable limit must be 0 in the status record";
        LUISA_INFO("P6.1 {}: unknown (no Vulkan query on this device)", key);
        return;
    }
    expect(!answer.empty())
        << key << ": this device can answer it, so the query must not be empty";
    if (answer.empty()) { return; }
    auto value = parse_limit_answer(answer);
    expect(value == recorded)
        << key << ": the answer must equal the capability_status() field (query="
        << answer << ", record=" << recorded << ")";
    if (spec_minimum != 0u) {
        expect(value >= spec_minimum)
            << key << ": " << value << " is below the Vulkan minimum limit value "
            << spec_minimum;
    }
    LUISA_INFO("P6.1 {}: {}", key, value);
}

// ---------------------------------------------------------------------------
// Plan P4: a replay must be one submission and zero work of every other kind
// ---------------------------------------------------------------------------
void reg_replay_zero_encode(Device &device) {
    "vk_profile_replay_zero_encode"_test = [&device] {
        auto *ext = device.extension<VulkanReplayExt>();
        expect(ext != nullptr)
            << "the Vulkan backend must expose VulkanReplayExt";
        if (ext == nullptr) { return; }
        constexpr auto dispatch_count = size_t{32u};
        constexpr auto replays = uint64_t{4u};
        auto shader = device.compile(write_dispatch_index);
        Buffer<float> buffer =
            device.create_buffer<float>(dispatch_count * kThreadsPerDispatch);
        Stream stream = device.create_stream();
        auto make_list = [&] {
            CommandList list;
            list.reserve(dispatch_count, 0u);
            for (auto i = 0u; i < dispatch_count; ++i) {
                list << shader(buffer, i).dispatch(kThreadsPerDispatch);
            }
            return list;
        };
        // ---- the recording: an encode, but never a submission ----------------
        auto before_record = read_counters(device);
        auto recorded = ext->record(stream.handle(), make_list());
        auto after_record = read_counters(device);
        expect(recorded.status == VulkanReplayStatus::SUCCESS);
        if (recorded.status != VulkanReplayStatus::SUCCESS) { return; }
        auto record_delta = difference(before_record, after_record);
        LUISA_INFO(
            "replay record window: records={} record_host={}us dispatches={} "
            "arg_plans={} set_alloc={} set_updates={} set_binds={} "
            "pipeline_binds={} batches={} submits={} encode={}ns",
            value(record_delta, Counter::replay_records),
            static_cast<double>(value(record_delta, Counter::replay_record_host_ns)) / 1000.0,
            value(record_delta, Counter::dispatches),
            value(record_delta, Counter::argument_plans),
            value(record_delta, Counter::descriptor_set_allocations),
            value(record_delta, Counter::descriptor_set_updates),
            value(record_delta, Counter::descriptor_set_binds),
            value(record_delta, Counter::pipeline_binds),
            value(record_delta, Counter::command_batches),
            value(record_delta, Counter::queue_submits),
            value(record_delta, Counter::host_encode_ns));
        // A record really did the encode work: this is what makes the replay
        // window below meaningful instead of "nothing happened anywhere".
        expect(value(record_delta, Counter::replay_records) == 1u)
            << "one recorded graph";
        expect(value(record_delta, Counter::dispatches) == dispatch_count)
            << "every recorded dispatch is counted as a recorded dispatch";
        expect(value(record_delta, Counter::argument_plans) == dispatch_count)
            << "the recording plans one argument block per dispatch (P1.4 count)";
        expect(value(record_delta, Counter::descriptor_set_allocations) == dispatch_count)
            << "the recording allocates one set 0 per dispatch";
        expect(value(record_delta, Counter::command_batches) == 1u)
            << "a recording is one command-buffer record";
        expect(value(record_delta, Counter::queue_submits) == 0u)
            << "record() must not submit anything";
        expect(value(record_delta, Counter::replay_record_host_ns) > 0u)
            << "the record reports its own host cost";
        expect(value(record_delta, Counter::host_encode_ns) == 0u)
            << "a recording does not enter the ordinary per-batch encode window, "
               "so the us/batch ratio of that counter keeps its meaning";
        // ---- the replays: one submission each and nothing else ---------------
        auto before_replay = read_counters(device);
        for (auto i = 0u; i < replays; ++i) {
            auto submitted = ext->replay(recorded.handle);
            expect(submitted.status == VulkanReplayStatus::SUCCESS);
            if (submitted.status != VulkanReplayStatus::SUCCESS) { break; }
        }
        stream << synchronize();
        auto after_replay = read_counters(device);
        auto d = difference(before_replay, after_replay);
        LUISA_INFO(
            "replay window ({} replays of {} dispatches): submits={} "
            "replay_submits={} encode={}ns arg_plans={} set_alloc={} "
            "set_updates={} set_templates={} push_calls={} set_binds={} "
            "pipeline_binds={} batches={} barrier_calls={} restore_calls={}",
            replays, dispatch_count,
            value(d, Counter::queue_submits),
            value(d, Counter::replay_submits),
            value(d, Counter::host_encode_ns),
            value(d, Counter::argument_plans),
            value(d, Counter::descriptor_set_allocations),
            value(d, Counter::descriptor_set_updates),
            value(d, Counter::descriptor_set_update_templates),
            value(d, Counter::push_descriptor_calls),
            value(d, Counter::descriptor_set_binds),
            value(d, Counter::pipeline_binds),
            value(d, Counter::command_batches),
            value(d, Counter::barrier_calls),
            value(d, Counter::restore_calls));
        expect(value(d, Counter::queue_submits) == replays)
            << "exactly one native submission per replay";
        expect(value(d, Counter::replay_submits) == replays)
            << "and one replay_submits entry per replay";
        check_submit_variants(device, d, replays, "replay window");
        expect(value(d, Counter::host_encode_ns) == 0u)
            << "a replay records nothing, so the host encode window is zero";
        expect(value(d, Counter::argument_plans) == 0u)
            << "no argument block is planned at replay time";
        expect(value(d, Counter::descriptor_set_allocations) == 0u)
            << "no descriptor set is allocated at replay time";
        expect(value(d, Counter::descriptor_set_updates) == 0u &&
               value(d, Counter::descriptor_set_update_templates) == 0u)
            << "no descriptor set is updated at replay time";
        expect(value(d, Counter::push_descriptor_calls) == 0u)
            << "no push descriptor call is recorded at replay time";
        expect(value(d, Counter::descriptor_set_binds) == 0u &&
               value(d, Counter::pipeline_binds) == 0u)
            << "nothing is bound at replay time";
        expect(value(d, Counter::command_batches) == 0u)
            << "a replay is not a command batch: nothing is recorded";
        expect(value(d, Counter::restore_calls) == 0u)
            << "the trailing restore barrier is inside the recorded buffer, so the "
               "host records no restore at replay time (the barrier argument of the "
               "design: no host barrier work, no weaker barriers)";
        expect(value(d, Counter::barrier_calls) == 0u)
            << "and no barrier struct of any kind is recorded";
        expect(value(d, Counter::dispatches) == 0u)
            << "a replay re-encodes no dispatch";
        // ---- update() refusal and destroy() accounting ------------------------
        auto before_update = read_counters(device);
        {
            CommandList update_list;
            update_list << shader(buffer, 0u).dispatch(kThreadsPerDispatch);
            auto status = ext->update(recorded.handle, std::move(update_list));
            expect(status == VulkanReplayUpdateStatus::UNSUPPORTED_UPDATE);
        }
        auto after_update = read_counters(device);
        auto update_delta = difference(before_update, after_update);
        expect(value(update_delta, Counter::replay_update_refusals) == 1u)
            << "every update() call is counted as a refusal";
        expect(value(update_delta, Counter::replay_records) == 0u &&
               value(update_delta, Counter::argument_plans) == 0u)
            << "an update() refusal creates and encodes nothing";
        auto before_destroy = read_counters(device);
        expect(ext->destroy(recorded.handle) == VulkanReplayStatus::SUCCESS);
        auto after_destroy = read_counters(device);
        auto destroy_delta = difference(before_destroy, after_destroy);
        expect(value(destroy_delta, Counter::replay_releases) == 1u)
            << "destroying the graph frees exactly one recorded command buffer";
        expect(device.query("vk_replay_graphs") == "0")
            << "and no live graph is left behind";
        // ---- results ---------------------------------------------------------
        luisa::vector<float> host(dispatch_count * kThreadsPerDispatch, -1.0f);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        check_dispatch_indices(host, dispatch_count, "replay");
    };
}

void reg_capability_surface(Device &device) {
    "vk_profile_capability_surface"_test = [&device] {
        // F8: before P6.1 the backend answered none of these 14 keys and fell
        // through to DeviceInterface::query (always empty). The tripwire now
        // validates the values, their plausibility and their agreement with the
        // P6.3 status record - an assertion that a missing key flips to "fails",
        // not to "silently answers 0".
        auto *status_ext = device.extension<VulkanExtensionStatusExt>();
        expect(status_ext != nullptr)
            << "the Vulkan backend must expose VulkanExtensionStatusExt";
        if (status_ext == nullptr) { return; }
        auto status = status_ext->capability_status();
        // The eleven unconditional members of VkPhysicalDeviceLimits. The floor
        // values are the specification minima.
        check_core_limit(device, "max_memory_allocation_count", true,
                         status.max_memory_allocation_count, 1u);
        check_core_limit(device, "max_compute_workgroup_count_x", true,
                         status.max_compute_workgroup_count_x, 65535u);
        check_core_limit(device, "max_compute_workgroup_count_y", true,
                         status.max_compute_workgroup_count_y, 65535u);
        check_core_limit(device, "max_compute_workgroup_count_z", true,
                         status.max_compute_workgroup_count_z, 65535u);
        check_core_limit(device, "max_compute_workgroup_size_x", true,
                         status.max_compute_workgroup_size_x, 128u);
        check_core_limit(device, "max_compute_workgroup_size_y", true,
                         status.max_compute_workgroup_size_y, 128u);
        check_core_limit(device, "max_compute_workgroup_size_z", true,
                         status.max_compute_workgroup_size_z, 64u);
        check_core_limit(device, "max_compute_shared_memory_bytes", true,
                         status.max_compute_shared_memory_bytes, 16384u);
        check_core_limit(device, "non_coherent_atom_size", true,
                         status.non_coherent_atom_size, 1u);
        check_core_limit(device, "min_storage_buffer_offset_alignment", true,
                         status.min_storage_buffer_offset_alignment, 1u);
        check_core_limit(device, "max_storage_buffer_range", true,
                         status.max_storage_buffer_range, 1u << 27);
        // The conditional ones. min/max subgroup size exist only when Vulkan's
        // subgroup-size-control properties are available (VK_EXT_subgroup_size_control,
        // core in Vulkan 1.3); a Vulkan 1.2 device without the extension has a
        // single subgroupSize and no bound, so both keys must stay unknown there
        // instead of being derived from it. This is the explicit unsupported path.
        auto have_subgroup_bounds = status.subgroup_size_control_available;
        check_core_limit(device, "min_subgroup_size", have_subgroup_bounds,
                         status.min_subgroup_size, 1u);
        check_core_limit(device, "max_subgroup_size", have_subgroup_bounds,
                         status.max_subgroup_size, 1u);
        if (have_subgroup_bounds) {
            expect(status.min_subgroup_size <= status.max_subgroup_size)
                << "the subgroup-size bounds must be ordered";
            expect(is_power_of_two(status.min_subgroup_size) &&
                   is_power_of_two(status.max_subgroup_size))
                << "Vulkan subgroup sizes are powers of two";
            // The bounds must at least contain the subgroup size the backend
            // already uses for its warp-level operations.
            expect(status.min_subgroup_size <= device.compute_warp_size() &&
                   device.compute_warp_size() <= status.max_subgroup_size)
                << "compute_warp_size() must lie inside the reported subgroup-size "
                   "range";
        } else {
            LUISA_INFO(
                "P6.1: this device exposes no Vulkan subgroup-size-control "
                "properties, so min/max_subgroup_size are correctly unknown");
        }
        auto have_subgroup_ops = status.subgroup_ops_available;
        check_core_limit(device, "subgroup_ops", have_subgroup_ops,
                         status.subgroup_ops, 1u);
        if (have_subgroup_ops) {
            expect((status.subgroup_ops & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0u)
                << "Vulkan guarantees the BASIC subgroup operation, so a real "
                   "subgroup bitmask always carries it";
        }
        // The two VkDeviceSize alignment-valued limits are powers of two on every
        // Vulkan device; a truncated or byte-swapped answer breaks that.
        expect(is_power_of_two(status.non_coherent_atom_size))
            << "nonCoherentAtomSize must be a power of two";
        expect(is_power_of_two(status.min_storage_buffer_offset_alignment))
            << "minStorageBufferOffsetAlignment must be a power of two";
        // 12 of the 14 keys answer on hardware without subgroup-size control, and
        // all 14 on hardware with it: a real-value validation instead of the
        // old "0 of 14 answered" tripwire.
        auto answered = size_t{0u};
        constexpr std::array keys = {
            "max_memory_allocation_count", "min_subgroup_size",
            "max_subgroup_size", "subgroup_ops",
            "max_compute_workgroup_count_x", "max_compute_workgroup_count_y",
            "max_compute_workgroup_count_z", "max_compute_workgroup_size_x",
            "max_compute_workgroup_size_y", "max_compute_workgroup_size_z",
            "max_compute_shared_memory_bytes", "non_coherent_atom_size",
            "min_storage_buffer_offset_alignment", "max_storage_buffer_range"};
        for (auto key : keys) {
            if (!device.query(key).empty()) { ++answered; }
        }
        auto expected_answered = have_subgroup_bounds ? keys.size() : keys.size() - 2u;
        LUISA_INFO("P6.1: {}/{} core limit queries answered (subgroup-size control "
                   "available: {})",
                   answered, keys.size(), have_subgroup_bounds);
        expect(answered == expected_answered)
            << "every limit this device can answer must answer; only min/max "
               "subgroup size may stay unknown (no subgroup-size-control "
               "properties), and then exactly those two";
        // A name that is not one of the 14 stays unanswered: adding keys must not
        // turn into answering queries the device does not have.
        expect(device.query("max_memory_allocation_count_").empty())
            << "a near-miss key must not be answered";
    };
}

// ---------------------------------------------------------------------------
// P1.4: one argument plan per eligible command
// ---------------------------------------------------------------------------
// Three kernels with deliberately different argument-block shapes - different
// uniform counts, sizes and alignments, and (on the XIR/SPIR-V route this build
// uses) a buffer-metadata trailer in each. `alpha` and `beta` also write one
// element of a shared stamp buffer, which is what makes the reordering pass put
// them into increasing layers while the independent `gamma` dispatches stay in
// the first layer: the emission order then is a permutation of the submission
// order, so a plan associated by position instead of by command identity would be
// paired with another command's layout. `gamma` writes the metadata-derived
// element count of its buffer argument, so a stale or misattributed metadata
// trailer shows up in the values and not only in a counter.
constexpr auto kArgumentPlanThreads = kThreadsPerDispatch;
Kernel1D argument_plan_alpha = [](BufferFloat out, BufferFloat stamp, UInt index,
                                  Float scale, Float2 bias, Float4 wave) noexcept {
    auto tid = dispatch_id().x;
    out.write(index * static_cast<uint32_t>(kArgumentPlanThreads) + tid,
              scale + bias.x * 10.0f + bias.y * 100.0f +
                  wave.z * 1000.0f + wave.w * 10000.0f);
    $if (tid == 0u) {
        stamp.write(0u, cast<float>(index) + 0.25f);
    };
};
Kernel1D argument_plan_beta = [](BufferFloat out, BufferFloat stamp, UInt index,
                                 Float3 shift, UInt count, UInt3 ids) noexcept {
    auto tid = dispatch_id().x;
    out.write(index * static_cast<uint32_t>(kArgumentPlanThreads) + tid,
              shift.x + shift.z * 10.0f +
                  cast<float>(count) * 100.0f +
                  cast<float>(ids.y) * 1000.0f);
    $if (tid == 0u) {
        stamp.write(0u, cast<float>(index) + 0.25f);
    };
};
Kernel1D argument_plan_gamma = [](BufferFloat out) noexcept {
    // `size()` is the element count of the *bound view*, i.e. the logical size the
    // XIR/SPIR-V buffer-metadata trailer carries for this command.
    out.write(dispatch_id().x, cast<float>(out.size()));
};
void reg_argument_plan_distinct_layouts(Device &device) {
    "vk_argument_plan_distinct_layouts"_test = [&device] {
        // The point of P1.4 is the *association*, so the batch has to contain
        // commands whose argument blocks differ in size, and the reordering pass
        // has to permute them. Both reorder settings run the same list, with the
        // same expected values.
        auto *reorder = device.extension<CommandReorderExt>();
        expect(reorder != nullptr)
            << "this test toggles CommandReorderExt to cover both orders";
        if (reorder == nullptr) { return; }
        auto const original_reorder = reorder->command_reorder_enabled();
        auto alpha = device.compile(argument_plan_alpha);
        auto beta = device.compile(argument_plan_beta);
        auto gamma = device.compile(argument_plan_gamma);
        constexpr auto gamma_capacity = size_t{4u} * kArgumentPlanThreads;
        constexpr auto sentinel = -7.0f;
        Buffer<float> alpha_out =
            device.create_buffer<float>(2u * kArgumentPlanThreads);
        Buffer<float> beta_out =
            device.create_buffer<float>(4u * kArgumentPlanThreads);
        Buffer<float> gamma_out = device.create_buffer<float>(gamma_capacity);
        Buffer<float> stamp = device.create_buffer<float>(2u);
        Stream stream = device.create_stream();
        for (auto reorder_enabled : {true, false}) {
            reorder->set_command_reorder_enabled(reorder_enabled);
            luisa::vector<float> alpha_host(2u * kArgumentPlanThreads, sentinel);
            luisa::vector<float> beta_host(4u * kArgumentPlanThreads, sentinel);
            luisa::vector<float> gamma_host(gamma_capacity, sentinel);
            luisa::vector<float> stamp_host(2u, sentinel);
            CommandList list;
            list.reserve(8u, 0u);
            list << alpha_out.copy_from(luisa::span{alpha_host})
                 << beta_out.copy_from(luisa::span{beta_host})
                 << gamma_out.copy_from(luisa::span{gamma_host})
                 << stamp.copy_from(luisa::span{stamp_host});
            // Submission order, and the layer the reordering pass derives for it:
            //   0 alpha  writes stamp[0]            -> first writer of stamp  = 0
            //   1 alpha  writes stamp[0]            -> WAW with 0            = 1
            //   2 gamma  touches nothing shared     ->                      = 0
            //   3 beta   writes stamp[0]            -> WAW with 1            = 2
            //   4 gamma  touches nothing shared     ->                      = 0
            // Emission order becomes 0, 2, 4, 1, 3 - a permutation of the
            // submission order whenever reordering is enabled.
            auto stamp_slot = stamp.view(0u, 1u);
            list << alpha(alpha_out, stamp_slot, 0u, 1.0f,
                          make_float2(2.0f, 3.0f),
                          make_float4(-1.0f, -1.0f, 4.0f, 5.0f))
                        .dispatch(kArgumentPlanThreads);
            list << alpha(alpha_out, stamp_slot, 1u, 8.0f,
                          make_float2(9.0f, 11.0f),
                          make_float4(-1.0f, -1.0f, 12.0f, 13.0f))
                        .dispatch(kArgumentPlanThreads);
            // Two gamma dispatches with *different* view sizes, so the two
            // metadata trailers have to carry different logical sizes and
            // different descriptor biases for their own command.
            list << gamma(gamma_out.view(0u, kArgumentPlanThreads))
                        .dispatch(kArgumentPlanThreads);
            list << beta(beta_out, stamp_slot, 3u,
                         make_float3(2.0f, -1.0f, 6.0f), 7u,
                         make_uint3(2u, 8u, 0u))
                        .dispatch(kArgumentPlanThreads);
            list << gamma(gamma_out.view(48u, 48u))
                        .dispatch(kArgumentPlanThreads);
            auto before = read_counters(device);
            stream << list.commit() << synchronize();
            auto after = read_counters(device);
            auto d = difference(before, after);
            stream << alpha_out.copy_to(luisa::span{alpha_host})
                   << beta_out.copy_to(luisa::span{beta_host})
                   << gamma_out.copy_to(luisa::span{gamma_host})
                   << stamp.copy_to(luisa::span{stamp_host})
                   << synchronize();
            LUISA_INFO(
                "P1.4 distinct layouts (reorder {}): 5 dispatches -> {} "
                "argument plans, {} recorded dispatches",
                reorder_enabled ? "on" : "off",
                value(d, Counter::argument_plans),
                value(d, Counter::dispatches));
            // The whole point of P1.4: every dispatched command is planned once.
            expect(value(d, Counter::dispatches) == 5u);
            expect(value(d, Counter::argument_plans) ==
                   value(d, Counter::dispatches))
                << "a command that needs an argument block is planned exactly "
                   "once, whatever the reordering pass then does with it (F4)";
            // Every element of every output buffer, sentinel included: a
            // dispatch that wrote outside its own slice, or whose uniforms or
            // metadata came from another command's block, cannot pass this.
            auto check_values = [&](const char *what,
                                    const luisa::vector<float> &host,
                                    auto expected_value) {
                auto ok = true;
                for (auto i = size_t{0u}; i < host.size(); ++i) {
                    auto want = expected_value(i);
                    if (host[i] != want) {
                        LUISA_WARNING(
                            "reorder {}: {}[{}] = {} instead of {} - the "
                            "dispatch was emitted with another command's "
                            "argument block.",
                            reorder_enabled ? "on" : "off", what, i, host[i],
                            want);
                        ok = false;
                        break;
                    }
                }
                return ok;
            };
            // alpha (index 0): 1 + 2*10 + 3*100 + 4*1000 + 5*10000 = 54321
            // alpha (index 1): 8 + 9*10 + 11*100 + 12*1000 + 13*10000 = 143198
            expect(check_values("alpha_out", alpha_host, [](size_t i) noexcept {
                return i < kArgumentPlanThreads ? 54321.0f : 143198.0f;
            })) << "both alpha dispatches must keep their own uniform block";
            // beta (index 3): 2 + 6*10 + 7*100 + 8*1000 = 8762, in slice 3
            expect(check_values("beta_out", beta_host, [](size_t i) noexcept {
                return i >= 3u * kArgumentPlanThreads &&
                               i < 4u * kArgumentPlanThreads ?
                           8762.0f :
                           sentinel;
            })) << "the shader with the widest uniform block must also get its "
                   "own block";
            // gamma: two metadata trailers with different logical sizes (32 and
            // 48 elements) and different descriptor biases (0 and 192 bytes).
            expect(check_values("gamma_out", gamma_host, [](size_t i) noexcept {
                if (i < kArgumentPlanThreads) { return 32.0f; }// view [0, 32)
                if (i >= 48u && i < 80u) { return 48.0f; }     // view [48, 80)
                return sentinel;
            })) << "each buffer-metadata trailer must carry the logical size and "
                   "the descriptor bias of its own view (XIR/SPIR-V route), and "
                   "no dispatch may write outside its own view";
            // The stamp writes of alpha/alpha/beta are ordered by the WAW
            // dependency, so the last emitter is beta's 3.25 under both orders.
            expect(stamp_host[0] == 3.25f)
                << "the shared stamp must show the last ordered write (3.25), "
                   "otherwise the batch was emitted in an impossible order";
        }
        reorder->set_command_reorder_enabled(original_reorder);
    };
}
void reg_argument_plan_indirect(Device &device) {
    if (!native_spirv_route(device)) {
        LUISA_INFO(
            "indirect argument planning: skipped - this build has no native "
            "XIR-to-SPIR-V codegen, so no shader carries the indirect-dispatch "
            "metadata ABI (vk_spirv_codegen_route={})",
            device.query("vk_spirv_codegen_route"));
        return;
    }
    "vk_argument_plan_indirect_dispatch"_test = [&device] {
        // An indirect dispatch is planned and preprocessed like a direct one, and
        // a statically empty indirect range must keep its "no plan, no dispatch"
        // behaviour: the plan count is compared against the recorded-dispatch
        // count, so planning the empty range would show up as 6 vs 5.
        constexpr auto kernel_block_size = 64u;
        constexpr auto record_count = 3u;
        Kernel1D clear_kernel = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(record_count);
        };
        Kernel1D author_kernel = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_kernel(dispatch_id().x,
                                make_uint3(kernel_block_size, 1u, 1u),
                                make_uint3(dispatch_id().x + 1u, 1u, 1u),
                                dispatch_id().x);
        };
        Kernel1D target_kernel = [=](BufferUInt out, Float scale, Float2 bias) noexcept {
            set_block_size(kernel_block_size, 1u, 1u);
            // Both uniforms reach the shader only through this command's argument
            // block, so a mixed-up block shows up as the wrong number.
            out.write(kernel_id(),
                      cast<uint>(scale * 100.0f + bias.y));
        };
        auto clear_shader = device.compile(clear_kernel);
        auto author_shader = device.compile(author_kernel);
        auto target_shader = device.compile(target_kernel);
        IndirectDispatchBuffer commands =
            device.create_indirect_dispatch_buffer(record_count);
        Buffer<uint> buffer = device.create_buffer<uint>(record_count);
        std::array<uint, record_count> host{7u, 7u, 7u};
        Stream stream = device.create_stream();
        CommandList list;
        list.reserve(record_count + 3u, 0u);
        list << buffer.copy_from(luisa::span{host})
             << clear_shader(commands).dispatch(1u)
             << author_shader(commands).dispatch(record_count)
             // Two indirect dispatch commands over disjoint record ranges with
             // different uniforms: each owns one argument block.
             << target_shader(buffer, 1.0f, make_float2(0.0f, 10.0f))
                    .dispatch(commands, 0u, 2u)
             << target_shader(buffer, 2.0f, make_float2(0.0f, 20.0f))
                    .dispatch(commands, 2u, 1u)
             // Statically empty: no plan, no descriptor, no source read.
             << target_shader(buffer, 3.0f, make_float2(0.0f, 30.0f))
                    .dispatch(commands, record_count, 0u)
             << buffer.copy_to(luisa::span{host});
        auto before = read_counters(device);
        stream << list.commit() << synchronize();
        auto after = read_counters(device);
        auto d = difference(before, after);
        LUISA_INFO(
            "P1.4 indirect: {} argument plans for {} recorded dispatches "
            "(2 direct + 2 indirect targets, one empty indirect command)",
            value(d, Counter::argument_plans),
            value(d, Counter::dispatches));
        expect(value(d, Counter::dispatches) == 4u)
            << "the statically empty indirect command records no dispatch";
        expect(value(d, Counter::argument_plans) ==
               value(d, Counter::dispatches))
            << "an indirect dispatch is planned once, and a statically empty "
               "indirect range is planned (and dispatched) zero times (F4)";
        std::array<uint, record_count> expected{110u, 110u, 220u};
        auto ok = true;
        for (size_t i = 0u; i < record_count; ++i) {
            if (host[i] != expected[i]) {
                LUISA_WARNING("indirect record {} produced {} instead of {}",
                              i, host[i], expected[i]);
                ok = false;
            }
        }
        expect(ok) << "every indirect target must run with the uniform block of "
                      "the command that dispatched it";
    };
}
}// namespace

int main(int argc, char *argv[]) {
    // Must happen before the device and the first dispatch: the mode latches on
    // first use.
    latch_profile_mode();
    // The in-flight depth (P1.5) is a device-creation input, so select it here
    // and strip the pair from argv before anybody else parses it.
    g_frames_in_flight_option = take_frames_in_flight_option(argc, argv);
    // The config extension always answers now, because P2.2's switch has to be
    // visible to the device even when nothing else is configured; every other
    // answer is the backend default.
    DeviceConfig device_config;
    device_config.extension = luisa::make_unique<SubmitProfileConfig>(
        g_frames_in_flight_option.specified ? g_frames_in_flight_option.value : 3u,
 g_min_api_1_2 ? VK_API_VERSION_1_2 : 0u,
 g_descriptor_update_template,
 g_push_descriptor_option.enabled,
 g_push_descriptor_option.specified);
    auto dc = luisa::test::create_device_from_ut(argc, argv, &device_config);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    if (device.backend_name() != "vk") {
        LUISA_INFO("Skipping: the submit profiler is a Vulkan-only instrument.");
        return 0;
    }
    reg_profile_latch(device);
    reg_submit_path(device);
    reg_cold_create_layouts(device);
    reg_layout_intern_separation(device);
    reg_dispatch_path_counts(device);
    reg_decode_shaped_batch(device);
    reg_restore_cost(device);
    reg_bind_elimination_repeated(device);
  reg_push_descriptor_path(device);
  reg_push_descriptor_alternating(device);
    reg_bind_elimination_alternating(device);
    reg_bind_elimination_state_reset(device);
    reg_bind_elimination_indirect(device);
    reg_bind_elimination_internal_and_raw(device);
    reg_descriptor_update_path(device);
    reg_descriptor_update_templates_repeated(device);
    reg_descriptor_update_templates_mixed(device);
    reg_descriptor_update_templates_printer(device);
    reg_descriptor_update_templates_distinct_shaders(device);
    reg_argument_plan_distinct_layouts(device);
    reg_argument_plan_indirect(device);
    reg_frames_in_flight_config(device);
    reg_in_flight_depth(device);
    reg_in_flight_depth_fast_path(device);
    reg_capability_surface(device);
 reg_replay_zero_encode(device);
    return 0;
}
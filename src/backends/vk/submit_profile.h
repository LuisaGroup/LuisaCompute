// Counter-only instrumentation of the Vulkan submit path (plan P0.2).
//
// Rules this file must keep:
//  * Counter-only. Every hook below is a bare increment on a path the backend
//    already executed; no backend decision reads a counter, so enabling the
//    counters can never skip a bind, drop a barrier or consult a cache
//    differently.
//  * Off by default, and "off" is the *build* default. The whole mode exists
//    only when the private compile-time macro LUISA_VULKAN_PROFILE_SUBMIT is
//    non-zero. It is deliberately not an XMake or CMake option and there is no
//    environment variable: a normal development binary contains no counter, no
//    thread-local mirror, no clock read and no profiling state at all (see the
//    `#else` branch below - a disabled build only gets empty inline hooks that
//    an optimizer removes completely). Enabling it is a source-level decision
//    of whoever is measuring, not a supported build configuration; see the
//    comments next to the profile test/benchmark registrations in
//    src/tests/CMakeLists.txt and src/tests/xmake.lua.
//  * Cheap when on. The hot path (`record_*`) increments a thread-local
//    mirror; `flush()` folds that mirror into the process-global atomics once
//    per command batch, so a dispatch never touches an atomic or a mutex.
//    (The first implementation of this instrument used a global atomic per
//    call site and cost +30.6 % of host submission time; see plan.md §2.1.)
//  * Readable through the ordinary capability channel: `Device::query` forwards
//    every `vk_profile_*` key here (see `query`).
//
// Counters are cumulative for the process lifetime; consumers compare two
// snapshots (see the tripwire test src/tests/unit/runtime/test_vk_submit_profile.cpp,
// which is itself gated by the same macro).
//
// Scope notes (deliberate, so a count can be trusted):
//  * `dispatches` counts recorded `ShaderDispatchCommand`s in the stream
//    command encoder: one per command, not per `vkCmdDispatch` a
//    multi-dispatch command emits, and not the backend's internal prepare /
//    bindless-update / acceleration-update kernels. A statically empty
//    indirect dispatch is skipped before it is counted, because the backend
//    records nothing for it either.
//  * `queue_submits` counts the native queue-submit calls of the dispatch path
//    (Event::_signal): one per submitted batch, whichever entry point performed
//    it (plan P2.1). It does not count the timeline-only wait submissions and not
//    the swapchain present path, and it never counts work a
//    `VulkanDeviceConfigExt::signal_semaphore`/`wait_semaphore` callback
//    submitted itself, because no backend call happened in that case.
//  * `queue_submit2` / `queue_submit_legacy` split `queue_submits` by native
//    entry point (vkQueueSubmit2 vs vkQueueSubmit), so one dispatch batch reads
//    `1 / 0` on a synchronization2 device and `0 / 1` on a Vulkan 1.2 floor.
//    `queue_submits == queue_submit2 + queue_submit_legacy` always holds.
//  * `timeline_wait_entries` counts every in-flight throttle decision
//    (VulkanDeviceConfigExt::max_frames_in_flight, default 3) in
//    `Stream::dispatch` and its present() twin. `timeline_waits` counts the
//    decisions that actually blocked the host, `timeline_wait_skips` the ones
//    the GPU completion watermark proved unnecessary (plan P1.5); before P1.5
//    there was no fast path, so `timeline_waits` was the entry count of the
//    hard-coded depth-2 stall.
//  * `argument_plans` counts `plan_argument_block` calls, i.e. the planning of
//  one per-dispatch argument block. Since P1.4 a command that needs an argument
//  block is planned exactly once (the sizing pass retains the plan and the
//  preprocessing pass reuses it), so a batch of compute/indirect dispatches
//  reads `argument_plans == dispatches`; before P1.4 the sizing and the
//  preprocessing pass each planned every dispatch, so it read 2 x dispatches.
//  A DSL raster draw plans one block per draw too, and a native raster draw
//  plans none (it has no saved-argument ABI).
//  * The layout creation counters are shader-scoped cold-start creations
//    (compute / raster / ray-tracing / native shaders), which is what F1 is
//    about. The device-global bindless and sampler layouts that `Device`
//    creates once are not part of them.
//  * `descriptor_set_updates` counts vkUpdateDescriptorSets calls and nothing
//  else (plan P2.2). A per-dispatch set-0 update that went through
//  vkUpdateDescriptorSetWithTemplate is counted by
//  `descriptor_set_update_templates` instead, so
//  `descriptor_set_updates + descriptor_set_update_templates` is the number of
//  set 0 updates recorded and `descriptor_set_updates` stays the honest legacy
//  number. `descriptor_update_template_creates` counts
//  vkCreateDescriptorUpdateTemplate calls (one per distinct write shape per
//  shader, i.e. a property of the shape, never of a counter), and
//  `descriptor_update_template_fallbacks` counts the dispatch updates that
//  stayed on the legacy call although the device could have used a template;
//  `vk_profile_descriptor_update_template_fallback_reason` names the most
//  recent cause. A device without the extension/switch makes none of the three
//  template counters move at all.
//  * `pipeline_cache_creates` counts every vkCreatePipelineCache the backend
//  issues. Since P1.3 the DSL shader paths share one cache owned by the
//  Device, so a compile contributes nothing; the remaining per-compile caches
//  are the native-shader paths (separate lifetime/ABI contract) and the
//  short-lived merge sources of `pipeline_cache_merge_source_creates`. A
//  report can therefore distinguish the device cache from merge sources
//  instead of calling every cache "a pipeline cache".
//  * The `restore_*` counters describe the per-submission restore barrier of
//  `ResourceBarrier::restore_states` (plan F5, §5.1: what it costs was
//  unmeasured). `restore_calls` is one per dispatched batch and one per
//  present() call, `restore_host_ns` the host time inside the call, the three
//  struct counters what that call recorded, and
//  `restore_state_buffers`/`restore_state_texture_mips` the tracker entries it
//  consumed (the state a cross-submission barrier cache would have to keep).
//  They are a subset of the funnel counters: the restore's structs are also
//  counted by `barrier_*_barriers`, and `restore_host_ns` is inside
//  `host_encode_ns`. `barrier_all_commands_barriers` alone *under*-counts a
//  restore whose destination scope a `VulkanDeviceConfigExt::after_states()`
//  entry narrowed, which is why the restore has its own counters.
#pragma once

#include <cstdint>

#include <luisa/core/stl/string.h>

// The single gate of this instrument. Everything below - the counter record,
// the thread-local mirror, the host clock read, the process-global fold and the
// `vk_profile_*` query surface - exists only when this macro is non-zero, which
// is why `#if` (and never `#ifdef`) is used throughout. It defaults to 0 here so
// a translation unit that includes this header without anything else still
// compiles to the no-op form; turning it on means defining it to 1 for the
// whole build that is being measured (backend library and the test/benchmark
// binaries alike), which is a private, throw-away measurement build.
#if !defined(LUISA_VULKAN_PROFILE_SUBMIT)
#define LUISA_VULKAN_PROFILE_SUBMIT 0
#endif

// Only the enabled branch reads a clock, so the header of a disabled build does
// not even pull in <chrono>.
#if LUISA_VULKAN_PROFILE_SUBMIT
#include <chrono>
#endif

namespace lc::vk {

#if LUISA_VULKAN_PROFILE_SUBMIT

// Every counter the mode exposes. Trivially copyable on purpose: a
// `thread_local` instance must be constant-initialized.
struct SubmitProfileCounters {
    uint64_t dispatches{0u};
    uint64_t descriptor_set_allocations{0u};
    uint64_t descriptor_set_updates{0u};
    // Plan P2.2: descriptor updates issued through
    // vkUpdateDescriptorSetWithTemplate, the templates created for them, and
    // the updates that stayed on vkUpdateDescriptorSets although the device
    // could have used a template.
    uint64_t descriptor_set_update_templates{0u};
    uint64_t descriptor_update_template_creates{0u};
    uint64_t descriptor_update_template_fallbacks{0u};
    // Plan P2.3: set-0 binding through vkCmdPushDescriptorSetKHR. `calls`
    // counts the native calls (one per dispatched set 0); `shaders` /
    // `classic_shaders` are the creation-time decision, one increment per
    // shader created on a push-capable device (did its set-0 layout get the
    // push bit or not). A shader that fell back on such a device keeps that
    // layout for its whole lifetime, so those two are counts of shaders, never
    // of dispatches.
    uint64_t push_descriptor_calls{0u};
    uint64_t push_descriptor_shaders{0u};
    uint64_t push_descriptor_classic_shaders{0u};
    uint64_t descriptor_set_binds{0u};
    uint64_t pipeline_binds{0u};
    uint64_t argument_plans{0u};
    uint64_t barrier_calls{0u};
    uint64_t barrier_memory_barriers{0u};
    uint64_t barrier_buffer_barriers{0u};
    uint64_t barrier_image_barriers{0u};
    uint64_t barrier_all_commands_barriers{0u};
    uint64_t descriptor_set_layout_creates{0u};
    uint64_t pipeline_layout_creates{0u};
    uint64_t pipeline_cache_creates{0u};
    uint64_t pipeline_cache_merge_source_creates{0u};
    uint64_t pipeline_cache_merges{0u};
    uint64_t pipeline_creates{0u};
    uint64_t command_batches{0u};
    uint64_t queue_submits{0u};
    // Subset of `queue_submits` per native entry point: vkQueueSubmit2 on a
    // sync2-capable device, vkQueueSubmit on the Vulkan 1.2 path.
    uint64_t queue_submit2{0u};
    uint64_t queue_submit_legacy{0u};
    // Every throttle decision (plan P1.5): a submission that considered the
    // in-flight bound. `timeline_waits` counts only the decisions that really
    // blocked the host, `timeline_wait_skips` the ones the GPU completion
    // watermark proved unnecessary, so entries == waits + skips always holds.
    uint64_t timeline_wait_entries{0u};
    uint64_t timeline_waits{0u};
    uint64_t timeline_wait_skips{0u};
    // The per-submission restore barrier (plan F5/§5.1). `restore_calls` counts
    // ResourceBarrier::restore_states invocations (one per dispatched batch, one
    // per present() call), `restore_host_ns` the host time spent inside it, and
    // the three struct counters the barriers that call recorded. They are a
    // *subset* of the funnel counters above (a restore struct whose destination
    // scope the config ext narrowed is still a restore struct), so
    // `restore_buffer_barriers + restore_image_barriers + restore_memory_barriers
    // <= barrier_memory_barriers + barrier_buffer_barriers + barrier_image_barriers`
    // and `restore_host_ns <= host_encode_ns` must hold in any batch that ends
    // with a restore. `restore_state_buffers` / `restore_state_texture_mips`
    // count the tracker entries the restore consumed (the size of the state a
    // cross-submission barrier cache would have to keep alive).
    uint64_t restore_calls{0u};
    uint64_t restore_host_ns{0u};
    uint64_t restore_memory_barriers{0u};
    uint64_t restore_buffer_barriers{0u};
    uint64_t restore_image_barriers{0u};
    uint64_t restore_state_buffers{0u};
    uint64_t restore_state_texture_mips{0u};
    uint64_t host_encode_ns{0u};
    uint64_t host_submit_ns{0u};
    // ---- replayable recording (plan P4) ------------------------------------
    // `replay_records` counts one successful record() (one recorded, *never
    // submitted* batch; its own host cost is `replay_record_host_ns`), and
    // `replay_submits` one replay() submission. A replay must move nothing else
    // of the encode/plan/descriptor families - that absence is what the
    // zero-work tripwire asserts - while `queue_submits` still counts its one
    // native submission. `replay_rejections` counts every refused record(),
    // `replay_update_refusals` every update() call (v1 always refuses),
    // `replay_invalidations` every graph a lifetime event invalidated and
    // `replay_releases` every recorded VkCommandBuffer the backend freed, so a
    // report can tell "no graph is live" from "graphs leak".
    uint64_t replay_records{0u};
    uint64_t replay_record_host_ns{0u};
    uint64_t replay_submits{0u};
    uint64_t replay_rejections{0u};
    uint64_t replay_update_refusals{0u};
    uint64_t replay_invalidations{0u};
    uint64_t replay_releases{0u};

    [[nodiscard]] constexpr bool empty() const noexcept {
        return dispatches == 0u &&
               descriptor_set_allocations == 0u &&
               descriptor_set_updates == 0u &&
               descriptor_set_update_templates == 0u &&
               descriptor_update_template_creates == 0u &&
               descriptor_update_template_fallbacks == 0u &&
               push_descriptor_calls == 0u &&
               push_descriptor_shaders == 0u &&
               push_descriptor_classic_shaders == 0u &&
               descriptor_set_binds == 0u &&
               pipeline_binds == 0u &&
               argument_plans == 0u &&
               barrier_calls == 0u &&
               barrier_memory_barriers == 0u &&
               barrier_buffer_barriers == 0u &&
               barrier_image_barriers == 0u &&
               barrier_all_commands_barriers == 0u &&
               descriptor_set_layout_creates == 0u &&
               pipeline_layout_creates == 0u &&
               pipeline_cache_creates == 0u &&
               pipeline_cache_merge_source_creates == 0u &&
               pipeline_cache_merges == 0u &&
               pipeline_creates == 0u &&
               command_batches == 0u &&
               queue_submits == 0u &&
               queue_submit2 == 0u &&
               queue_submit_legacy == 0u &&
               timeline_wait_entries == 0u &&
               timeline_waits == 0u &&
               timeline_wait_skips == 0u &&
               restore_calls == 0u &&
               restore_host_ns == 0u &&
               restore_memory_barriers == 0u &&
               restore_buffer_barriers == 0u &&
               restore_image_barriers == 0u &&
               restore_state_buffers == 0u &&
               restore_state_texture_mips == 0u &&
               host_encode_ns == 0u &&
               host_submit_ns == 0u &&
               replay_records == 0u &&
               replay_record_host_ns == 0u &&
               replay_submits == 0u &&
               replay_rejections == 0u &&
               replay_update_refusals == 0u &&
               replay_invalidations == 0u &&
               replay_releases == 0u;
    }
};

namespace submit_profile {

// Compile-time true: in this build the instrument *is* the code below, so the
// hooks never test a runtime switch. Kept as a callable anyway, because the
// unchanged backend call sites use it to skip a whole measurement block (see
// `detail::profile_pipeline_barrier` in command_buffer_sync.h).
[[nodiscard]] constexpr bool enabled() noexcept { return true; }

// Per-thread mirror the hot path increments. Non-atomic by design: only the
// owning thread ever touches it, and `flush()` folds it under the global lock.
inline thread_local SubmitProfileCounters tls_counters{};

[[nodiscard]] inline SubmitProfileCounters &mirror() noexcept {
    return tls_counters;
}

inline void record_dispatch() noexcept {
    ++tls_counters.dispatches;
}

inline void record_descriptor_set_allocation() noexcept {
    ++tls_counters.descriptor_set_allocations;
}

// One vkUpdateDescriptorSets call and nothing else (plan P2.2 keeps this
// counter meaning exactly the legacy entry point).
inline void record_descriptor_set_update() noexcept {
    ++tls_counters.descriptor_set_updates;
}

// One vkUpdateDescriptorSetWithTemplate call for a per-dispatch set-0 update.
inline void record_descriptor_set_update_template() noexcept {
    ++tls_counters.descriptor_set_update_templates;
}

// One vkCmdPushDescriptorSetKHR (plan P2.3). Recorded by the single funnel
// detail::push_descriptor_set, so it counts the native calls a batch really
// recorded, not the dispatches that asked to push.
inline void record_push_descriptor_call() noexcept {
    ++tls_counters.push_descriptor_calls;
}

// The creation-time set-0 decision of one shader on a push-capable device (plan
// P2.3): one increment per shader, never per dispatch. `pushed` is the layout
// the shader was created with and keeps for its whole lifetime.
inline void record_push_descriptor_shader(bool pushed) noexcept {
    if (pushed) {
        ++tls_counters.push_descriptor_shaders;
    } else {
        ++tls_counters.push_descriptor_classic_shaders;
    }
}

// One vkCreateDescriptorUpdateTemplate. A template is created once per
// distinct write shape of a shader, so this counts shapes, not dispatches.
inline void record_descriptor_update_template_create() noexcept {
    ++tls_counters.descriptor_update_template_creates;
}

// The reason of the most recent template fallback on this thread. Kept in the
// thread-local mirror like every other hot-path observation, so recording a
// fallback costs no atomic and no lock; `flush()` folds it into the global slot.
inline thread_local const char *tls_fallback_reason{nullptr};

// A set-0 update that used the legacy call although the device supports
// templates, because this particular write list could not be represented (see
// descriptor_update_template.h). `reason` is one of that file's static status
// strings; the driver-facing path stays the legacy one.
inline void record_descriptor_update_template_fallback(const char *reason) noexcept {
    ++tls_counters.descriptor_update_template_fallbacks;
    tls_fallback_reason = reason == nullptr ? "unknown" : reason;
}

// The reason reported by the most recent fallback, or "" when none happened.
[[nodiscard]] luisa::string_view
descriptor_update_template_fallback_reason() noexcept;

inline void record_descriptor_set_bind() noexcept {
    ++tls_counters.descriptor_set_binds;
}

inline void record_pipeline_bind() noexcept {
    ++tls_counters.pipeline_binds;
}

inline void record_argument_plan() noexcept {
    ++tls_counters.argument_plans;
}

inline void record_barrier_call() noexcept {
    ++tls_counters.barrier_calls;
}

inline void record_memory_barriers(uint64_t count) noexcept {
    tls_counters.barrier_memory_barriers += count;
}

inline void record_buffer_barriers(uint64_t count) noexcept {
    tls_counters.barrier_buffer_barriers += count;
}

inline void record_image_barriers(uint64_t count) noexcept {
    tls_counters.barrier_image_barriers += count;
}

inline void record_all_commands_barriers(uint64_t count) noexcept {
    tls_counters.barrier_all_commands_barriers += count;
}

inline void record_descriptor_set_layout_create() noexcept {
    ++tls_counters.descriptor_set_layout_creates;
}

inline void record_pipeline_layout_create() noexcept {
    ++tls_counters.pipeline_layout_creates;
}

inline void record_pipeline_cache_create() noexcept {
    ++tls_counters.pipeline_cache_creates;
}

// Subset of `pipeline_cache_creates`: a short-lived cache that exists only to be
// handed to vkMergePipelineCaches (the P1.3 per-shader seed path). Counted
// separately so a report can tell the one device cache apart from merge sources.
inline void record_pipeline_cache_merge_source_create() noexcept {
    ++tls_counters.pipeline_cache_merge_source_creates;
}

// Successful vkMergePipelineCaches calls: the number of per-shader PSO seeds
// that really reached the device cache.
inline void record_pipeline_cache_merge() noexcept {
    ++tls_counters.pipeline_cache_merges;
}

inline void record_pipeline_create() noexcept {
    ++tls_counters.pipeline_creates;
}

inline void record_command_batch() noexcept {
    ++tls_counters.command_batches;
}

// One native queue submission of the dispatch path. `sync2` says which entry
// point performed it (plan P2.1): vkQueueSubmit2 or the legacy vkQueueSubmit. A
// submission a config-ext callback performed is never recorded, because the
// backend made no native call.
inline void record_queue_submit(bool sync2) noexcept {
    ++tls_counters.queue_submits;
    if (sync2) {
        ++tls_counters.queue_submit2;
    } else {
        ++tls_counters.queue_submit_legacy;
    }
}

// One entry into the in-flight throttle (plan P1.5): the submission had to
// consider the bound, whether or not it then blocked.
inline void record_timeline_wait_entry() noexcept {
    ++tls_counters.timeline_wait_entries;
}

// The throttle really blocked the host (the target timeline value had not
// completed yet).
inline void record_timeline_wait() noexcept {
    ++tls_counters.timeline_waits;
}

// The throttle was entered but the GPU had already completed the target, so the
// host waited for nothing. Recorded separately so a test can tell "the bound is
// never reached" (entries == 0) from "the bound is reached and always
// unnecessary" (skips == entries).
inline void record_timeline_wait_skip() noexcept {
    ++tls_counters.timeline_wait_skips;
}

// One ResourceBarrier::restore_states invocation (plan F5/§5.1). Recorded on
// entry, so a restore that finds no state to restore is still counted as the
// per-submission cost it is.
inline void record_restore_call() noexcept {
    ++tls_counters.restore_calls;
}

// The barrier structs that restore_states recorded, by kind. A restore struct
// whose destination scope the config ext narrowed (`after_states`) is *not* an
// ALL_COMMANDS struct, which is exactly why the restore needs its own counters
// instead of reusing `barrier_all_commands_barriers`.
inline void record_restore_barriers(uint64_t memory, uint64_t buffers,
                                    uint64_t images) noexcept {
    tls_counters.restore_memory_barriers += memory;
    tls_counters.restore_buffer_barriers += buffers;
    tls_counters.restore_image_barriers += images;
}

// The tracker entries the restore consumed: tracked buffers and tracked texture
// mips. This is the size of the state a cross-submission barrier cache would
// have to keep alive, measured at the point where it is dropped.
inline void record_restore_state(uint64_t buffers, uint64_t texture_mips) noexcept {
    tls_counters.restore_state_buffers += buffers;
    tls_counters.restore_state_texture_mips += texture_mips;
}

inline void record_host_restore_ns(uint64_t ns) noexcept {
    tls_counters.restore_host_ns += ns;
}

inline void record_host_encode_ns(uint64_t ns) noexcept {
    tls_counters.host_encode_ns += ns;
}

inline void record_host_submit_ns(uint64_t ns) noexcept {
    tls_counters.host_submit_ns += ns;
}

// ---- replayable recording (plan P4). Counter-only, like every hook above:
// no backend decision reads any of these, so enabling them cannot change what
// is recorded or replayed.
// One successful record(): a batch that was encoded into a persistent command
// buffer and not submitted. `record_host_ns` is the host cost of that encode and
// is deliberately *not* folded into `host_encode_ns` (which stays the ordinary
// per-batch encode window, so its us/batch ratio keeps its meaning).
inline void record_replay_record(uint64_t record_host_ns) noexcept {
    ++tls_counters.replay_records;
    tls_counters.replay_record_host_ns += record_host_ns;
}

// One replay() submission. The native queue submission itself is counted by
// `record_queue_submit` inside Event::_signal, exactly like an ordinary batch.
inline void record_replay_submit() noexcept {
    ++tls_counters.replay_submits;
}

// One record() that was refused with a typed reason.
inline void record_replay_rejection() noexcept {
    ++tls_counters.replay_rejections;
}

// One update() call. v1 refuses every one of them (a recording encodes its
// uniform bytes and descriptor contents), so this count equals the number of
// update() calls a caller made.
inline void record_replay_update_refusal() noexcept {
    ++tls_counters.replay_update_refusals;
}

inline void record_replay_invalidations(uint64_t count) noexcept {
    tls_counters.replay_invalidations += count;
}

// One recorded VkCommandBuffer freed (graph destruction or device teardown).
// `replay_records - replay_releases` is the number of live recordings.
inline void record_replay_release() noexcept {
    ++tls_counters.replay_releases;
}

// Host timing helpers. `timing_begin` and every `timing_end_*` are the only
// clock reads of the instrument, so a batch that is not timed touches no clock.
[[nodiscard]] inline int64_t timing_begin() noexcept {
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

[[nodiscard]] inline uint64_t timing_elapsed_ns(int64_t begin) noexcept {
    auto end = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    return end > begin ? static_cast<uint64_t>(end - begin) : 0u;
}

inline void timing_end_encode(int64_t begin) noexcept {
    record_host_encode_ns(timing_elapsed_ns(begin));
}

inline void timing_end_submit(int64_t begin) noexcept {
    record_host_submit_ns(timing_elapsed_ns(begin));
}

inline void timing_end_restore(int64_t begin) noexcept {
    record_host_restore_ns(timing_elapsed_ns(begin));
}

// Fold the calling thread's mirror into the process-global totals. Called once
// per command batch from `Stream::dispatch` and, defensively, before every
// query.
void flush() noexcept;

// Fold the calling thread's mirror, then read a consistent copy of the global
// totals. Another thread's not-yet-flushed mirror is not included: counters are
// per-batch, so a query observes every batch that completed before it.
[[nodiscard]] SubmitProfileCounters snapshot() noexcept;

// One-line human-readable dump: counts, per-dispatch ratios and host µs.
[[nodiscard]] luisa::string report(const SubmitProfileCounters &counters) noexcept;

// Answers a `vk_profile_*` capability key, or an empty string for anything it
// does not know (the caller then falls through to its own keys).
[[nodiscard]] luisa::string query(luisa::string_view key) noexcept;

}// namespace submit_profile

#else// !LUISA_VULKAN_PROFILE_SUBMIT

namespace submit_profile {

// Disabled build: the instrument does not exist. Every hook keeps the exact
// signature the (unchanged) backend call sites use but is an empty `inline`
// function, so an ordinary binary carries no counter record, no thread-local
// mirror, no atomic, no clock read, no global table and no report/query
// surface - a standard optimizer removes the calls entirely, and nothing here
// can be observed at run time. The parameters are deliberately unnamed: there
// is nothing to observe, and a named-but-unused parameter would only be a
// warning.

// Compile-time false: this build cannot be turned into a profiling build by any
// runtime input. Call sites that guard a whole measurement block with
// `if (!enabled()) { return; }` therefore compile the block away.
[[nodiscard]] constexpr bool enabled() noexcept { return false; }

inline void record_dispatch() noexcept {}
inline void record_descriptor_set_allocation() noexcept {}
inline void record_descriptor_set_update() noexcept {}
inline void record_descriptor_set_update_template() noexcept {}
inline void record_push_descriptor_call() noexcept {}
inline void record_push_descriptor_shader(bool /*pushed*/) noexcept {}
inline void record_descriptor_update_template_create() noexcept {}
inline void record_descriptor_update_template_fallback(const char * /*reason*/) noexcept {}
inline void record_descriptor_set_bind() noexcept {}
inline void record_pipeline_bind() noexcept {}
inline void record_argument_plan() noexcept {}
inline void record_barrier_call() noexcept {}
inline void record_memory_barriers(uint64_t /*count*/) noexcept {}
inline void record_buffer_barriers(uint64_t /*count*/) noexcept {}
inline void record_image_barriers(uint64_t /*count*/) noexcept {}
inline void record_all_commands_barriers(uint64_t /*count*/) noexcept {}
inline void record_descriptor_set_layout_create() noexcept {}
inline void record_pipeline_layout_create() noexcept {}
inline void record_pipeline_cache_create() noexcept {}
inline void record_pipeline_cache_merge_source_create() noexcept {}
inline void record_pipeline_cache_merge() noexcept {}
inline void record_pipeline_create() noexcept {}
inline void record_command_batch() noexcept {}
inline void record_queue_submit(bool /*sync2*/) noexcept {}
inline void record_timeline_wait_entry() noexcept {}
inline void record_timeline_wait() noexcept {}
inline void record_timeline_wait_skip() noexcept {}
inline void record_restore_call() noexcept {}
inline void record_restore_barriers(uint64_t /*memory*/, uint64_t /*buffers*/,
                                    uint64_t /*images*/) noexcept {}
inline void record_restore_state(uint64_t /*buffers*/,
                                 uint64_t /*texture_mips*/) noexcept {}
inline void record_host_restore_ns(uint64_t /*ns*/) noexcept {}
inline void record_host_encode_ns(uint64_t /*ns*/) noexcept {}
inline void record_host_submit_ns(uint64_t /*ns*/) noexcept {}
inline void record_replay_record(uint64_t /*record_host_ns*/) noexcept {}
inline void record_replay_submit() noexcept {}
inline void record_replay_rejection() noexcept {}
inline void record_replay_update_refusal() noexcept {}
inline void record_replay_invalidations(uint64_t /*count*/) noexcept {}
inline void record_replay_release() noexcept {}

// No clock is read: a begin stamp is a constant 0 and every elapsed-time helper
// returns 0, so the call sites' timing windows vanish. Nothing is folded and
// nothing is queried either.
[[nodiscard]] constexpr int64_t timing_begin() noexcept { return 0; }
[[nodiscard]] constexpr uint64_t timing_elapsed_ns(int64_t /*begin*/) noexcept { return 0u; }
inline void timing_end_encode(int64_t /*begin*/) noexcept {}
inline void timing_end_submit(int64_t /*begin*/) noexcept {}
inline void timing_end_restore(int64_t /*begin*/) noexcept {}

inline void flush() noexcept {}

// No reason is ever recorded, so the answer is always empty.
[[nodiscard]] inline luisa::string_view
descriptor_update_template_fallback_reason() noexcept { return {}; }

// `Device::query` forwards every `vk_profile_*` key here; in a disabled build
// there is nothing to report, so the answer is the empty string and the caller
// reports the key as unknown instead of inventing a number.
[[nodiscard]] inline luisa::string query(luisa::string_view /*key*/) noexcept {
    return {};
}

}// namespace submit_profile

#endif// LUISA_VULKAN_PROFILE_SUBMIT

}// namespace lc::vk
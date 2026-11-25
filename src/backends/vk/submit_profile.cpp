// Implementation of the counter-only Vulkan submit profiler (plan P0.2).
// The hot path lives inline in submit_profile.h; everything here runs once per
// command batch (flush) or on demand (snapshot / report / query).
//
// The whole file is behind the instrument's private compile-time gate (see
// submit_profile.h): with LUISA_VULKAN_PROFILE_SUBMIT at 0 - the default of every
// ordinary build - this translation unit contributes nothing but the include
// above, so no binary links a global counter table, a fold mutex, a fallback
// reason slot or the report/query formatting. There is no environment variable
// and no build option to change that; enabling it is a source/compiler-level
// decision of whoever measures (see src/tests/CMakeLists.txt and
// src/tests/xmake.lua for why no build configuration is offered on purpose).
#include "submit_profile.h"

#if LUISA_VULKAN_PROFILE_SUBMIT

#include <atomic>
#include <mutex>

#include <luisa/core/stl/format.h>

namespace lc::vk::submit_profile {

namespace {

struct GlobalCounters {
    std::atomic<uint64_t> dispatches{0u};
    std::atomic<uint64_t> descriptor_set_allocations{0u};
    std::atomic<uint64_t> descriptor_set_updates{0u};
    std::atomic<uint64_t> descriptor_set_update_templates{0u};
    std::atomic<uint64_t> descriptor_update_template_creates{0u};
    std::atomic<uint64_t> descriptor_update_template_fallbacks{0u};
    std::atomic<uint64_t> push_descriptor_calls{0u};
    std::atomic<uint64_t> push_descriptor_shaders{0u};
    std::atomic<uint64_t> push_descriptor_classic_shaders{0u};
    std::atomic<uint64_t> descriptor_set_binds{0u};
    std::atomic<uint64_t> pipeline_binds{0u};
    std::atomic<uint64_t> argument_plans{0u};
    std::atomic<uint64_t> barrier_calls{0u};
    std::atomic<uint64_t> barrier_memory_barriers{0u};
    std::atomic<uint64_t> barrier_buffer_barriers{0u};
    std::atomic<uint64_t> barrier_image_barriers{0u};
    std::atomic<uint64_t> barrier_all_commands_barriers{0u};
    std::atomic<uint64_t> descriptor_set_layout_creates{0u};
    std::atomic<uint64_t> pipeline_layout_creates{0u};
    std::atomic<uint64_t> pipeline_cache_creates{0u};
    std::atomic<uint64_t> pipeline_cache_merge_source_creates{0u};
    std::atomic<uint64_t> pipeline_cache_merges{0u};
    std::atomic<uint64_t> pipeline_creates{0u};
    std::atomic<uint64_t> command_batches{0u};
    std::atomic<uint64_t> queue_submits{0u};
    std::atomic<uint64_t> queue_submit2{0u};
    std::atomic<uint64_t> queue_submit_legacy{0u};
    std::atomic<uint64_t> timeline_wait_entries{0u};
    std::atomic<uint64_t> timeline_waits{0u};
    std::atomic<uint64_t> timeline_wait_skips{0u};
    std::atomic<uint64_t> restore_calls{0u};
    std::atomic<uint64_t> restore_host_ns{0u};
    std::atomic<uint64_t> restore_memory_barriers{0u};
    std::atomic<uint64_t> restore_buffer_barriers{0u};
    // Phase 3B (opt D): tracked buffers the restore published through its single
    // coalesced VkMemoryBarrier2 instead of one buffer struct each.
    std::atomic<uint64_t> restore_coalesced_buffers{0u};
    std::atomic<uint64_t> restore_image_barriers{0u};
    std::atomic<uint64_t> restore_state_buffers{0u};
    std::atomic<uint64_t> restore_state_texture_mips{0u};
    std::atomic<uint64_t> host_encode_ns{0u};
    std::atomic<uint64_t> host_submit_ns{0u};
    std::atomic<uint64_t> replay_records{0u};
    std::atomic<uint64_t> replay_record_host_ns{0u};
    std::atomic<uint64_t> replay_submits{0u};
    std::atomic<uint64_t> replay_rejections{0u};
    std::atomic<uint64_t> replay_update_refusals{0u};
    std::atomic<uint64_t> replay_invalidations{0u};
    std::atomic<uint64_t> replay_releases{0u};
};

// Trivially destructible aggregate: never destroyed, so a late hook cannot touch
// a destroyed atomic during static teardown.
GlobalCounters &globals() noexcept {
    static GlobalCounters value;
    return value;
}

// Why the most recent per-dispatch set-0 update stayed on the legacy call
// (plan P2.2). Only ever written on that path, and the value is a pointer to one
// of descriptor_update_template.h's static status strings, so recording it costs
// one relaxed store and needs no allocation or lock.
std::atomic<const char *> &fallback_reason_slot() noexcept {
    static std::atomic<const char *> value{nullptr};
    return value;
}

// Serializes "fold a mirror into the globals" against "read a consistent copy",
// so a snapshot can never mix a half-folded mirror with a folded one.
std::mutex &fold_mutex() noexcept {
    static std::mutex value;
    return value;
}

void add(std::atomic<uint64_t> &dst, uint64_t &src) noexcept {
    auto value = src;
    src = 0u;
    if (value != 0u) {
        dst.fetch_add(value, std::memory_order_relaxed);
    }
}

// Requires fold_mutex().
void fold(SubmitProfileCounters &c) noexcept {
    auto &g = globals();
    add(g.dispatches, c.dispatches);
    add(g.descriptor_set_allocations, c.descriptor_set_allocations);
    add(g.descriptor_set_updates, c.descriptor_set_updates);
    add(g.descriptor_set_update_templates, c.descriptor_set_update_templates);
    add(g.descriptor_update_template_creates, c.descriptor_update_template_creates);
    add(g.descriptor_update_template_fallbacks, c.descriptor_update_template_fallbacks);
    add(g.push_descriptor_calls, c.push_descriptor_calls);
    add(g.push_descriptor_shaders, c.push_descriptor_shaders);
    add(g.push_descriptor_classic_shaders, c.push_descriptor_classic_shaders);
    add(g.descriptor_set_binds, c.descriptor_set_binds);
    add(g.pipeline_binds, c.pipeline_binds);
    add(g.argument_plans, c.argument_plans);
    add(g.barrier_calls, c.barrier_calls);
    add(g.barrier_memory_barriers, c.barrier_memory_barriers);
    add(g.barrier_buffer_barriers, c.barrier_buffer_barriers);
    add(g.barrier_image_barriers, c.barrier_image_barriers);
    add(g.barrier_all_commands_barriers, c.barrier_all_commands_barriers);
    add(g.descriptor_set_layout_creates, c.descriptor_set_layout_creates);
    add(g.pipeline_layout_creates, c.pipeline_layout_creates);
    add(g.pipeline_cache_creates, c.pipeline_cache_creates);
    add(g.pipeline_cache_merge_source_creates, c.pipeline_cache_merge_source_creates);
    add(g.pipeline_cache_merges, c.pipeline_cache_merges);
    add(g.pipeline_creates, c.pipeline_creates);
    add(g.command_batches, c.command_batches);
    add(g.queue_submits, c.queue_submits);
    add(g.queue_submit2, c.queue_submit2);
    add(g.queue_submit_legacy, c.queue_submit_legacy);
    add(g.timeline_wait_entries, c.timeline_wait_entries);
    add(g.timeline_waits, c.timeline_waits);
    add(g.timeline_wait_skips, c.timeline_wait_skips);
    add(g.restore_calls, c.restore_calls);
    add(g.restore_host_ns, c.restore_host_ns);
    add(g.restore_memory_barriers, c.restore_memory_barriers);
    add(g.restore_buffer_barriers, c.restore_buffer_barriers);
    add(g.restore_coalesced_buffers, c.restore_coalesced_buffers);
    add(g.restore_image_barriers, c.restore_image_barriers);
    add(g.restore_state_buffers, c.restore_state_buffers);
    add(g.restore_state_texture_mips, c.restore_state_texture_mips);
    add(g.host_encode_ns, c.host_encode_ns);
    add(g.host_submit_ns, c.host_submit_ns);
    add(g.replay_records, c.replay_records);
    add(g.replay_record_host_ns, c.replay_record_host_ns);
    add(g.replay_submits, c.replay_submits);
    add(g.replay_rejections, c.replay_rejections);
    add(g.replay_update_refusals, c.replay_update_refusals);
    add(g.replay_invalidations, c.replay_invalidations);
    add(g.replay_releases, c.replay_releases);
    // The fallback reason of the batch being folded. Sticky on purpose: the
    // query reports the most recent cause, not a per-batch one.
    if (auto reason = tls_fallback_reason; reason != nullptr) {
        fallback_reason_slot().store(reason, std::memory_order_relaxed);
    }
}

[[nodiscard]] luisa::string count_string(uint64_t value) noexcept {
    return luisa::format("{}", value);
}

[[nodiscard]] double per_dispatch(uint64_t value,
                                  uint64_t dispatches) noexcept {
    if (dispatches == 0u) { return 0.0; }
    return static_cast<double>(value) / static_cast<double>(dispatches);
}

[[nodiscard]] double us_per(uint64_t ns, uint64_t count) noexcept {
    if (count == 0u) { return 0.0; }
    return static_cast<double>(ns) / 1000.0 / static_cast<double>(count);
}

}// namespace

luisa::string_view descriptor_update_template_fallback_reason() noexcept {
    auto value = fallback_reason_slot().load(std::memory_order_relaxed);
    return value == nullptr ? luisa::string_view{} : luisa::string_view{value};
}

void flush() noexcept {
    auto &mine = mirror();
    if (mine.empty()) { return; }
    std::lock_guard lock{fold_mutex()};
    fold(mine);
}

SubmitProfileCounters snapshot() noexcept {
    // Fold the querying thread's own mirror first, then read the totals under
    // the same lock so no other thread can be in the middle of a fold.
    flush();
    SubmitProfileCounters result;
    auto &g = globals();
    std::lock_guard lock{fold_mutex()};
    result.dispatches = g.dispatches.load(std::memory_order_relaxed);
    result.descriptor_set_allocations = g.descriptor_set_allocations.load(std::memory_order_relaxed);
    result.descriptor_set_updates = g.descriptor_set_updates.load(std::memory_order_relaxed);
    result.descriptor_set_update_templates = g.descriptor_set_update_templates.load(std::memory_order_relaxed);
    result.descriptor_update_template_creates = g.descriptor_update_template_creates.load(std::memory_order_relaxed);
    result.descriptor_update_template_fallbacks = g.descriptor_update_template_fallbacks.load(std::memory_order_relaxed);
    result.push_descriptor_calls = g.push_descriptor_calls.load(std::memory_order_relaxed);
    result.push_descriptor_shaders = g.push_descriptor_shaders.load(std::memory_order_relaxed);
    result.push_descriptor_classic_shaders = g.push_descriptor_classic_shaders.load(std::memory_order_relaxed);
    result.descriptor_set_binds = g.descriptor_set_binds.load(std::memory_order_relaxed);
    result.pipeline_binds = g.pipeline_binds.load(std::memory_order_relaxed);
    result.argument_plans = g.argument_plans.load(std::memory_order_relaxed);
    result.barrier_calls = g.barrier_calls.load(std::memory_order_relaxed);
    result.barrier_memory_barriers = g.barrier_memory_barriers.load(std::memory_order_relaxed);
    result.barrier_buffer_barriers = g.barrier_buffer_barriers.load(std::memory_order_relaxed);
    result.barrier_image_barriers = g.barrier_image_barriers.load(std::memory_order_relaxed);
    result.barrier_all_commands_barriers = g.barrier_all_commands_barriers.load(std::memory_order_relaxed);
    result.descriptor_set_layout_creates = g.descriptor_set_layout_creates.load(std::memory_order_relaxed);
    result.pipeline_layout_creates = g.pipeline_layout_creates.load(std::memory_order_relaxed);
    result.pipeline_cache_creates = g.pipeline_cache_creates.load(std::memory_order_relaxed);
    result.pipeline_cache_merge_source_creates = g.pipeline_cache_merge_source_creates.load(std::memory_order_relaxed);
    result.pipeline_cache_merges = g.pipeline_cache_merges.load(std::memory_order_relaxed);
    result.pipeline_creates = g.pipeline_creates.load(std::memory_order_relaxed);
    result.command_batches = g.command_batches.load(std::memory_order_relaxed);
    result.queue_submits = g.queue_submits.load(std::memory_order_relaxed);
    result.queue_submit2 = g.queue_submit2.load(std::memory_order_relaxed);
    result.queue_submit_legacy = g.queue_submit_legacy.load(std::memory_order_relaxed);
    result.timeline_wait_entries = g.timeline_wait_entries.load(std::memory_order_relaxed);
    result.timeline_waits = g.timeline_waits.load(std::memory_order_relaxed);
    result.timeline_wait_skips = g.timeline_wait_skips.load(std::memory_order_relaxed);
    result.restore_calls = g.restore_calls.load(std::memory_order_relaxed);
    result.restore_host_ns = g.restore_host_ns.load(std::memory_order_relaxed);
    result.restore_memory_barriers = g.restore_memory_barriers.load(std::memory_order_relaxed);
    result.restore_buffer_barriers = g.restore_buffer_barriers.load(std::memory_order_relaxed);
    result.restore_coalesced_buffers = g.restore_coalesced_buffers.load(std::memory_order_relaxed);
    result.restore_image_barriers = g.restore_image_barriers.load(std::memory_order_relaxed);
    result.restore_state_buffers = g.restore_state_buffers.load(std::memory_order_relaxed);
    result.restore_state_texture_mips = g.restore_state_texture_mips.load(std::memory_order_relaxed);
    result.host_encode_ns = g.host_encode_ns.load(std::memory_order_relaxed);
    result.host_submit_ns = g.host_submit_ns.load(std::memory_order_relaxed);
    result.replay_records = g.replay_records.load(std::memory_order_relaxed);
    result.replay_record_host_ns = g.replay_record_host_ns.load(std::memory_order_relaxed);
    result.replay_submits = g.replay_submits.load(std::memory_order_relaxed);
    result.replay_rejections = g.replay_rejections.load(std::memory_order_relaxed);
    result.replay_update_refusals = g.replay_update_refusals.load(std::memory_order_relaxed);
    result.replay_invalidations = g.replay_invalidations.load(std::memory_order_relaxed);
    result.replay_releases = g.replay_releases.load(std::memory_order_relaxed);
    return result;
}

luisa::string report(const SubmitProfileCounters &c) noexcept {
    return luisa::format(
        "enabled={} dispatches={} batches={} submits={} (submit2={} legacy={}) "
        "timeline_waits={} (entries={} skips={}) | "
        "per-dispatch: desc_set_alloc={:.2f} desc_set_update={:.2f} "
        "desc_set_update_template={:.2f} push_descriptor={:.2f} "
          "desc_set_bind={:.2f} "
        "pipeline_bind={:.2f} arg_plan={:.2f} "
        "barrier_calls={:.2f} | "
        "barrier_structs: memory={} buffer={} image={} all_commands={} | "
        "descriptor updates: legacy={} template={} template_creates={} "
        "template_fallbacks={} | "
        "push descriptors: calls={} shaders={} classic_shaders={} | "
        "create: dsl_layouts={} pipeline_layouts={} pipeline_caches={} "
        "pipeline_cache_merge_sources={} pipeline_cache_merges={} "
        "pipelines={} | "
"host: encode={:.1f}us/batch submit={:.1f}us/submit | "
            "restore: calls={} host={:.2f}us/call structs(mem={} buf={} img={}) "
            "state(buf={} mips={}) | "
            // Plan P4: a replay must show one submit and nothing else; the record
            // that produced it is the only place encode/plan work appears.
            "replay: records={} rejects={} submits={} update_refusals={} "
            "invalidations={} releases={} record_host={:.1f}us/record",
        // The mode is a compile-time constant in this build (this file does not
        // exist otherwise), so the report always says enabled=1.
        1,
        c.dispatches, c.command_batches, c.queue_submits,
        c.queue_submit2, c.queue_submit_legacy,
        c.timeline_waits, c.timeline_wait_entries, c.timeline_wait_skips,
        per_dispatch(c.descriptor_set_allocations, c.dispatches),
        per_dispatch(c.descriptor_set_updates, c.dispatches),
        per_dispatch(c.descriptor_set_update_templates, c.dispatches),
        per_dispatch(c.push_descriptor_calls, c.dispatches),
        per_dispatch(c.descriptor_set_binds, c.dispatches),
        per_dispatch(c.pipeline_binds, c.dispatches),
        per_dispatch(c.argument_plans, c.dispatches),
        per_dispatch(c.barrier_calls, c.dispatches),
        c.barrier_memory_barriers, c.barrier_buffer_barriers,
        c.barrier_image_barriers, c.barrier_all_commands_barriers,
        c.descriptor_set_updates, c.descriptor_set_update_templates,
        c.descriptor_update_template_creates,
        c.descriptor_update_template_fallbacks,
        c.push_descriptor_calls, c.push_descriptor_shaders,
        c.push_descriptor_classic_shaders,
        c.descriptor_set_layout_creates, c.pipeline_layout_creates,
        c.pipeline_cache_creates, c.pipeline_cache_merge_source_creates,
        c.pipeline_cache_merges, c.pipeline_creates,
us_per(c.host_encode_ns, c.command_batches),
          us_per(c.host_submit_ns, c.queue_submits),
          c.restore_calls, us_per(c.restore_host_ns, c.restore_calls),
          c.restore_memory_barriers, c.restore_buffer_barriers,
c.restore_image_barriers, c.restore_state_buffers,
            c.restore_state_texture_mips,
            c.replay_records, c.replay_rejections, c.replay_submits,
            c.replay_update_refusals, c.replay_invalidations, c.replay_releases,
            us_per(c.replay_record_host_ns, c.replay_records));
}

luisa::string query(luisa::string_view key) noexcept {
    if (!key.starts_with("vk_profile_")) { return {}; }
    auto c = snapshot();
    if (key == "vk_profile_enabled") {
        // Compile-time constant: a binary without the macro has no query surface
        // at all (see the disabled branch of submit_profile.h).
        return "1";
    }
    if (key == "vk_profile_report") { return report(c); }
    if (key == "vk_profile_dispatches") { return count_string(c.dispatches); }
    if (key == "vk_profile_descriptor_set_allocations") { return count_string(c.descriptor_set_allocations); }
    if (key == "vk_profile_descriptor_set_updates") { return count_string(c.descriptor_set_updates); }
    if (key == "vk_profile_descriptor_set_update_templates") { return count_string(c.descriptor_set_update_templates); }
    if (key == "vk_profile_descriptor_update_template_creates") { return count_string(c.descriptor_update_template_creates); }
    if (key == "vk_profile_descriptor_update_template_fallbacks") { return count_string(c.descriptor_update_template_fallbacks); }
    if (key == "vk_profile_push_descriptor_calls") { return count_string(c.push_descriptor_calls); }
    if (key == "vk_profile_push_descriptor_shaders") { return count_string(c.push_descriptor_shaders); }
    if (key == "vk_profile_push_descriptor_classic_shaders") { return count_string(c.push_descriptor_classic_shaders); }
    if (key == "vk_profile_descriptor_update_template_fallback_reason") {
        return luisa::string{descriptor_update_template_fallback_reason()};
    }
    if (key == "vk_profile_descriptor_set_binds") { return count_string(c.descriptor_set_binds); }
    if (key == "vk_profile_pipeline_binds") { return count_string(c.pipeline_binds); }
    if (key == "vk_profile_argument_plans") { return count_string(c.argument_plans); }
    if (key == "vk_profile_barrier_calls") { return count_string(c.barrier_calls); }
    if (key == "vk_profile_barrier_memory_barriers") { return count_string(c.barrier_memory_barriers); }
    if (key == "vk_profile_barrier_buffer_barriers") { return count_string(c.barrier_buffer_barriers); }
    if (key == "vk_profile_barrier_image_barriers") { return count_string(c.barrier_image_barriers); }
    if (key == "vk_profile_barrier_all_commands_barriers") { return count_string(c.barrier_all_commands_barriers); }
    if (key == "vk_profile_descriptor_set_layout_creates") { return count_string(c.descriptor_set_layout_creates); }
    if (key == "vk_profile_pipeline_layout_creates") { return count_string(c.pipeline_layout_creates); }
    if (key == "vk_profile_pipeline_cache_creates") { return count_string(c.pipeline_cache_creates); }
    if (key == "vk_profile_pipeline_cache_merge_source_creates") { return count_string(c.pipeline_cache_merge_source_creates); }
    if (key == "vk_profile_pipeline_cache_merges") { return count_string(c.pipeline_cache_merges); }
    if (key == "vk_profile_pipeline_creates") { return count_string(c.pipeline_creates); }
    if (key == "vk_profile_command_batches") { return count_string(c.command_batches); }
    if (key == "vk_profile_queue_submits") { return count_string(c.queue_submits); }
    if (key == "vk_profile_queue_submit2") { return count_string(c.queue_submit2); }
    if (key == "vk_profile_queue_submit_legacy") { return count_string(c.queue_submit_legacy); }
    if (key == "vk_profile_timeline_wait_entries") { return count_string(c.timeline_wait_entries); }
    if (key == "vk_profile_timeline_waits") { return count_string(c.timeline_waits); }
    if (key == "vk_profile_timeline_wait_skips") { return count_string(c.timeline_wait_skips); }
    if (key == "vk_profile_restore_calls") { return count_string(c.restore_calls); }
 if (key == "vk_profile_restore_host_ns") { return count_string(c.restore_host_ns); }
 if (key == "vk_profile_restore_memory_barriers") { return count_string(c.restore_memory_barriers); }
 if (key == "vk_profile_restore_buffer_barriers") { return count_string(c.restore_buffer_barriers); }
 if (key == "vk_profile_restore_coalesced_buffers") { return count_string(c.restore_coalesced_buffers); }
 if (key == "vk_profile_restore_image_barriers") { return count_string(c.restore_image_barriers); }
 if (key == "vk_profile_restore_state_buffers") { return count_string(c.restore_state_buffers); }
 if (key == "vk_profile_restore_state_texture_mips") { return count_string(c.restore_state_texture_mips); }
 if (key == "vk_profile_host_encode_ns") { return count_string(c.host_encode_ns); }
    if (key == "vk_profile_host_submit_ns") { return count_string(c.host_submit_ns); }
      // Plan P4: replayable recording. `vk_profile_replay_live_graphs` is derived
      // (records - releases), so a leak is visible without a bespoke tool.
      if (key == "vk_profile_replay_records") { return count_string(c.replay_records); }
      if (key == "vk_profile_replay_record_host_ns") { return count_string(c.replay_record_host_ns); }
      if (key == "vk_profile_replay_submits") { return count_string(c.replay_submits); }
      if (key == "vk_profile_replay_rejections") { return count_string(c.replay_rejections); }
      if (key == "vk_profile_replay_update_refusals") { return count_string(c.replay_update_refusals); }
      if (key == "vk_profile_replay_invalidations") { return count_string(c.replay_invalidations); }
      if (key == "vk_profile_replay_releases") { return count_string(c.replay_releases); }
      if (key == "vk_profile_replay_live_graphs") {
          return count_string(c.replay_records >= c.replay_releases ?
                                  c.replay_records - c.replay_releases :
                                  0u);
      }
    // An unknown vk_profile_* key is reported as unknown rather than silently
    // forwarded to another backend's key.
    return {};
}

}// namespace lc::vk::submit_profile

#endif// LUISA_VULKAN_PROFILE_SUBMIT

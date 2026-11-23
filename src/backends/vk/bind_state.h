// Per-command-buffer elimination of redundant Vulkan pipeline and descriptor-set
// binds (plan P1.2).
//
// Why this exists
// ---------------
// Every compute/RT dispatch, raster draw and internal kernel (indirect
// preparation, bindless upload, acceleration-update) used to record its whole
// bind sequence unconditionally, so the same shader dispatched N times in one
// command buffer paid N x (vkCmdBindDescriptorSets + vkCmdBindPipeline) (plan §4,
// F3: `pipeline_binds == descriptor_set_binds == dispatches`). This tracker
// remembers what is already bound at each Vulkan bind point of the command
// buffer being recorded and lets the caller skip a bind that would not change
// anything.
//
// What may be skipped
// -------------------
// * A pipeline bind is redundant only when the *same VkPipeline handle* is
//   already bound at that bind point. Nothing here compares structures: two
//   pipelines compiled from identical SPIR-V are still two objects, and binding
//   the second one must be recorded.
// * A descriptor-set bind is redundant only when the exact same
// VkPipelineLayout *object* is already recorded and every set index of the
// requested range either already holds exactly the requested VkDescriptorSet
// handle or the request is not comparable at all. Layout object identity is the
// only thing P1.1 licenses: a structural (hash/field) comparison is never used to
// decide that two layouts are "compatible enough" (plan §9 risk row).
// * The tracked state is per *set index*, not per bind range (plan P2.3). That is
// what makes a global-set bind stay skippable when set 0 is pushed: a push
// replaces exactly one slot, so a later `vkCmdBindDescriptorSets` of the unchanged
// sampler/bindless sets at their real first set is genuinely redundant even though
// the local set changed. A bind with dynamic offsets is compared by its whole
// range instead (a dynamic offset means what it means for the set it was recorded
// with), and a slot last written by a push never matches a classic handle, because
// no VkDescriptorSet value describes a push descriptor bank.
// * Bind points are tracked separately (`compute`, `graphics`, `ray_tracing`):
//   Vulkan keeps one pipeline binding and one descriptor binding per bind point,
//   so a compute bind says nothing about the graphics state.
// * A changed pipeline with the same exact layout object keeps the descriptor
//   knowledge: the layout is what governs the bound sets, and it did not change.
//
// What invalidates the tracked state
// ----------------------------------
// * `reset()` at the start of every recording (`CommandBuffer::begin`) and on
//   `CommandBufferState::reset`, so no binding is ever carried across command
//   buffers, submissions, pool resets or command-buffer recycling.
// * A pipeline bind whose layout is a *different object* than the layout the
// tracked descriptor sets were recorded with drops the descriptor knowledge.
// The two layouts may well be structurally identical - and therefore
// compatible - but "may be" is not a licence here, so the next dispatch binds
// its sets again. The same rule applies to a push: Vulkan poisons every push
// descriptor as soon as push descriptors of a different descriptor set layout are
// set, so a push with another layout object also forgets the tracked slots.
// * `invalidate_all()` after anything that hands the raw VkCommandBuffer to code
// this tracker cannot see (`VKCustomCmd::execute`, `vkCmdCudaLaunchKernelNV`, the
// swapchain's present blit). That includes a *raw* push descriptor call made by
// such code: it changes a descriptor slot this tracker did not record, so the
// knowledge is dropped rather than repaired. The backend's own pushes go through
// `cmd_push_descriptor_set` and are tracked exactly.
// * A bind that cannot be represented (more sets or more dynamic offsets than
//   the tracker stores, or no sets at all) drops the knowledge instead of
//   guessing. Nothing in the backend reaches those limits today; the fallback
//   exists so a future call site degrades to "always bind" instead of to a wrong
//   skip.
// * Render pass boundaries do *not* invalidate anything. Per the Vulkan
//   specification the bound pipeline and descriptor sets are command-buffer
//   state that survives a render pass instance, and a skipped bind cannot
//   introduce a pipeline the untracked code would not have bound anyway. The
//   render pass only constrains which pipelines are *usable* inside it, which is
//   a pipeline-creation contract (one pipeline per render pass today), not a
//   binding one.
//
// Correctness notes
// -----------------
// * Updates are never skipped. `vkUpdateDescriptorSets` stays exactly where it
//   was, so a newly allocated or rewritten set is always complete before the
//   command that uses it. A skipped bind only re-uses set *objects* that are
//   still bound, so a later update of those objects is visible exactly as it was
//   before this optimisation.
// * Handle identity is stable for the whole recording, which is what makes
//   comparing handles instead of objects sound:
//   - a VkDescriptorSet is allocated from the command buffer state's own pools
//     and is only freed by `vkResetDescriptorPool` when the command buffer is
//     retired, so two sets of one recording never share a handle (the bindless
//     path's descriptor recycling is deferred to the same reset, see
//     `BindlessArray::_defer_descriptor_recycling`);
//   - a VkPipeline and the VkPipelineLayout/VkDescriptorSetLayout it was built
//     from are owned by the `Shader`, and the backend dereferences that shader
//     while recording (`shader->pipeline()`, `shader->pipeline_layout()`), so it
//     cannot be destroyed - and its handles cannot be recycled - inside the
//     recording. DSL dispatch pipelines are additionally retained by
//     `CommandBufferState::retain_pipeline` until the command buffer completes.
//   The consequence a future call site has to keep: never hand the tracker a
//   handle whose object can be destroyed and re-created under the same value
//   while the same command buffer is still being recorded.
// * Counters describe the recorded calls, not the requests: the two
//   `detail::cmd_bind_*` helpers call `submit_profile::record_*` only when they
//   actually record the Vulkan entry point (plan P0.2 contract).
// * Fail open is not an option: an unexpected bind point, an unrepresentable
//   bind or a flush request always results in the call being recorded.
//
// This header is deliberately free of Vulkan *entry points*: the state machine
// only compares handles, so the transition rules are covered exhaustively on the
// host (src/tests/unit/ext/test_vk_bind_state.cpp) without a device. The two
// `cmd_bind_*` helpers at the bottom are the only places that record the calls.
#pragma once

#include <volk.h>
#include <cstddef>
#include <cstdint>

#include <luisa/core/logging.h>

#include "layout_intern.h"// detail::handle_to_uint64, shared with the interning table
#include "submit_profile.h"

namespace lc::vk::detail {

// The three bind points this backend records into. `count` is the array size.
enum class BindPoint : uint32_t {
    compute = 0u,
    graphics = 1u,
    ray_tracing = 2u,
    count = 3u
};

inline constexpr auto kBindPointCount = static_cast<size_t>(BindPoint::count);

// Upper bound of the descriptor-set range and dynamic-offset array one bind point
// can remember. A compute dispatch binds at most five sets today (the local
// argument set, the immutable-sampler heap and up to three bindless heaps) and no
// call site passes a dynamic offset at all, so both limits are far above the real
// usage; exceeding them only means "do not track" (always bind).
inline constexpr uint32_t kMaxTrackedDescriptorSets = 8u;
inline constexpr uint32_t kMaxTrackedDynamicOffsets = 16u;

// What is known about one bind point. Handles are normalized to uint64_t (see
// detail::handle_to_uint64) so that "unknown" is unambiguous: a real handle is
// never 0, and every `*_known` flag gates its field.
struct BindPointState {
    // --- pipeline binding ------------------------------------------------
    bool pipeline_known{false};
    uint64_t pipeline{0u};
    // The VkPipelineLayout of the last recorded vkCmdBindPipeline. Only used to
    // decide whether the tracked descriptor state is still decidable.
    uint64_t pipeline_layout{0u};
    // --- descriptor-set binding -----------------------------------------
    bool descriptor_known{false};
    uint64_t descriptor_layout{0u};
    // Slots [first_set, first_set + set_count) of the last recorded *range*
    // bind. Kept as a description of that call (diagnostics and the host test);
    // the skip decision uses the per-slot state below, because a push replaces
    // one slot without touching the others and a sub-range bind can therefore be
    // redundant even when its range differs from the last one.
    uint32_t descriptor_first_set{0u};
    uint32_t descriptor_set_count{0u};
    uint64_t descriptor_sets[kMaxTrackedDescriptorSets]{};
    // The flat dynamic-offset array of the recorded bind, in the order Vulkan
    // consumes it (set by set).
    uint32_t dynamic_offset_count{0u};
    uint32_t dynamic_offsets[kMaxTrackedDynamicOffsets]{};
    // --- per-slot descriptor state (plan P2.3) ---------------------------
    // Vulkan keeps one bound descriptor set per set index. Tracking that array
    // directly is what lets a bind of the *global* sets keep being skipped when
    // set 0 is pushed: a push replaces slot 0 only.
    //   slot_known[i]  - slot i has a known content,
    //   slot_pushed[i] - that content is a push descriptor bank (no VkDescriptorSet
    //                    value describes it, so no classic bind can ever be
    //                    identical to it, while a later push always records),
    //   slot_handle[i] - the VkDescriptorSet bound to slot i.
    bool slot_known[kMaxTrackedDescriptorSets]{};
    bool slot_pushed[kMaxTrackedDescriptorSets]{};
    uint64_t slot_handle[kMaxTrackedDescriptorSets]{};
};

[[nodiscard]] constexpr size_t bind_point_index(BindPoint point) noexcept {
    return static_cast<size_t>(point);
}

// Translates a Vulkan bind point into the tracked one. Returns false for a bind
// point this backend never records into, which makes the caller fall back to
// "flush everything and bind".
[[nodiscard]] inline bool to_bind_point(
    VkPipelineBindPoint bind_point, BindPoint &result) noexcept {
    switch (bind_point) {
        case VK_PIPELINE_BIND_POINT_COMPUTE:
            result = BindPoint::compute;
            return true;
        case VK_PIPELINE_BIND_POINT_GRAPHICS:
            result = BindPoint::graphics;
            return true;
        // Ray tracing is a separate bind point in Vulkan, so an RT dispatch and a
        // compute dispatch never share a tracked pipeline.
        case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
            result = BindPoint::ray_tracing;
            return true;
        default:
            return false;
    }
}

// The complete bind state of one command buffer.
class BindState {
public:
    // Drops every binding. Called at the start of a recording (a fresh command
    // buffer has no bindings of its own) and whenever the command buffer is
    // reset for recycling.
    void reset() noexcept { invalidate_all(); }

    // Drops the binding of every bind point.
    void invalidate_all() noexcept {
        for (auto &point : _points) { point = BindPointState{}; }
    }

    // Drops the binding of one bind point.
    void invalidate(BindPoint point) noexcept {
        _points[bind_point_index(point)] = BindPointState{};
    }

    // Records a pipeline bind decision. Returns true when the caller must record
    // vkCmdBindPipeline, false when the exact same pipeline handle is already
    // bound at this bind point and the call can be skipped. A skipped bind leaves
    // every field untouched.
    [[nodiscard]] bool bind_pipeline(BindPoint point, uint64_t pipeline,
                                     uint64_t pipeline_layout) noexcept {
        auto &state = _points[bind_point_index(point)];
        // Descriptor bindings are only meaningful for the pipeline layout they
        // were recorded with. A different layout object - compatible or not, and
        // we refuse to find out from the structure - makes the tracked sets
        // undecidable, so they are forgotten and the next dispatch re-binds them.
        if (state.descriptor_known && state.descriptor_layout != pipeline_layout) {
            forget_descriptors(state);
        }
        if (state.pipeline_known && state.pipeline == pipeline) {
            // Same object: its layout is the same object too.
            return false;
        }
        state.pipeline_known = true;
        state.pipeline = pipeline;
        state.pipeline_layout = pipeline_layout;
        return true;
    }

    // Records a descriptor-set bind decision. Returns true when the caller must
    // record vkCmdBindDescriptorSets, false when the identical bind is already
    // recorded at this bind point.
    //
    // `sets` and `dynamic_offsets` may be null only when their count is zero or
    // the bind is unrepresentable (the caller always passes the real arrays for a
    // representable bind).
[[nodiscard]] bool bind_descriptor_sets(
        BindPoint point, uint64_t layout,
        uint32_t first_set, uint32_t set_count,
        uint64_t const *sets,
        uint32_t dynamic_offset_count,
        uint32_t const *dynamic_offsets) noexcept {
        auto &state = _points[bind_point_index(point)];
        // An empty bind or one larger than the tracked range is never remembered:
        // tracking it would need a mapping from dynamic offsets to sets that this
        // conservative tracker does not model.
        auto const representable =
            set_count != 0u &&
            set_count <= kMaxTrackedDescriptorSets &&
            static_cast<uint64_t>(first_set) + set_count <= kMaxTrackedDescriptorSets &&
            dynamic_offset_count <= kMaxTrackedDynamicOffsets;
        if (!representable) {
            forget_descriptors(state);
            return true;
        }
        // A classic bind under a *different* pipeline layout object must not inherit
        // the slots tracked under the old one: those handles describe sets that were
        // bound with another layout, so comparing a later request against them would
        // claim an identity that does not exist (and `bind_pipeline` cannot repair it,
        // because ordinary encoding binds descriptors before the pipeline). Forget the
        // whole per-slot/range state first, then record this bind under the new layout.
        // A partial-range bind under the *same* layout keeps merging with the slots it
        // did not touch, which is the only case where that is legal.
        if (state.descriptor_known && state.descriptor_layout != layout) {
            forget_descriptors(state);
        }
        if (state.descriptor_known && state.descriptor_layout == layout) {
            auto same_dynamic_offsets = state.dynamic_offset_count == dynamic_offset_count;
            for (auto i = uint32_t{0u}; same_dynamic_offsets && i < dynamic_offset_count; ++i) {
                if (state.dynamic_offsets[i] != dynamic_offsets[i]) {
                    same_dynamic_offsets = false;
                }
            }
            if (same_dynamic_offsets && dynamic_offset_count != 0u) {
                // A dynamic offset only means what it means for the set it was
                // recorded with, so a bind with dynamic offsets is compared by
                // its whole range instead of per slot.
                same_dynamic_offsets =
                    state.descriptor_first_set == first_set &&
                    state.descriptor_set_count == set_count;
            }
            if (same_dynamic_offsets) {
                // Exact per-slot redundancy: every slot of the requested range
                // already holds exactly this VkDescriptorSet under this exact
                // layout object. A slot last written by a push never matches,
                // because no VkDescriptorSet value describes a push bank.
                auto identical = true;
                for (auto i = uint32_t{0u}; i < set_count; ++i) {
                    auto const slot = static_cast<size_t>(first_set) + i;
                    if (!state.slot_known[slot] || state.slot_pushed[slot] ||
                        state.slot_handle[slot] != sets[i]) {
                        identical = false;
                        break;
                    }
                }
                if (identical) { return false; }
            }
        }
        state.descriptor_known = true;
        state.descriptor_layout = layout;
        state.descriptor_first_set = first_set;
        state.descriptor_set_count = set_count;
        for (auto i = uint32_t{0u}; i < set_count; ++i) {
            auto const slot = static_cast<size_t>(first_set) + i;
            state.slot_known[slot] = true;
            state.slot_pushed[slot] = false;
            state.slot_handle[slot] = sets[i];
        }
        for (auto i = uint32_t{0u}; i < set_count; ++i) {
            state.descriptor_sets[i] = sets[i];
        }
        state.dynamic_offset_count = dynamic_offset_count;
        for (auto i = uint32_t{0u}; i < dynamic_offset_count; ++i) {
            state.dynamic_offsets[i] = dynamic_offsets[i];
        }
        return true;
    }

    // Records that a vkCmdPushDescriptorSetKHR replaced the contents of `set` at
    // this bind point for `layout` (plan P2.3). Pushing is always recorded: it
    // changes state by definition, and the pushed content has no VkDescriptorSet
    // value that a later classic bind could be compared against.
    //
    // Vulkan poisons every push descriptor as soon as push descriptors with a
    // *different* descriptor set layout are set, and a classic slot tracked under
    // another pipeline layout is not comparable with this one either, so a layout
    // change forgets the whole slot array first - exactly like
    // `bind_pipeline` does for a changed layout object.
    void push_descriptor_sets(BindPoint point, uint64_t layout, uint32_t set) noexcept {
        auto &state = _points[bind_point_index(point)];
        if (!state.descriptor_known || state.descriptor_layout != layout) {
            forget_descriptors(state);
            state.descriptor_known = true;
            state.descriptor_layout = layout;
        }
        if (set >= kMaxTrackedDescriptorSets) {
            // A set index this tracker does not model: forget everything instead of
            // guessing. `descriptor_known` stays false, so the next bind of any slot
            // is recorded.
            forget_descriptors(state);
        } else {
            state.slot_known[set] = true;
            state.slot_pushed[set] = true;
            state.slot_handle[set] = 0u;
        }
        // The last-range description no longer describes the bound state: one
        // slot was replaced by something that is not a VkDescriptorSet.
        state.descriptor_first_set = 0u;
        state.descriptor_set_count = 0u;
        for (auto &handle : state.descriptor_sets) { handle = 0u; }
    }

    // Vulkan-bind-point overload, so a caller that only has the native enum (the
    // command encoder) does not re-implement the mapping. An untracked bind point
    // drops every binding, exactly like the bind helpers below.
    void push_descriptor_sets(VkPipelineBindPoint bind_point, uint64_t layout,
                              uint32_t set) noexcept {
        auto point = BindPoint::compute;
        if (!to_bind_point(bind_point, point)) {
            invalidate_all();
            return;
        }
        push_descriptor_sets(point, layout, set);
    }

    [[nodiscard]] BindPointState const &point(BindPoint point) const noexcept {
        return _points[bind_point_index(point)];
    }

private:
    // Drops the tracked descriptor state of one bind point (the per-slot array and
    // the last-range description) while leaving the pipeline binding alone. Used
    // whenever the tracker can no longer prove what a slot holds: a changed
    // pipeline layout object, a push with another layout, a bind it cannot
    // represent.
    static void forget_descriptors(BindPointState &state) noexcept {
        state.descriptor_known = false;
        state.descriptor_layout = 0u;
        state.descriptor_first_set = 0u;
        state.descriptor_set_count = 0u;
        for (auto &handle : state.descriptor_sets) { handle = 0u; }
        state.dynamic_offset_count = 0u;
        for (auto &known : state.slot_known) { known = false; }
        for (auto &pushed : state.slot_pushed) { pushed = false; }
        for (auto &handle : state.slot_handle) { handle = 0u; }
    }

    BindPointState _points[kBindPointCount]{};
};

// ---------------------------------------------------------------------------
// The only place the backend may bind while recording.
// ---------------------------------------------------------------------------
// Both helpers translate the Vulkan bind point, ask the tracker, record the call
// when the tracker says the state would change, and count exactly the recorded
// calls (plan P0.2: a counter must describe a real vkCmd*, not an intention).
// Plan P2.3 adds a third state transition that is not a bind: a
// vkCmdPushDescriptorSetKHR replaces the content of one set index and is tracked
// through BindState::push_descriptor_sets by the `detail::push_descriptor_set`
// funnel, which is the only place the backend may push.

inline bool cmd_bind_pipeline(BindState &state, VkCommandBuffer command_buffer,
                              VkPipelineBindPoint bind_point, VkPipeline pipeline,
                              VkPipelineLayout pipeline_layout) noexcept {
    auto point = BindPoint::compute;
    if (!to_bind_point(bind_point, point)) {
        // A bind point this backend does not track: whatever it was, the recorded
        // state is no longer trustworthy for any bind point.
        state.invalidate_all();
        vkCmdBindPipeline(command_buffer, bind_point, pipeline);
        submit_profile::record_pipeline_bind();
        return true;
    }
    if (!state.bind_pipeline(point, handle_to_uint64(pipeline),
                             handle_to_uint64(pipeline_layout))) {
        return false;
    }
    vkCmdBindPipeline(command_buffer, bind_point, pipeline);
    submit_profile::record_pipeline_bind();
    return true;
}

inline bool cmd_bind_descriptor_sets(
    BindState &state, VkCommandBuffer command_buffer,
    VkPipelineBindPoint bind_point, VkPipelineLayout pipeline_layout,
    uint32_t first_set, uint32_t set_count, VkDescriptorSet const *sets,
    uint32_t dynamic_offset_count = 0u,
    uint32_t const *dynamic_offsets = nullptr) noexcept {
    auto point = BindPoint::compute;
    if (!to_bind_point(bind_point, point)) {
        state.invalidate_all();
        vkCmdBindDescriptorSets(command_buffer, bind_point, pipeline_layout,
                                first_set, set_count, sets,
                                dynamic_offset_count, dynamic_offsets);
        submit_profile::record_descriptor_set_bind();
        return true;
    }
    // The tracker stores normalized handles; an untrackable bind is passed to it
    // with the local buffer it can ignore, so the decision and the recorded call
    // always describe the same arguments.
    uint64_t handles[kMaxTrackedDescriptorSets]{};
    auto copied = set_count < kMaxTrackedDescriptorSets ?
                      set_count :
                      kMaxTrackedDescriptorSets;
    for (auto i = uint32_t{0u}; i < copied; ++i) {
        handles[i] = handle_to_uint64(sets[i]);
    }
    if (!state.bind_descriptor_sets(point, handle_to_uint64(pipeline_layout),
                                    first_set, set_count, handles,
                                    dynamic_offset_count, dynamic_offsets)) {
        return false;
    }
    vkCmdBindDescriptorSets(command_buffer, bind_point, pipeline_layout,
                            first_set, set_count, sets,
                            dynamic_offset_count, dynamic_offsets);
    submit_profile::record_descriptor_set_bind();
    return true;
}


}// namespace lc::vk::detail
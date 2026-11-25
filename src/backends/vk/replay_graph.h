// Native state and registry of one replayable recording (plan P4, narrow v1).
//
// A graph owns:
//   * one native primary `VkCommandBuffer` recorded once (SIMULTANEOUS_USE), the
//     dedicated `CommandBufferState` it was recorded into (command pool without
//     the reset bit, descriptor pools and sets, argument staging, retained
//     `VkPipeline`s, image views) - i.e. everything the recorded commands still
//     reference inside the backend;
//   * explicit references to the interned pipeline/descriptor-set layouts its
//     recorded shaders use. P1.1 destroys an interned layout when its last shader
//     releases it, and the recorded buffer (and the descriptor sets allocated from
//     those layouts) outlives the shader, so the graph re-acquires the reference
//     **by id** instead of relying on the shader;
//   * the *native identities* (bit-cast `VkBuffer` values) of every buffer the
//     recorded commands name. Nothing here is a pointer into a `Stream`, a
//     `Buffer` or a `Shader`; a graph outlives them all and reports a typed error
//     instead of touching released memory.
//
// The registry is a device-private detail, not an extension: `Device` owns it, so
// teardown works even when a caller never asked for `VulkanReplayExt`.
#pragma once

#include <volk.h>

#include <mutex>

#include <luisa/core/stl/string.h>
#include <luisa/vstl/common.h>

#include <luisa/backends/ext/vk_replay_ext.h>

namespace lc::vk {

class Device;
class Stream;
struct CommandBufferState;

namespace detail {

// One recorded, never-re-recorded batch. Frees the native command buffer before
// its state (the pool owns the buffer, so the buffer has to go first) and never
// resets or re-begins anything: a graph is submitted, not re-recorded.
struct RecordedBatch {
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
    // Raw on purpose: the definition of `CommandBufferState` (and therefore of
    // its destructor) lives in stream.h, which this header deliberately avoids so
    // that device.h can include it.
    CommandBufferState *state{nullptr};

    RecordedBatch() noexcept = default;
    RecordedBatch(VkCommandBuffer buffer, CommandBufferState *state) noexcept
        : command_buffer{buffer}, state{state} {}
    RecordedBatch(RecordedBatch const &) = delete;
    RecordedBatch &operator=(RecordedBatch const &) = delete;
    RecordedBatch(RecordedBatch &&rhs) noexcept
        : command_buffer{rhs.command_buffer}, state{rhs.state} {
        rhs.command_buffer = VK_NULL_HANDLE;
        rhs.state = nullptr;
    }
    RecordedBatch &operator=(RecordedBatch &&rhs) noexcept;
    ~RecordedBatch() noexcept { reset(); }

    // Frees the native command buffer and its state exactly once (idempotent).
    void reset() noexcept;
    [[nodiscard]] bool empty() const noexcept {
        return command_buffer == VK_NULL_HANDLE && state == nullptr;
    }
};

struct ReplayGraph {
    // The owning device. Only used to free native state (the logical device and
    // the layout interner); a device always outlives its graphs because `~Device`
    // drains the registry, so this is not a lifetime hazard. Nothing in the graph
    // names a `Stream`, a `Buffer` or a `Shader`.
    Device *device{nullptr};
    uint64_t handle{0u};
    uint64_t stream_handle{0u};
    uint32_t dispatches{0u};
    uint32_t buffer_copies{0u};
    RecordedBatch batch;
    // Native buffer identities the recording names (bit-cast VkBuffer values).
    vstd::vector<uint64_t> watched_buffers;
    // Interned layout ids this recording depends on, retained by id.
    //
    // Both are *deduplicated vectors*, not one id per kind: a graph may hold several
    // compute shaders with structurally different pipeline layouts, so "the graph's
    // pipeline layout" is not a single object. Retaining only the first dispatch's
    // layout would let a later shader's destruction destroy a pipeline layout (and,
    // through it, its descriptor-set layouts) that the recorded command buffer still
    // binds.
    vstd::vector<uint64_t> desc_set_layout_ids;
    vstd::vector<uint64_t> pipeline_layout_ids;
    luisa::compute::VulkanReplayInvalidationReason invalidation{
        luisa::compute::VulkanReplayInvalidationReason::NONE};
    uint64_t submits{0u};
    uint64_t last_timeline_value{0u};
    luisa::string message;
    // Frees the recorded command buffer and its state first (which is what frees
    // the descriptor sets allocated from the interned layouts) and only then gives
    // the interned pipeline/descriptor-set layout references back. Retain-by-id and
    // release-by-id are symmetric: every id here was retained exactly once by
    // `record()`.
    ~ReplayGraph() noexcept;
};

// One mutex for every operation; the mutex is a leaf (no callback, no Vulkan
// call and no device lookup runs under it), so it cannot participate in a lock
// cycle with `Device::_stream_mtx` or a stream's `_dispatch_mtx`.
class ReplayRegistry {
public:
    // Phase 3C: the hot-path view of one graph, used by
    // `VulkanReplayExt::replay()` - the call a device-greedy decode loop makes once
    // per token. It carries exactly what a submission needs and nothing else:
    // `snapshot()` is for the diagnostic entry points (`status()`, `destroy()`,
    // `update()`), which may copy the free-form `message`; this one copies the
    // message only when the graph is invalidated (i.e. on the error path), so a
    // successful replay allocates no string at all.
    struct ReplayHandle {
        bool found{false};
        uint64_t stream_handle{0u};
        uint64_t command_buffer{0u};// bit-cast VkCommandBuffer, 0 when released
        luisa::compute::VulkanReplayInvalidationReason invalidation{
            luisa::compute::VulkanReplayInvalidationReason::NONE};
        // Only filled when `invalidation != NONE`; empty otherwise.
        luisa::string message;
    };

    struct Snapshot {
        bool found{false};
        uint64_t stream_handle{0u};
        uint64_t command_buffer{0u};// bit-cast VkCommandBuffer, 0 when released
        uint32_t dispatches{0u};
        uint32_t buffer_copies{0u};
        // How many interned layouts the graph holds. The number of *distinct*
        // pipeline layouts is the observable that proves a multi-shader graph retains
        // every one of them, not just the first.
        uint32_t interned_pipeline_layouts{0u};
        uint32_t interned_descriptor_set_layouts{0u};
        luisa::compute::VulkanReplayInvalidationReason invalidation{
            luisa::compute::VulkanReplayInvalidationReason::NONE};
        uint64_t submits{0u};
        uint64_t last_timeline_value{0u};
        luisa::string message;
    };

private:
    mutable std::mutex _mtx;
    vstd::unordered_map<uint64_t, ReplayGraph *> _graphs;
    uint64_t _next_handle{1u};

public:
    ReplayRegistry() noexcept = default;
    ReplayRegistry(ReplayRegistry const &) = delete;
    ReplayRegistry(ReplayRegistry &&) = delete;
    ReplayRegistry &operator=(ReplayRegistry const &) = delete;
    ReplayRegistry &operator=(ReplayRegistry &&) = delete;
    ~ReplayRegistry() noexcept;

    // Takes ownership of `graph` and assigns its handle. Returns 0 (and deletes
    // nothing) when the handle space is exhausted.
    [[nodiscard]] uint64_t add(ReplayGraph *graph) noexcept;
    [[nodiscard]] Snapshot snapshot(uint64_t handle) const noexcept;
    // The allocation-free hot path of `snapshot()`: no `luisa::string` copy unless
    // the graph is invalidated, no diagnostic counters, no layout counts. All the
    // safety checks of `replay()` are preserved - a missing handle, an invalidated
    // graph and a released command buffer are all reported through the fields.
    [[nodiscard]] ReplayHandle prepare_replay(uint64_t handle) const noexcept;
    // Removes the graph from the registry and hands it to the caller, who then
    // owns (and must delete) it.
    [[nodiscard]] ReplayGraph *take(uint64_t handle) noexcept;
    void record_submission(uint64_t handle, uint64_t timeline_value) noexcept;
    // Marks `handle` as invalidated unless it already was; returns true when this
    // call changed it (so the caller can count one invalidation per graph).
    bool invalidate_graph(uint64_t handle,
                          luisa::compute::VulkanReplayInvalidationReason reason,
                          luisa::string_view message) noexcept;
    // Invalidates every graph naming one of the given native buffer identities.
    size_t invalidate_buffers(const uint64_t *native_ids, size_t count,
                              luisa::string_view message) noexcept;
    size_t invalidate_stream(uint64_t stream_handle,
                             luisa::string_view message) noexcept;
    // Destroys every graph (device teardown). The caller must have synchronized
    // the owning streams first; this only marks and frees native state.
    size_t destroy_all_instances(luisa::string_view message) noexcept;
    // Stream handles that currently own at least one graph (teardown sync set).
    [[nodiscard]] vstd::vector<uint64_t> stream_handles() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
};

}// namespace detail
}// namespace lc::vk
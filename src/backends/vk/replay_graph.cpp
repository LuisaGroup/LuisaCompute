// Implementation of the replay graph registry and of the native teardown of one
// recorded batch (plan P4, narrow v1).
//
// The only native calls here are the ones that free a graph's Vulkan objects.
// Everything else is bookkeeping, so this translation unit is the single place a
// reviewer has to check for "freed exactly once" and for "never resets or
// re-begins a recorded buffer".
#include "replay_graph.h"

#include "device.h"
#include "stream.h"
#include "submit_profile.h"

namespace lc::vk::detail {

RecordedBatch &RecordedBatch::operator=(RecordedBatch &&rhs) noexcept {
    if (this != &rhs) {
        reset();
        command_buffer = rhs.command_buffer;
        state = rhs.state;
        rhs.command_buffer = VK_NULL_HANDLE;
        rhs.state = nullptr;
    }
    return *this;
}

void RecordedBatch::reset() noexcept {
    // Detach both halves first: every path below (including the failure paths) has
    // to leave this object empty, and `~RecordedBatch` calls this function a second
    // time on an already-reset batch.
    auto buffer = command_buffer;
    command_buffer = VK_NULL_HANDLE;
    auto *recorded_state = state;
    state = nullptr;
    if (buffer == VK_NULL_HANDLE && recorded_state == nullptr) { return; }
    if (recorded_state == nullptr) {
        // `CommandBuffer::release_native_batch` moves the buffer *and* the state out
        // together, so this can only be a recorder bug. The owning pool of `buffer`
        // is unknown here: freeing it against a guessed pool would corrupt the
        // backend allocator, and dropping it silently would hide the bug. Fail
        // closed instead of guessing.
        LUISA_ERROR(
            "Vulkan replay: a recorded batch holds a native command buffer without "
            "the command-buffer state that owns it; the pool is unknown, so the "
            "buffer is not freed.");
    }
    // The state frees everything the recorded commands still reference - the command
    // pool (and with it every buffer allocated from it), the descriptor pools and
    // sets, the staging buffers, the retained PipelineRefs and the image views -
    // exactly once, with the allocator it was created with (`vstd::make_unique` in
    // the `CommandBuffer` constructor). Holding it in a `vstd::unique_ptr` is what
    // makes "freed exactly once, never twice" independent of the paths below.
    auto holder = vstd::unique_ptr<CommandBufferState>{recorded_state};
    if (buffer == VK_NULL_HANDLE) {
        // A batch whose recording never produced a native buffer still owns its
        // state (pool, descriptor pools, retained pipelines); the holder destroys it
        // on return. Nothing was freed, so no release is counted.
        return;
    }
    if (recorded_state->device == nullptr) {
        // Registry teardown that happened after the logical device was gone: the
        // pool cannot be destroyed and no Vulkan entry point may be called. The
        // state holder only drops the host-side objects and the leak is reported
        // here rather than turned into a use-after-free of the logical device.
        LUISA_WARNING(
            "Vulkan replay: a recorded batch is being dropped without its logical "
            "device; its native command buffer, pool and descriptor pools leak.");
        return;
    }
    // The buffer was submitted `submits` times and never reset, so it is in
    // the executable state; freeing it is legal exactly once no submission
    // still refers to it - which the graph's BUSY check established. Freeing it
    // explicitly (and only then destroying the pool with the holder above) is what
    // makes the release observable exactly once; the pool destruction afterwards
    // only frees whatever is left.
    vkFreeCommandBuffers(recorded_state->device->logic_device(), recorded_state->pool, 1, &buffer);
    submit_profile::record_replay_release();
}

ReplayGraph::~ReplayGraph() noexcept {
    // Order is load-bearing (see replay_graph.h): the recorded buffer references
    // descriptor sets allocated from the interned layouts, so the buffer and its
    // state (which frees the descriptor pools) go first, and only then the layout
    // references are handed back.
    batch.reset();
    if (device == nullptr) { return; }
    auto vk_device = device->logic_device();
    if (vk_device == VK_NULL_HANDLE) { return; }
    auto &interner = device->layout_interner();
    // A pipeline layout references its descriptor-set layouts, so it is released
    // first - the same order `LayoutInterner::destroy_all` uses.
    for (auto id : pipeline_layout_ids) {
        interner.release_pipeline_layout(id, vk_device);
    }
    pipeline_layout_ids.clear();
    for (auto id : desc_set_layout_ids) {
        interner.release_descriptor_set_layout(id, vk_device);
    }
    desc_set_layout_ids.clear();
}

ReplayRegistry::~ReplayRegistry() noexcept {
    // `~Device` drains every graph explicitly before this point, while the logical
    // device-and-layout state its destructor needs is still alive, so this loop is
    // empty in practice. If it is not, freeing native state here would dereference
    // a destroyed logical device, so the graphs are dropped instead and the leak is
    // reported: a silent use-after-free would be worse than an honest message.
    if (!_graphs.empty()) {
        LUISA_WARNING(
            "Vulkan replay: {} graph(s) reached registry teardown without being "
            "drained; their native state cannot be freed after the logical device "
            "is gone.",
            _graphs.size());
        for (auto &&entry : _graphs) {
            entry.second->device = nullptr;
            // The recorded batch's state carries its own `Device *` (that is what
            // `RecordedBatch::reset` frees the native buffer against). Clearing it
            // is what turns this already-broken path from "call vkFreeCommandBuffers
            // / vkDestroyCommandPool on a destroyed logical device" into the leak
            // this warning reports.
            if (entry.second->batch.state != nullptr) {
                entry.second->batch.state->device = nullptr;
            }
            delete entry.second;
        }
        _graphs.clear();
    }
}

uint64_t ReplayRegistry::add(ReplayGraph *graph) noexcept {
    if (graph == nullptr) { return 0u; }
    std::lock_guard lock{_mtx};
    auto handle = _next_handle++;
    if (handle == 0u) {// exhausted
        _next_handle = 1u;
        handle = _next_handle++;
    }
    graph->handle = handle;
    _graphs.try_emplace(handle, graph);
    return handle;
}

ReplayRegistry::Snapshot ReplayRegistry::snapshot(uint64_t handle) const noexcept {
    Snapshot result{};
    std::lock_guard lock{_mtx};
    auto iter = _graphs.find(handle);
    if (iter == _graphs.end()) { return result; }
    auto *graph = iter->second;
    result.found = true;
    result.stream_handle = graph->stream_handle;
    result.command_buffer = graph->batch.command_buffer == VK_NULL_HANDLE ?
                                0u :
                                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(graph->batch.command_buffer));
    result.dispatches = graph->dispatches;
    result.buffer_copies = graph->buffer_copies;
    result.interned_pipeline_layouts =
        static_cast<uint32_t>(graph->pipeline_layout_ids.size());
    result.interned_descriptor_set_layouts =
        static_cast<uint32_t>(graph->desc_set_layout_ids.size());
    result.invalidation = graph->invalidation;
    result.submits = graph->submits;
    result.last_timeline_value = graph->last_timeline_value;
    result.message = graph->message;
    return result;
}

ReplayRegistry::ReplayHandle ReplayRegistry::prepare_replay(uint64_t handle) const noexcept {
    // Phase 3C: the `replay()` fast path. Same lock, same lookups, but it fills only
    // the four fields a submission needs and deliberately skips the diagnostics
    // (`dispatches`, layout counts, `submits`, `last_timeline_value`). The only
    // string it may copy is the invalidation message, and only on the error path -
    // a successful replay therefore performs no allocation here at all.
    ReplayHandle result{};
    std::lock_guard lock{_mtx};
    auto iter = _graphs.find(handle);
    if (iter == _graphs.end()) { return result; }
    auto *graph = iter->second;
    result.found = true;
    result.stream_handle = graph->stream_handle;
    result.command_buffer = graph->batch.command_buffer == VK_NULL_HANDLE ?
                                0u :
                                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(graph->batch.command_buffer));
    result.invalidation = graph->invalidation;
    if (result.invalidation != luisa::compute::VulkanReplayInvalidationReason::NONE) {
        result.message = graph->message;
    }
    return result;
}

ReplayGraph *ReplayRegistry::take(uint64_t handle) noexcept {
    std::lock_guard lock{_mtx};
    auto iter = _graphs.find(handle);
    if (iter == _graphs.end()) { return nullptr; }
    auto *graph = iter->second;
    _graphs.erase(iter);
    return graph;
}

void ReplayRegistry::record_submission(uint64_t handle, uint64_t timeline_value) noexcept {
    std::lock_guard lock{_mtx};
    auto iter = _graphs.find(handle);
    if (iter == _graphs.end()) { return; }
    ++iter->second->submits;
    iter->second->last_timeline_value = timeline_value;
}

bool ReplayRegistry::invalidate_graph(
    uint64_t handle,
    luisa::compute::VulkanReplayInvalidationReason reason,
    luisa::string_view message) noexcept {
    std::lock_guard lock{_mtx};
    auto iter = _graphs.find(handle);
    if (iter == _graphs.end()) { return false; }
    auto *graph = iter->second;
    if (graph->invalidation != luisa::compute::VulkanReplayInvalidationReason::NONE) {
        return false;
    }
    graph->invalidation = reason;
    graph->message = luisa::string{message};
    return true;
}

size_t ReplayRegistry::invalidate_buffers(
    const uint64_t *native_ids, size_t count,
    luisa::string_view message) noexcept {
    if (native_ids == nullptr || count == 0u) { return 0u; }
    auto invalidated = size_t{0u};
    {
        std::lock_guard lock{_mtx};
        for (auto &&entry : _graphs) {
            auto *graph = entry.second;
            if (graph->invalidation != luisa::compute::VulkanReplayInvalidationReason::NONE) {
                continue;
            }
            auto hit = false;
            for (auto watched : graph->watched_buffers) {
                for (auto i = size_t{0u}; i < count; ++i) {
                    if (watched == native_ids[i]) {
                        hit = true;
                        break;
                    }
                }
                if (hit) { break; }
            }
            if (hit) {
                graph->invalidation = luisa::compute::VulkanReplayInvalidationReason::RESOURCE_DESTROYED;
                graph->message = luisa::string{message};
                ++invalidated;
            }
        }
    }
    if (invalidated != 0u) { submit_profile::record_replay_invalidations(invalidated); }
    return invalidated;
}

size_t ReplayRegistry::invalidate_stream(
    uint64_t stream_handle, luisa::string_view message) noexcept {
    auto invalidated = size_t{0u};
    {
        std::lock_guard lock{_mtx};
        for (auto &&entry : _graphs) {
            auto *graph = entry.second;
            if (graph->stream_handle != stream_handle) { continue; }
            if (graph->invalidation != luisa::compute::VulkanReplayInvalidationReason::NONE) {
                continue;
            }
            graph->invalidation = luisa::compute::VulkanReplayInvalidationReason::STREAM_DESTROYED;
            graph->message = luisa::string{message};
            ++invalidated;
        }
    }
    if (invalidated != 0u) { submit_profile::record_replay_invalidations(invalidated); }
    return invalidated;
}

size_t ReplayRegistry::destroy_all_instances(luisa::string_view message) noexcept {
    vstd::vector<ReplayGraph *> released;
    {
        std::lock_guard lock{_mtx};
        released.reserve(_graphs.size());
        for (auto &&entry : _graphs) {
            released.emplace_back(entry.second);
        }
        _graphs.clear();
        for (auto *graph : released) {
            if (graph->invalidation == luisa::compute::VulkanReplayInvalidationReason::NONE) {
                graph->invalidation = luisa::compute::VulkanReplayInvalidationReason::DEVICE_TEARDOWN;
                graph->message = luisa::string{message};
            }
        }
    }
    for (auto *graph : released) {
        delete graph;
    }
    if (!released.empty()) {
        submit_profile::record_replay_invalidations(released.size());
    }
    return released.size();
}

vstd::vector<uint64_t> ReplayRegistry::stream_handles() const noexcept {
    vstd::vector<uint64_t> handles;
    std::lock_guard lock{_mtx};
    handles.reserve(_graphs.size());
    for (auto &&entry : _graphs) {
        auto handle = entry.second->stream_handle;
        auto known = false;
        for (auto existing : handles) {
            if (existing == handle) {
                known = true;
                break;
            }
        }
        if (!known) { handles.emplace_back(handle); }
    }
    return handles;
}

size_t ReplayRegistry::size() const noexcept {
    std::lock_guard lock{_mtx};
    return _graphs.size();
}

}// namespace lc::vk::detail
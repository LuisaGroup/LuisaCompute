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
    if (command_buffer != VK_NULL_HANDLE && state != nullptr && state->device != nullptr) {
        // The buffer was submitted `submits` times and never reset, so it is in
        // the executable state; freeing it is legal exactly once no submission
        // still refers to it - which the graph's BUSY check established.
        vkFreeCommandBuffers(state->device->logic_device(), state->pool, 1, &command_buffer);
        submit_profile::record_replay_release();
    }
    command_buffer = VK_NULL_HANDLE;
    if (state != nullptr) {
        // The state's destructor releases the retained PipelineRefs and destroys
        // the command pool (and, with it, every command buffer allocated from it)
        // plus the descriptor pools, which is also what frees the descriptor sets
        // the recorded buffer bound. It does not touch the interned layouts: the
        // graph releases those after this. The holder carries the allocator the
        // state was created with (`vstd::make_unique` in the CommandBuffer
        // constructor), so the deallocation matches the allocation.
        vstd::unique_ptr<CommandBufferState> holder{state};
        state = nullptr;
    }
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
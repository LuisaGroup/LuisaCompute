#pragma once
#include "resource.h"
#include "event.h"
#include "texture.h"
#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include "indirect_copy_command_buffer.h"
#include <luisa/core/stl/memory.h>
#include <volk.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/vstl/stack_allocator.h>
#include <luisa/vstl/functional.h>
#include "../common/command_reorder_visitor.h"
#include "bindless_array.h"
#include "shader.h"
#include "resource_barrier.h"
#include "command_buffer_ownership.h"
#include "bind_state.h"
#include "argument_plan_scratch.h"

namespace lc::vk {
class Event;
class Stream;
class Swapchain;
using namespace luisa::compute;
class CommandBuffer;
struct PipelineRef;
namespace detail {
// plan P4: the native command buffer plus the state it was recorded into, moved
// out of a `CommandBuffer` and owned by a replay graph (replay_graph.h).
struct RecordedBatch;
}// namespace detail
namespace temp_buffer {
struct DeferredStagingFrees {
    vstd::vector<vstd::unique_ptr<UploadBuffer>> upload_buffers;
    vstd::vector<vstd::unique_ptr<ReadbackBuffer>> readback_buffers;
    void clear() noexcept {
        upload_buffers.clear();
        readback_buffers.clear();
    }
    [[nodiscard]] bool empty() const noexcept {
        return upload_buffers.empty() && readback_buffers.empty();
    }
};
template<typename Pack>
class Visitor : public vstd::StackAllocatorVisitor {
public:
    Device *device;
    vstd::vector<vstd::unique_ptr<Pack>> *deferred_frees{nullptr};
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
    Pack *create(uint64_t size);
};
class DefaultBufferDeferredVisitor : public vstd::StackAllocatorVisitor {
public:
    Device *device;
    CommandBuffer *cmdbuffer;
    vstd::unordered_map<uint64_t, vstd::unique_ptr<DefaultBuffer>> buffers;
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
};
class BufferAllocatorBase {
protected:
    ~BufferAllocatorBase() = default;
public:
    virtual BufferView allocate(size_t size) = 0;
    virtual BufferView allocate(size_t size, size_t align) = 0;
};
template<typename T>
class BufferAllocator : public BufferAllocatorBase {
public:
    static constexpr size_t kLargeBufferSize = 65536ull;
    struct LargeBuffer {
        vstd::unique_ptr<T> buffer;
        size_t capacity{0u};
        size_t used_offset{0u};
        uint64_t last_used_epoch{0u};
    };
    vstd::StackAllocator alloc;
    vstd::vector<LargeBuffer> active_large_buffers;
    vstd::vector<LargeBuffer> free_large_buffers;
    uint64_t clear_epoch{0u};

    Visitor<T> visitor;
    BufferView allocate(size_t size) override;
    BufferView allocate(size_t size, size_t align) override;
    void clear(vstd::vector<vstd::unique_ptr<T>> *deferred_frees = nullptr);
    void flush_host();
    BufferAllocator(size_t init_capacity);
    ~BufferAllocator();
private:
    BufferView _allocate_large(size_t size, size_t align);
};
}// namespace temp_buffer
struct CommandBufferState {
    VkCommandPool pool{};
    Device *device{};
    temp_buffer::BufferAllocator<UploadBuffer> upload_alloc;
    temp_buffer::BufferAllocator<ReadbackBuffer> readback_alloc;
    VkDescriptorPool desc_pool{};
    vstd::vector<VkDescriptorPool> overflow_desc_pools;
    size_t active_desc_pool_index{0u};
    uint32_t next_overflow_max_sets{8192u};
    uint32_t next_overflow_per_type{8192u};
    vstd::vector<PipelineRef *> retained_pipelines;
    // VK_KHR_copy_memory_indirect negotiation data of this state
    // (copy_command_plan.h decides with them, this struct only carries them):
    // VkPhysicalDeviceCopyMemoryIndirectPropertiesKHR::supportedQueues and the
    // VkQueueFlags of the family this state's command pool belongs to. Both are
    // queried once in init(); a zero mask means "not enabled/queried", which
    // keeps the direct copy path.
    VkQueueFlags copy_memory_indirect_supported_queues{0u};
    VkQueueFlags command_pool_queue_flags{0u};
    // Command array of the indirect copy batching fast path, created on the
    // first batch that reaches it and reused for the lifetime of this state.
    luisa::unique_ptr<IndirectCopyCommandBuffer> indirect_copy_commands;
    vstd::vector<VkImageView> img_views;
    vstd::vector<std::pair<void *, vstd::func_ptr_t<void(Stream *, CommandBufferState *, void *)>>> dispose_pool;
    vstd::vector<vstd::function<void()>> callbacks;
    // What is already bound at each Vulkan bind point of the command buffer this
    // state belongs to (plan P1.2). Reset when a recording starts, so a recycled
    // state never carries a binding across command buffers or submissions; see
    // bind_state.h for what may be skipped and what invalidates it.
    detail::BindState bind_state;
    CommandBufferState();
    ~CommandBufferState();
    // `pool_flags` lets a replay recording (plan P4) create its command pool
    // without VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: such a buffer is
    // begun once and only ever resubmitted, so nothing may reset it individually.
    // Every existing caller keeps the reset-capable pool.
    void init(Device &device, StreamTag tag,
              VkCommandPoolCreateFlags pool_flags =
                  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    [[nodiscard]] VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
    void retain_pipeline(PipelineRef *ref);
    void reset(
        Stream *stream,
        Device &device,
        temp_buffer::DeferredStagingFrees *deferred_frees = nullptr);
    template<typename TT>
        requires(!std::is_trivially_destructible_v<TT> && !std::is_reference_v<TT>)
    void dispose_after_flush(TT &&value) {
        auto ptr = vengine_malloc(sizeof(std::remove_cvref_t<TT>));
        new (ptr) TT(std::forward<TT>(value));
        dispose_pool.emplace_back(
            ptr,
            [](Stream *, CommandBufferState *, void *ptr) {
                std::destroy_at(reinterpret_cast<std::remove_cvref_t<TT> *>(ptr));
                vengine_free(ptr);
            });
    }
};

class CommandBuffer : public Resource {
    Stream &_stream;
    VkCommandBuffer _cmdbuffer;
    vstd::unique_ptr<CommandBufferState> _state;
    detail::CommandBufferOwnership _ownership{
        detail::CommandBufferOwnership::BACKEND};

public:
    luisa::function<void(luisa::string_view)> *logger{};
    vstd::vector<VkDescriptorSet> *desc_sets{};
    vstd::vector<std::byte> *uniform_data{};
    // The argument blocks of the layer being recorded: which command owns which
    // block of `uniform_data` (plan P1.4). The command identity is carried so the
    // encoding pass can assert that it binds the block of the command it encodes.
    vstd::vector<detail::DispatchArgumentBlock> *dispatch_offsets{};
    vstd::vector<VkWriteDescriptorSet> *write_desc_sets{};
    vstd::StackAllocator *scratch_buffer_alloc{};
    vstd::vector<uint4> *bindless_cache{};
    vstd::StackAllocator *temp_desc{};

    ResourceBarrier *resource_barrier{};
    using Resource::operator bool;
    // `ownership == PERSISTENT` (plan P4) allocates from its own state's pool,
    // never borrows a config-ext command buffer, and is never reset or recycled.
    explicit CommandBuffer(Stream &stream,
                           detail::CommandBufferOwnership ownership =
                               detail::CommandBufferOwnership::BACKEND) noexcept;
    CommandBuffer(CommandBuffer &&) noexcept;
    ~CommandBuffer();
    [[nodiscard]] auto cmdbuffer() const { return _cmdbuffer; }
    [[nodiscard]] bool retire_and_recycle(temp_buffer::DeferredStagingFrees *deferred_frees = nullptr);
    // `flags` is forwarded to VkCommandBufferBeginInfo; only the replay recorder
    // passes VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT (plan P4), so an
    // ordinary batch keeps the flags-less begin info it always used.
    void begin(VkCommandBufferUsageFlags flags = 0u);
    void end();
    // Moves the native buffer and its state out of this object, leaving it empty
    // and owning nothing. The caller (a replay graph) becomes responsible for
    // freeing both; the destructor of this object no longer touches them.
    //
    // Only a `PERSISTENT` buffer (a replay recording) may be handed out this way,
    // and `replay_graph.cpp` is its only caller: the graph frees the buffer against
    // `state->pool`, which is wrong for a `BACKEND` buffer (the retirement FIFO or
    // `~CommandBuffer` already own it) and illegal for a `BORROWED` one (the pool
    // belongs to the config ext). Asserted, never guessed.
    [[nodiscard]] detail::RecordedBatch release_native_batch() noexcept;
    auto states() const { return _state.get(); }
    void execute(vstd::span<const luisa::unique_ptr<Command>> cmds);
};
struct ReorderFuncTable {
    [[nodiscard]] uint64_t canonical_buffer_handle(
        uint64_t handle) const noexcept;
    [[nodiscard]] uint64_t canonical_texture_handle(
        uint64_t handle) const noexcept;
    void traverse_bindless_resources(
        uint64_t bindless_handle,
        ReorderBindlessResourceVisitor visitor) const noexcept;
    [[nodiscard]] size_t bindless_pending_count(
        uint64_t bindless_handle) const noexcept {
        auto bindless = reinterpret_cast<BindlessArray *>(bindless_handle);
        std::lock_guard lock{bindless->mtx};
        return bindless->pending_resource_count();
    }
    [[nodiscard]] bool bindless_has_pending_buffers(
        uint64_t bindless_handle) const noexcept {
        auto bindless = reinterpret_cast<BindlessArray *>(bindless_handle);
        std::lock_guard lock{bindless->mtx};
        return bindless->has_pending_buffers();
    }
    template<typename F>
    void with_bindless_pending_lookup(
        uint64_t bindless_handle, F &&fn) const noexcept {
        auto bindless = reinterpret_cast<BindlessArray *>(bindless_handle);
        std::lock_guard lock{bindless->mtx};
        fn([bindless](uint64_t canonical_handle, bool is_buffer) noexcept {
            return bindless->contains_pending_canonical(canonical_handle, is_buffer);
        });
    }
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {
        auto cs = reinterpret_cast<Shader *>(shader_handle);
        auto arguments = cs->saved_arguments();
        LUISA_ASSERT(
            argument_index < arguments.size(),
            "Vulkan command reordering requested shader argument {} from "
            "a saved table containing only {} entries.",
            argument_index, arguments.size());
        return arguments[argument_index].var_usage;
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept;
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::BufferModification> modifications) const noexcept;
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> modifications) const noexcept;
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> modifications) const noexcept;
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {
        auto cs = reinterpret_cast<Shader *>(handle);
        return cs->captured();
    }
    luisa::span<const Argument> raster_shader_bindings(uint64_t handle) const noexcept {
        auto shader = reinterpret_cast<Shader *>(handle);
        return shader->captured();
    }
};

class Stream : public Resource {
    struct SyncExt {
        Event *evt;
        uint64_t value;
    };
    struct NotifyEvt {
        Event *evt;
        uint64_t value;
    };
    using Callbacks = luisa::vector<luisa::move_only_function<void()>>;
    using AsyncCmd = vstd::variant<
        Callbacks,
        CommandBuffer,
        SyncExt,
        NotifyEvt>;
    Event _evt;
    VkQueue _queue;
    std::atomic_bool _enabled{true};
    std::mutex _dispatch_mtx;
    luisa::spin_mutex _mtx;
    vstd::LockFreeArrayQueue<CommandBuffer> _cmdbuffers;
    vstd::vector<VkDescriptorSet> _desc_sets;
    vstd::SingleThreadArrayQueue<AsyncCmd> _exec;
    ResourceBarrier _resource_barrier;
    vstd::vector<std::byte> _uniform_data;
    vstd::vector<detail::DispatchArgumentBlock> _dispatch_offsets;
    vstd::VEngineMallocVisitor _temp_desc_visitor;
    vstd::StackAllocator _temp_desc;
    temp_buffer::DefaultBufferDeferredVisitor _scratch_buffer_alloc_visitor;
    vstd::StackAllocator _scratch_buffer_alloc;
    vstd::vector<VkWriteDescriptorSet> _write_desc_sets;
    vstd::vector<uint4> _bindless_cache;
    StreamTag _stream_tag;
    luisa::spin_mutex *_queue_mtx;
public:
    luisa::function<void(luisa::string_view)> logger;
    CommandReorderVisitor<ReorderFuncTable, true> reorder;
    // Stream-scoped scratch of the single argument-planning pass (plan P1.4):
    // the plan of every command of the current batch plus the two trailer
    // vectors it is emitted with. `CommandBuffer::execute` is its only user and
    // runs under `_dispatch_mtx`, which is what serializes it; see
    // argument_plan_scratch.h for the lifetime and reentrancy contract.
    detail::ArgumentScratch argument_scratch;
    [[nodiscard]] auto queue() const { return _queue; }
    [[nodiscard]] auto stream_tag() const { return _stream_tag; }
    Stream(Device *device, StreamTag tag);
    ~Stream();
    luisa::spin_mutex &queue_mtx() { return *_queue_mtx; };

    void dispatch(
        vstd::span<const luisa::unique_ptr<Command>> cmds,
        Callbacks &&callbacks,
        vstd::span<const SwapchainPresent> presents,
        bool inqueue_limit);
    void present(
        Texture const *tex,
        uint mip,
        Swapchain *swapchain,
        bool inqueue_limit);
    void update_sparse_resources(luisa::vector<SparseUpdateTile> &&textures_update) noexcept;
    void sync();
    void remove_resource_state(Resource const *resource) noexcept;
    void signal(Event *event, uint64_t value);
    void wait(Event *event, uint64_t value);

    // ---- replayable recording (plan P4) -----------------------------------
    // Records `cmds` once into a dedicated primary command buffer (SIMULTANEOUS_
    // USE, its own state, pool without the reset bit) and hands the native batch
    // to the caller without submitting it. It runs exactly the same prefix and
    // suffix as `Stream::dispatch` - `_begin_batch_record` / `_finish_batch_record`
    // below are the statements lifted out of it - so the barriers, the config-ext
    // before/after states, the descriptor and argument encoding and the trailing
    // restore are identical to an ordinary batch; only the submit is missing.
    // Takes `_dispatch_mtx` for the whole call, like `dispatch`.
    [[nodiscard]] detail::RecordedBatch record_replay_batch(
        vstd::span<const luisa::unique_ptr<Command>> cmds);

    // Submits one already-recorded command buffer on this stream through the
    // ordinary timeline path: throttle, one reserved timeline value, exactly one
    // `Event::_signal` (one native submission) and the SyncExt/NotifyEvt pair.
    // Nothing is re-recorded and no command buffer enters the retirement FIFO, so
    // no `CommandBufferState::reset` can touch a replay graph. Returns the
    // timeline value the submission signalled. Takes `_dispatch_mtx`.
    [[nodiscard]] uint64_t submit_recorded(VkCommandBuffer command_buffer,
                                          bool inqueue_limit);

    // Non-blocking: has the submission that signalled `timeline_value` retired on
    // the host (the stream's retirement thread notified it)? Used by a replay
    // graph's BUSY check; it never waits.
    [[nodiscard]] bool submission_complete(uint64_t timeline_value) const noexcept;

private:
    // The record prefix of one batch, extracted behaviour-neutrally out of
    // `Stream::dispatch` (plan P4): acquire or construct the command buffer, wire
    // the stream-scoped scratch and the barrier tracker into it, inject the config
    // ext's before-states and begin recording.
    //
    // `ownership` is the only thing that differs between an ordinary batch and a
    // replay recording:
    //   * `BACKEND` (the default, what every ordinary dispatch passes) keeps the
    //     historical acquisition bit for bit: a recycled buffer from the stream's
    //     retirement FIFO if there is one, a fresh backend-owned buffer otherwise.
    //   * `PERSISTENT` (the replay recorder) neither dequeues the FIFO nor borrows
    //     a config-ext buffer. It constructs a dedicated backend-owned buffer whose
    //     pool has no reset bit, because the native buffer *and* its state are
    //     handed to a replay graph that resubmits the buffer N times and frees both
    //     exactly once. Recycling a retired state (already restored, descriptor
    //     pools reset) or taking a one-shot borrowed buffer would make the graph
    //     free native state from a pool it does not own.
    // `flags` is forwarded to the begin info and is only non-zero for the replay
    // recorder.
    [[nodiscard]] CommandBuffer _begin_batch_record(
        detail::CommandBufferOwnership ownership =
            detail::CommandBufferOwnership::BACKEND,
        VkCommandBufferUsageFlags flags = 0u);
    // The record suffix of one batch, extracted the same way: apply the config
    // ext's after-states, record the trailing restore and end the recording.
    void _finish_batch_record(CommandBuffer &cmdbuffer);
    // In-flight throttle of one submission (plan P1.5): both the dispatch path and
    // the present path call it before reserving the next timeline value, so their
    // arithmetic cannot drift apart. The bound comes from
    // `Device::max_frames_in_flight()` and the engine lives in in_flight_depth.h;
    // this function only decides whether the host has to block, and records the
    // decision under the submit profiler's three timeline counters.
    void _throttle_in_flight();
    [[nodiscard]] bool _execute_external_command_buffer(
        VkCommandBuffer command_buffer) noexcept;
    std::thread _thd;
};

}// namespace lc::vk

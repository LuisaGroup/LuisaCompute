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
#include "../common/host_transfer_range.h"
#include "bindless_array.h"
#include "shader.h"
#include "resource_barrier.h"
#include "command_buffer_ownership.h"
#include "bind_state.h"
#include "argument_plan_scratch.h"
#include "vk_gpu_timeline.h"

namespace lc::vk {
class Event;
class Stream;
class Swapchain;
class Blas;
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
    // Phase 3B (opt B): membership of `retained_pipelines` (so `retain_pipeline`
    // is O(1) instead of a linear scan) plus the "same pipeline as last time"
    // fast path. Both are dropped together with the vector in `reset()`.
    vstd::HashMap<PipelineRef *, uint8_t> retained_pipeline_set;
    PipelineRef *_last_retained_pipeline{nullptr};
    // Phase 3B (opt A): per-state reuse cache of set-0 descriptor sets.
    //
    // A `VkDescriptorSet` is allocated from this state's `desc_pool` and stays
    // valid until that pool is reset, so a batch that allocates 364 sets only to
    // reset the pool on retirement pays 364 `vkAllocateDescriptorSets` calls per
    // batch for nothing. The cache hands out the sets of the previous batch
    // instead: within one batch every dispatch still gets a *distinct* handle
    // (`cursor` advances), so the update-before-bind order and the
    // "do not touch a set a pending command buffer references" rule of the
    // previous batch are untouched - reuse only ever happens across batches, i.e.
    // after `reset()` ran on the retirement thread, which itself only runs once the
    // submitted command buffer completed (the `SyncExt` entry precedes the
    // `CommandBuffer` entry in the stream's retirement queue).
    //
    // The bucket is keyed by `Shader::desc_set_layout_id()` - the interned
    // identity of the layout (0 = "not interned", recorded through
    // `has_uncached_desc_allocs` so such a state falls back to the pool reset) and
    // is dropped whenever `Device::shader_layout_destroy_generation()` moved,
    // because a destroyed layout invalidates every set created from it.
    struct CachedDescriptorBinding {
        VkDescriptorType type{VK_DESCRIPTOR_TYPE_MAX_ENUM};
        uint64_t handle{0u};
        VkDeviceSize offset{0u};
        VkDeviceSize range{0u};
        [[nodiscard]] bool operator==(const CachedDescriptorBinding &) const noexcept = default;
    };
    struct CachedDescriptorSetEntry {
        VkDescriptorSet set{VK_NULL_HANDLE};
        uint64_t last_shader_id{0u};
        luisa::vector<CachedDescriptorBinding> bindings;
    };
    struct CachedDescriptorSetBucket {
        vstd::vector<CachedDescriptorSetEntry> sets;
        size_t cursor{0u};
    };
    vstd::HashMap<uint64_t, CachedDescriptorSetBucket> desc_set_cache;
    // Phase 3C (defect 3): descriptor sets that were dropped from the cache without
    // the pool being reset (their layout died, see `reset_native`). They still
    // occupy their slot in `desc_pool`/`overflow_desc_pools` and can never be handed
    // out again, so a host that destroys and recreates shaders in a loop (a decode
    // loop that re-specializes with a new buffer set, a live-reload, ...) would leak
    // one pool's worth of sets per round without leaking a single set object. Once
    // this reaches the threshold of `reset_native` the state takes the pool-reset
    // path instead of rewinding the cursors, which reclaims every orphan.
    size_t orphaned_desc_sets{0u};
    // Generation of `Device::shader_layout_destroy_generation()` the cache was
    // built under (see `reset`).
    uint64_t cached_layout_generation{0u};
    uint64_t cached_resource_generation{0u};
    // True once anything was allocated through a path that passes a layout id of
    // 0 (indirect preparation, native/custom shaders). Such a set is not in any
    // bucket, so the pool has to be reset on retirement; the state then also drops
    // the cache instead of pretending it could reuse it.
    bool has_uncached_desc_allocs{false};
    // Scratch of the per-id invalidation walk in `reset()` (kept to avoid a
    // per-retirement allocation; its content never outlives the call).
    vstd::vector<uint64_t> destroyed_layout_scratch;
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
    VkQueryPool timestamp_pool{VK_NULL_HANDLE};
    uint32_t timestamp_pool_capacity{0u};
    uint32_t timestamp_valid_bits{0u};
    uint32_t timestamp_cursor{0u};
    bool has_submitted_timestamps{false};
    bool timestamp_resettable{false};
    uint64_t tl_submit_seq{0u};
    uint64_t tl_stream_handle{0u};
    uint32_t tl_stream_tag{0u};
    uint64_t tl_fence{0u};
    uint64_t tl_cpu_record_begin_ns{0u};
    uint64_t tl_cpu_submit_begin_ns{0u};
    uint64_t tl_cpu_submit_end_ns{0u};
    uint64_t tl_cpu_fence_complete_ns{0u};
    uint32_t tl_cb_q_begin{0xffffffffu};
    uint32_t tl_cb_q_end{0xffffffffu};
    uint32_t tl_cmd_count{0u};
    uint32_t tl_layer_count{0u};
    uint32_t tl_truncated_spans{0u};
    luisa::vector<TimelineLayerRecord> tl_layers;
    uint64_t host_retire_fence{0u};
    CommandBufferState();
    ~CommandBufferState();
    // `pool_flags` lets a replay recording (plan P4) create its command pool
    // without VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: such a buffer is
    // begun once and only ever resubmitted, so nothing may reset it individually.
    // Every existing caller keeps the reset-capable pool.
    void init(Device &device, StreamTag tag,
              VkCommandPoolCreateFlags pool_flags =
                  VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    void ensure_timestamp_pool(Device &device, StreamTag tag, uint32_t min_capacity);
    // `layout_id` is `Shader::desc_set_layout_id(set)` (0 when the layout is not
    // an interned, device-owned one). A non-zero id enables the reuse cache of
    // this state; 0 always allocates a fresh set and marks the state so its pools
    // are reset on retirement.
    [[nodiscard]] VkDescriptorSet allocate_descriptor_set(
        uint64_t layout_id, VkDescriptorSetLayout layout,
        uint64_t shader_id = 0u,
        CachedDescriptorSetEntry **out_entry = nullptr);
    void retain_pipeline(PipelineRef *ref);
    void reset(
        Stream *stream,
        Device &device,
        temp_buffer::DeferredStagingFrees *deferred_frees = nullptr);
    // Phase 3B (opt C): the two halves of `reset`. `reset_host_visible` runs the
    // part `Stream::sync()` has to observe before it returns (readback callbacks,
    // deferred disposals, temporary image views); `reset_native` runs the rest
    // (retained pipelines, staging allocations, descriptor pools, bind state),
    // which the retirement thread defers until after it notified the event.
    // `reset()` is still the synchronous combination of both.
    void reset_host_visible(Stream *stream, Device &device);
    void reset_native(
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
    [[nodiscard]] Stream &stream() const noexcept { return _stream; }
    [[nodiscard]] bool is_persistent_recording() const noexcept {
        return _ownership == detail::CommandBufferOwnership::PERSISTENT;
    }
    [[nodiscard]] bool retire_and_recycle(temp_buffer::DeferredStagingFrees *deferred_frees = nullptr);
    // Phase 3B (opt C): the retirement split into the half `Stream::sync()` waits
    // for (`retire_host_visible`) and the recycling half (`recycle_native`, whose
    // return value is the same "push it back into the stream's FIFO" answer
    // `retire_and_recycle` gives). Both are no-ops on a moved-from or
    // already-released buffer, and both refuse a PERSISTENT recording exactly like
    // `retire_and_recycle` does.
    void retire_host_visible();
    [[nodiscard]] bool recycle_native(temp_buffer::DeferredStagingFrees *deferred_frees = nullptr);
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
    template<typename F>
    void with_bindless_pending_state(
        uint64_t bindless_handle, F &&fn) const noexcept {
        auto bindless = reinterpret_cast<BindlessArray *>(bindless_handle);
        std::lock_guard lock{bindless->mtx};
        fn(bindless->pending_resource_count(),
           bindless->has_pending_buffers(),
           bindless->has_pending_textures(),
           [bindless](uint64_t canonical_handle, bool is_buffer) noexcept {
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
    luisa::spin_mutex _pending_compaction_mtx;
    vstd::vector<Blas *> _pending_compaction_blases;
    vstd::vector<EventWaitEntry> _pending_waits;
    struct PendingHostDownload {
        HostByteRange range;
        uint64_t fence;
    };
    luisa::vector<PendingHostDownload> _pending_host_downloads;
    std::atomic<uint32_t> _online_timing_refs{0u};
    std::atomic<uint64_t> _online_last_cb_gpu_ns{0u};
    std::atomic<uint64_t> _online_last_cb_gap_ns{0u};
    std::atomic<uint64_t> _online_completed_cbs{0u};
    uint64_t _online_prev_cb_end_ticks{0u};
    bool _online_has_prev_cb_end{false};
    void _flush_pending_waits();
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
    void enable_online_timing() noexcept {
        _online_timing_refs.fetch_add(1u, std::memory_order_acq_rel);
    }
    void disable_online_timing() noexcept {
        uint32_t cur = _online_timing_refs.load(std::memory_order_relaxed);
        while (cur != 0u &&
               !_online_timing_refs.compare_exchange_weak(
                   cur, cur - 1u, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        }
    }
    [[nodiscard]] bool online_timing_enabled() const noexcept {
        return _online_timing_refs.load(std::memory_order_relaxed) != 0u;
    }
    void record_online_cb_timing(
        uint64_t begin_ticks, uint64_t end_ticks,
        uint32_t valid_bits, float period_ns) noexcept {
        if (valid_bits == 0u || period_ns <= 0.0f) { return; }
        const uint64_t mask =
            valid_bits >= 64u ? ~0ull : ((1ull << valid_bits) - 1ull);
        const uint64_t dur_ticks = (end_ticks - begin_ticks) & mask;
        const uint64_t gpu_ns = static_cast<uint64_t>(
            static_cast<double>(dur_ticks) * static_cast<double>(period_ns));
        uint64_t gap_ns = 0ull;
        if (_online_has_prev_cb_end) {
            const uint64_t raw_gap =
                (begin_ticks - _online_prev_cb_end_ticks) & mask;
            const uint64_t half =
                valid_bits >= 64u ? (1ull << 63) : (1ull << (valid_bits - 1u));
            if (raw_gap < half) {
                gap_ns = static_cast<uint64_t>(
                    static_cast<double>(raw_gap) * static_cast<double>(period_ns));
            }
        }
        _online_prev_cb_end_ticks = end_ticks;
        _online_has_prev_cb_end = true;
        _online_last_cb_gpu_ns.store(gpu_ns, std::memory_order_relaxed);
        _online_last_cb_gap_ns.store(gap_ns, std::memory_order_relaxed);
        _online_completed_cbs.fetch_add(1ull, std::memory_order_release);
    }
    void read_online_cb_timing(
        uint64_t &gpu_ns, uint64_t &gap_ns, uint64_t &completed_cbs) const noexcept {
        completed_cbs = _online_completed_cbs.load(std::memory_order_acquire);
        gpu_ns = _online_last_cb_gpu_ns.load(std::memory_order_relaxed);
        gap_ns = _online_last_cb_gap_ns.load(std::memory_order_relaxed);
    }
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
    uint64_t execute_immediate(
        luisa::move_only_function<void(CommandBuffer &)> &&fn,
        bool wait_after = true);
    void register_pending_compaction(Blas *blas) noexcept;
    void unregister_pending_compaction(Blas *blas) noexcept;
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
    void _wait_host_retired_fence(uint64_t target) noexcept;
    [[nodiscard]] uint64_t _dispatch_chunk(
        vstd::span<const luisa::unique_ptr<Command>> cmds,
        luisa::vector<luisa::move_only_function<void()>> &&callbacks,
        vstd::span<const SwapchainPresent> presents,
        bool inqueue_limit);
    [[nodiscard]] bool _execute_external_command_buffer(
        VkCommandBuffer command_buffer) noexcept;
    // How long the retirement thread keeps polling before it blocks, latched at
    // construction from `LUISA_VK_RETIRE_SPIN_US` (default 200 us, converted to a
    // pause-loop bound). Declared *before* `_thd` because the thread reads it as
    // soon as it starts, and members are initialized in declaration order. It is a
    // bound, not a semantic: the thread always ends up either draining the queue or
    // sleeping on `_work_epoch`.
    uint32_t _retire_spin_iterations{0u};
    // Phase 3B (opt C): wakeup epoch of the retirement thread. Every producer that
    // enqueues into `_exec` bumps it under `_mtx` and notifies after unlocking, so
    // the thread can block instead of spinning with `std::this_thread::yield()` for
    // the whole time the GPU is busy. The value is the *first* of the two things the
    // retirement loop looks at (before `loop_cmd()` and before it blocks), so a
    // wakeup can never be lost: a bump that already happened makes the epoch
    // mismatch visible and the following `wait` returns immediately.
    //
    // Declaration order (phase 3C defect 1): `_work_epoch` and
    // `_retire_spin_iterations` are declared - and therefore constructed - *before*
    // `_thd`, because the thread starts running inside `_thd`'s constructor and
    // reads both. Declaring `_work_epoch` after `_thd` made the thread read a
    // not-yet-constructed atomic (member-initialization-order UB).
    std::atomic_uint64_t _host_retired_fence{0u};
    std::atomic_uint64_t _work_epoch{0u};
    std::thread _thd;
    // Bumped under `_mtx` by every producer that enqueues into `_exec` (so the
    // enqueue and the epoch are one atomic step from the thread's point of view)...
    void _bump_work_epoch_locked() noexcept {
        _work_epoch.fetch_add(1u, std::memory_order_release);
    }
    // ...and notified only after `_mtx` is released: `notify_all` wakes a parked
    // thread, which immediately tries to take `_mtx` in `loop_cmd()`.
    void _notify_retirement_thread() noexcept {
        _work_epoch.notify_all();
    }
};

}// namespace lc::vk

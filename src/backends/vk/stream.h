#pragma once
#include "resource.h"
#include "event.h"
#include "texture.h"
#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include <volk.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/vstl/stack_allocator.h>
#include "../common/command_reorder_visitor.h"
#include "shader.h"
#include "resource_barrier.h"

namespace lc::vk {
class Event;
class Stream;
class Swapchain;
using namespace luisa::compute;
class CommandBuffer;
namespace temp_buffer {
template<typename Pack>
class Visitor : public vstd::StackAllocatorVisitor {
public:
    Device *device;
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
    Pack *Create(size_t size);
};
class DefaultBufferDeferredVisitor : public vstd::StackAllocatorVisitor {
public:
    Device *device;
    CommandBuffer *cmdbuffer;
    vstd::unordered_map<uint64_t, vstd::unique_ptr<DefaultBuffer>> _buffers;
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
    static constexpr size_t kLargeBufferSize = 65536ull;
    vstd::StackAllocator alloc;
    vstd::vector<vstd::unique_ptr<T>> largeBuffers;

public:
    Visitor<T> visitor;
    BufferView allocate(size_t size) override;
    BufferView allocate(size_t size, size_t align) override;
    void clear();
    BufferAllocator(size_t initCapacity);
    ~BufferAllocator();
};
}// namespace temp_buffer
struct CommandBufferState {
    VkCommandPool _pool{};
    Device *device{};
    temp_buffer::BufferAllocator<UploadBuffer> upload_alloc;
    temp_buffer::BufferAllocator<ReadbackBuffer> readback_alloc;
    VkDescriptorPool _desc_pool;
    vstd::vector<VkImageView> img_views;
    vstd::vector<std::pair<void *, vstd::func_ptr_t<void(Stream *, CommandBufferState *, void *)>>> _dispose_pool;
    vstd::vector<vstd::function<void()>> _callbacks;
    CommandBufferState();
    ~CommandBufferState();
    void init(Device &device, StreamTag tag);
    void reset(Stream *stream, Device &device);
    template<typename TT>
        requires(!std::is_trivially_destructible_v<TT> && !std::is_reference_v<TT>)
    void dispose_after_flush(TT &&value) {
        auto ptr = vengine_malloc(sizeof(std::remove_cvref_t<TT>));
        new (ptr) TT(std::forward<TT>(value));
        _dispose_pool.emplace_back(
            ptr,
            [](Stream *, CommandBufferState *, void *ptr) {
                std::destroy_at(reinterpret_cast<std::remove_cvref_t<TT> *>(ptr));
                vengine_free(ptr);
            });
    }
};

class CommandBuffer : public Resource {
    Stream &stream;
    VkCommandBuffer _cmdbuffer;
    vstd::unique_ptr<CommandBufferState> _state;

public:
    luisa::function<void(luisa::string_view)> *logger;
    vstd::vector<VkDescriptorSet> *desc_sets;
    vstd::vector<std::byte> *uniform_data;
    vstd::vector<std::pair<size_t, size_t>> *dispatch_offsets;
    vstd::vector<VkWriteDescriptorSet> *write_desc_sets;
    vstd::StackAllocator *scratch_buffer_alloc;
    vstd::vector<uint4> *bindless_cache;
    vstd::StackAllocator *temp_desc;

    ResourceBarrier *resource_barrier;
    using Resource::operator bool;
    CommandBuffer(Stream &stream);
    CommandBuffer(CommandBuffer &&);
    ~CommandBuffer();
    [[nodiscard]] auto cmdbuffer() const { return _cmdbuffer; }
    void reset();
    void begin();
    void end();
    auto states() const { return _state.get(); }
    void execute(vstd::span<const luisa::unique_ptr<Command>> cmds);
};
struct ReorderFuncTable {
    bool is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept;
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {
        auto cs = reinterpret_cast<Shader *>(shader_handle);
        return cs->saved_arguments()[argument_index].varUsage;
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept;
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::BufferModification> modifications) const noexcept;
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> modifications) const noexcept;
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> modifications) const noexcept;
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {
        auto cs = reinterpret_cast<Shader *>(handle);
        return cs->captured();
    }
    void lock_bindless(uint64_t bindless_handle) const noexcept;
    void unlock_bindless(uint64_t bindless_handle) const noexcept;
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
    vstd::vector<VkDescriptorSet> desc_sets;
    vstd::SingleThreadArrayQueue<AsyncCmd> _exec;
    ResourceBarrier resource_barrier;
    vstd::vector<std::byte> uniform_data;
    vstd::vector<std::pair<size_t, size_t>> dispatch_offsets;
    vstd::VEngineMallocVisitor temp_desc_visitor;
    vstd::StackAllocator temp_desc;
    temp_buffer::DefaultBufferDeferredVisitor scratch_buffer_alloc_visitor;
    vstd::StackAllocator scratch_buffer_alloc;
    vstd::vector<VkWriteDescriptorSet> write_desc_sets;
    vstd::vector<uint4> bindless_cache;
    StreamTag _stream_tag;
    luisa::spin_mutex *_queue_mtx;
public:
    luisa::function<void(luisa::string_view)> logger;
    CommandReorderVisitor<ReorderFuncTable, true> reorder;
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
    void signal(Event *event, uint64_t value);
    void wait(Event *event, uint64_t value);
private:
    std::thread _thd;
};

}// namespace lc::vk

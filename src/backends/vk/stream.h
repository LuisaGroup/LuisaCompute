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
#include <luisa/vstl/functional.h>
#include "../common/command_reorder_visitor.h"
#include "shader.h"
#include "resource_barrier.h"
#include "command_buffer_ownership.h"

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
    vstd::StackAllocator alloc;
    vstd::vector<vstd::unique_ptr<T>> large_buffers;

    Visitor<T> visitor;
    BufferView allocate(size_t size) override;
    BufferView allocate(size_t size, size_t align) override;
    void clear();
    BufferAllocator(size_t init_capacity);
    ~BufferAllocator();
};
}// namespace temp_buffer
struct CommandBufferState {
    VkCommandPool pool{};
    Device *device{};
    temp_buffer::BufferAllocator<UploadBuffer> upload_alloc;
    temp_buffer::BufferAllocator<ReadbackBuffer> readback_alloc;
    VkDescriptorPool desc_pool{};
    vstd::vector<VkImageView> img_views;
    vstd::vector<std::pair<void *, vstd::func_ptr_t<void(Stream *, CommandBufferState *, void *)>>> dispose_pool;
    vstd::vector<vstd::function<void()>> callbacks;
    CommandBufferState();
    ~CommandBufferState();
    void init(Device &device, StreamTag tag);
    void reset(Stream *stream, Device &device);
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
    vstd::vector<std::pair<size_t, size_t>> *dispatch_offsets{};
    vstd::vector<VkWriteDescriptorSet> *write_desc_sets{};
    vstd::StackAllocator *scratch_buffer_alloc{};
    vstd::vector<uint4> *bindless_cache{};
    vstd::StackAllocator *temp_desc{};

    ResourceBarrier *resource_barrier{};
    using Resource::operator bool;
    explicit CommandBuffer(Stream &stream) noexcept;
    CommandBuffer(CommandBuffer &&) noexcept;
    ~CommandBuffer();
    [[nodiscard]] auto cmdbuffer() const { return _cmdbuffer; }
    [[nodiscard]] bool retire_and_recycle();
    void begin();
    void end();
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
    // Work graph is DX12-only; Vulkan stubs satisfy the concept but are never called.
    luisa::span<const Argument> work_graph_bindings(uint64_t) const noexcept { return {}; }
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
    vstd::vector<std::pair<size_t, size_t>> _dispatch_offsets;
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
private:
    [[nodiscard]] bool _execute_external_command_buffer(
        VkCommandBuffer command_buffer) noexcept;
    std::thread _thd;
};

}// namespace lc::vk

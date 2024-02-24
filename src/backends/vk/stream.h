#pragma once
#include "resource.h"
#include "event.h"
#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include <vulkan/vulkan.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/vstl/stack_allocator.h>
#include "../common/command_reorder_visitor.h"
#include "shader.h"

namespace lc::vk {
class Event;
class Stream;
using namespace luisa::compute;
class CommandBuffer;
namespace temp_buffer {
template<typename Pack>
class Visitor : public vstd::StackAllocatorVisitor {
public:
    Device *device;
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
};
template<typename T>
class BufferAllocator {
    static constexpr size_t kLargeBufferSize = 65536ull;
    vstd::StackAllocator alloc;
    vstd::vector<vstd::unique_ptr<T>> largeBuffers;

public:
    Visitor<T> visitor;
    BufferView allocate(size_t size);
    BufferView allocate(size_t size, size_t align);
    void clear();
    BufferAllocator(size_t initCapacity);
    ~BufferAllocator();
};
}// namespace temp_buffer
struct CommandBufferState {
    temp_buffer::BufferAllocator<UploadBuffer> upload_alloc;
    temp_buffer::BufferAllocator<DefaultBuffer> default_alloc;
    temp_buffer::BufferAllocator<ReadbackBuffer> readback_alloc;
    vstd::vector<VkDescriptorSet> _desc_sets;
    CommandBufferState();
    void reset(Device &device);
};
class CommandBuffer : public Resource {
    Stream &stream;
    VkCommandBuffer _cmdbuffer;
    vstd::unique_ptr<CommandBufferState> _state;

public:
    using Resource::operator bool;
    CommandBuffer(Stream &stream);
    CommandBuffer(CommandBuffer &&);
    ~CommandBuffer();
    [[nodiscard]] auto cmdbuffer() const { return _cmdbuffer; }
    void reset();
    void begin();
    void end();
    void execute(vstd::span<const luisa::unique_ptr<Command>> cmds);
};
struct ReorderFuncTable {
    bool is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept {
        return false;
    }
    Usage get_usage(uint64_t shader_handle, size_t argument_index) const noexcept {
        using namespace lc::hlsl;
        auto cs = reinterpret_cast<Shader *>(shader_handle);
        switch (cs->binds()[argument_index].type) {
            case ShaderVariableType::ConstantBuffer:
            case ShaderVariableType::SRVTextureHeap:
            case ShaderVariableType::SRVBufferHeap:
            case ShaderVariableType::CBVBufferHeap:
            case ShaderVariableType::SamplerHeap:
            case ShaderVariableType::StructuredBuffer:
            case ShaderVariableType::ConstantValue:
                return Usage::READ;
            default:
                return Usage::READ_WRITE;
        }
    }
    void update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {
    }
    luisa::span<const Argument> shader_bindings(uint64_t handle) const noexcept {
        auto cs = reinterpret_cast<Shader *>(handle);
        return cs->captured();
    }
    void lock_bindless(uint64_t bindless_handle) const noexcept {}
    void unlock_bindless(uint64_t bindless_handle) const noexcept {}
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
    VkCommandPool _pool;
    VkQueue _queue;
    std::atomic_bool _enabled{true};
    std::condition_variable _cv;
    std::mutex _mtx;
    vstd::LockFreeArrayQueue<CommandBuffer> _cmdbuffers;
    vstd::LockFreeArrayQueue<AsyncCmd> _exec;

public:
    CommandReorderVisitor<ReorderFuncTable, true> reorder;
    [[nodiscard]] auto queue() const { return _queue; }
    [[nodiscard]] auto pool() const { return _pool; }
    Stream(Device *device, StreamTag tag);
    ~Stream();
    void dispatch(
        vstd::span<const luisa::unique_ptr<Command>> cmds,
        Callbacks &&callbacks,
        bool inqueue_limit);
    void sync();
    void signal(Event *event, uint64_t value);
    void wait(Event *event, uint64_t value);
private:
    std::thread _thd;
};

}// namespace lc::vk

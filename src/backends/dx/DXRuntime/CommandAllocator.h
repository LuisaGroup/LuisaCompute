#pragma once
#include <luisa/vstl/functional.h>
#include <DXRuntime/CommandBuffer.h>
#include <luisa/vstl/stack_allocator.h>
#include <Resource/UploadBuffer.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <dxgi1_4.h>
namespace lc::dx {
class CommandQueue;
class IPipelineEvent;
class CommandAllocator final : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    friend class CommandBuffer;

private:
    template<typename Pack>
    class Visitor : public vstd::StackAllocatorVisitor {
    public:
        CommandAllocator *self;
        uint64 allocate(uint64 size) override;
        vstd::unique_ptr<Pack> create(uint64 size);
        void deallocate(uint64 handle) override;
    };
    class DescHeapVisitor : public vstd::StackAllocatorVisitor {
    public:
        D3D12_DESCRIPTOR_HEAP_TYPE type;
        Device *device;
        uint64 allocate(uint64 size) override;
        void deallocate(uint64 handle) override;
        DescHeapVisitor(Device *device, D3D12_DESCRIPTOR_HEAP_TYPE type) : type(type), device(device) {}
    };
    template<typename T>
    class BufferAllocator {
        static constexpr size_t kLargeBufferSize = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        vstd::StackAllocator _alloc;
        vstd::vector<vstd::unique_ptr<T>> _large_buffers;

    public:
        Visitor<T> visitor;
        BufferView allocate(size_t size);
        BufferView allocate(size_t size, size_t align);
        void clear();
        BufferAllocator(size_t initCapacity);
        ~BufferAllocator();
    };
    Device *_device;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> _allocator;
    mutable vstd::optional<CommandBuffer> _cbuffer;
    D3D12_COMMAND_LIST_TYPE _type;
    GpuAllocator *_resource_allocator;
    vstd::LockFreeArrayQueue<vstd::function<void()>> _execute_after_complete;
    vstd::vector<vstd::unique_ptr<Resource>> _res_dispose_list;
    vstd::spin_mutex _res_dispose_list_mtx;

    DescHeapVisitor _rtv_visitor;
    DescHeapVisitor _dsv_visitor;
    BufferAllocator<UploadBuffer> _upload_allocator;
    BufferAllocator<DefaultBuffer> _default_allocator;
    BufferAllocator<ReadbackBuffer> _readback_allocator;
    vstd::unique_ptr<DefaultBuffer> _scratch_buffer;
    //TODO: allocate commandbuffer
    CommandAllocator(Device *device, GpuAllocator *resourceAllocator, D3D12_COMMAND_LIST_TYPE type);

public:
    vstd::StackAllocator rtv_allocator;
    vstd::StackAllocator dsv_allocator;

    template<typename Func>
        requires(luisa::is_constructible_v<vstd::function<void()>, Func &&>)
    void execute_after_complete(Func &&func) {
        _execute_after_complete.enqueue(std::forward<Func>(func));
    }
    void dispose_after_complete(vstd::unique_ptr<Resource> &&res) {
        std::lock_guard lck{_res_dispose_list_mtx};
        _res_dispose_list.emplace_back(std::move(res));
    }
    ID3D12CommandAllocator *allocator();
    Device *device() const noexcept { return _device; }
    D3D12_COMMAND_LIST_TYPE type() const { return _type; }
    ~CommandAllocator();
    CommandBuffer *get_buffer() const;
    void execute(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex, luisa::span<std::pair<IDXGISwapChain *, bool>> swapChains, bool cmdlist_is_empty);
    void complete(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    DefaultBuffer const *allocate_scratch_buffer(size_t targetSize);
    BufferView get_temp_readback_buffer(uint64 size, size_t align = 0);
    BufferView get_temp_upload_buffer(uint64 size, size_t align = 0);
    BufferView get_temp_default_buffer(uint64 size, size_t align = 0);
    void reset(CommandQueue *queue);
    KILL_COPY_CONSTRUCT(CommandAllocator)
    KILL_MOVE_CONSTRUCT(CommandAllocator)
};
class IPipelineEvent : public vstd::IOperatorNewBase {
public:
    virtual ~IPipelineEvent() = default;
};
}// namespace lc::dx

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
        vstd::unique_ptr<Pack> Create(uint64 size);
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
        vstd::StackAllocator alloc;
        vstd::vector<vstd::unique_ptr<T>> largeBuffers;

    public:
        Visitor<T> visitor;
        BufferView Allocate(size_t size);
        BufferView Allocate(size_t size, size_t align);
        void Clear();
        BufferAllocator(size_t initCapacity);
        ~BufferAllocator();
    };
    Device *device;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    mutable vstd::optional<CommandBuffer> cbuffer;
    D3D12_COMMAND_LIST_TYPE type;
    GpuAllocator *resourceAllocator;
    vstd::LockFreeArrayQueue<vstd::function<void()>> executeAfterComplete;
    vstd::vector<vstd::unique_ptr<Resource>> resDisposeList;
    vstd::spin_mutex resDisposeListMtx;

    DescHeapVisitor rtvVisitor;
    DescHeapVisitor dsvVisitor;
    BufferAllocator<UploadBuffer> uploadAllocator;
    BufferAllocator<DefaultBuffer> defaultAllocator;
    BufferAllocator<ReadbackBuffer> readbackAllocator;
    vstd::unique_ptr<DefaultBuffer> scratchBuffer;
    //TODO: allocate commandbuffer
    CommandAllocator(Device *device, GpuAllocator *resourceAllocator, D3D12_COMMAND_LIST_TYPE type);

public:
    vstd::StackAllocator rtvAllocator;
    vstd::StackAllocator dsvAllocator;

    template<typename Func>
        requires(std::is_constructible_v<vstd::function<void()>, Func &&>)
    void ExecuteAfterComplete(Func &&func) {
        executeAfterComplete.push(std::forward<Func>(func));
    }
    void DisposeAfterComplete(vstd::unique_ptr<Resource> &&res) {
        std::lock_guard lck{resDisposeListMtx};
        resDisposeList.emplace_back(std::move(res));
    }
    ID3D12CommandAllocator *Allocator() const { return allocator.Get(); }
    D3D12_COMMAND_LIST_TYPE Type() const { return type; }
    ~CommandAllocator();
    CommandBuffer *GetBuffer() const;
    void Execute(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    void ExecuteAndPresent(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex, IDXGISwapChain3 *swapchain, bool vsync);
    void Complete(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    DefaultBuffer const *AllocateScratchBuffer(size_t targetSize);
    BufferView GetTempReadbackBuffer(uint64 size, size_t align = 0);
    BufferView GetTempUploadBuffer(uint64 size, size_t align = 0);
    BufferView GetTempDefaultBuffer(uint64 size, size_t align = 0);
    void Reset(CommandQueue *queue);
    KILL_COPY_CONSTRUCT(CommandAllocator)
    KILL_MOVE_CONSTRUCT(CommandAllocator)
};
class IPipelineEvent : public vstd::IOperatorNewBase {
public:
    virtual ~IPipelineEvent() = default;
};
}// namespace lc::dx

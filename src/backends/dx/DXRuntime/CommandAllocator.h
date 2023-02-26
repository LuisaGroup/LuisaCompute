#pragma once
#include <DXRuntime/CommandAllocatorBase.h>
namespace toolhub::directx {
class CommandQueue;
class IPipelineEvent;
class CommandAllocator final : public CommandAllocatorBase {
    friend class CommandQueue;
    friend class CommandBuffer;

private:
    template<typename Pack>
    class Visitor : public vstd::StackAllocatorVisitor {
    public:
        CommandAllocator *self;
        uint64 Allocate(uint64 size) override;
        vstd::unique_ptr<Pack> Create(uint64 size);
        void DeAllocate(uint64 handle) override;
    };
    class DescHeapVisitor : public vstd::StackAllocatorVisitor {
    public:
        D3D12_DESCRIPTOR_HEAP_TYPE type;
        Device *device;
        uint64 Allocate(uint64 size) override;
        void DeAllocate(uint64 handle) override;
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
    ~CommandAllocator();
    DefaultBuffer const *AllocateScratchBuffer(size_t targetSize);
    BufferView GetTempReadbackBuffer(uint64 size, size_t align = 0);
    BufferView GetTempUploadBuffer(uint64 size, size_t align = 0);
    BufferView GetTempDefaultBuffer(uint64 size, size_t align = 0);
    void Reset(CommandQueue *queue) override;
    KILL_COPY_CONSTRUCT(CommandAllocator)
    KILL_MOVE_CONSTRUCT(CommandAllocator)
};
class IPipelineEvent : public vstd::IOperatorNewBase {
public:
    virtual ~IPipelineEvent() = default;
};
}// namespace toolhub::directx
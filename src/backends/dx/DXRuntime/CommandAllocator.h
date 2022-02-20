#pragma once
#include <vstl/functional.h>
#include <DXRuntime/CommandBuffer.h>
#include <vstl/StackAllocator.h>
#include <Resource/UploadBuffer.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/ReadbackBuffer.h>
#include <vstl/LockFreeArrayQueue.h>
namespace toolhub::directx {
class CommandQueue;
class IPipelineEvent;
class CommandAllocator final : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    friend class CommandBuffer;

private:
    static constexpr size_t TEMP_SIZE = 4ull * 1024ull * 1024ull;
    template<typename Pack>
    class Visitor : public vstd::StackAllocatorVisitor {
    public:
        CommandAllocator *self;
        uint64 Allocate(uint64 size) override;
        void DeAllocate(uint64 handle) override;
    };
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    vstd::unique_ptr<CommandBuffer> cbuffer;
    vstd::LockFreeArrayQueue<vstd::move_only_func<void()>> executeAfterComplete;
    Visitor<ReadbackBuffer> rbVisitor;
    Visitor<DefaultBuffer> dbVisitor;
    Visitor<UploadBuffer> ubVisitor;
    vstd::StackAllocator uploadAllocator;
    vstd::StackAllocator defaultAllocator;
    vstd::StackAllocator readbackAllocator;
    Device *device;
    D3D12_COMMAND_LIST_TYPE type;
    IGpuAllocator *resourceAllocator;
    vstd::unique_ptr<DefaultBuffer> scratchBuffer;
    //TODO: allocate commandbuffer
    CommandAllocator(Device *device, IGpuAllocator *resourceAllocator, D3D12_COMMAND_LIST_TYPE type);
    void Execute(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    void Complete(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex);
    vstd::StackAllocator::Chunk Allocate(vstd::StackAllocator &allocator, uint64 size, size_t align);

public:
    ~CommandAllocator();
    ID3D12CommandAllocator *Allocator() const { return allocator.Get(); }
    D3D12_COMMAND_LIST_TYPE Type() const { return type; }
    void Reset(CommandQueue *queue);
    template<typename Func>
        requires(std::is_constructible_v<vstd::function<void()>, Func &&>)
    void ExecuteAfterComplete(Func &&func) {
        executeAfterComplete.Push(std::forward<Func>(func));
    }
    DefaultBuffer const *AllocateScratchBuffer(size_t targetSize);
    CommandBuffer* GetBuffer() const;
    BufferView GetTempReadbackBuffer(uint64 size, size_t align = 0);
    BufferView GetTempUploadBuffer(uint64 size, size_t align = 0);
    BufferView GetTempDefaultBuffer(uint64 size, size_t align = 0);
    KILL_COPY_CONSTRUCT(CommandAllocator)
    KILL_MOVE_CONSTRUCT(CommandAllocator)
};
class IPipelineEvent : public vstd::IOperatorNewBase {
public:
    virtual ~IPipelineEvent() = default;
};
}// namespace toolhub::directx
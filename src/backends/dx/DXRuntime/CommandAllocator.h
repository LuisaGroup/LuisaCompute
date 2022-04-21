#pragma once
#include <DXRuntime/CommandAllocatorBase.h>
namespace toolhub::directx {
class CommandQueue;
class IPipelineEvent;
class CommandAllocator final : public CommandAllocatorBase {
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

    Visitor<ReadbackBuffer> rbVisitor;
    Visitor<DefaultBuffer> dbVisitor;
    Visitor<UploadBuffer> ubVisitor;
    vstd::StackAllocator uploadAllocator;
    vstd::StackAllocator defaultAllocator;
    vstd::StackAllocator readbackAllocator;
    vstd::unique_ptr<DefaultBuffer> scratchBuffer;
    //TODO: allocate commandbuffer
    CommandAllocator(Device *device, IGpuAllocator *resourceAllocator, D3D12_COMMAND_LIST_TYPE type);
    vstd::StackAllocator::Chunk Allocate(vstd::StackAllocator &allocator, uint64 size, size_t align);

public:
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
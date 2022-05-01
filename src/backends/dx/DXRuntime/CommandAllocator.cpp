
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandQueue.h>
namespace toolhub::directx {
void CommandAllocatorBase::Execute(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    ID3D12CommandList *cmdList = cbuffer->CmdList();
    queue->Queue()->ExecuteCommandLists(
        1,
        &cmdList);
    ThrowIfFailed(queue->Queue()->Signal(fence, fenceIndex));
}
void CommandAllocatorBase::ExecuteAndPresent(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex, IDXGISwapChain3 *swapchain) {
    ID3D12CommandList *cmdList = cbuffer->CmdList();
    queue->Queue()->ExecuteCommandLists(
        1,
        &cmdList);
    ThrowIfFailed(swapchain->Present(0, 0));
    ThrowIfFailed(queue->Queue()->Signal(fence, fenceIndex));
}

void CommandAllocatorBase::Complete(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    uint64 completeValue;
    device->WaitFence(fence, fenceIndex);
    while (auto evt = executeAfterComplete.Pop()) {
        (*evt)();
    }
}
void CommandAllocatorBase::Complete_Async(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    uint64 completeValue;
    device->WaitFence_Async(fence, fenceIndex);
    while (auto evt = executeAfterComplete.Pop()) {
        (*evt)();
    }
}


CommandBuffer *CommandAllocatorBase::GetBuffer() const {
    return cbuffer;
}
CommandAllocatorBase::CommandAllocatorBase(
    Device *device,
    IGpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : type(type),
      resourceAllocator(resourceAllocator),
      device(device) {
    ThrowIfFailed(
        device->device->CreateCommandAllocator(type, IID_PPV_ARGS(allocator.GetAddressOf())));
    cbuffer.New(
        device,
        this);
}
CommandAllocator::CommandAllocator(
    Device *device,
    IGpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : CommandAllocatorBase(device, resourceAllocator, type),
      uploadAllocator(TEMP_SIZE, &ubVisitor),
      readbackAllocator(TEMP_SIZE, &rbVisitor),
      defaultAllocator(TEMP_SIZE, &dbVisitor) {
    rbVisitor.self = this;
    ubVisitor.self = this;
    dbVisitor.self = this;
}
CommandAllocator::~CommandAllocator() {
    cbuffer.Delete();
}
void CommandAllocatorBase::Reset(CommandQueue *queue) {
    ThrowIfFailed(
        allocator->Reset());
    cbuffer->Reset();
}
void CommandAllocator::Reset(CommandQueue *queue) {
    readbackAllocator.Clear();
    uploadAllocator.Clear();
    defaultAllocator.Clear();
    CommandAllocatorBase::Reset(queue);
}

DefaultBuffer const *CommandAllocator::AllocateScratchBuffer(size_t targetSize) {
    if (scratchBuffer) {
        if (scratchBuffer->GetByteSize() < targetSize) {
            size_t allocSize = scratchBuffer->GetByteSize();
            while (allocSize < targetSize) {
                allocSize = std::max<size_t>(allocSize + 1, allocSize * 1.5f);
            }
            executeAfterComplete.Push([s = std::move(scratchBuffer)]() {});
            allocSize = CalcAlign(allocSize, 65536);
            scratchBuffer = vstd::create_unique(new DefaultBuffer(device, allocSize, device->defaultAllocator, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        return scratchBuffer.get();
    } else {
        targetSize = CalcAlign(targetSize, 65536);
        scratchBuffer = vstd::create_unique(new DefaultBuffer(device, targetSize, device->defaultAllocator, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        return scratchBuffer.get();
    }
}

template<typename Pack>
uint64 CommandAllocator::Visitor<Pack>::Allocate(uint64 size) {
    auto packPtr = new Pack(
        self->device,
        size,
        self->resourceAllocator);
    return reinterpret_cast<uint64>(packPtr);
}
template<typename Pack>
void CommandAllocator::Visitor<Pack>::DeAllocate(uint64 handle) {
    delete reinterpret_cast<Pack *>(handle);
}
vstd::StackAllocator::Chunk CommandAllocator::Allocate(
    vstd::StackAllocator &allocator,
    uint64 size,
    size_t align) {
    if (align <= 1) {
        return allocator.Allocate(size);
    }
    return allocator.Allocate(size, align);
}

BufferView CommandAllocator::GetTempReadbackBuffer(uint64 size, size_t align) {
    auto chunk = Allocate(readbackAllocator, size, align);
    auto package = reinterpret_cast<ReadbackBuffer *>(chunk.handle);
    return {
        package,
        chunk.offset,
        size};
}

BufferView CommandAllocator::GetTempUploadBuffer(uint64 size, size_t align) {
    auto chunk = Allocate(uploadAllocator, size, align);
    auto package = reinterpret_cast<UploadBuffer *>(chunk.handle);
    return {
        package,
        chunk.offset,
        size};
}
BufferView CommandAllocator::GetTempDefaultBuffer(uint64 size, size_t align) {
    auto chunk = Allocate(defaultAllocator, size, align);
    auto package = reinterpret_cast<DefaultBuffer *>(chunk.handle);
    return {
        package,
        chunk.offset,
        size};
}
CommandAllocatorBase::~CommandAllocatorBase() {
}
}// namespace toolhub::directx
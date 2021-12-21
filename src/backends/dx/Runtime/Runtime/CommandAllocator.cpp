#pragma vengine_package vengine_directx
#include <Runtime/CommandAllocator.h>
namespace toolhub::directx {
namespace detail {

}// namespace detail
void CommandAllocator::CollectBuffer(CommandBuffer *buffer) {
    bufferPool.push_back(buffer);
}
void CommandAllocator::Execute(
    ID3D12CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    queue->ExecuteCommandLists(
        executeCache.size(),
        executeCache.data());
    ThrowIfFailed(queue->Signal(fence, fenceIndex));
    ThrowIfFailed(queue->Wait(fence, fenceIndex));
}
void CommandAllocator::Complete(
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    if (fence->GetCompletedValue() < fenceIndex) {
        LPCWSTR falseValue = 0;
        HANDLE eventHandle = CreateEventEx(nullptr, falseValue, false, EVENT_ALL_ACCESS);
        auto disp = vstd::create_disposer([&] { CloseHandle(eventHandle); });
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
    }
    while (auto evt = executeAfterComplete.Pop()) {
        (*evt)();
    }
    tempEvent.Clear();
}
vstd::unique_ptr<CommandBuffer> CommandAllocator::GetBuffer() {
    auto dev = [&] {
        if (bufferPool.empty())
            return bufferAllocator.New(device, this);
        else
            return bufferPool.erase_last();
    }();
    executeCache.push_back(dev->CmdList());
    return vstd::create_unique(dev);
}
CommandAllocator::CommandAllocator(
    Device *device,
    IGpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : bufferAllocator(32, false),
      type(type),
      resourceAllocator(resourceAllocator),
      device(device),
      uploadAllocator(TEMP_SIZE, &ubVisitor),
      readbackAllocator(TEMP_SIZE, &rbVisitor),
      defaultAllocator(TEMP_SIZE, &dbVisitor) {
    rbVisitor.self = this;
    ubVisitor.self = this;
    ThrowIfFailed(
        device->device->CreateCommandAllocator(type, IID_PPV_ARGS(allocator.GetAddressOf())));
}
CommandAllocator::~CommandAllocator() {
    for (auto &&i : bufferPool) {
        i->~CommandBuffer();
    }
}
IPipelineEvent *CommandAllocator::AddOrGetTempEvent(void const *ptr, vstd::function<IPipelineEvent *()> const &func) {
    auto ite = tempEvent.Emplace(ptr, vstd::LazyEval<vstd::function<IPipelineEvent *()> const &>(func));
    return ite.Value().get();
}
void CommandAllocator::Reset() {
    readbackAllocator.Clear();
    uploadAllocator.Clear();
    defaultAllocator.Clear();
    ThrowIfFailed(
        allocator->Reset());
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
BufferView CommandAllocator::GetTempReadbackBuffer(uint64 size) {
    auto chunk = readbackAllocator.Allocate(size);
    auto package = reinterpret_cast<ReadbackBuffer *>(chunk.handle);
    return {
        package,
        chunk.offset,
        size};
}
BufferView CommandAllocator::GetTempUploadBuffer(uint64 size) {
    auto chunk = uploadAllocator.Allocate(size);
    auto package = reinterpret_cast<UploadBuffer *>(chunk.handle);
    return {
        package,
        chunk.offset,
        size};
}
BufferView CommandAllocator::GetTempDefaultBuffer(uint64 size) {
    auto chunk = defaultAllocator.Allocate(size);
    auto package = reinterpret_cast<UploadBuffer *>(chunk.handle);
    return {
        package,
        chunk.offset,
        size};
}

}// namespace toolhub::directx
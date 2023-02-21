#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/CommandQueue.h>
namespace toolhub::directx {
template<typename Pack>
uint64 CommandAllocator::Visitor<Pack>::Allocate(uint64 size) {
    auto packPtr = new Pack(
        self->device,
        size,
        self->resourceAllocator);
    return reinterpret_cast<uint64>(packPtr);
}
template<typename Pack>
vstd::unique_ptr<Pack> CommandAllocator::Visitor<Pack>::Create(uint64 size) {
    return vstd::make_unique<Pack>(
        self->device,
        size,
        self->resourceAllocator);
}

template<typename Pack>
void CommandAllocator::Visitor<Pack>::DeAllocate(uint64 handle) {
    delete reinterpret_cast<Pack *>(handle);
}
template<typename T>
void CommandAllocator::BufferAllocator<T>::Clear() {
    largeBuffers.clear();
    alloc.Dispose();
}
template<typename T>
CommandAllocator::BufferAllocator<T>::BufferAllocator(size_t initCapacity)
    : alloc(initCapacity, &visitor) {
}
template<typename T>
CommandAllocator::BufferAllocator<T>::~BufferAllocator() {
}
template<typename T>
BufferView CommandAllocator::BufferAllocator<T>::Allocate(size_t size) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.Allocate(size);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = largeBuffers.emplace_back(visitor.Create(size));
        return BufferView(v.get(), 0, size);
    }
}
template<typename T>
BufferView CommandAllocator::BufferAllocator<T>::Allocate(size_t size, size_t align) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.Allocate(size, align);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = largeBuffers.emplace_back(visitor.Create(size));
        return BufferView(v.get(), 0, size);
    }
}
// void CommandAllocatorBase::WaitExternQueue(ID3D12Fence *fence, uint64 fenceIndex) {
//     if (device->deviceSettings) {
//         auto after_queue = device->deviceSettings->GetQueue();
//         if (after_queue) {
//             after_queue->Wait(fence, fenceIndex);
//         }
//     }
// }
void CommandAllocatorBase::Execute(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    ID3D12CommandList *cmdList = cbuffer->CmdList();
    if (cbuffer->ContainedCmdList()) {
        queue->Queue()->ExecuteCommandLists(
            1,
            &cmdList);
        ThrowIfFailed(queue->Queue()->Signal(fence, fenceIndex));
    } else {
        device->deviceSettings->SignalAfterCmdDispatchFence(fence, fenceIndex);
    }
}
void CommandAllocatorBase::ExecuteAndPresent(CommandQueue *queue, ID3D12Fence *fence, uint64 fenceIndex, IDXGISwapChain3 *swapchain, bool vsync) {
    auto present = [&]() {
        if (vsync) {
            ThrowIfFailed(swapchain->Present(1, 0));
        } else {
            ThrowIfFailed(swapchain->Present(0, DXGI_PRESENT_ALLOW_TEARING));
        }
    };
    ID3D12CommandList *cmdList = cbuffer->CmdList();
    if (cbuffer->ContainedCmdList()) {
        queue->Queue()->ExecuteCommandLists(
            1,
            &cmdList);
        present();
        ThrowIfFailed(queue->Queue()->Signal(fence, fenceIndex));
    } else {
        present();
        device->deviceSettings->SignalAfterCmdDispatchFence(fence, fenceIndex);
    }
}

void CommandAllocatorBase::Complete(
    CommandQueue *queue,
    ID3D12Fence *fence,
    uint64 fenceIndex) {
    device->WaitFence(fence, fenceIndex);
    while (auto evt = executeAfterComplete.Pop()) {
        (*evt)();
    }
}

CommandBuffer *CommandAllocatorBase::GetBuffer() const {
    return cbuffer;
}
CommandAllocatorBase::CommandAllocatorBase(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : device(device),
      type(type),
      resourceAllocator(resourceAllocator) {
    ThrowIfFailed(
        device->device->CreateCommandAllocator(type, IID_PPV_ARGS(allocator.GetAddressOf())));
    cbuffer.create(
        device,
        this);
}
static size_t TEMP_SIZE = 1024ull * 1024ull;
CommandAllocator::CommandAllocator(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : CommandAllocatorBase(device, resourceAllocator, type),
      rtvVisitor(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV),
      dsvVisitor(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV),
      uploadAllocator(TEMP_SIZE),
      defaultAllocator(TEMP_SIZE),
      readbackAllocator(TEMP_SIZE),
      rtvAllocator(64, &rtvVisitor),
      dsvAllocator(64, &dsvVisitor) {
    uploadAllocator.visitor.self = this;
    defaultAllocator.visitor.self = this;
    readbackAllocator.visitor.self = this;
}

CommandAllocator::~CommandAllocator() {
    cbuffer.destroy();
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
    rtvAllocator.Clear();
    dsvAllocator.Clear();
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
            scratchBuffer = vstd::create_unique(new DefaultBuffer(device, allocSize, device->defaultAllocator.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        return scratchBuffer.get();
    } else {
        targetSize = CalcAlign(targetSize, 65536);
        scratchBuffer = vstd::create_unique(new DefaultBuffer(device, targetSize, device->defaultAllocator.get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        return scratchBuffer.get();
    }
}

BufferView CommandAllocator::GetTempReadbackBuffer(uint64 size, size_t align) {
    if (align <= 1) [[likely]] {
        return readbackAllocator.Allocate(size);
    } else {
        return readbackAllocator.Allocate(size, align);
    }
}

BufferView CommandAllocator::GetTempUploadBuffer(uint64 size, size_t align) {
    if (align <= 1) [[likely]] {
        return uploadAllocator.Allocate(size);
    } else {
        return uploadAllocator.Allocate(size, align);
    }
}
BufferView CommandAllocator::GetTempDefaultBuffer(uint64 size, size_t align) {
    if (align <= 1) [[likely]] {
        return defaultAllocator.Allocate(size);
    } else {
        return defaultAllocator.Allocate(size, align);
    }
}

uint64 CommandAllocator::DescHeapVisitor::Allocate(uint64 size) {
    return reinterpret_cast<uint64>(new DescriptorHeap(
        device,
        type,
        size, false));
}
void CommandAllocator::DescHeapVisitor::DeAllocate(uint64 handle) {
    delete reinterpret_cast<DescriptorHeap *>(handle);
}
CommandAllocatorBase::~CommandAllocatorBase() {
}
}// namespace toolhub::directx
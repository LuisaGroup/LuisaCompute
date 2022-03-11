#pragma vengine_package vengine_directx
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/IGpuAllocator.h>
#include <Api/LCEvent.h>
namespace toolhub::directx {
CommandQueue::CommandQueue(
    Device *device,
    IGpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : device(device),
      type(type),
      resourceAllocator(resourceAllocator), thd([this] { ExecuteThread(); }) {
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = type;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
    ThrowIfFailed(device->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(queue.GetAddressOf())));
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&cmdFence)));
}
CommandQueue::AllocatorPtr CommandQueue::CreateAllocator(size_t maxAllocCount) {
    if (maxAllocCount != std::numeric_limits<size_t>::max()) {
        std::unique_lock lck(mtx);
        while (lastFrame - executedFrame > maxAllocCount) {
            mainCv.wait(lck);
        }
    }
    auto newPtr = allocatorPool.Pop();
    if (newPtr) {
        return std::move(*newPtr);
    }
    return AllocatorPtr(new CommandAllocator(device, resourceAllocator, type));
}
void CommandQueue::AddEvent(LCEvent const *evt) {
    executedAllocators.Push(evt, uint64(evt->fenceIndex));
    waitCv.notify_one();
}
void CommandQueue::AddEvent(LCEvent const *evt, luisa::move_only_function<void()> &&func) {
    executedAllocators.Push(new StreamCallback(evt, uint64(evt->fenceIndex), std::move(func)));
    waitCv.notify_one();
}

void CommandQueue::ExecuteThread() {
    while (enabled) {
        auto ExecuteAllocator = [&](AllocatorPtr &b) {
            b->Complete(this, cmdFence.Get(), executedFrame + 1);
            b->Reset(this);
            allocatorPool.Push(std::move(b));
            executedFrame++;
            mainCv.notify_all();
        };
        auto ExecuteEvent = [&](std::pair<LCEvent const *, uint64> &pair) {
            auto evt = pair.first;
            auto tarFrame = pair.second;
            evt->SyncTarget(tarFrame);
            {
                std::lock_guard lck(evt->globalMtx);
                evt->finishedEvent = tarFrame;
            }
            evt->cv.notify_one();
        };
        auto ExecuteStreamCallback = [&](vstd::unique_ptr < StreamCallback>& ptr) {
            auto &&callback = *ptr;
            callback.evt->SyncTarget(callback.tarFrame);
            callback.callback();
            {
                std::lock_guard lck(callback.evt->globalMtx);
                callback.evt->finishedEvent = callback.tarFrame;
            }
            callback.evt->cv.notify_one();
        };
        while (auto b = executedAllocators.Pop()) {
            b->multi_visit(
                ExecuteAllocator,
                ExecuteEvent,
                ExecuteStreamCallback);
        }
        std::unique_lock lck(mtx);
        while (enabled && executedAllocators.Length() == 0)
            waitCv.wait(lck);
    }
}

CommandQueue::~CommandQueue() {
    {
        std::lock_guard lck(mtx);
        enabled = false;
    }
    waitCv.notify_one();
    thd.join();
}
uint64 CommandQueue::Execute(AllocatorPtr &&alloc) {
    alloc->Execute(this, cmdFence.Get(), lastFrame + 1);
    executedAllocators.Push(std::move(alloc));
    {
        std::lock_guard lck(mtx);
        lastFrame++;
    }
    waitCv.notify_one();
    return lastFrame;
}
void CommandQueue::Complete(uint64 fence) {
    std::unique_lock lck(mtx);
    while (executedFrame < fence) {
        mainCv.wait(lck);
    }
}
}// namespace toolhub::directx
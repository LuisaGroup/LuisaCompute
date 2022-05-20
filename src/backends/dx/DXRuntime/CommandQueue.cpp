
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
            ExecuteDuringWaiting();
            mainCv.wait_for(lck, std::chrono::milliseconds(1));
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
    {
        std::scoped_lock lock{mtx};
    }
    waitCv.notify_one();
}
void CommandQueue::Callback(vstd::move_only_func<void()> &&f) {
    executedAllocators.Push(std::move(f));
    {
        std::scoped_lock lock{mtx};
    }
    waitCv.notify_one();
}
void CommandQueue::ExecuteDuringWaiting() {
    //TODO
}

void CommandQueue::ExecuteThread() {
    while (enabled) {
        auto ExecuteAllocator = [&](std::pair<AllocatorPtr, uint64> &b) {
            b.first->Complete(this, cmdFence.Get(), b.second);
            b.first->Reset(this);
            allocatorPool.Push(std::move(b.first));
            {
                std::scoped_lock lck(mtx);
                executedFrame = b.second;
            }
            mainCv.notify_all();
        };

        auto ExecuteEvent = [&](std::pair<LCEvent const *, uint64> &pair) {
            auto evt = pair.first;
            auto tarFrame = pair.second;
            device->WaitFence(evt->fence.Get(), tarFrame);
            {
                std::scoped_lock lck(evt->globalMtx);
                evt->finishedEvent = tarFrame;
            }
            evt->cv.notify_all();
        };
        while (auto b = executedAllocators.Pop()) {
            b->multi_visit(
                ExecuteAllocator,
                [&](auto &&f) { f(); },
                ExecuteEvent);
        }
        std::unique_lock lck(mtx);
        while (enabled && executedAllocators.Length() == 0)
            waitCv.wait(lck);
    }
}
void CommandQueue::ForceSync(
    AllocatorPtr &alloc,
    CommandBuffer &cb) {
    cb.Close();
    Complete();
    auto curFrame = ++lastFrame;
    alloc->Execute(this, cmdFence.Get(), curFrame);
    alloc->Complete_Async(this, cmdFence.Get(), curFrame);
    alloc->Reset(this);
    {
        std::scoped_lock lck(mtx);
        executedFrame = curFrame;
    }
    cb.Reset();
}
CommandQueue::~CommandQueue() {
    {
        std::scoped_lock lck(mtx);
        enabled = false;
    }
    waitCv.notify_one();
    thd.join();
}

uint64 CommandQueue::Execute(AllocatorPtr &&alloc) {
    auto curFrame = ++lastFrame;
    alloc->Execute(this, cmdFence.Get(), curFrame);
    executedAllocators.Push(std::move(alloc), curFrame);
    {
        std::scoped_lock lock{mtx};
    }
    waitCv.notify_one();
    return curFrame;
}
void CommandQueue::ExecuteEmpty(AllocatorPtr &&alloc) {
    alloc->Reset(this);
}

uint64 CommandQueue::ExecuteAndPresent(AllocatorPtr &&alloc, IDXGISwapChain3 *swapChain) {
    auto curFrame = ++lastFrame;
    alloc->ExecuteAndPresent(this, cmdFence.Get(), curFrame, swapChain);
    executedAllocators.Push(std::move(alloc), curFrame);
    {
        std::scoped_lock lock{mtx};
    }
    waitCv.notify_one();
    return curFrame;
}

void CommandQueue::Complete(uint64 fence) {
    std::unique_lock lck(mtx);
    while (executedFrame < fence) {
        ExecuteDuringWaiting();
        mainCv.wait_for(lck, std::chrono::milliseconds(1));
    }
}
void CommandQueue::Complete() {
    std::unique_lock lck(mtx);
    while (executedAllocators.Length() > 0) {
        ExecuteDuringWaiting();
        mainCv.wait_for(lck, std::chrono::milliseconds(1));
    }
}

}// namespace toolhub::directx
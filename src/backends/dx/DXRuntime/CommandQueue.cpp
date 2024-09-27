#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <Resource/GpuAllocator.h>
#include <DXApi/LCEvent.h>
namespace lc::dx {
CommandQueue::CommandQueue(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : device(device),
      resourceAllocator(resourceAllocator),
      type(type),
      thd([this] { ExecuteThread(); }) {
    auto CreateQueue = [&] {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = type;
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
        switch (type) {
            case D3D12_COMMAND_LIST_TYPE_DIRECT:
                queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
                break;
            case D3D12_COMMAND_LIST_TYPE_COMPUTE:
            case D3D12_COMMAND_LIST_TYPE_COPY:
                queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
                break;
        }
        ThrowIfFailed(device->device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(queue.GetAddressOf())));
    };
    if (device->deviceSettings) {
        queue = {device->deviceSettings->CreateQueue(type), false};
        if (!queue) [[unlikely]] {
            CreateQueue();
        }
    } else {
        CreateQueue();
    }
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&cmdFence)));
}
CommandQueue::AllocatorPtr CommandQueue::CreateAllocator(size_t maxAllocCount) {
    if (maxAllocCount != std::numeric_limits<uint64>::max()) {
        if (lastFrame > maxAllocCount)
            Complete(lastFrame - maxAllocCount);
    }
    auto newPtr = allocatorPool.pop();
    if (newPtr) {
        (*newPtr)->GetBuffer()->UpdateCommandBuffer(device);
        return std::move(*newPtr);
    }
    return AllocatorPtr(new CommandAllocator(device, resourceAllocator, type));
}

void CommandQueue::AddEvent(LCEvent const *evt, uint64 fenceIdx) {
    ++lastFrame;
    mtx.lock();
    executedAllocators.push(evt, fenceIdx, true);
    mtx.unlock();
    waitCv.notify_one();
}

void CommandQueue::ExecuteThread() {
    while (enabled || executedAllocators.length() != 0) {
        uint64_t fence;
        bool wakeupThread;
        auto Weakup = [&] {
            if (wakeupThread) {
                uint64 prev_value = executedFrame;
                while (prev_value < fence && !executedFrame.compare_exchange_weak(prev_value, fence)) {
                    std::this_thread::yield();
                }
            }
        };
        auto ExecuteAllocator = [&](AllocatorPtr &b) {
            b->Complete(this, cmdFence.Get(), fence);
            b->Reset(this);
            allocatorPool.push(std::move(b));
            Weakup();
        };
        auto ExecuteCallbacks = [&](vstd::vector<vstd::function<void()>> &vec) {
            for (auto &&i : vec) {
                i();
            }
            Weakup();
        };

        auto ExecuteEvent = [&](LCEvent const *evt) {
            device->WaitFence(evt->fence.Get(), fence);
            {
                std::lock_guard lck(evt->eventMtx);
                evt->finishedEvent = std::max(fence, evt->finishedEvent);
            }
            evt->cv.notify_all();
            if (wakeupThread) {
                executedFrame++;
            }
        };
        auto ExecuteHandle = [&](WaitFence) {
            device->WaitFence(cmdFence.Get(), fence);
            Weakup();
        };
        while (true) {
            vstd::optional<CallbackEvent> b;
            {
                std::lock_guard lck{mtx};
                b = executedAllocators.pop();
            }
            if (!b) break;
            fence = b->fence;
            wakeupThread = b->wakeupThread;
            b->evt.multi_visit(
                ExecuteAllocator,
                ExecuteCallbacks,
                ExecuteEvent,
                ExecuteHandle);
        }
        std::unique_lock lck(mtx);
        while (enabled && executedAllocators.length() == 0) {
            waitCv.wait(lck);
        }
    }
}
void CommandQueue::ForceSync(
    AllocatorPtr &alloc,
    CommandBuffer &cb) {
    cb.Close();
    Complete();
    auto curFrame = ++lastFrame;
    alloc->Execute(this, cmdFence.Get(), curFrame);
    alloc->Complete(this, cmdFence.Get(), curFrame);
    alloc->Reset(this);
    executedFrame = curFrame;

    cb.Reset();
}
CommandQueue::~CommandQueue() {
    {
        std::lock_guard lck(mtx);
        enabled = false;
    }
    waitCv.notify_one();
    thd.join();
}
void CommandQueue::WaitFrame(uint64 lastFrame) {
    if (lastFrame > 0)
        queue->Wait(cmdFence.Get(), lastFrame);
}
void CommandQueue::Signal() {
    auto curFrame = ++lastFrame;
    ThrowIfFailed(queue->Signal(cmdFence.Get(), curFrame));
    mtx.lock();
    executedAllocators.push(WaitFence{}, curFrame, true);
    mtx.unlock();
    waitCv.notify_one();
}
void CommandQueue::Execute(AllocatorPtr &&alloc) {
    auto curFrame = ++lastFrame;
    alloc->Execute(this, cmdFence.Get(), curFrame);
    mtx.lock();
    executedAllocators.push(std::move(alloc), curFrame, true);
    mtx.unlock();
    waitCv.notify_one();
}
void CommandQueue::ExecuteCallbacks(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks) {
    auto curFrame = ++lastFrame;
    alloc->Execute(this, cmdFence.Get(), curFrame);
    mtx.lock();
    executedAllocators.push(std::move(alloc), curFrame, false);
    executedAllocators.push(std::move(callbacks), curFrame, true);
    mtx.unlock();
    waitCv.notify_one();
}
void CommandQueue::ExecuteEmpty(AllocatorPtr &&alloc) {
    alloc->Reset(this);
    allocatorPool.push(std::move(alloc));
}

void CommandQueue::ExecuteEmptyCallbacks(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks) {
    alloc->Reset(this);
    allocatorPool.push(std::move(alloc));
    auto curFrame = ++lastFrame;
    mtx.lock();
    executedAllocators.push(std::move(callbacks), curFrame, true);
    mtx.unlock();
    waitCv.notify_one();
}

void CommandQueue::ExecuteAndPresent(AllocatorPtr &&alloc, IDXGISwapChain3 *swapChain, bool vsync) {
    auto curFrame = ++lastFrame;
    alloc->ExecuteAndPresent(this, cmdFence.Get(), curFrame, swapChain, vsync);
    mtx.lock();
    executedAllocators.push(std::move(alloc), curFrame, true);
    mtx.unlock();
    waitCv.notify_one();
}

void CommandQueue::Complete(uint64 fence) {
    while (executedFrame < fence) {
        std::this_thread::yield();
    }
}
void CommandQueue::Complete() {
    Complete(lastFrame);
}

}// namespace lc::dx

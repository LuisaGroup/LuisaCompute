
#include <DXApi/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
namespace lc::dx {
LCEvent::LCEvent(Device *device)
    : device(device) {
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence)));
}
LCEvent::~LCEvent() {
}

void LCEvent::Sync() const {
    std::unique_lock lck(eventMtx);
    if (currentThreadSync) {
        uint64_t currentFenceIndex = fenceIndex;
        lck.unlock();
        device->WaitFence(fence.Get(), currentFenceIndex);
        lck.lock();
        finishedEvent = std::max(finishedEvent, currentFenceIndex);
    } else {
        while (finishedEvent < fenceIndex) {
            cv.wait(lck);
        }
    }
}
void LCEvent::Signal(CommandQueue *queue) const {
    std::lock_guard lck(eventMtx);
    currentThreadSync = false;
    queue->Queue()->Signal(fence.Get(), ++fenceIndex);
    queue->AddEvent(this);
}
void LCEvent::Wait(CommandQueue *queue) const {
    std::lock_guard lck(eventMtx);
    queue->Queue()->Wait(fence.Get(), fenceIndex);
}
}// namespace lc::dx
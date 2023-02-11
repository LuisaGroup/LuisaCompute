
#include <DXApi/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
namespace toolhub::directx {
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
    while (finishedEvent < fenceIndex) {
        cv.wait(lck);
    }
}
void LCEvent::Signal(CommandQueue *queue) const {
    this->queue = queue;
    ++fenceIndex;
    queue->Queue()->Signal(fence.Get(), fenceIndex);
    queue->AddEvent(this);
}
void LCEvent::Wait(CommandQueue *queue) const {
    queue->Queue()->Wait(fence.Get(), fenceIndex);
}
}// namespace toolhub::directx
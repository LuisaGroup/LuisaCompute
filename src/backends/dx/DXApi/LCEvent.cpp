
#include <DXApi/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/DStorageCommandQueue.h>
namespace lc::dx {
LCEvent::LCEvent(Device *device)
    : Resource(device) {
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
    std::lock_guard lck(eventMtx);
    queue->Queue()->Signal(fence.Get(), ++fenceIndex);
    queue->AddEvent(this);
}
void LCEvent::Signal(DStorageCommandQueue *queue) const {
    std::lock_guard lck(eventMtx);
    queue->Signal(fence.Get(), fenceIndex);
    queue->AddEvent(this);
}
void LCEvent::Wait(CommandQueue *queue) const {
    std::lock_guard lck(eventMtx);
    queue->Queue()->Wait(fence.Get(), fenceIndex);
}
bool LCEvent::IsComplete() const{
    std::lock_guard lck(eventMtx);
    return finishedEvent >= fenceIndex;
}
}// namespace lc::dx

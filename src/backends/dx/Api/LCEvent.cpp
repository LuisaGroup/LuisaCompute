#pragma vengine_package vengine_directx
#include <Api/LCEvent.h>
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
void LCEvent::SyncTarget(uint64 tar) const {
    if (tar > 0 && fence->GetCompletedValue() < tar) {
        ThrowIfFailed(fence->SetEventOnCompletion(tar, device->EventHandle()));
        WaitForSingleObject(device->EventHandle(), INFINITE);
    }
}

void LCEvent::Sync() const {
    std::unique_lock lck(globalMtx);
    while (finishedEvent < fenceIndex) {
        cv.wait(lck);
    }
}
void LCEvent::Signal(CommandQueue *queue) const {
    ++fenceIndex;
    queue->Queue()->Signal(fence.Get(), fenceIndex);
    queue->AddEvent(this);
}
void LCEvent::Signal(CommandQueue *queue, luisa::move_only_function<void()> &&func) const {
    ++fenceIndex;
    queue->Queue()->Signal(fence.Get(), fenceIndex);
    queue->AddEvent(this, std::move(func));
}

void LCEvent::Wait(CommandQueue *queue) const {
    queue->Queue()->Wait(fence.Get(), fenceIndex);
}
}// namespace toolhub::directx
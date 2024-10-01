#include <DXApi/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/DStorageCommandQueue.h>
namespace lc::dx {
LCEvent::LCEvent(Device *device, bool shared)
    : Resource(device),
      cond_var(!device->useFiber) {
    ThrowIfFailed(device->device->CreateFence(
        0,
        shared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence)));
}
LCEvent::~LCEvent() {
}

void LCEvent::Sync(uint64_t fenceIdx) const {
    cond_var.wait([&]() { return finishedEvent >= fenceIdx; });
}
void LCEvent::Signal(CommandQueue *queue, uint64 fenceIdx) const {
    std::lock_guard lck(cond_var);
    queue->Queue()->Signal(fence.Get(), fenceIdx);
    lastFence = std::max(lastFence, fenceIdx);
    queue->AddEvent(this, fenceIdx);
}
void LCEvent::Signal(DStorageCommandQueue *queue, uint64 fenceIdx) const {
    std::lock_guard lck(cond_var);
    queue->Signal(fence.Get(), fenceIdx);
    lastFence = std::max(lastFence, fenceIdx);
    queue->AddEvent(this, fenceIdx);
}
void LCEvent::Wait(CommandQueue *queue, uint64 fenceIdx) const {
    std::lock_guard lck(cond_var);
    queue->Queue()->Wait(fence.Get(), fenceIdx);
}
bool LCEvent::IsComplete(uint64 fenceIdx) const {
    std::lock_guard lck(cond_var);
    return finishedEvent >= fenceIdx;
}
}// namespace lc::dx

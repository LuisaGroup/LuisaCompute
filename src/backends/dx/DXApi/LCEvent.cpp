#include <DXApi/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/DStorageCommandQueue.h>
#include <luisa/core/intrin.h>
namespace lc::dx {
LCEvent::LCEvent(Device *device, bool shared)
    : Resource(device) {
    ThrowIfFailed(device->device->CreateFence(
        0,
        shared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&_fence)));
}
LCEvent::~LCEvent() {
    if (!_fence) return;
    if (_fence->GetCompletedValue() < last_fence) {
        device->wait_fence(_fence.Get(), last_fence);
    }
}

void LCEvent::sync(uint64_t fenceIdx) const {
    for (;;) {
        if (finished_event.load(std::memory_order_acquire) >= fenceIdx) {
            return;
        }
        for (uint32_t spin = 0u; spin < 4096u; ++spin) {
            if (finished_event.load(std::memory_order_acquire) >= fenceIdx) {
                return;
            }
            LUISA_INTRIN_PAUSE();
        }
        const auto cur = finished_event.load(std::memory_order_acquire);
        if (cur >= fenceIdx) {
            return;
        }
        finished_event.wait(cur, std::memory_order_acquire);
    }
}
void LCEvent::signal(CommandQueue *queue, uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    if (!device->device_settings || !device->device_settings->SignalFence(queue->queue(), _fence.Get(), fenceIdx))
        ThrowIfFailed(queue->queue()->Signal(_fence.Get(), fenceIdx));
    last_fence = std::max(last_fence, fenceIdx);
    queue->add_event(this, fenceIdx);
}
void LCEvent::signal(DStorageCommandQueue *queue, uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    queue->Signal(_fence.Get(), fenceIdx);
    last_fence = std::max(last_fence, fenceIdx);
    queue->AddEvent(this, fenceIdx);
}
void LCEvent::wait(CommandQueue *queue, uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    if (!device->device_settings || !device->device_settings->WaitFence(queue->queue(), _fence.Get(), fenceIdx))
        ThrowIfFailed(queue->queue()->Wait(_fence.Get(), fenceIdx));
}
bool LCEvent::is_complete(uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    return finished_event >= fenceIdx;
}
}// namespace lc::dx

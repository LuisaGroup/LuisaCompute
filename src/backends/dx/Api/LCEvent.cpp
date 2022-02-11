#pragma vengine_package vengine_directx
#include <Api/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
namespace toolhub::directx {
LCEvent::LCEvent(Device *device) {
    ThrowIfFailed(device->device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence)));
}
LCEvent::~LCEvent() {
}
void LCEvent::Sync() {
    if (fenceIndex > 0 && fence->GetCompletedValue() < fenceIndex) {
        LPCWSTR falseValue = 0;
        HANDLE eventHandle = CreateEventEx(nullptr, falseValue, false, EVENT_ALL_ACCESS);
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }
}
void LCEvent::Signal(CommandQueue *queue) {
    ++fenceIndex;
    std::lock_guard lck(queue->GetMutex());
    queue->Queue()->Signal(fence.Get(), fenceIndex);
}
void LCEvent::Wait(CommandQueue *queue) {
    std::lock_guard lck(queue->GetMutex());
    queue->Queue()->Wait(fence.Get(), fenceIndex);
}
}// namespace toolhub::directx
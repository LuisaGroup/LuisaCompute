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
    if (fence->GetCompletedValue() < fenceIndex) {
        LPCWSTR falseValue = 0;
        HANDLE eventHandle = CreateEventEx(nullptr, falseValue, false, EVENT_ALL_ACCESS);
        auto disp = vstd::create_disposer([&] { CloseHandle(eventHandle); });
        ThrowIfFailed(fence->SetEventOnCompletion(fenceIndex, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
    }
}
void LCEvent::Signal(CommandQueue *queue) {
    ++fenceIndex;
    queue->Queue()->Signal(fence.Get(), fenceIndex);
}
void LCEvent::Wait(CommandQueue *queue) {
    queue->Queue()->Wait(fence.Get(), fenceIndex);
}
}// namespace toolhub::directx
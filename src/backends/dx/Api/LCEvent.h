#pragma once
#include <DXRuntime/Device.h>
namespace toolhub::directx {
class CommandQueue;
class LCEvent : public vstd::IOperatorNewBase {
    ComPtr<ID3D12Fence> fence;
    uint64 fenceIndex = 0;

public:
    LCEvent(Device *device);
    ~LCEvent();
    void Sync();
    void Signal(CommandQueue *queue);
    void Wait(CommandQueue *queue);
};
}// namespace toolhub::directx
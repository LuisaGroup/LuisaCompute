#pragma once
#include <DXRuntime/Device.h>
namespace toolhub::directx {
class CommandQueue;
class LCEvent : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    mutable CommandQueue *queue;
    ComPtr<ID3D12Fence> fence;
    mutable uint64 fenceIndex = 1;
    Device *device;
    mutable std::mutex globalMtx;
    mutable std::condition_variable cv;
    mutable uint64 finishedEvent = 1;

public:
    LCEvent(Device *device);
    ~LCEvent();
    void Sync() const;
    void Signal(CommandQueue *queue) const;
    void Wait(CommandQueue *queue) const;
};
}// namespace toolhub::directx
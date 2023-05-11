#pragma once
#include <DXRuntime/Device.h>
namespace lc::dx {
class CommandQueue;
class LCEvent : public vstd::IOperatorNewBase {
    friend class CommandQueue;

public:
    ComPtr<ID3D12Fence> fence;
    mutable size_t fenceIndex = 1;
    Device *device;
    mutable bool currentThreadSync{false};
    mutable std::mutex eventMtx;
    mutable std::condition_variable cv;
    mutable uint64 finishedEvent = 1;

    ID3D12Fence *Fence() const { return fence.Get(); }
    LCEvent(Device *device);
    ~LCEvent();
    void Sync() const;
    void Signal(CommandQueue *queue) const;
    void Wait(CommandQueue *queue) const;
};
}// namespace lc::dx
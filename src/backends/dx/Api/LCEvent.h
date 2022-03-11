#pragma once
#include <DXRuntime/Device.h>
namespace toolhub::directx {
class CommandQueue;
class LCEvent : public vstd::IOperatorNewBase {
    friend class CommandQueue;
    ComPtr<ID3D12Fence> fence;
    mutable std::atomic_uint64_t fenceIndex = 1;
    Device *device;
    mutable std::mutex globalMtx;
    mutable std::condition_variable cv;
    mutable uint64 finishedEvent = 1;
    void SyncTarget(uint64 tar) const;

public:
    LCEvent(Device *device);
    ~LCEvent();
    void Sync() const;
    void Signal(CommandQueue *queue) const;
    void Signal(CommandQueue *queue, luisa::move_only_function<void()> &&func) const;
    void Wait(CommandQueue *queue) const;
};
}// namespace toolhub::directx
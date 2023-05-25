#pragma once
#include <DXRuntime/Device.h>
#include <Resource/Resource.h>
namespace lc::dx {
class CommandQueue;
class DStorageCommandQueue;
class LCEvent : public Resource {
public:
    ComPtr<ID3D12Fence> fence;
    mutable uint64 fenceIndex = 0;
    mutable std::mutex eventMtx;
    mutable std::condition_variable cv;
    mutable uint64 finishedEvent = 0;
    Tag GetTag() const override {return Tag::Event;}
    ID3D12Fence *Fence() const { return fence.Get(); }
    LCEvent(Device *device);
    ~LCEvent();
    void Sync() const;
    void Signal(CommandQueue *queue) const;
    void Signal(DStorageCommandQueue *queue) const;
    void Wait(CommandQueue *queue) const;
};
}// namespace lc::dx
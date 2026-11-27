#pragma once
#include <DXRuntime/Device.h>
#include <Resource/Resource.h>
namespace lc::dx {
class CommandQueue;
class DStorageCommandQueue;
class LCEvent : public Resource {
    /// @brief References held by the frontend handle (one) plus one per queued
    /// command-queue callback (see `CommandQueue::add_event`).
    /// @details Destroying an event is allowed while the submissions that refer to
    /// its timeline are still in flight -- the frontend contract is that retiring an
    /// event retires those submissions -- and the queues publish their completion
    /// into this object from their own thread, so the object has to outlive every
    /// callback that still points at it. `LCDevice::destroy_event` therefore only
    /// drops the handle's own reference; the last callback deletes the event.
    mutable std::atomic_uint32_t _ref_count{1u};

public:
    ComPtr<ID3D12Fence> _fence;
    mutable std::atomic_uint64_t finished_event = 0;
    mutable luisa::spin_mutex event_mtx;
    mutable uint64_t last_fence = 0;
    Tag get_tag() const override { return Tag::Event; }
    ID3D12Fence *fence() const { return _fence.Get(); }
    /// @brief Take an extra reference (a queued command-queue callback).
    void retain() const noexcept { _ref_count.fetch_add(1u, std::memory_order_relaxed); }
    /// @brief Drop a reference, deleting the event with the last one.
    void release() const noexcept {
        if (_ref_count.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
            delete const_cast<LCEvent *>(this);
        }
    }
    LCEvent(Device *device, bool shared = false);
    ~LCEvent();
    void sync(uint64_t fence) const;
    void signal(CommandQueue *queue, uint64_t fenceIdx) const;
    void signal(DStorageCommandQueue *queue, uint64_t fenceIdx) const;
    void wait(CommandQueue *queue, uint64_t fenceIdx) const;
    bool is_complete(uint64_t fenceIdx) const;
};
}// namespace lc::dx

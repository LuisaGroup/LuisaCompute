#pragma once

#include <atomic>

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/stream_event.h>

namespace luisa::compute {

class Device;

class LUISA_RUNTIME_API Event final : public Resource {

public:
    struct LUISA_RUNTIME_API Signal {
        uint64_t handle;
        uint64_t fence;
        void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
    };
    struct LUISA_RUNTIME_API Wait {
        uint64_t handle;
        uint64_t fence;
        void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
    };

private:
    friend class Device;
    friend class DxCudaInterop;
    friend class VkCudaInterop;
    /// \brief Fence this handle has signaled last.
    /// \details Copies of one `Event` are handles to *one* device-side timeline, so they
    /// must agree on the counter: a signal issued through one copy is exactly what a
    /// `wait()`/`synchronize()` through another copy has to wait for, and two copies must
    /// never hand the backend the same fence value (Vulkan requires a signal value
    /// strictly greater than the current one, and a D3D12 fence is monotonic in the same
    /// way). The value therefore lives in `ResourceRefCount::state` as soon as a second
    /// handle exists -- the block is created by the first copy, seeded with the value
    /// below -- and only in this member while the instance is the sole owner.
    mutable std::atomic_uint64_t _fence{0u};

    /// \brief Fence any handle of this event has signaled last.
    [[nodiscard]] uint64_t _last_fence() const noexcept {
        if (auto block = _shared_block()) {
            return block->state.load(std::memory_order_acquire);
        }
        return _fence.load(std::memory_order_acquire);
    }
    /// \brief Claim the next fence value of the timeline all handles share.
    [[nodiscard]] uint64_t _next_fence() const noexcept {
        if (auto block = _shared_block()) {
            return block->state.fetch_add(1u, std::memory_order_acq_rel) + 1u;
        }
        return ++_fence;
    }
    explicit Event(DeviceInterface *device) noexcept;
    explicit Event(DeviceInterface *device, const ResourceCreationInfo &info) noexcept;

public:
    Event() noexcept = default;
    ~Event() noexcept override { _release(); }
    using Resource::operator bool;
    /// \brief Hand the ownership of the underlying event over to the caller.
    /// \details `Resource::release()` dismantles the shared block the fence lives in, so
    /// the value moves back into this instance first: the released handle is invalid, but
    /// it keeps reporting the fence it handed over instead of falling back to zero.
    [[nodiscard]] ResourceCreationInfo release() noexcept {
        _fence.store(_last_fence(), std::memory_order_release);
        return Resource::release();
    }
    Event(Event &&) noexcept;
    Event(Event const &) noexcept;
    Event &operator=(Event &&rhs) noexcept;
    Event &operator=(Event const &rhs) noexcept;
    [[nodiscard]] Signal signal() const noexcept;
    [[nodiscard]] Wait wait(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
    [[nodiscard]] bool is_completed(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
    [[nodiscard]] uint64_t last_fence() const noexcept { return _last_fence(); }
    void synchronize(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;

protected:
    void destroy() noexcept override;
    /// \brief Hand the fence the first copy will share over to the block.
    void _init_shared_state(ResourceRefCount &block) const noexcept override {
        block.state.store(_fence.load(std::memory_order_relaxed),
                          std::memory_order_release);
    }
};

LUISA_MARK_STREAM_EVENT_TYPE(Event::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(Event::Wait)

class LUISA_RUNTIME_API TimelineEvent final : public Resource {

public:
    using Signal = Event::Signal;
    using Wait = Event::Wait;

private:
    friend class Device;
    friend class DxCudaInterop;
    friend class VkCudaInterop;
    explicit TimelineEvent(DeviceInterface *device, const ResourceCreationInfo &info) noexcept;
    explicit TimelineEvent(DeviceInterface *device) noexcept;

public:
    TimelineEvent() noexcept = default;
    ~TimelineEvent() noexcept override { _release(); }
    using Resource::operator bool;
    using Resource::release;
    TimelineEvent(TimelineEvent &&) noexcept;
    TimelineEvent(TimelineEvent const &) noexcept;
    TimelineEvent &operator=(TimelineEvent &&) noexcept;
    TimelineEvent &operator=(TimelineEvent const &) noexcept;
    [[nodiscard]] Signal signal(uint64_t fence) const noexcept;
    [[nodiscard]] Wait wait(uint64_t fence) const noexcept;
    [[nodiscard]] bool is_completed(uint64_t fence) const noexcept;
    void synchronize(uint64_t fence) const noexcept;

protected:
    void destroy() noexcept override;
};

}// namespace luisa::compute

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
    mutable std::atomic_uint64_t _fence{0u};
    explicit Event(DeviceInterface *device) noexcept;
    explicit Event(DeviceInterface *device, const ResourceCreationInfo &info) noexcept;

public:
    Event() noexcept = default;
    ~Event() noexcept override { _release(); }
    using Resource::operator bool;
    using Resource::release;
    Event(Event &&) noexcept;
    Event(Event const &) noexcept;
    Event &operator=(Event &&rhs) noexcept;
    Event &operator=(Event const &rhs) noexcept;
    [[nodiscard]] Signal signal() const noexcept;
    [[nodiscard]] Wait wait(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
    [[nodiscard]] bool is_completed(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
    [[nodiscard]] uint64_t last_fence() const noexcept { return _fence.load(); }
    void synchronize(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;

protected:
    void destroy() noexcept override;
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

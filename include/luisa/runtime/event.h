#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/stream_event.h>

namespace luisa::compute {

class Device;

class LC_RUNTIME_API Event final : public Resource {

public:
    struct LC_RUNTIME_API Signal {
        uint64_t handle;
        uint64_t fence;
        void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
    };
    struct LC_RUNTIME_API Wait {
        uint64_t handle;
        uint64_t fence;
        void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
    };

private:
    friend class Device;
    mutable std::atomic_uint64_t _fence;
    explicit Event(DeviceInterface *device) noexcept;

public:
    Event() noexcept = default;
    ~Event() noexcept override;
    using Resource::operator bool;
    Event(Event &&) noexcept;
    Event(Event const &) noexcept = delete;
    Event &operator=(Event &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Event &operator=(Event const &) noexcept = delete;
    [[nodiscard]] Signal signal() const noexcept;
    [[nodiscard]] Wait wait(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
    [[nodiscard]] bool is_completed(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
    [[nodiscard]] uint64_t last_fence() const noexcept { return _fence.load(); }
    void synchronize(uint64_t fence = std::numeric_limits<uint64_t>::max()) const noexcept;
};

LUISA_MARK_STREAM_EVENT_TYPE(Event::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(Event::Wait)

class LC_RUNTIME_API TimelineEvent final : public Resource {

public:
    using Signal = Event::Signal;
    using Wait = Event::Wait;

private:
    friend class Device;
    explicit TimelineEvent(DeviceInterface *device) noexcept;

public:
    TimelineEvent() noexcept = default;
    ~TimelineEvent() noexcept override;
    using Resource::operator bool;
    TimelineEvent(TimelineEvent &&) noexcept;
    TimelineEvent(TimelineEvent const &) noexcept = delete;
    TimelineEvent &operator=(TimelineEvent &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    TimelineEvent &operator=(TimelineEvent const &) noexcept = delete;
    [[nodiscard]] Signal signal(uint64_t fence) const noexcept;
    [[nodiscard]] Wait wait(uint64_t fence) const noexcept;
    [[nodiscard]] bool is_completed(uint64_t fence) const noexcept;
    void synchronize(uint64_t fence) const noexcept;
};

}// namespace luisa::compute

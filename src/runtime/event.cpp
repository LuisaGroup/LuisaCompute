#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>

namespace luisa::compute {

void Event::Signal::operator()(DeviceInterface *device,
                               uint64_t stream_handle) const && noexcept {
    device->signal_event(handle, stream_handle, fence);
}

void Event::Wait::operator()(DeviceInterface *device,
                             uint64_t stream_handle) const && noexcept {
    device->wait_event(handle, stream_handle, fence);
}

// counting event
Event Device::create_event() noexcept {
    return _create<Event>();
}

Event::Event(DeviceInterface *device, const ResourceCreationInfo &info) noexcept
    : Resource{device, Tag::EVENT, info},
      _fence{0u} {
}

Event::Event(DeviceInterface *device) noexcept
    : Event{device, [&] {
                auto info = device->create_event();
#ifdef LUISA_ENABLE_SAFE_MODE
                if (!info.valid()) {
                    LUISA_ERROR("Failed to create event.");
                }
#endif
                return info;
            }()} {}

Event::Event(Event &&rhs) noexcept
    : Resource{std::move(rhs)},
      _fence{rhs.last_fence()} {}

Event::Event(Event const &rhs) noexcept
    : Resource{rhs},
      _fence{rhs.last_fence()} {}

Event &Event::operator=(Event &&rhs) noexcept {
    if (this != &rhs) [[likely]] {
        Resource::operator=(std::move(rhs));
        _fence.store(rhs.last_fence());
    }
    return *this;
}

Event &Event::operator=(Event const &rhs) noexcept {
    if (this != &rhs) [[likely]] {
        Resource::operator=(rhs);
        _fence.store(rhs.last_fence());
    }
    return *this;
}

void Event::synchronize(uint64_t fence) const noexcept {
    _check_is_valid();
    if (fence == std::numeric_limits<uint64_t>::max()) { fence = last_fence(); }
    device()->synchronize_event(handle(), fence);
}

void Event::destroy() noexcept {
    if (*this) { device()->destroy_event(handle()); }
}

bool Event::is_completed(uint64_t fence) const noexcept {
    _check_is_valid();
    if (fence == std::numeric_limits<uint64_t>::max()) { fence = last_fence(); }
    return device()->is_event_completed(handle(), fence);
}

[[nodiscard]] Event::Signal Event::signal() const noexcept {
    _check_is_valid();
    return {handle(), ++_fence};
}

[[nodiscard]] Event::Wait Event::wait(uint64_t fence) const noexcept {
    _check_is_valid();
    if (fence == std::numeric_limits<uint64_t>::max()) { fence = last_fence(); }
    return {handle(), fence};
}

// timeline event
TimelineEvent Device::create_timeline_event() noexcept {
    return _create<TimelineEvent>();
}

TimelineEvent::TimelineEvent(DeviceInterface *device, const ResourceCreationInfo &info) noexcept
    : Resource{device, Tag::EVENT, info} {
}

TimelineEvent::TimelineEvent(DeviceInterface *device) noexcept
    : TimelineEvent{device, [&] {
                        auto info = device->create_event();
#ifdef LUISA_ENABLE_SAFE_MODE
                        if (!info.valid()) {
                            LUISA_ERROR("Failed to create timeline event.");
                        }
#endif
                        return info;
                    }()} {}

void TimelineEvent::destroy() noexcept {
    if (*this) { device()->destroy_event(handle()); }
}

TimelineEvent::TimelineEvent(TimelineEvent &&event) noexcept
    : Resource{std::move(event)} {}

TimelineEvent::TimelineEvent(TimelineEvent const &event) noexcept
    : Resource{event} {}

TimelineEvent &TimelineEvent::operator=(TimelineEvent &&event) noexcept {
    Resource::operator=(std::move(event));
    return *this;
}

TimelineEvent &TimelineEvent::operator=(TimelineEvent const &event) noexcept {
    Resource::operator=(event);
    return *this;
}

TimelineEvent::Signal TimelineEvent::signal(uint64_t fence) const noexcept {
    _check_is_valid();
    return {handle(), fence};
}

TimelineEvent::Wait TimelineEvent::wait(uint64_t fence) const noexcept {
    _check_is_valid();
    return {handle(), fence};
}

bool TimelineEvent::is_completed(uint64_t fence) const noexcept {
    _check_is_valid();
    return device()->is_event_completed(handle(), fence);
}

void TimelineEvent::synchronize(uint64_t fence) const noexcept {
    _check_is_valid();
    device()->synchronize_event(handle(), fence);
}

}// namespace luisa::compute

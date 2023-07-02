//
// Created by Mike Smith on 2021/4/8.
//

#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>

namespace luisa::compute {

Event Device::create_event() noexcept {
    return _create<Event>();
}

Event::Event(DeviceInterface *device) noexcept
    : Resource{device, Tag::EVENT, device->create_event()} {}

Event::Event(Event &&rhs) noexcept
    : Resource{std::move(rhs)},
      _fence{rhs.last_fence()} {}

void Event::synchronize(uint64_t fence) const noexcept {
    if (fence == std::numeric_limits<uint64_t>::max()) { fence = last_fence(); }
    device()->synchronize_event(handle(), fence);
}

Event::~Event() noexcept {
    if (*this) { device()->destroy_event(handle()); }
}

void Event::Signal::operator()(DeviceInterface *device,
                               uint64_t stream_handle) const && noexcept {
    device->signal_event(handle, stream_handle, fence);
}

void Event::Wait::operator()(DeviceInterface *device,
                             uint64_t stream_handle) const && noexcept {
    device->wait_event(handle, stream_handle, fence);
}

bool Event::is_completed(uint64_t fence) const noexcept {
    if (fence == std::numeric_limits<uint64_t>::max()) { fence = last_fence(); }
    return device()->is_event_completed(handle(), fence);
}

[[nodiscard]] Event::Signal Event::signal() const noexcept {
    return {handle(), ++_fence};
}

[[nodiscard]] Event::Wait Event::wait(uint64_t fence) const noexcept {
    if (fence == std::numeric_limits<uint64_t>::max()) { fence = last_fence(); }
    return {handle(), fence};
}

}// namespace luisa::compute

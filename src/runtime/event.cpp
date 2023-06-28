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
    : Resource{
          device,
          Tag::EVENT,
          device->create_event()} {}

Event::Event(Event &&rhs) noexcept
    : Resource{std::move(rhs)},
      _fence{rhs._fence.load()} {
}

void Event::synchronize() const noexcept {
    device()->synchronize_event(handle());
}

Event::~Event() noexcept {
    if (*this) { device()->destroy_event(handle()); }
}

void Event::Signal::operator()(
    DeviceInterface *device,
    uint64_t stream_handle) && noexcept {
    device->signal_event(handle, stream_handle, fence);
}

void Event::Wait::operator()(
    DeviceInterface *device,
    uint64_t stream_handle) && noexcept {
    device->wait_event(handle, stream_handle, fence);
}

bool Event::is_completed(uint64_t fence) const noexcept {
    return device()->is_event_completed(handle(), fence);
}

}// namespace luisa::compute

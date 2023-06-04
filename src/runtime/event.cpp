//
// Created by Mike Smith on 2021/4/8.
//

#include <runtime/device.h>
#include <runtime/event.h>

namespace luisa::compute {

Event Device::create_event() noexcept {
    return _create<Event>();
}

Event::Event(DeviceInterface *device) noexcept
    : Resource{
          device,
          Tag::EVENT,
          device->create_event()} {}

void Event::synchronize() const noexcept {
    device()->synchronize_event(handle());
}

Event::~Event() noexcept {
    if (*this) { device()->destroy_event(handle()); }
}
void StreamEvent<Event::Wait>::execute(
    DeviceInterface *device,
    uint64_t stream_handle,
    const Event::Wait &wait) noexcept {
    device->wait_event(wait.handle, stream_handle);
}
void StreamEvent<Event::Signal>::execute(
    DeviceInterface *device,
    uint64_t stream_handle,
    const Event::Signal &signal) noexcept {
    device->signal_event(signal.handle, stream_handle);
}
}// namespace luisa::compute

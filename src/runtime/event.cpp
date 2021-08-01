//
// Created by Mike Smith on 2021/4/8.
//

#include <runtime/device.h>
#include <runtime/event.h>

namespace luisa::compute {

Event Device::create_event() noexcept {
    return _create<Event>();
}

Event::Event(Device::Interface *device) noexcept
    : Resource{
        device,
        Tag::EVENT,
        device->create_event()} {}

void Event::synchronize() const noexcept {
    device()->synchronize_event(handle());
}

}// namespace luisa::compute

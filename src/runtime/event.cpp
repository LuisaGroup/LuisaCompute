//
// Created by Mike Smith on 2021/4/8.
//

#include <runtime/device.h>
#include <runtime/event.h>

namespace luisa::compute {

Event::Event(Device::Handle device) noexcept
    : _device{std::move(device)},
      _handle{_device->create_event()} {}

Event::~Event() noexcept { _destroy(); }

Event &Event::operator=(Event &&rhs) noexcept {
    if (this != &rhs) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
    }
    return *this;
}

void Event::synchronize() const noexcept {
    _device->synchronize_event(_handle);
}

void Event::_destroy() noexcept {
    if (*this) { _device->destroy_event(_handle); }
}

}// namespace luisa::compute

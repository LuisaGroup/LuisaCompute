//
// Created by Mike Smith on 2021/4/8.
//

#include <runtime/device.h>
#include <runtime/event.h>

namespace luisa::compute {

Event::Event(Device &device) noexcept
    : _device{device.impl()},
      _handle{device.impl()->create_event()} {}

Event::~Event() noexcept {
    if (_device != nullptr) {
        _device->destroy_event(_handle);
    }
}

Event::Event(Event &&another) noexcept
    : _device{another._device},
      _handle{another._handle} { another._device = nullptr; }

Event &Event::operator=(Event &&rhs) noexcept {
    if (this != &rhs) {
        _device->destroy_event(_handle);
        _device = rhs._device;
        _handle = rhs._handle;
        rhs._device = nullptr;
    }
    return *this;
}

void Event::synchronize() const noexcept {
    _device->synchronize_event(_handle);
}

}// namespace luisa::compute

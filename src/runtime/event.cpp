//
// Created by Mike Smith on 2021/4/8.
//

#include <runtime/device.h>
#include <runtime/event.h>

namespace luisa::compute {

Event::Event(Device &device) noexcept
    : _device{&device},
      _handle{device.create_event()} {}

Event::~Event() noexcept {
    if (_device != nullptr) {
        _device->dispose_event(_handle);
    }
}

Event::Event(Event &&another) noexcept
    : _device{another._device},
      _handle{another._handle} { another._device = nullptr; }

Event &Event::operator=(Event &&rhs) noexcept {
    if (this != &rhs) {
        _device->dispose_event(_handle);
        _device = rhs._device;
        _handle = rhs._handle;
        rhs._device = nullptr;
    }
    return *this;
}

CommandHandle Event::signal() const noexcept {
    return EventSignalCommand::create(_handle);
}

CommandHandle Event::wait() const noexcept {
    return EventWaitCommand::create(_handle);
}

void Event::synchronize() const noexcept {
    _device->synchronize_event(_handle);
}

}// namespace luisa::compute
